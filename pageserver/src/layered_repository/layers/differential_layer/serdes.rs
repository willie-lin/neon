use anyhow::{anyhow, Result};
use bitmaps::{Bitmap, Bits, BitsImpl};
use log::debug;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::mem::size_of;
use std::num::NonZeroU32;
use std::ops::Deref;
use lazy_static::lazy_static;
use varuint::WriteVarint;
use zenith_utils::bitpacking_lsns::LsnPacker;
use zenith_utils::lsn::Lsn;
use zenith_utils::multi_bitmap::{
    BitmappedMapPage, BitmapStorageType, LayeredBitmap, MBitmap
};
use zenith_utils::vec_map::VecMap;

use crate::layered_repository::filebufutils::{
    get_buf, get_buf_mut, read_slice_from_blocky_file, write_buf,
    write_item_to_blocky_file, write_slice_to_blocky_file, BufPage,
};
use crate::layered_repository::inmemory_layer::InMemoryLayerInner;
use crate::layered_repository::layers::differential_layer::file_format::{BranchPtrData, BranchPtrType, InfoPageV1, InfoPageVersions, LineageBranchInfo, LineageInfo, PageInitiatedLineageBranchHeader, PageOnlyLineageBranchHeader, UnalignFileOffset, VersionInfo, WalInitiatedLineageBranchHeader, WalOnlyLineageBranchHeader};
use crate::layered_repository::layers::differential_layer::LineagePointer;
use crate::layered_repository::layers::LAYER_PAGE_SIZE;
use crate::layered_repository::page_versions::PageVersionPtr;
use crate::layered_repository::storage_layer::{
    AsVersionType, PageVersion, VersionType, RELISH_SEG_SIZE,
};
use crate::{read_from_file, write_to_file, val_to_read, val_to_write};
use crate::branches::BranchInfo;
use crate::layered_repository::layers::differential_layer::wal_stream::WalStream;
use crate::repository::WALRecord;

lazy_static! {
    static ref BITMAP_HEIGHT: usize = f64::from(RELISH_SEG_SIZE as u32).log2().ceil() as usize;
}

pub fn write_latest_metadata<T>(
    file: &mut BufWriter<T>,
    mut offset: u64,
    metadata: &InfoPageV1,
) -> Result<u64>
    where
        T: Write + Seek
{
    let version_number = 1u32;
    return write_to_file!(file, offset, version_number, metadata);
}

pub fn read_metadata<T>(
    file: &mut BufReader<T>,
    mut offset: usize,
    metadata: &mut InfoPageVersions,
) -> Result<usize>
    where
        T: Write + Seek
{
    let mut version_number = 1u32;
    let mut res = read_from_file!(file, offset, version_number);
    res?;

    match version_number {
        1 => {
            let mut metadata_v1: InfoPageV1 = InfoPageV1::default();
            res = read_from_file!(file, offset, metadata_v1);
            res?;
            *metadata = InfoPageVersions::V1(metadata_v1);
        },
        _ => {
            return Err(anyhow!("Unsupported metapage number {}", version_number));
        }
    }

    return res;
}


/// Write out the old page versions section.
/// Uses 128 bytes, plus max 15 bytes alignment and 128+4 bytes per section of 1024
/// blocks in the base relation with changes, plus 8 bytes per changed block.
///
/// base_blockno is assumed to be empty, and the start of this section.
///
/// Access time to one field is O(log(n)) with log base 1028.
/// returns the next free block number.
pub fn write_old_page_versions_section<T>(
    file: &mut BufWriter<T>,
    mut offset: u64,
    previous_versions: &LayeredBitmap<Lsn>,
) -> Result<u64>
    where
        T: Write + Seek
{
    write_bitmap_page_recursive(
        file,
        offset,
        offset,
        previous_versions,
        &|it| it,
    )
}

pub fn write_page_lineages_section<T>(
    file: &mut BufWriter<T>,
    mut offset: u64,
    previous_inner: &InMemoryLayerInner,
    page_lineages: &mut LayeredBitmap<LineagePointer>,
) -> Result<u64>
    where
        T: Write + Seek
{
    let iterator = previous_inner.page_changes_vecmap_iterator();

    let closure = |ptr: &PageVersionPtr| previous_inner.get_page_version(*ptr);

    for (pageno, map) in iterator {
        page_lineages.set(
            (*pageno % RELISH_SEG_SIZE) as usize,
            LineagePointer(UnalignFileOffset::from(offset)),
        );
        offset = write_vecmap_lineage_section(
            file,
            offset,
            map,
            &closure,
        )?;
    }

    return Ok(offset);
}

/// Write out one lineage from the lineage section.
///
/// Note that in VecMap, LSN are stored in increasing order, but in the Lineage
/// section, they are stored in decreasing order. This means that we don't generally
/// need to seek to find the latest Lineage, which helps with performance.
/// Note that this is all WAL for one page in one block.
pub fn write_vecmap_lineage_section<'a, F, T>(
    file: &mut BufWriter<F>,
    mut offset: u64,
    map: &VecMap<Lsn, T>,
    to_pageversion: &dyn Fn(&T) -> PageVersion,
) -> Result<u64>
    where
        F: Write + Seek,
        T: Copy + Clone + AsVersionType,
{
    let struct_location: u64 = offset;

    // phase 1: Iterate over the map, and construct branches based on that data.
    let mut branches = Vec::<(Lsn, T, Vec<(Lsn, T)>)>::new();

    {
        let mut active = None;

        for (lsn, version) in map.as_slice().iter() {
            match version.as_version_type() {
                VersionType::Init => {
                    if active.is_some() {
                        branches.push(active.unwrap());
                    }
                    active = Some((*lsn, *version, vec![]));
                }
                VersionType::Delta => {
                    assert!(active.is_some());
                    active.as_mut().unwrap().2.push((*lsn, *version));
                }
            }
        }
        if active.is_some() {
            branches.push(active.unwrap());
        }
    }

    // Note that we expect to receive our WAL records in order of creation (increasing LSN).
    // To get the latest page version close to the seek position, we reverse the order of the
    // branches so that the latest branch pointer is likely to be co-located with the lineage
    // header.
    branches.reverse();
    // remove mut modifier from here on.
    let branches = branches;

    // Phase 2: construct branch info arrays to store in the main Lineage index
    let n_branches = branches.len();
    let mut branch_infos = vec![LineageBranchInfo::default(); n_branches];

    // Phase 3: write the lineage to the file
    let latest_lineage_lsn = {
        let newest_branch = &branches[0];
        if newest_branch.2.len() > 0 {
            newest_branch.2[newest_branch.2.len() - 1].0
        } else {
            newest_branch.0
        }
    };

    let descriptor = LineageInfo {
        latest_lsn: latest_lineage_lsn,
        num_branches: n_branches as u32,
        last_img: 0,
    };

    {
        let branch_infos_slice = &branch_infos[..];
        let res = write_to_file!(file, struct_location, descriptor, branch_infos_slice []);
        offset = res?;
    }

    for (branch_index, (&start_lsn, head, versions)) in branches.iter().enumerate() {
        let branch_ptr_data = write_branch(file, file.stream_position() as u64, head, start_lsn, &versions[..])?;
        branch_infos.push(LineageBranchInfo {
            start_lsn,
            branch_ptr_data,
        });
    }

    {
        let branch_infos_slice = &branch_infos[..];
        let res = write_to_file!(file, struct_location, descriptor, branch_infos_slice []);
        res?;
    }

    return Ok(offset);
}

pub fn write_branch<File>(
    file: &mut BufWriter<File>,
    base_offset: u64,
    head: &PageVersion,
    lsn: Lsn,
    versions: &[(Lsn, &WALRecord)],
) -> Result<BranchPtrData>
where
    File: Write,
{
    if !versions.is_empty() {
        let lsn_compressor = LsnPacker::new(lsn);

        for (lsn, _) in versions {
            lsn_compressor.add_lsn(*lsn);
        }

        let (_, num_items, meta, compressed) = lsn_compressor.finish();

        WriteVarint::<u32>::write_varint(file, num_items as u32)?;
        // We don't need to store the length of the metadata, nor that of the compressed data:
        // The length of the metadata is implied by it's contents and num_items; and the length
        // of the compressed data is implied by the contents of the metadata.
        // Do note, howeveer, that this does mean we need to buffer whilst reading, and need to
        // determine the size of our data whilst processing that data.
        file.write_all(&meta[..])?;
        file.write_all(&compressed[..])?;
    };

    let mut stream = WalStream::new(file);
    let typ = match head {
        PageVersion::Page(bytes) => {
            stream.write_page(bytes)?;
            BranchPtrType::PageImage.with_pagecount(versions.len())
        },
        PageVersion::Wal(rec) => {
            stream.write_record(rec)?;
            BranchPtrType::WalRecord.with_pagecount(versions.len())
        },
    };

    for (_, rec) in versions {
        stream.write_record(rec)?;
    }

    Ok(BranchPtrData {
        item_ptr: UnalignFileOffset::from(base_offset),
        typ,
        extra_typ_info: 0,
    })
}


pub fn write_bitmap_page_recursive<'a, File, T, R, M>(
    file: &mut BufWriter<File>,
    base_offset: u64,
    mut offset: u64,
    page: &BitmappedMapPage<T>,
    mapper: &M,
) -> Result<u64>
where
    File: Write,
    T: Clone,
    R: Copy,
    M: Fn(&T) -> R,
{
    match page {
        BitmappedMapPage::LeafPage { data } => {
            let bitmap = data.page_get_bitmap();
            let values = data.page_get_values();
            let transformed_values: Vec<R> = values.iter().map(|it| mapper(it)).collect();
            let write_sec = &transformed_values[..];

            let res = write_to_file!(file, offset, bitmap [], write_sec []);
            offset = res?;
        }
        BitmappedMapPage::InnerPage { data, .. } => {
            let bitmap = data.page_get_bitmap();
            let start_offset = offset;
            let mut num_items = data.page_len();
            let mut items_arr = vec![0u32; num_items];
            {
                let arr = &mut items_arr[..];
                let res = write_to_file!(file, offset, bitmap [], arr []);
                res?;
            }

            for (i, item) in data.page_get_values().iter().enumerate() {
                offset = write_bitmap_page_recursive(
                    file,
                    offset,
                    offset,
                    item,
                    mapper,
                )?;
                let pos = offset - base_offset;
                items_arr[i] = pos as u32;
            }

            let last_offset = offset;
            offset = start_offset;

            {
                let items = &items_arr[..];
                let res = write_to_file!(file, offset, bitmap [], items []);
                res?;
            }
            offset = last_offset;
        }
    }

    return Ok(offset);
}

/// Write out the new page versions section.
/// Basically a compressed bitmap, which stores one UnalignedFileOffset per bit.
/// Bit positions are relative to the start of the layer's segment.
/// Uses up 128 bytes, plus max 15 bytes alignment and 128+4 bytes per section of 1024
/// blocks in the base relation with changes, plus 6 bytes per block.
///
/// base_blockno is assumed to be empty, and the start of this section.
///
/// Access time to one field is O(log(n)) with log base 1028.
/// returns the next free block number.
pub fn write_page_versions_map<T>(
    file: &mut BufWriter<T>,
    offset: u64,
    map: &LayeredBitmap<LineagePointer>,
) -> Result<u64>
    where
        T: Write + Seek,
{
    write_bitmap_page_recursive(
        file,
        offset,
        offset,
        map.backing_store(),
        &|&ptr| ptr,
    )
}

pub fn read_bitmap_recursive<T, B, R, F>(
    file: &mut BufReader<T>,
    mut offset: usize,
    map: &mut BitmappedMapPage<B>,
    mapper: &F,
) -> Result<()>
    where
        T: Read + Seek,
        B: Clone,
        R: Copy + Default,
        F: Fn(&R) -> B,
{
    let mut bits_store: BitmapStorageType = Default::default();
    let base_offset = offset;
    let res = read_from_file!(file, offset, bits_store);
    let mut offset = res?;

    let bitmap = MBitmap::from_value(bits_store);

    match map {
        BitmappedMapPage::LeafPage { data } => {
            let mut stored = vec![<R as Default>::default(); bitmap.len()];
            {
                let arr = &mut stored[..];
                let res = read_from_file!(file, offset, arr []);
                offset = res?;
            }

            for (i, value) in Iterator::zip(bitmap.into_iter(), stored.iter()) {
                data.page_set(i, mapper(value));
            }
        }
        BitmappedMapPage::InnerPage {
            data,
            n_items,
            layer,
            mem_usage ,
        } => {
            let mut stored = vec![0u32; bitmap.len()];
            {
                let arr = &mut stored[..];
                let res = read_from_file!(file, offset, arr []);
                offset = res?;
            }

            for (i, value) in Iterator::zip(bitmap.into_iter(), stored.iter()) {
                let mut lower_page = BitmappedMapPage::new(*layer - 1);

                read_bitmap_recursive(
                    file,
                    base_offset + *value as usize,
                    &mut lower_page,
                    mapper,
                )?;

                *n_items += lower_page.page_len();
                *mem_usage += lower_page.get_mem_usage();

                data.page_set(i, Box::new(lower_page));
            }
        }
    }

    return Ok(());
}

/// Read out the new page versions section.
/// Basically a compressed bitmap, which stores one UnalignedFileOffset per set bit.
/// Bit positions are block numbers relative to the start of the layer's segment.
/// base_blockno is assumed to be empty, and the start of this section.
///
/// Access time to one field is O(log(n)) with log base 1028.
/// returns the next free block number.
pub fn read_page_versions_map<T>(
    file: &mut BufReader<T>,
    offset: usize,
    map: &mut LayeredBitmap<LineagePointer>,
)
    where
        T: Read + Seek,
{
    let mut bitmap_layer = BitmappedMapPage::<LineagePointer>::new(*BITMAP_HEIGHT);

    let _ = read_bitmap_recursive(
        file,
        offset,
        &mut bitmap_layer,
        &|ptr: &UnalignFileOffset| LineagePointer(*ptr),
    );

    *map = LayeredBitmap::new_from_page(bitmap_layer);
}

/// Read out the old page versions section into the map.
/// Basically a compressed bitmap, which stores one UnalignedFileOffset per set bit.
/// Bit positions are block numbers relative to the start of the layer's segment.
/// base_blockno is assumed to be empty, and the start of this section.
///
/// Note: replaces the map with a new one.
/// Access time to one field is O(log(n)) with log base 1028.
/// returns the next free block number.
pub fn read_old_page_versions_section(
    file: &mut File,
    base_offno: usize,
    map: &mut LayeredBitmap<Lsn>,
) {
    let mut bitmap_layer = BitmappedMapPage::<Lsn>::new(*BITMAP_HEIGHT);

    let _ = read_bitmap_recursive(
        file,
        base_offno,
        &mut bitmap_layer,
        &|lsn: &Lsn| *lsn,
    );

    *map = LayeredBitmap::new_from_page(bitmap_layer);
}

pub fn read_lineage(
    file: &mut File,
    offset: usize,
    restore_point: Option<Lsn>,
    seg_max_lsn: Lsn,
) -> Option<(Vec<(Lsn, PageVersion)>, Lsn)> {
    const PTRS_PER_PAGE: usize = LAYER_PAGE_SIZE / size_of::<LineageBranchInfo>();
    let mut lineage_info: LineageInfo = LineageInfo::default();

    // If we need the latest version, we know that we only need one Branch's info.
    if restore_point.is_none() {
        let mut branch_info = LineageBranchInfo::default();
        let res = read_from_file!(file, offset, lineage_info, branch_info);
        if res.is_err() {
            return None;
        }

        return Some((
            branch_info.iter(file).collect(),
            seg_max_lsn,
        ))
    }

    // safe, because None case is handled above.
    let restore_point = restore_point.unwrap();

    // Note: On the following read, we potentially read past the end of this
    // lineage's pointers. That should be safe, as this should have at least
    // one more page worth of bytes trailing the section.

    // That is _generally_ ok, as this saves us at least one syscall in the
    // general case of requesting (one of) the latest branchinfos.
    let mut branch_ptrs_buf = vec![LineageBranchInfo::default(); PTRS_PER_PAGE];

    {
        let buf = &mut branch_ptrs_buf[..];
        let res = read_from_file!(file, offset, lineage_info, buf []);
        if res.is_err() {
            return None;
        }
    }

    // We always have at least one branch; so this is always safe.
    if branch_ptrs_buf[0].start_lsn < restore_point {
        return Some((
            branch_ptrs_buf[0].iter(file).collect(),
            seg_max_lsn,
        ))
    }

    let num_branches = lineage_info.num_branches;

    assert!(
        lineage_info.num_branches > 0,
        "Each lineage should always have at least one branch"
    );

    let branch_infos_len: usize = (lineage_info.num_branches as usize) * size_of::<LineageBranchInfo>();
    let (mut local_branch_ptrs, _) = branch_ptrs_buf.split_at(PTRS_PER_PAGE.min(num_branches as usize));

    let branch_array_base_offset: usize = offset + size_of::<LineageInfo>();

    fn read_local_branchinfos<'a>(
        file: &File,
        offset: usize,
        max_len: usize,
        branches: &'a mut Vec<LineageBranchInfo>,
        branch_array_base_offset: usize
    ) -> Result<(&'a [LineageBranchInfo], usize)> {
        let page_start_item_offset = (offset / PTRS_PER_PAGE) * PTRS_PER_PAGE;
        let page_start_offset = page_start_item_offset * size_of::<LineageBranchInfo>();

        let n_in_page = PTRS_PER_PAGE.min(max_len - page_start_offset);

        let read_start = page_start_offset + branch_array_base_offset;
        branches.resize(n_in_page, LineageBranchInfo::default());
        let buf = &mut branches[..n_in_page];
        let res = read_from_file!(file, read_start, buf []);
        res?;

        return Ok((buf, page_start_item_offset));
    }

    let mut buf_offset = 0usize;

    let mut min = 0usize;
    let mut max = lineage_info.num_branches as usize - 1;
    let mut mid;
    let mut next_changed_lsn = seg_max_lsn;

    loop {
        mid = (min + max) / 2;

        if (buf_offset..(buf_offset + local_branch_ptrs.len())).contains(&mid) {
            match local_branch_ptrs[mid - buf_offset].start_lsn.cmp(&restore_point) {
                Ordering::Less => {
                    min = mid + 1;
                }
                Ordering::Equal => {
                    min = mid;
                    max = mid;
                }
                Ordering::Greater => {
                    max = mid - 1;
                    next_changed_lsn = local_branch_ptrs[mid - buf_offset].start_lsn;
                }
            }
        } else if mid < buf_offset {
            match local_branch_ptrs[0].start_lsn.cmp(&restore_point) {
                Ordering::Less => {
                    min = buf_offset + 1;
                }
                Ordering::Equal => {
                    min = buf_offset;
                    max = buf_offset;
                }
                Ordering::Greater => {
                    max = buf_offset;
                }
            }
            mid = (min + max) / 2;
            if mid < buf_offset {
                branch_ptrs_buf.clear();
            }
        } else { // mid > buf_offset + local_branch_ptrs.len()
            match local_branch_ptrs[local_branch_ptrs.len() - 1].start_lsn.cmp(&restore_point) {
                Ordering::Less => {
                    min = local_branch_ptrs.len() + buf_offset;
                }
                Ordering::Equal => {
                    min = local_branch_ptrs.len() + buf_offset - 1;
                    max = local_branch_ptrs.len() + buf_offset - 1;
                }
                Ordering::Greater => {
                    max = local_branch_ptrs.len() + buf_offset - 1;
                }
            }
            mid = (min + max) / 2;
            if mid > buf_offset + local_branch_ptrs.len() - 1 {
                let res = read_local_branchinfos(file, mid, branch_infos_len, &mut branch_ptrs_buf, branch_array_base_offset)?;
                local_branch_ptrs = res.0;
                buf_offset = res.1;
            }
        }

        if min >= max {
            break;
        }

        if min >= buf_offset && max < buf_offset + local_branch_ptrs.len() {
            let point = local_branch_ptrs[(min - buf_offset) .. (max - buf_offset)].binary_search_by(|branch_ptr| {
                branch_ptr.start_lsn.cmp(&restore_point).reverse()
            });
            match point {
                Ok(index) => {
                    min = min + index;
                    max = min;
                    mid = min;
                }
                Err(index) => {
                    min = min + index;
                    max = min;
                    mid = min;
                }
            }
            break;
        }

        let res = read_local_branchinfos(file, mid, branch_infos_len, &mut branch_ptrs_buf, branch_array_base_offset)?;
        local_branch_ptrs = res.0;
        buf_offset = res.1;
    }
    let my_branchinfo = &local_branch_ptrs[min - buf_offset];

    let data = extract_branch_data(
        file,
        my_branchinfo,
        Some(restore_point),
        &mut next_changed_lsn,
    );

    return Some((data, next_changed_lsn));
}

/// Find the branch info that we need to restore the block to the LSN in restore_point.
///
/// Stores an LSN in lsn_cutoff for the values that we know of where the returned branchinfo
/// is not valid for: lsn_cutoff > restore_point >= result.start_lsn. Note that this is a hint,
/// not a guarantee.
///
/// restore_point may be None, in which case return the newest page version.
///
/// Note that this lsn_cutoff will not always be set; e.g. when the LSN requested is greater than
/// the highest branch point of any branch it will not be changed.

fn extract_branch_data(
    file: &mut File,
    branch_info: &LineageBranchInfo,
    restore_point: Option<Lsn>,
    lsn_cutoff: &mut Lsn,
) -> Vec<(Lsn, PageVersion)> {
    let mut iter = branch_info.iter(file);

    let mut result = Vec::with_capacity(iter.size_hint().0);

    while let Some((lsn, pageversion)) = iter.next() {
        result.push((lsn, pageversion));

        if let Some(restore_lsn) = restore_point {
            if let Some(peeked) = iter.peek_lsn() {
                if peeked > restore_lsn {
                    *lsn_cutoff = peeked;
                    break;
                }
            }
        }
    }

    return result;
}
