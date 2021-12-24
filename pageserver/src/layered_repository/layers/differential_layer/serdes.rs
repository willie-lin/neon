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
use zenith_utils::lsn::Lsn;
use zenith_utils::multi_bitmap::{BitmappedMapPage, LayeredBitmap};
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

lazy_static! {
    static ref BITMAP_HEIGHT: usize = f64::from(RELISH_SEG_SIZE as u32).log2().ceil() as usize;
}

pub fn write_latest_metadata<T>(
    file: &mut BufWriter<T>,
    mut offset: usize,
    metadata: &InfoPageV1,
) -> Result<usize>
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
    mut offset: usize,
    previous_versions: &LayeredBitmap<Lsn>,
) -> Result<usize>
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
    mut offset: usize,
    previous_inner: &InMemoryLayerInner,
    page_lineages: &mut LayeredBitmap<LineagePointer>,
) -> Result<usize>
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
    mut offset: usize,
    map: &VecMap<Lsn, T>,
    to_pageversion: &dyn Fn(&T) -> PageVersion,
) -> Result<usize>
    where
        F: Write + Seek,
        T: Copy + Clone + AsVersionType,
{
    let struct_location: usize = offset;

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
        res = write_to_file!(file, struct_location, descriptor, branch_infos_slice []);
        offset = res?;
    }

    for (branch_index, (lsn, head, versions)) in branches.iter().enumerate() {
        let mut version_infos = Vec::with_capacity(versions.len());
        let mut data_length = 0u64;

        let head_page_version: PageVersion = to_pageversion(head);

        if versions.len() == 0 {
            let my_start= offset;
            let typ: BranchPtrType;
            match &head_page_version {
                PageVersion::Page(bytes) => {
                    let header = PageOnlyLineageBranchHeader {
                        length: NonZeroU32::try_from(bytes.len() as u32).unwrap(),
                    };
                    let bytes = &bytes[..];
                    let res = write_to_file!(file, offset, header, bytes []);
                    offset = res?;
                    typ = BranchPtrType::PageImage;
                }
                PageVersion::Wal(rec) => {
                    let header = WalOnlyLineageBranchHeader {
                        length: NonZeroU32::try_from(rec.rec.len() as u32).unwrap(),
                        main_data_offset: rec.main_data_offset,
                    };
                    let bytes = &rec.rec[..];
                    let res = write_to_file!(file, offset, header, bytes []);
                    offset = res?;
                    typ = BranchPtrType::WalRecord;
                }
            };

            branch_infos[branch_index].branch_ptr_data.item_ptr =
                UnalignFileOffset::from(my_start);
            branch_infos[branch_index].branch_ptr_data.typ = typ;
            branch_infos[branch_index].start_lsn = *lsn;
            continue; // that's it for this branch
        }

        // Note: this allocates all WALRecords into memory, which is not great.
        // The alternative, however, is to double-parse the WAL records, which is
        // random IO and not great as well.
        let mapped_tail_versions = versions
            .iter()
            .map(|(lsn, rec)| {
                (
                    *lsn,
                    match to_pageversion(rec) {
                        PageVersion::Page(_) => Err(anyhow!("PageVersion::Page in tail should be impossible"))?,
                        PageVersion::Wal(rec) => rec,
                    },
                )
            })
            .collect::<Vec<_>>();

        for (local_lsn, record) in mapped_tail_versions.iter() {
            let length = NonZeroU32::try_from(record.rec.len() as u32).unwrap();
            let main_data_offset = record.main_data_offset;

            version_infos.push(VersionInfo {
                lsn: local_lsn.0.to_ne_bytes(),
                length,
                main_data_offset,
            });

            data_length += u32::from(length) as u64;
        }
        let typ: BranchPtrType;
        let header_bytes: &[u8];
        let struct_start = offset;
        offset = match &head_page_version {
            PageVersion::Page(bytes) => {
                let header = PageInitiatedLineageBranchHeader {
                    num_entries: NonZeroU32::try_from(versions.len() as u32)?,
                    head: PageOnlyLineageBranchHeader {
                        length: NonZeroU32::try_from(bytes.len() as u32)?,
                    },
                };
                typ = BranchPtrType::PageInitiatedBranch;
                header_bytes = &bytes[..];
                write_to_file!(file, offset, header)
            }
            PageVersion::Wal(rec) => {
                typ = BranchPtrType::WalInitiatedBranch;

                let header = WalInitiatedLineageBranchHeader {
                    num_entries: NonZeroU32::try_from(versions.len() as u32)?,
                    head: WalOnlyLineageBranchHeader {
                        length: NonZeroU32::try_from(rec.rec.len() as u32)?,
                        main_data_offset: rec.main_data_offset,
                    },
                };

                header_bytes = &rec.rec[..];
                write_to_file!(file, offset, header)
            }
        }?;
        branch_infos[branch_index].branch_ptr_data.item_ptr =
            UnalignFileOffset::from(struct_start);
        branch_infos[branch_index].branch_ptr_data.typ = typ;
        branch_infos[branch_index].start_lsn = *lsn;

        {
            let arr = &version_infos[..];
            let res = write_to_file!(file, offset, version_infos [], header_bytes []);
            offset = res?;
        }

        for (_, rec) in mapped_tail_versions {
            let bytes = &rec.rec[..];
            let res = write_to_file!(file, offset, bytes []);
            offset = res?;
        }
    }

    {
        let branch_infos_slice = &branch_infos[..];
        res = write_to_file!(file, struct_location, descriptor, branch_infos_slice []);
        res?;
    }

    return Ok(offset);
}

pub fn write_bitmap_page_recursive<'a, File, T, R, M>(
    file: &mut BufWriter<File>,
    base_offset: usize,
    mut offset: usize,
    page: &BitmappedMapPage<T>,
    mapper: &M,
) -> Result<usize>
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
    offset: usize,
    map: &LayeredBitmap<LineagePointer>,
) -> Result<usize>
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
    let mut bits_store: <BitsImpl<RELISH_SEG_SIZE> as Bits>::Store = Default::default();
    let base_offset = offset;
    let res = read_from_file!(file, offset, bitmap);
    let offset = res?;

    let bitmap = Bitmap::<RELISH_SEG_SIZE>::from_value(bits_store);

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
    file: &File,
    offset: usize,
    restore_point: Option<Lsn>,
    seg_max_lsn: Lsn,
) -> Option<(Vec<(Lsn, PageVersion)>, Lsn)> {
    const PTRS_PER_PAGE: usize = LAYER_BLOCK_SIZE / size_of::<LineageBranchInfo>();
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

    if local_branch_ptrs[0].start_lsn < restore_point {
        return Some((
            local_branch_ptrs[0].iter(file).collect(),
            seg_max_lsn,
        ))
    }

    let num_branches = lineage_info.num_branches;

    assert!(
        lineage_info.num_branches > 0,
        "Each lineage should always have at least one branch"
    );

    const BRANCH_INFOS_LEN: usize = (lineage_info.num_branches as usize) * size_of::<LineageBranchInfo>();
    let (mut local_branch_ptrs, _) = branch_ptrs_buf.split_at(PTRS_PER_PAGE.min(num_branches as usize));

    const BRANCH_ARRAY_BASE_OFFSET: usize = offset + size_of::<LineageInfo>();

    fn read_local_branchinfos(offset: usize, branches: &mut Vec<LineageBranchInfo>) -> Result<(&[LineageBranchInfo], usize)> {
        let page_start_item_offset = (offset / PTRS_PER_PAGE) * PTRS_PER_PAGE;
        let page_start_offset = page_start_item_offset * size_of::<LineageBranchInfo>();

        let n_in_page = PTRS_PER_PAGE.min(BRANCH_INFOS_LEN - page_start_offset);

        let read_start = page_start_offset + BRANCH_ARRAY_BASE_OFFSET;
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
                let res = read_local_branchinfos(mid, &mut branch_ptrs_buf)?;
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

        let res = read_local_branchinfos(mid, &mut branch_ptrs_buf)?;
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
fn find_relevant_branchinfo<'a>(
    file: &File,
    lineage_info: LineageInfo,
    offset: usize,
    restore_point: Lsn,
    lsn_cutoff: &mut Lsn,
) -> Option<LineageBranchInfo> {
    #[inline]
    fn get_local_branch_infos<'b>(
        offset: usize,
        n_local_items: u32,
        buf: &'b BufPage,
    ) -> &'b [LineageBranchInfo] {
        let len = n_local_items as usize * size_of::<LineageBranchInfo>();

        debug_assert!(len + offset <= LAYER_PAGE_SIZE);

        let datas: &'b [u8] = &buf.data()[offset..(offset + len)];
        let (head, infos, tail) = unsafe { datas.align_to::<LineageBranchInfo>() };

        debug_assert_eq!(tail.len(), 0);
        debug_assert_eq!(head.len(), 0);
        return infos;
    }

    const BRANCHES_PER_PAGE: usize = LAYER_PAGE_SIZE / size_of::<LineageBranchInfo>();

    debug_assert!(lineage_info.num_branches > 0);

    let mut left = 0usize;
    let mut right = if restore_point.is_some() {
        lineage_info.num_branches as usize - 1
    } else {
        0
    };
    let base_page: u32;
    let mut is_local = false;

    let n_page_local_branches = lineage_info
        .num_branches
        .min(((LAYER_PAGE_SIZE - offset as usize) / size_of::<LineageBranchInfo>()) as u32);

    let mut local_branch_infos =
        get_local_branch_infos(offset as usize, n_page_local_branches, &*buf);

    let cutoff_lsn = restore_point.unwrap_or(Lsn::MAX);

    // Page-local specialization for the first page of branches
    if local_branch_infos.len() > 0 {
        let newest_branch_info = &local_branch_infos[0];

        if let Some(lsn) = restore_point {
            // fast path: It is highly likely that we request the latest version of a page,
            if newest_branch_info.start_lsn <= lsn {
                return Some(*newest_branch_info);
            }

            // If the LSN is not on this first page, then we need to check the rest of the pages.
            let oldest_local_branch_info = &local_branch_infos[local_branch_infos.len() - 1];

            if oldest_local_branch_info.start_lsn > lsn {
                left = 0;
                base_page = blockno + 1;
                right -= local_branch_infos.len();
                *lsn_cutoff = oldest_local_branch_info.start_lsn;
                is_local = false;
            } else {
                is_local = true;
                base_page = blockno;
                right = local_branch_infos.len() - 1;
            }
        } else {
            return Some(*newest_branch_info);
        }
    } else {
        base_page = blockno + 1;
    }

    loop {
        if is_local {
            break;
        }
        let mid = (left + right) / 2;
        let buf_offno = mid / BRANCHES_PER_PAGE;

        *buf = get_buf(file, base_page + (buf_offno as u32), my_buf);

        let n_local_valid_branches =
            BRANCHES_PER_PAGE.min(right - (buf_offno * BRANCHES_PER_PAGE) + 1) as u32;

        local_branch_infos = get_local_branch_infos(0, n_local_valid_branches, &*buf);
        if left == right {
            is_local = true;
            break;
        }

        let local_offset = mid - (buf_offno * BRANCHES_PER_PAGE);
        let item = &local_branch_infos[local_offset];

        match item.start_lsn.cmp(&cutoff_lsn).reverse() {
            Ordering::Less => {
                right = mid;

                if local_offset != 0 {
                    let item = &local_branch_infos[0];
                    match item.start_lsn.cmp(&cutoff_lsn).reverse() {
                        Ordering::Less => right = buf_offno * BRANCHES_PER_PAGE,
                        Ordering::Equal => {
                            return Some(*item);
                        }
                        Ordering::Greater => {
                            left = buf_offno * BRANCHES_PER_PAGE;
                            *lsn_cutoff = item.start_lsn;
                            is_local = true;
                        }
                    }
                }
            }
            Ordering::Greater => {
                left = mid + 1;
                *lsn_cutoff = item.start_lsn;

                if local_offset != (n_local_valid_branches as usize) {
                    let item = &local_branch_infos[n_local_valid_branches as usize - 1];
                    match item.start_lsn.cmp(&cutoff_lsn).reverse() {
                        Ordering::Less => {
                            right =
                                buf_offno * BRANCHES_PER_PAGE + n_local_valid_branches as usize - 1;
                            is_local = true;
                        }
                        Ordering::Equal => {
                            return Some(*item);
                        }
                        Ordering::Greater => {
                            left = (buf_offno + 1) * BRANCHES_PER_PAGE;
                            // The following condition is only possible if this last comparison was
                            // on the very last page of the entries; in which we've just compared
                            // the last branchinfo of this lineage and found that it does not
                            // contain our LSN.
                            *lsn_cutoff = item.start_lsn;
                            if left > right {
                                return None;
                            }
                        }
                    }
                }
            }
            Ordering::Equal => {
                return Some(*item);
            }
        }
    }

    debug_assert!(is_local);

    let restore_point = restore_point.unwrap();

    return match local_branch_infos
        .binary_search_by(|item| item.start_lsn.cmp(&restore_point).reverse())
    {
        Ok(off) => {
            if off > 0 {
                *lsn_cutoff = local_branch_infos[off - 1].start_lsn;
            }

            Some(local_branch_infos[off])
        }
        Err(off) => {
            // The LSN is on this page, thus we must not be pointing past the end of the
            // local_brance_infos array.
            debug_assert!(off < local_branch_infos.len());
            if off > 0 {
                *lsn_cutoff = local_branch_infos[off - 1].start_lsn;
            }

            Some(local_branch_infos[off])
        }
    };
}

fn extract_branch_data(
    file: &File,
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
