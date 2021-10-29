use anyhow::Result;
use bitmaps::Bitmap;
use log::debug;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fs::File;
use std::io::Write;
use std::mem::size_of;
use std::num::NonZeroU32;
use zenith_utils::lsn::Lsn;
use zenith_utils::multi_bitmap::{BitmappedMapPage, LayeredBitmap};
use zenith_utils::vec_map::VecMap;

use crate::layered_repository::filebufutils::{
    get_buf, get_buf_mut, read_item_from_blocky_file, read_slice_from_blocky_file, write_buf,
    write_item_to_blocky_file, write_slice_to_blocky_file, BufPage,
};
use crate::layered_repository::inmemory_layer::InMemoryLayerInner;
use crate::layered_repository::layers::differential_layer::file_format::{
    BranchPtrData, BranchPtrType, InfoPageV1, LineageBranchInfo, LineageInfo,
    PageInitiatedLineageBranchHeader, PageOnlyLineageBranchHeader, UnalignFileOffset, VersionInfo,
    WalInitiatedLineageBranchHeader, WalOnlyLineageBranchHeader,
};
use crate::layered_repository::layers::differential_layer::LineagePointer;
use crate::layered_repository::layers::LAYER_PAGE_SIZE;
use crate::layered_repository::page_versions::PageVersionPtr;
use crate::layered_repository::storage_layer::{
    AsVersionType, PageVersion, VersionType, RELISH_SEG_SIZE,
};

pub fn write_latest_metadata(
    file: &mut File,
    blockno: u32,
    metadata: &InfoPageV1,
    buf: &RwLock<BufPage>,
) -> Result<()> {
    let mut buf = get_buf_mut(file, blockno, buf);

    let page_base_ptr = buf.data_mut();

    let (mut version_pointer, mut meta_pointer) = page_base_ptr.split_at_mut(size_of::<u32>());

    assert!(size_of::<InfoPageV1>() < LAYER_PAGE_SIZE as usize - size_of::<u32>());

    let _ = version_pointer.write(&mut 1u32.to_le_bytes())?;

    let metadataref = unsafe {
        let (head, data, tail) = meta_pointer.align_to_mut::<InfoPageV1>();
        assert!(head.is_empty());
        assert!(!data.is_empty());
        &mut data[0]
    };

    *metadataref = *metadata;

    write_buf(file, blockno, &*buf);
    Ok(())
}

pub fn read_metadata(file: &mut File, blockno: u32, buf: &RwLock<BufPage>) -> InfoPageV1 {
    let buffer = get_buf(file, blockno, buf);

    let (version, meta) = buffer.data().split_at(size_of::<u32>());

    let version = unsafe { version.align_to::<u32>().1[0] };

    match version {
        1 => {
            let meta = unsafe {
                let (head, data, tail) = meta.align_to::<InfoPageV1>();
                assert!(head.is_empty());
                assert!(!data.is_empty());
                data[0]
            };
            debug!("{:?}", meta);
            return meta;
        }
        _ => panic!("Unknown version: {}", version),
    }
}

/// Write out the old page versions section.
/// Uses up 128 bytes, plus max 15 bytes alignment and 128+4 bytes per section of 1024
/// blocks in the base relation with changes, plus 8 bytes per changed block.
///
/// base_blockno is assumed to be empty, and the start of this section.
///
/// Access time to one field is O(log(n)) with log base 1028.
/// returns the next free block number.
pub fn write_old_page_versions_section(
    file: &mut File,
    base_blockno: u32,
    previous_inner: &InMemoryLayerInner,
) -> u32 {
    let my_buf = RwLock::new(BufPage::default());
    let mut buf = my_buf.write();
    let mutated_pages = previous_inner.changed_since_snapshot().backing_store();
    let (map_start, map_end) = write_bitmap_page_recursive(
        file,
        base_blockno,
        0u32,
        0u16,
        &mut buf,
        &my_buf,
        mutated_pages,
        &|&lsn| lsn,
    );

    assert_eq!(map_start, (base_blockno, 0u16));

    return map_end.0 + 1;
}

/// returns the next free block number.
pub fn write_page_lineages_section(
    file: &mut File,
    base_blockno: u32,
    previous_inner: &InMemoryLayerInner,
    lineage_block_offsets: &mut LayeredBitmap<LineagePointer>,
) -> u32 {
    let my_buf: RwLock<BufPage> = RwLock::new(BufPage::default());

    let iterator = previous_inner.page_changes_vecmap_iterator();
    let mut blockno = base_blockno;
    let mut offset = 0u16;
    let mut buf: RwLockWriteGuard<BufPage> = get_buf_mut(file, blockno, &my_buf);

    let closure = |ptr: &PageVersionPtr| previous_inner.get_page_version(*ptr);
    for (pageno, map) in iterator {
        let (object_start, used_up_to) =
            write_vecmap_lineage_section(file, blockno, offset, &map, &mut buf, &my_buf, &closure);

        lineage_block_offsets.set(
            (*pageno % RELISH_SEG_SIZE) as usize,
            LineagePointer(UnalignFileOffset::new(object_start.0, object_start.1)),
        );

        blockno = used_up_to.0;
        offset = used_up_to.1;
    }

    if offset == 0 {
        return blockno;
    }
    return blockno + 1;
}

/// Write out one lineage from the lineage section.
///
/// Note that in VecMap, LSN are stored in increasing order, but in the Lineage
/// section, they are stored in decreasing order. This means that we don't generally
/// need to seek to find the latest Lineage, which helps with performance.
/// Note that this is all WAL for one page in one block.
pub fn write_vecmap_lineage_section<'a, T>(
    file: &mut File,
    start_block: u32,
    start_offset: u16,
    map: &VecMap<Lsn, T>,
    buf: &mut RwLockWriteGuard<'a, BufPage>,
    my_buf: &'a RwLock<BufPage>,
    to_pageversion: &dyn Fn(&T) -> PageVersion,
) -> ((u32, u16), (u32, u16))
where
    T: Copy + Clone + AsVersionType,
{
    let struct_location: (u32, u16);

    let mut pos: (u32, u16) = (start_block, start_offset);

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
    let branches = branches;

    // Phase 2: construct branch info arrays to store in the main Lineage index
    let n_branches = branches.len();
    let mut branch_infos = Vec::<LineageBranchInfo>::with_capacity(n_branches);

    for (lsn, _head, _branch) in branches.iter() {
        branch_infos.push(LineageBranchInfo {
            start_lsn: *lsn,
            // Will later be filled in with data during branch serialization.
            branch_ptr_data: BranchPtrData {
                item_ptr: UnalignFileOffset::new(0, 0),
                tail_padding: 0,
                typ: BranchPtrType::PageImage,
            },
        });
    }

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
    let (start, end) =
        write_item_to_blocky_file::<LineageInfo>(file, pos.0, pos.1, &descriptor, buf, my_buf);
    struct_location = start;
    pos = end;

    pos = write_slice_to_blocky_file::<LineageBranchInfo>(
        file,
        pos.0,
        pos.1,
        &branch_infos,
        buf,
        my_buf,
    );

    for (branch_index, (_, head, versions)) in branches.iter().enumerate() {
        let mut data = Vec::with_capacity(versions.len());
        let mut data_length = 0u64;

        let head_page_version = to_pageversion(head);

        if versions.len() == 0 {
            let my_start;
            let typ: BranchPtrType;
            let bytes = match &head_page_version {
                PageVersion::Page(bytes) => {
                    let header = PageOnlyLineageBranchHeader {
                        length: NonZeroU32::try_from(bytes.len() as u32).unwrap(),
                    };
                    let res = write_item_to_blocky_file(file, pos.0, pos.1, &header, buf, my_buf);

                    my_start = res.0;
                    pos = res.1;
                    typ = BranchPtrType::PageImage;

                    bytes
                }
                PageVersion::Wal(rec) => {
                    let header = WalOnlyLineageBranchHeader {
                        length: NonZeroU32::try_from(rec.rec.len() as u32).unwrap(),
                        main_data_offset: rec.main_data_offset,
                    };

                    let res = write_item_to_blocky_file(file, pos.0, pos.1, &header, buf, my_buf);

                    my_start = res.0;
                    pos = res.1;
                    typ = BranchPtrType::WalRecord;

                    &rec.rec
                }
            };

            branch_infos[branch_index].branch_ptr_data.item_ptr =
                UnalignFileOffset::new(my_start.0, my_start.1);
            branch_infos[branch_index].branch_ptr_data.typ = typ;

            pos = write_slice_to_blocky_file(file, pos.0, pos.1, &bytes[..], buf, my_buf);

            continue; // no header, only direct data.
        }

        // Note: this allocates all WALRecords into memory, which is not great.
        // The alternative, however, is to double-parse the WAL records, which is
        // random IO and not great as well.
        let mapped_iter = versions
            .iter()
            .map(|(lsn, rec)| {
                (
                    *lsn,
                    match to_pageversion(rec) {
                        PageVersion::Page(_) => unreachable!("PageVersion::Page in tail"),
                        PageVersion::Wal(rec) => rec,
                    },
                )
            })
            .collect::<Vec<_>>();
        ();

        for (lsn, record) in mapped_iter.iter() {
            let length = NonZeroU32::try_from(record.rec.len() as u32).unwrap();
            let main_data_offset = record.main_data_offset;

            data.push(VersionInfo {
                lsn: lsn.0.to_ne_bytes(),
                length,
                main_data_offset,
            });

            data_length += u32::from(length) as u64;
        }

        let typ: BranchPtrType;
        let (data_start, end) = match head_page_version {
            PageVersion::Page(bytes) => {
                typ = BranchPtrType::PageInitiatedBranch;
                write_item_to_blocky_file(
                    file,
                    pos.0,
                    pos.1,
                    &PageInitiatedLineageBranchHeader {
                        num_entries: NonZeroU32::try_from(versions.len() as u32).unwrap(),
                        head: PageOnlyLineageBranchHeader {
                            length: NonZeroU32::try_from(bytes.len() as u32).unwrap(),
                        },
                    },
                    buf,
                    my_buf,
                )
            }
            PageVersion::Wal(rec) => {
                typ = BranchPtrType::WalInitiatedBranch;
                write_item_to_blocky_file(
                    file,
                    pos.0,
                    pos.1,
                    &WalInitiatedLineageBranchHeader {
                        num_entries: NonZeroU32::try_from(versions.len() as u32).unwrap(),
                        head: WalOnlyLineageBranchHeader {
                            length: NonZeroU32::try_from(rec.rec.len() as u32).unwrap(),
                            main_data_offset: rec.main_data_offset,
                        },
                    },
                    buf,
                    my_buf,
                )
            }
        };

        branch_infos[branch_index].branch_ptr_data.item_ptr =
            UnalignFileOffset::new(data_start.0, data_start.1);
        branch_infos[branch_index].branch_ptr_data.typ = typ;

        pos = end;

        pos = write_slice_to_blocky_file::<VersionInfo>(file, pos.0, pos.1, &data[..], buf, my_buf);

        for (_, rec) in mapped_iter {
            pos = write_slice_to_blocky_file::<u8>(file, pos.0, pos.1, &rec.rec[..], buf, my_buf);
        }
    }

    return (struct_location, pos);
}

pub fn write_bitmap_page_recursive<'a, T, R, F>(
    file: &mut File,
    base_blockno: u32,
    start_blockno: u32,
    start_offset: u16,
    buf: &mut RwLockWriteGuard<'a, BufPage>,
    my_buf: &'a RwLock<BufPage>,
    page: &BitmappedMapPage<T>,
    mapper: &F,
) -> ((u32, u16), (u32, u16))
where
    T: Clone,
    R: Copy,
    F: Fn(&T) -> R,
{
    let my_map_blockno;
    let my_map_offset;
    let mut my_blockno = base_blockno + start_blockno;
    let mut my_offset = start_offset;

    match page {
        BitmappedMapPage::LeafPage { data } => {
            let bitmap = data.page_get_bitmap();

            let (my_start, end) = write_item_to_blocky_file::<[u128; 8]>(
                file, my_blockno, my_offset, bitmap, buf, my_buf,
            );

            my_map_blockno = my_start.0;
            my_map_offset = my_start.1;

            my_blockno = end.0;
            my_offset = end.1;

            let values = data.page_get_values();

            let transformed_values: Vec<R> = values.iter().map(|it| mapper(it)).collect();

            let res = write_slice_to_blocky_file(
                file,
                my_blockno,
                my_offset,
                &transformed_values[..],
                buf,
                my_buf,
            );
            my_blockno = res.0;
            my_offset = res.1;
        }
        BitmappedMapPage::InnerPage { data, .. } => {
            let bitmap = data.page_get_bitmap();

            let (my_start, end) = write_item_to_blocky_file::<[u128; 8]>(
                file, my_blockno, my_offset, bitmap, buf, my_buf,
            );

            my_map_blockno = my_start.0;
            my_map_offset = my_start.1;

            my_blockno = end.0;
            my_offset = end.1;

            let mut num_items = data.page_len();
            let mut items_arr = vec![0u32; num_items];

            let data_start = (my_blockno, my_offset);

            let res = write_slice_to_blocky_file::<u32>(
                file, my_blockno, my_offset, &items_arr, buf, my_buf,
            );

            my_blockno = res.0;
            my_offset = res.1;

            for (i, item) in data.page_get_values().iter().enumerate() {
                let item = &**item;
                let (map, end_of_write) = write_bitmap_page_recursive(
                    file,
                    base_blockno,
                    my_blockno - base_blockno,
                    my_offset,
                    buf,
                    my_buf,
                    item,
                    mapper,
                );
                my_blockno = end_of_write.0;
                my_offset = end_of_write.1;

                let pos: u32 = ((((map.0 - base_blockno) as usize)
                    * ((LAYER_PAGE_SIZE as usize) / size_of::<u128>()))
                    | ((map.1 as usize) / size_of::<u128>())) as u32;
                items_arr[i] = pos;
            }

            let last_data_end = (my_blockno, my_offset);

            my_blockno = data_start.0;
            my_offset = data_start.1;

            // Result is ignored as this does not extend the data section;
            // we return to last_data_end instead, as that is the current free pointer.
            let _ = write_slice_to_blocky_file::<u32>(
                file, my_blockno, my_offset, &items_arr, buf, my_buf,
            );

            my_blockno = last_data_end.0;
            my_offset = last_data_end.1;
        }
    }
    return ((my_map_blockno, my_map_offset), (my_blockno, my_offset));
}

fn read_bitmap_recursive<'a, T, P, Map>(
    file: &File,
    my_map_page: &mut BitmappedMapPage<T>,
    base_blockno: u32,
    start_blockno: u32,
    start_offset: u16,
    buf: &mut RwLockReadGuard<'a, BufPage>,
    my_buf: &'a RwLock<BufPage>,
    map: &Map,
) -> (u32, u16)
where
    T: Sized + Clone,
    Map: Fn(P) -> T,
    P: Default + Clone + Copy,
{
    let mut blockno = base_blockno + start_blockno;
    let mut offset = start_offset;

    let mut bits = [0u128; 8];
    let res =
        read_item_from_blocky_file::<[u128; 8]>(file, blockno, offset, &mut bits, buf, my_buf);
    blockno = res.0;
    offset = res.1;
    let bitmap = Bitmap::<1024>::from(bits);

    match my_map_page {
        BitmappedMapPage::LeafPage { data } => {
            let mut values = vec![P::default(); bitmap.len()];

            let res =
                read_slice_from_blocky_file::<P>(file, blockno, offset, &mut values, buf, my_buf);
            blockno = res.0;
            offset = res.1;

            for (i, value) in Iterator::zip(bitmap.into_iter(), values.into_iter()) {
                data.page_set(i, map(value));
            }
        }
        BitmappedMapPage::InnerPage {
            data,
            n_items,
            layer,
            mem_usage,
        } => {
            let mut values = Vec::<u32>::with_capacity(bitmap.len());
            values.fill(0u32);

            let res =
                read_slice_from_blocky_file::<u32>(file, blockno, offset, &mut values, buf, my_buf);
            blockno = res.0;
            offset = res.1;

            for i in bitmap.into_iter() {
                let the_block = values[i] / (LAYER_PAGE_SIZE as u32) + base_blockno;
                let the_offset = (values[i] % (LAYER_PAGE_SIZE as u32)) as u16;
                let mut map_item = BitmappedMapPage::new(*layer - 1);

                let _ = read_bitmap_recursive(
                    file,
                    &mut map_item,
                    base_blockno,
                    the_block,
                    the_offset,
                    buf,
                    my_buf,
                    &|it| map(it),
                );

                *n_items += map_item.page_len();
                *mem_usage += map_item.get_mem_usage();
                let prev = data.page_set(i, Box::new(map_item));
                assert!(prev.is_none());
            }
        }
    }

    return (blockno, offset);
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
pub fn write_page_versions_map(
    file: &mut File,
    base_blockno: u32,
    map: &LayeredBitmap<LineagePointer>,
) -> u32 {
    let my_buf = RwLock::new(BufPage::default());
    let mut buf = my_buf.write();
    let (map_start, map_end) = write_bitmap_page_recursive(
        file,
        base_blockno,
        0u32,
        0u16,
        &mut buf,
        &my_buf,
        map.backing_store(),
        &|&ptr| ptr,
    );

    assert_eq!(map_start, (base_blockno, 0u16));

    return map_end.0 + 1;
}

/// Read out the new page versions section.
/// Basically a compressed bitmap, which stores one UnalignedFileOffset per set bit.
/// Bit positions are block numbers relative to the start of the layer's segment.
/// base_blockno is assumed to be empty, and the start of this section.
///
/// Access time to one field is O(log(n)) with log base 1028.
/// returns the next free block number.
pub fn read_page_versions_map(
    file: &mut File,
    base_blockno: u32,
    map: &mut LayeredBitmap<LineagePointer>,
) {
    let mut bitmap_layer = BitmappedMapPage::<LineagePointer>::new(0);
    let mut my_buf: RwLock<BufPage> = RwLock::new(BufPage::default());
    let mut buf = my_buf.read();

    let _ = read_bitmap_recursive(
        file,
        &mut bitmap_layer,
        base_blockno,
        0u32,
        0u16,
        &mut buf,
        &my_buf,
        &|ptr| LineagePointer(ptr),
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
    base_blockno: u32,
    map: &mut LayeredBitmap<Lsn>,
) {
    let mut bitmap_layer = BitmappedMapPage::<Lsn>::new(0);
    let mut my_buf: RwLock<BufPage> = RwLock::new(BufPage::default());
    let mut buf = my_buf.read();

    let _ = read_bitmap_recursive(
        file,
        &mut bitmap_layer,
        base_blockno,
        0u32,
        0u16,
        &mut buf,
        &my_buf,
        &|lsn| lsn,
    );

    *map = LayeredBitmap::new_from_page(bitmap_layer);
}

pub fn read_lineage(
    file: &File,
    blockno: u32,
    offset: u16,
    restore_point: Option<Lsn>,
) -> Option<(Vec<(Lsn, PageVersion)>, Lsn)> {
    let mut my_buf: RwLock<BufPage> = RwLock::new(BufPage::default());
    let mut buf = get_buf(file, blockno, &my_buf);

    let (_, start) = buf.data().split_at(offset as usize);
    let (header_data, tail) = start.split_at(size_of::<LineageInfo>());

    let lineage_info = {
        let (align, lineage_header, alig) = unsafe { header_data.align_to::<LineageInfo>() };
        assert_eq!(align.len(), 0);
        assert_eq!(alig.len(), 0);
        assert_eq!(lineage_header.len(), 1);
        lineage_header[0]
    };

    let branch_infos_len = (lineage_info.num_branches as usize) * size_of::<LineageBranchInfo>();
    let all_local = tail.len() >= branch_infos_len;

    assert!(
        lineage_info.num_branches > 0,
        "Each lineage should always have at least one branch"
    );

    let mut next_changed_lsn = Lsn(u64::MAX);

    let my_branchinfo = find_relevant_branchinfo(
        file,
        lineage_info,
        blockno,
        offset + (size_of::<LineageInfo>() as u16),
        restore_point,
        &mut next_changed_lsn,
        &mut buf,
        &my_buf,
    );

    if my_branchinfo.is_none() {
        return None;
    }

    let branch = my_branchinfo.unwrap().clone();

    let data = extract_branch_data(file, &branch, restore_point, &mut next_changed_lsn);

    return Some((vec![], next_changed_lsn));
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
    blockno: u32,
    offset: u16,
    restore_point: Option<Lsn>,
    lsn_cutoff: &mut Lsn,
    buf: &mut RwLockReadGuard<'a, BufPage>,
    my_buf: &'a RwLock<BufPage>,
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
