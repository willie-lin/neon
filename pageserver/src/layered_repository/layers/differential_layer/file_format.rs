use crate::layered_repository::layers::LAYER_PAGE_SIZE;
use static_assertions::{const_assert, const_assert_eq};
use std::mem::size_of;
use std::num::NonZeroU32;
use zenith_utils::lsn::Lsn;

#[repr(C, align(2))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct UnalignFileOffset {
    blk_hi: u16,
    blk_lo: u16,
    offset: u16,
}

impl UnalignFileOffset {
    pub fn new(blockno: u32, offset: u16) -> Self {
        Self {
            blk_hi: (blockno >> 16) as u16,
            blk_lo: (blockno & 0xFFFF) as u16,
            offset,
        }
    }

    pub fn blockno(&self) -> u32 {
        (self.blk_hi as u32) << 16 | (self.blk_lo as u32)
    }

    pub fn offset(&self) -> u16 {
        self.offset
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct InfoPageV1 {
    pub old_page_versions_map_start: u32,
    pub new_page_versions_map_start: u32,
    pub page_lineages_start: u32,
    pub page_images_start: u32,
    pub seg_lengths_start: u32,
}

// Used for size assertion checks.
#[repr(C)]
union InfoPageVersions {
    v1: InfoPageV1,
}

const_assert!(size_of::<InfoPageVersions>() + size_of::<u32>() <= LAYER_PAGE_SIZE,);

///
/// LineageBranchInfo holds data for one branch of the page's lineage, and
/// points to one of the following (determined by branch_ptr_data.typ):
///
///  - A WAL record that initializes the page
///  - A Full-Page-Image
///  - One of the two above, with records that apply to the page initialized
///     by the record or page image.
///
/// Note that data_length is only the sum of the data payloads of the records;
/// and is thus exclusive of the size of the LineageBranchInfo struct itself, and
/// also exclusive of any LineageBranchHeader and VersionInfo structs containing metadata.
#[repr(C, align(8))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LineageBranchInfo {
    pub start_lsn: Lsn,
    pub branch_ptr_data: BranchPtrData,
}

///
/// Version branch header
///
/// This is the header for a version branch. Following this header there are num_entries aligned
/// VersionInfo entries, potentially spilling onto following pages.
///
/// head_item_len and head_item_data_off are used to reconstruct the first record of the branch, the
/// type of which is detailed in the BranchInfo struct above.
#[repr(C, align(4))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct WalInitiatedLineageBranchHeader {
    pub num_entries: NonZeroU32,
    pub head: WalOnlyLineageBranchHeader,
}
#[repr(C, align(4))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct WalOnlyLineageBranchHeader {
    pub length: NonZeroU32,
    pub main_data_offset: NonZeroU32,
}
///
/// Version branch header
///
/// This is the header for a version branch. Following this header there are num_entries aligned
/// VersionInfo entries, potentially spilling onto following pages.
///
/// head_item_len and head_item_data_off are used to reconstruct the first record of the branch, the
/// type of which is detailed in the BranchInfo struct above.
#[repr(C, align(4))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PageInitiatedLineageBranchHeader {
    pub num_entries: NonZeroU32,
    pub head: PageOnlyLineageBranchHeader,
}
#[repr(C, align(4))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PageOnlyLineageBranchHeader {
    pub length: NonZeroU32,
}

/// A version info; detailing how long the written data is.
///
/// Note that this does not contain info on what type the underlying data is,
/// as that will always be a WAL record without will_init set.
///
/// main_data_offset is the offset of the main data in the record.
#[repr(C, align(4))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct VersionInfo {
    pub lsn: [u8; 8],
    pub length: NonZeroU32,
    pub main_data_offset: NonZeroU32,
}

impl Default for VersionInfo {
    fn default() -> Self {
        Self {
            lsn: [0; 8],
            length: NonZeroU32::new(1).unwrap(),
            main_data_offset: NonZeroU32::new(1).unwrap(),
        }
    }
}

// expected struct size: 16 bytes;
const_assert_eq!(std::mem::size_of::<VersionInfo>(), 16usize);

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BranchPtrType {
    PageImage = 1,
    WalRecord,
    PageInitiatedBranch,
    WalInitiatedBranch,
}

#[repr(C, align(8))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BranchPtrData {
    pub item_ptr: UnalignFileOffset,
    pub tail_padding: u8,
    pub typ: BranchPtrType,
}

// expected struct size: 8 bytes;
const_assert_eq!(std::mem::size_of::<BranchPtrData>(), 8usize);

#[repr(C, align(8))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LineageInfo {
    pub latest_lsn: Lsn,
    pub num_branches: u32,
    pub last_img: u32,
}
// expected struct size: 16 bytes;
const_assert_eq!(std::mem::size_of::<LineageInfo>(), 16usize);
