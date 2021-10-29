use crate::layered_repository::layers::meta::{LayerMetadata, PageFindResult};
use std::iter;
use std::mem::size_of;
use zenith_utils::lsn::Lsn;

pub struct SnapshotMetadata {
    my_lsn: Lsn,
}

impl SnapshotMetadata {
    pub fn new(lsn: Lsn) -> Self {
        Self { my_lsn: lsn }
    }
}

impl LayerMetadata for SnapshotMetadata {
    type PerPageData = ();

    fn query_page(&self, _block_num: u32) -> PageFindResult<()> {
        return PageFindResult::InThisLayer(&());
    }

    fn is_snapshot(&self) -> bool {
        return true;
    }

    fn get_latest_block_lsn(&self, _block_num: u32) -> Option<Lsn> {
        Some(self.my_lsn)
    }

    fn get_page_data(&self, _block_num: u32) -> Option<&Self::PerPageData> {
        return Some(&());
    }

    fn get_latest_lsn_since_snapshot(&self) -> Box<dyn Iterator<Item = (usize, Lsn)> + '_> {
        Box::new(iter::empty())
    }

    fn num_mutated_since_snapshot(&self) -> u32 {
        0
    }

    fn get_mem_usage(&self) -> usize {
        size_of::<Self>()
    }
}
