use crate::layered_repository::storage_layer::RELISH_SEG_SIZE;
use zenith_utils::lsn::Lsn;

pub(crate) mod differential_meta;
pub(crate) mod snapshot_meta;

const BLOCKS_PER_SEGMENT: u32 = RELISH_SEG_SIZE;

pub enum PageFindResult<'a, T> {
    /// The page was found in this layer.
    InThisLayer(&'a T),
    /// The page was last modified at last_modified, so look there.
    InOtherLayer { last_modified: Lsn },
    /// The page wasn't modified since last full snapshot
    UsePreviousSnapshot,
    /// We can't say who has info on the page, but it definitely isn't stored in this layer.
    ///
    /// Retained for older style Layers, where no index for page versions since last snapshot was
    /// kept.
    TryPreviousLayer,
}

pub trait LayerMetadata {
    type PerPageData;

    /// Check where we can find page restoration info for this page.
    fn query_page(&self, block_num: u32) -> PageFindResult<Self::PerPageData>;

    fn is_snapshot(&self) -> bool;

    /// Get the latest lsn for a page that we know about.
    fn get_latest_block_lsn(&self, block_num: u32) -> Option<Lsn>;

    /// Get access to the page's data stored in this layer.
    /// None if not stored in this layer.
    fn get_page_data(&self, block_num: u32) -> Option<&Self::PerPageData>;

    /// Map the block_num to a local offset in this segment.
    fn blocknum_to_local(block_num: u32) -> usize {
        return (block_num % BLOCKS_PER_SEGMENT) as usize;
    }

    /// Get all pages that have been modified since the last snapshot,
    /// with their respective latest LSNs.
    fn get_latest_lsn_since_snapshot(&self) -> Box<dyn Iterator<Item = (usize, Lsn)> + '_>;

    fn num_mutated_since_snapshot(&self) -> u32;

    fn get_mem_usage(&self) -> usize;
}

pub trait MutLayerMetadata: LayerMetadata {
    fn add_mutated_page(&mut self, block_num: u32, metadata: <Self as LayerMetadata>::PerPageData);

    fn get_page_data_mut(
        &mut self,
        block_num: u32,
    ) -> Option<&mut <Self as LayerMetadata>::PerPageData>;
}
