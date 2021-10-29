use crate::layered_repository::layers::meta::{LayerMetadata, MutLayerMetadata, PageFindResult};
use crate::layered_repository::storage_layer::{Layer, RELISH_SEG_SIZE};
use std::fmt::Debug;
use zenith_utils::lsn::Lsn;
use zenith_utils::multi_bitmap::{BitMapIterator, LayeredBitmap};
use zenith_utils::vec_map::VecMap;

pub trait AsLatestLsn {
    type Arg: Clone + Debug;
    fn as_latest_lsn(&self, arg: &Self::Arg) -> Lsn;
}

/// Layer metadata for layers that do not have enough information to restore all
/// pages in the layer.
/// Efficiency heavily relies on block-based relations.
///
/// T can be used to store data at the page level.
#[derive(Debug, Clone)]
pub struct DifferentialMetadata<T>
where
    T: Sized + Clone + AsLatestLsn,
{
    /// Bitmap defining which pages have been modified in this layer, and can
    /// thus be restored using this layer only.
    /// Because we only store booleans (yes/no), this maps to unit type.
    pages_contained: LayeredBitmap<T>,
    /// This contain an LSN for each page that was modified since the last
    /// snapshot, but have not been modified in this Layer. As such, this Lsn
    /// should be enough to find the correct Layer for page reconstruction;
    ///
    /// Note: As a consequence of this, LSNs stored in this map are of the
    /// range '(last snapshot's LSN, start of this layer)', that is, both
    /// exclusive.
    pages_modified_since_snapshot: LayeredBitmap<Lsn>,

    /// contains the argument used to restore LSN from the T type.
    lsn_arg: <T as AsLatestLsn>::Arg,
}

impl<T> DifferentialMetadata<T>
where
    T: Sized + Clone + AsLatestLsn,
{
    pub fn get_old_versions_map(&self) -> &LayeredBitmap<Lsn> {
        return &self.pages_modified_since_snapshot;
    }

    pub fn get_versions_map(&self) -> &LayeredBitmap<T> {
        return &self.pages_contained;
    }

    pub fn get_largest_modified_page(&self) -> Option<usize> {
        let mod_since_snapshot = self.pages_modified_since_snapshot.last();
        let contained = self.pages_contained.last();

        if mod_since_snapshot.is_none() {
            return contained;
        }

        if contained.is_none() {
            return mod_since_snapshot;
        }

        return Some(contained.unwrap().max(mod_since_snapshot.unwrap()));
    }
}

pub struct IncompleteMetadataIter<'a, T>
where
    T: Sized + Clone + AsLatestLsn,
{
    inner: BitMapIterator<'a, T>,
    arg: &'a <T as AsLatestLsn>::Arg,
}

impl<'a, T> IncompleteMetadataIter<'a, T>
where
    T: Sized + Clone + AsLatestLsn,
{
    pub fn new(map: &'a LayeredBitmap<T>, arg: &'a <T as AsLatestLsn>::Arg) -> Self {
        Self {
            inner: map.iter(),
            arg,
        }
    }
}

impl<'a, T> Iterator for IncompleteMetadataIter<'a, T>
where
    T: Sized + Clone + AsLatestLsn,
{
    type Item = (usize, Lsn);

    fn next(&mut self) -> Option<Self::Item> {
        return self
            .inner
            .next()
            .map(|it| (it.0, it.1.as_latest_lsn(self.arg)));
    }
}

impl<T> DifferentialMetadata<T>
where
    T: Sized + Clone + AsLatestLsn,
{
    pub fn new(lsn_arg: <T as AsLatestLsn>::Arg) -> Self {
        Self {
            pages_contained: LayeredBitmap::new(RELISH_SEG_SIZE as usize),
            pages_modified_since_snapshot: LayeredBitmap::new(RELISH_SEG_SIZE as usize),
            lsn_arg,
        }
    }

    pub fn new_from_layer(previous_layer: &dyn Layer, lsn_arg: <T as AsLatestLsn>::Arg) -> Self {
        let mut res = Self {
            pages_contained: LayeredBitmap::new(RELISH_SEG_SIZE as usize),
            pages_modified_since_snapshot: LayeredBitmap::new(RELISH_SEG_SIZE as usize),
            lsn_arg,
        };

        if previous_layer.is_incremental() {
            if let Some(iter) = previous_layer.latest_page_versions_since_snapshot() {
                for (blknum, lsn) in iter {
                    res.pages_modified_since_snapshot
                        .set((blknum % RELISH_SEG_SIZE) as usize, lsn);
                }
            }
        }

        return res;
    }

    pub fn new_with_maps(
        pages_contained: LayeredBitmap<T>,
        pages_modified_since_snapshot: LayeredBitmap<Lsn>,
        lsn_arg: <T as AsLatestLsn>::Arg,
    ) -> Self {
        Self {
            pages_contained,
            pages_modified_since_snapshot,
            lsn_arg,
        }
    }

    pub fn pages(&self) -> BitMapIterator<T> {
        self.pages_contained.iter()
    }
}

impl AsLatestLsn for Lsn {
    type Arg = ();

    fn as_latest_lsn(&self, _arg: &Self::Arg) -> Lsn {
        *self
    }
}

impl<T> LayerMetadata for DifferentialMetadata<T>
where
    T: Sized + Clone + AsLatestLsn,
{
    type PerPageData = T;

    fn query_page(&self, block_num: u32) -> PageFindResult<'_, T> {
        let offset = <Self as LayerMetadata>::blocknum_to_local(block_num);

        return if let Some(data) = self.pages_contained.get(offset) {
            PageFindResult::InThisLayer(data)
        } else if let Some(lsn) = self.pages_modified_since_snapshot.get(offset) {
            PageFindResult::InOtherLayer {
                last_modified: *lsn,
            }
        } else {
            PageFindResult::UsePreviousSnapshot
        };
    }

    fn is_snapshot(&self) -> bool {
        return false;
    }

    fn get_latest_block_lsn(&self, block_num: u32) -> Option<Lsn> {
        let offset = <Self as LayerMetadata>::blocknum_to_local(block_num);

        return if let Some(inner) = self.pages_contained.get(offset) {
            Some(inner.as_latest_lsn(&self.lsn_arg))
        } else {
            self.pages_modified_since_snapshot
                .get(offset)
                .map(|lsn| *lsn)
        };
    }

    fn get_page_data(&self, block_num: u32) -> Option<&T> {
        return self
            .pages_contained
            .get(<Self as LayerMetadata>::blocknum_to_local(block_num));
    }

    fn get_latest_lsn_since_snapshot(&self) -> Box<dyn Iterator<Item = (usize, Lsn)> + '_> {
        Box::new(IncompleteMetadataIter {
            inner: self.pages_modified_since_snapshot.iter(),
            arg: &(),
        })
    }

    fn num_mutated_since_snapshot(&self) -> u32 {
        return (self.pages_modified_since_snapshot.len() + self.pages_contained.len()) as u32;
    }

    fn get_mem_usage(&self) -> usize {
        return self.pages_contained.get_mem_usage()
            + self.pages_modified_since_snapshot.get_mem_usage();
    }
}

impl<Y: Clone> DifferentialMetadata<VecMap<Lsn, Y>> {
    pub fn all_page_versions(&self) -> Box<dyn Iterator<Item = (usize, Lsn, &'_ Y)> + '_> {
        let iterator = self.pages_contained.iter();

        return Box::new(iterator.flat_map(|(offno, map)| {
            map.as_slice()
                .iter()
                .map(move |(lsn, data)| (offno, *lsn, data))
        }));
    }
}

impl<T> MutLayerMetadata for DifferentialMetadata<T>
where
    T: Sized + Clone + AsLatestLsn,
{
    fn add_mutated_page(&mut self, block_num: u32, metadata: T) {
        let offset = <Self as LayerMetadata>::blocknum_to_local(block_num);
        if self.pages_contained.get(offset).is_some() {
            return;
        }
        self.pages_modified_since_snapshot.reset(offset);
        self.pages_contained.set(offset, metadata);
    }

    fn get_page_data_mut(&mut self, block_num: u32) -> Option<&mut T> {
        return self
            .pages_contained
            .get_mut(<Self as LayerMetadata>::blocknum_to_local(block_num));
    }
}
