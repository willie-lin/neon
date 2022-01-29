use bitmaps;
use bitmaps::{Bitmap, Bits, BitsImpl};
use core::mem::size_of;
use std::fmt::{Debug, Formatter};

pub const PER_LAYER_SPECIFICITY_BITS: usize = 10;
pub const N_BITS_PER_PAGE: usize = 1024; // 2usize.pow(PER_LAYER_SPECIFICITY_BITS as u32);
const LAYER_BITMASK: usize = N_BITS_PER_PAGE - 1usize;

pub type BitmapStorageType = <BitsImpl<N_BITS_PER_PAGE> as Bits>::Store;
pub type MBitmap = Bitmap<N_BITS_PER_PAGE>;

fn le_popcnt(it: Bitmap<N_BITS_PER_PAGE>, len: usize) -> usize {
    (Bitmap::<N_BITS_PER_PAGE>::mask(len) & it).len()
}

fn destruct(layer: usize, index: usize) -> (usize, usize) {
    let local = index
        .checked_shr((layer * PER_LAYER_SPECIFICITY_BITS) as u32)
        .unwrap_or(0)
        & LAYER_BITMASK;
    let lower = index % (N_BITS_PER_PAGE.pow(layer as u32));
    return (local, lower);
}

fn construct(layer: usize, local: usize, lower: usize) -> usize {
    return lower + (local << (layer * PER_LAYER_SPECIFICITY_BITS));
}

#[derive(Clone)]
pub struct LayerPageData<T: Sized + Clone> {
    map: Bitmap<N_BITS_PER_PAGE>,
    data: Vec<T>,
}

impl<T: Sized + Clone> Default for LayerPageData<T> {
    fn default() -> Self {
        LayerPageData {
            map: Default::default(),
            data: vec![],
        }
    }
}

impl<T: Sized + Clone> LayerPageData<T> {
    pub fn mem_usage(&self) -> usize {
        assert!(cfg!(memory_profiling = "on"));
        return size_of::<Bitmap<N_BITS_PER_PAGE>>() + self.data.len() * size_of::<T>();
    }

    pub fn page_get(&self, offset: usize) -> Option<&T> {
        if self.map.get(offset) {
            let vec_off = le_popcnt(self.map.clone(), offset);
            Some(&self.data[vec_off])
        } else {
            None
        }
    }

    pub fn page_get_mut(&mut self, offset: usize) -> Option<&mut T> {
        if self.map.get(offset) {
            let vec_off = le_popcnt(self.map.clone(), offset);
            self.data.get_mut(vec_off)
        } else {
            None
        }
    }

    pub fn page_set(&mut self, offset: usize, val: T) -> Option<T> {
        let vec_off = le_popcnt(self.map.clone(), offset);
        return if self.map.get(offset) {
            let old = std::mem::replace(&mut self.data[vec_off], val);

            Some(old)
        } else {
            self.data.insert(vec_off, val);
            self.map.set(offset, true);
            None
        };
    }

    pub fn page_reset(&mut self, offset: usize) -> Option<T> {
        let vec_off = le_popcnt(self.map.clone(), offset);
        return if self.map.get(offset) {
            self.map.set(offset, false);
            Some(self.data.remove(vec_off))
        } else {
            None
        };
    }

    pub fn page_len(&self) -> usize {
        return self.data.len();
    }

    pub fn page_get_bitmap(&self) -> &<BitsImpl<N_BITS_PER_PAGE> as Bits>::Store {
        self.map.as_value()
    }

    pub fn page_get_values(&self) -> &[T] {
        &self.data.as_slice()
    }
}

#[derive(Clone)]
pub enum BitmappedMapPage<T: Sized + Clone> {
    LeafPage {
        data: LayerPageData<T>,
    },
    InnerPage {
        data: LayerPageData<Box<BitmappedMapPage<T>>>,
        n_items: usize,
        layer: usize,
        mem_usage: usize,
    },
}

impl<T: Sized + Clone> BitmappedMapPage<T> {
    pub fn from_data(data: LayerPageData<T>) -> Self {
        BitmappedMapPage::LeafPage { data }
    }

    fn page_get(&self, index: usize) -> Option<&T> {
        match self {
            BitmappedMapPage::LeafPage { data } => data.page_get(index),
            BitmappedMapPage::InnerPage { data, layer, .. } => {
                let (local, next) = destruct(*layer, index);

                if let Some(res) = data.page_get(local) {
                    res.as_ref().page_get(next)
                } else {
                    None
                }
            }
        }
    }

    fn page_get_mut(&mut self, index: usize) -> Option<&mut T> {
        match self {
            BitmappedMapPage::LeafPage { data } => data.page_get_mut(index),
            BitmappedMapPage::InnerPage { data, layer, .. } => {
                let (local, next) = destruct(*layer, index);

                if let Some(res) = data.page_get_mut(local) {
                    res.as_mut().page_get_mut(next)
                } else {
                    None
                }
            }
        }
    }

    pub fn get_mem_usage(&self) -> usize {
        match self {
            BitmappedMapPage::LeafPage { data } => data.mem_usage(),
            BitmappedMapPage::InnerPage { mem_usage, .. } => *mem_usage,
        }
    }

    fn page_set(&mut self, index: usize, value: T) -> Option<T> {
        match self {
            BitmappedMapPage::LeafPage { data } => data.page_set(index, value),
            BitmappedMapPage::InnerPage {
                data,
                layer,
                n_items,
                mem_usage,
            } => {
                let (local, lower) = destruct(*layer, index);

                let upper_res: Option<T>;

                if let Some(res) = data.page_get_mut(local) {
                    if cfg!(memory_profiling = "on") {
                        *mem_usage -= res.get_mem_usage();
                    }

                    upper_res = res.as_mut().page_set(lower, value);

                    if cfg!(memory_profiling = "on") {
                        *mem_usage += res.get_mem_usage();
                    }
                } else {
                    data.page_set(local, Box::new(BitmappedMapPage::new(*layer - 1)));

                    if cfg!(memory_profiling = "on") {
                        *mem_usage += size_of::<T>();
                    }

                    let page = data.page_get_mut(local).unwrap();

                    upper_res = page.as_mut().page_set(lower, value);

                    if cfg!(memory_profiling = "on") {
                        *mem_usage += page.get_mem_usage() + size_of::<T>();
                    }
                }
                // If the child didn't have an entry at this place, we inserted a new item.
                // In that case, increase our count by one.
                if upper_res.is_none() {
                    *n_items += 1;
                }
                return upper_res;
            }
        }
    }

    fn page_reset(&mut self, index: usize) -> Option<T> {
        match self {
            BitmappedMapPage::LeafPage { data } => data.page_reset(index),
            BitmappedMapPage::InnerPage {
                data,
                n_items,
                layer,
                mem_usage,
            } => {
                let (local, lower) = destruct(*layer, index);

                if let Some(page) = data.page_get_mut(local) {
                    if cfg!(memory_profiling = "on") {
                        *mem_usage -= page.get_mem_usage();
                    }
                    let upper_res: Option<T> = page.as_mut().page_reset(lower);

                    if upper_res.is_some() {
                        *n_items = *n_items - 1;

                        if page.as_mut().page_len() == 0 {
                            data.page_reset(local);

                            if cfg!(memory_profiling = "on") {
                                // We lost one reference, and all memory from the dropped page
                                *mem_usage -= size_of::<T>();
                            }
                        } else {
                            if cfg!(memory_profiling = "on") {
                                *mem_usage += page.get_mem_usage();
                            }
                        }
                    }
                    return upper_res;
                } else {
                    None
                }
            }
        }
    }

    pub fn page_len(&self) -> usize {
        match self {
            BitmappedMapPage::LeafPage { data } => data.page_len(),
            BitmappedMapPage::InnerPage { n_items, .. } => *n_items,
        }
    }

    fn first_index(&self) -> Option<usize> {
        match self {
            BitmappedMapPage::LeafPage { data } => data.map.first_index(),
            BitmappedMapPage::InnerPage { data, layer, .. } => {
                data.map.first_index().and_then(|upper| {
                    data.page_get(upper).and_then(|page| {
                        page.as_ref()
                            .first_index()
                            .map(|lower| construct(*layer, upper, lower))
                    })
                })
            }
        }
    }

    fn last_index(&self) -> Option<usize> {
        match self {
            BitmappedMapPage::LeafPage { data } => data.map.last_index(),
            BitmappedMapPage::InnerPage { data, layer, .. } => {
                data.map.last_index().and_then(|upper| {
                    data.page_get(upper).and_then(|page| {
                        page.as_ref()
                            .last_index()
                            .map(|lower| construct(*layer, upper, lower))
                    })
                })
            }
        }
    }

    fn next_index(&self, index: usize) -> Option<usize> {
        match self {
            BitmappedMapPage::LeafPage { data } => {
                return data.map.next_index(index);
            }
            BitmappedMapPage::InnerPage { data, layer, .. } => {
                let (local, lower) = destruct(*layer, index);

                let result;
                if lower + 1 < N_BITS_PER_PAGE {
                    result = data
                        .page_get(local)
                        .and_then(|b| b.as_ref().next_index(lower))
                } else {
                    result = None
                }

                if result.is_some() {
                    return result;
                }

                return data.map.next_index(local).and_then(|index| {
                    data.page_get(index)
                        .and_then(|page| page.as_ref().first_index())
                });
            }
        }
    }

    pub fn new(height: usize) -> BitmappedMapPage<T> {
        if height == 0 {
            BitmappedMapPage::LeafPage {
                data: Default::default(),
            }
        } else {
            BitmappedMapPage::InnerPage {
                data: Default::default(),
                n_items: 0,
                layer: height,
                mem_usage: size_of::<BitmappedMapPage<T>>(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayeredBitmap<T: Sized + Clone> {
    data: BitmappedMapPage<T>,
    size: usize,
}

impl<T: Sized + Clone> Debug for BitmappedMapPage<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BitmappedMapPage ({})", self.page_len())
    }
}

impl<T: Sized + Clone> LayeredBitmap<T> {
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.page_get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.page_get_mut(index)
    }

    pub fn get_mem_usage(&self) -> usize {
        return self.data.get_mem_usage();
    }

    pub fn set(&mut self, index: usize, value: T) -> Option<T> {
        self.data.page_set(index, value)
    }

    pub fn reset(&mut self, index: usize) -> Option<T> {
        self.data.page_reset(index)
    }

    pub fn len(&self) -> usize {
        self.data.page_len()
    }

    pub fn new(size: usize) -> LayeredBitmap<T> {
        // We need ceil(log2(size)) bits to represent all possible items. Because each layer
        // represents PER_LAYER_SPECIFICITY_BITS, we need ceil(log2(size) /
        // PER_LAYER_SPECIFICITY_BITS) layers. Following calculation does that calculation using
        // (mostly) integer arithmatic.
        //
        // TODO: Move to usize::log2 when that becomes stable
        // ref: https://github.com/rust-lang/rust/issues/70887.
        let depth = (((size as f64).log2() as usize) + PER_LAYER_SPECIFICITY_BITS - 1)
            / PER_LAYER_SPECIFICITY_BITS;
        LayeredBitmap {
            data: BitmappedMapPage::<T>::new(depth - 1),
            size,
        }
    }

    pub fn new_from_page(page: BitmappedMapPage<T>) -> LayeredBitmap<T> {
        let size = match &page {
            BitmappedMapPage::LeafPage { .. } => N_BITS_PER_PAGE,
            BitmappedMapPage::InnerPage { layer, .. } => {
                N_BITS_PER_PAGE.pow((*layer as u32) + 1u32)
            }
        };

        LayeredBitmap { data: page, size }
    }

    pub fn iter(&self) -> BitMapIterator<T> {
        return BitMapIterator {
            map: self,
            current_index: None,
        };
    }

    fn next(&self, index: usize) -> Option<usize> {
        self.data.next_index(index)
    }
    pub fn first(&self) -> Option<usize> {
        self.data.first_index()
    }
    pub fn last(&self) -> Option<usize> {
        self.data.last_index()
    }

    pub fn backing_store(&self) -> &BitmappedMapPage<T> {
        &self.data
    }

    pub fn block_size(&self) -> usize {
        N_BITS_PER_PAGE
    }
    pub fn max_length(&self) -> usize {
        self.size
    }
}

impl LayeredBitmap<()> {
    pub fn get_bool(&self, index: usize) -> bool {
        self.data.page_get(index).is_some()
    }

    pub fn set_bool(&mut self, index: usize, value: bool) -> bool {
        if value {
            self.data.page_set(index, ()).is_some()
        } else {
            self.data.page_reset(index).is_some()
        }
    }

    pub fn reset_bool(&mut self, index: usize) -> bool {
        self.data.page_reset(index).is_some()
    }
}

pub struct BitMapIterator<'a, T: Sized + Clone> {
    map: &'a LayeredBitmap<T>,
    current_index: Option<usize>,
}

impl<'a, T: Sized + Clone> Iterator for BitMapIterator<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let result;
        match self.current_index {
            None => {
                result = self.map.first();
            }
            Some(index) => {
                if index == usize::MAX {
                    return None;
                }
                result = self.map.next(index);
            }
        }

        match result {
            None => {
                self.current_index = Some(usize::MAX);
                None
            }
            Some(index) => {
                self.current_index = Some(index);
                self.map.get(index).map(|value| (index, value))
            }
        }
    }
}

impl<'a, T: Sized + Clone> IntoIterator for &'a LayeredBitmap<T> {
    type Item = (usize, &'a T);
    type IntoIter = BitMapIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        return BitMapIterator {
            map: self,
            current_index: None,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_checks() {
        const INDEX: usize = 0;

        let mut map: LayeredBitmap<()> = LayeredBitmap::new(20);
        assert_eq!(map.len(), 0);
        map.set(INDEX, ());
        map.set(INDEX + 1, ());
        assert_eq!(map.len(), 2);
        map.set(INDEX, ());
        assert_eq!(map.len(), 2);

        // Also correctly decrease lengths
        map.reset(INDEX);
        assert_eq!(map.len(), 1);
        map.reset(INDEX);
        assert_eq!(map.len(), 1);
        map.reset(INDEX + 1);
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn length_checks_multilayer() {
        const INDEX: usize = 0;

        let mut map: LayeredBitmap<()> = LayeredBitmap::new(20);
        assert_eq!(map.len(), 0);
        map.set(INDEX, ());
        map.set(INDEX + 1, ());
        assert_eq!(map.len(), 2);
        map.set(INDEX, ());
        assert_eq!(map.len(), 2);

        // Also correctly decrease lengths
        map.reset(INDEX);
        assert_eq!(map.len(), 1);
        map.reset(INDEX);
        assert_eq!(map.len(), 1);
        map.reset(INDEX + 1);
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn data_checks() {
        const INDEX: usize = 0;

        let mut map: LayeredBitmap<u8> = LayeredBitmap::new(20);
        let res = map.set(INDEX, 1);
        assert_eq!(res, None);
        let res = map.get(INDEX);
        assert_eq!(res, Option::<&u8>::Some(&1));
        let res = map.set(INDEX, 2);
        assert_eq!(res, Some(1));
        map.set(INDEX + 1, 3);
        let res = map.get(INDEX + 1);
        assert_eq!(res, Some(&3));
    }

    #[test]
    fn iter_checks() {
        const INDEX: usize = 0;

        let mut map: LayeredBitmap<u8> = LayeredBitmap::new(20);
        map.set(INDEX, 1);
        map.set(INDEX + 1, 2);
        map.set(INDEX + 3, 3);

        assert_eq!(
            map.into_iter().collect::<Vec<(usize, &u8)>>(),
            [(INDEX, &1), (INDEX + 1, &2), (INDEX + 3, &3)]
        );
    }

    #[test]
    fn data_checks_multilayer() {
        const INDEX: usize = 3usize * N_BITS_PER_PAGE;

        let mut map: LayeredBitmap<u8> = LayeredBitmap::new(20usize * N_BITS_PER_PAGE);
        let res = map.set(INDEX, 1);
        assert_eq!(res, None);
        let res = map.get(INDEX);
        assert_eq!(res, Option::<&u8>::Some(&1));
        let res = map.set(INDEX, 2);
        assert_eq!(res, Some(1));
        map.set(INDEX + 1, 3);
        let res = map.get(INDEX + 1);
        assert_eq!(res, Some(&3));
    }
}
