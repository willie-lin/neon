use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use postgres_ffi::pg_constants::BLCKSZ;
use std::fs::File;
use std::mem::{align_of, size_of};
use std::ops::Deref;
use std::os::unix::fs::FileExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(align(8192))]
pub struct BufPage([u8; BLCKSZ as usize]);

impl BufPage {
    pub fn data_mut(&mut self) -> &mut [u8; BLCKSZ as usize] {
        &mut self.0
    }

    pub fn data(&self) -> &[u8; BLCKSZ as usize] {
        &self.0
    }
}

impl Default for BufPage {
    fn default() -> Self {
        Self([0u8; BLCKSZ as usize])
    }
}

pub fn get_buf_mut<'a>(
    file: &mut File,
    blockno: u32,
    buf_with_lock: &'a RwLock<BufPage>,
) -> RwLockWriteGuard<'a, BufPage> {
    let mut buf = buf_with_lock.write();

    extend_file_if_needed(file, blockno);
    file.read_exact_at(&mut buf.0, (blockno as u64) * (BLCKSZ as u64))
        .unwrap();

    return buf;
}

pub fn get_buf<'a>(
    file: &File,
    blockno: u32,
    buf_with_lock: &'a RwLock<BufPage>,
) -> RwLockReadGuard<'a, BufPage> {
    let mut buf = buf_with_lock.write();

    if file.metadata().unwrap().len() < (blockno as u64 + 1) * (BLCKSZ as u64) {
        buf.0.fill(0u8);
        return RwLockWriteGuard::downgrade(buf);
    }
    file.read_exact_at(&mut buf.0, (blockno as u64) * (BLCKSZ as u64))
        .unwrap();

    return RwLockWriteGuard::downgrade(buf);
}

pub fn write_buf(file: &mut File, blockno: u32, buf: &BufPage) {
    extend_file_if_needed(file, blockno);
    file.write_at(&buf.0, (blockno as u64) * (BLCKSZ as u64))
        .unwrap();
}

pub fn extend_file_if_needed(file: &mut File, blockno: u32) {
    if file.metadata().unwrap().len() < (blockno as u64 + 1) * (BLCKSZ as u64) {
        file.set_len((blockno as u64 + 1) * (BLCKSZ as u64))
            .unwrap();
    }
}

pub fn write_item_to_blocky_file<'a, T>(
    file: &mut File,
    cur_blockno: u32,
    cur_offset: u16,
    data: &T,
    buf: &mut RwLockWriteGuard<'a, BufPage>,
    my_buf: &'a RwLock<BufPage>,
) -> ((u32, u16), (u32, u16))
where
    T: Copy,
{
    let mut my_blockno = cur_blockno;
    let mut my_offset = cur_offset;
    assert_eq!(size_of::<T>() % align_of::<T>(), 0);
    assert!(size_of::<T>() <= BLCKSZ as usize);

    if ((BLCKSZ - my_offset) as usize) < size_of::<T>() {
        write_buf(file, my_blockno, buf.deref());
        my_blockno += 1;
        my_offset = 0;
        let tmp_lock = RwLock::new(BufPage::default());
        *buf = unsafe { std::mem::transmute(tmp_lock.write()) };
        *buf = get_buf_mut(file, my_blockno, &my_buf);
    }

    if (my_offset as usize) % align_of::<T>() != 0 {
        my_offset += (align_of::<T>() - ((my_offset as usize) % align_of::<T>())) as u16;
    }

    let start = (my_blockno, my_offset);

    let (_, unused_data) = buf.data_mut().split_at_mut(my_offset as usize);

    // Safety: We know the data section is aligned, and has space available.
    let (padding, data_ptr, _tail) = unsafe { unused_data.align_to_mut::<T>() };

    debug_assert_eq!(padding.len(), 0);
    debug_assert!(data_ptr.len() > 0);

    data_ptr[0] = *data;
    my_offset += size_of::<T>() as u16;

    return (start, (my_blockno, my_offset));
}

pub fn read_item_from_blocky_file<'a, T>(
    file: &File,
    cur_blockno: u32,
    cur_offset: u16,
    data: &mut T,
    buf: &mut RwLockReadGuard<'a, BufPage>,
    my_buf: &'a RwLock<BufPage>,
) -> (u32, u16)
where
    T: Copy,
{
    let mut my_blockno = cur_blockno;
    let mut my_offset = cur_offset;
    assert_eq!(size_of::<T>() % align_of::<T>(), 0);
    assert!(size_of::<T>() <= BLCKSZ as usize);

    if ((BLCKSZ - my_offset) as usize) < size_of::<T>() {
        my_blockno += 1;
        my_offset = 0;
        *buf = get_buf(file, my_blockno, &my_buf);
    }

    if (my_offset as usize) % align_of::<T>() != 0 {
        my_offset += (align_of::<T>() - ((my_offset as usize) % align_of::<T>())) as u16;
    }

    let (_, unused_data) = buf.data().split_at(my_offset as usize);

    // Safety: We know the data section is aligned, and has space available.
    let (padding, data_ptr, _tail) = unsafe { unused_data.align_to::<T>() };
    debug_assert_eq!(padding.len(), 0);
    debug_assert!(data_ptr.len() > 0);

    *data = data_ptr[0];
    my_offset += size_of::<T>() as u16;

    return (my_blockno, my_offset);
}

#[must_use]
pub fn write_slice_to_blocky_file<'a, T>(
    file: &mut File,
    cur_blockno: u32,
    cur_offset: u16,
    data: &[T],
    buf: &mut RwLockWriteGuard<'a, BufPage>,
    my_buf: &'a RwLock<BufPage>,
) -> (u32, u16)
where
    T: Copy,
{
    let mut remaining = data.len();
    let mut my_blockno = cur_blockno;
    let mut my_offset = cur_offset;

    debug_assert_eq!(size_of::<T>() % align_of::<T>(), 0);

    if (my_offset as usize) % align_of::<T>() != 0 {
        my_offset += (align_of::<T>() - ((my_offset as usize) % align_of::<T>())) as u16;
    }

    while remaining > 0 {
        if ((BLCKSZ - my_offset) as usize) < size_of::<T>() {
            write_buf(file, my_blockno, buf.deref());
            my_blockno += 1;
            my_offset = 0;
            let tmp_lock = RwLock::new(BufPage::default());
            *buf = unsafe { std::mem::transmute(tmp_lock.write()) };
            *buf = get_buf_mut(file, my_blockno, my_buf);
        }

        let (_, overwritable_data) = buf.data_mut().split_at_mut(my_offset as usize);
        // Safety: The written data is always u128; and aligned.
        let (padding, storage_array, end) = unsafe { overwritable_data.align_to_mut::<T>() };

        debug_assert_eq!(padding.len(), 0);
        debug_assert_eq!((my_offset as usize) % align_of::<T>(), 0);

        let start = data.len() - remaining;
        let added = remaining.min(storage_array.len());
        let end = start + added;

        storage_array[..added].copy_from_slice(&data[start..end]);
        remaining -= added;
        my_offset += (added * size_of::<T>()) as u16;
    }

    return (my_blockno, my_offset);
}

#[must_use]
pub fn read_slice_from_blocky_file<'a, T>(
    file: &File,
    cur_blockno: u32,
    cur_offset: u16,
    buffer: &mut [T],
    buf: &mut RwLockReadGuard<'a, BufPage>,
    my_buf: &'a RwLock<BufPage>,
) -> (u32, u16)
where
    T: Copy,
{
    let mut my_blockno = cur_blockno;
    let mut my_offset = cur_offset;

    if (my_offset as usize) % align_of::<T>() != 0 {
        my_offset += (align_of::<T>() - ((my_offset as usize) % align_of::<T>())) as u16;
    }

    let mut remaining = buffer.len();

    debug_assert_eq!(size_of::<T>() % align_of::<T>(), 0);

    while remaining > 0 {
        if ((BLCKSZ - my_offset) as usize) < size_of::<T>() {
            my_blockno += 1;
            my_offset = 0;
            *buf = get_buf(file, my_blockno, my_buf);
        }

        let (_, readable_data) = buf.data().split_at(my_offset as usize);
        // Safety: The written data is always u128; and aligned.
        let (padding, storage_array, end) = unsafe { readable_data.align_to::<T>() };

        debug_assert_eq!(padding.len(), 0);
        debug_assert_eq!((my_offset as usize) % align_of::<T>(), 0);

        let start = buffer.len() - remaining;
        let added = remaining.min(storage_array.len());
        let end = start + added;

        let data_buf = &mut buffer[start..end];
        data_buf.copy_from_slice(&storage_array[..added]);
        remaining -= added;
        my_offset += (added * size_of::<T>()) as u16;
    }

    return (my_blockno, my_offset);
}
