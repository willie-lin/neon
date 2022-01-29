use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use postgres_ffi::pg_constants::BLCKSZ;
use std::fs::File;
use std::io::{BufWriter, IoSlice, IoSliceMut, Write};
use std::mem::{align_of, MaybeUninit, size_of};
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
    mut buf: RwLockWriteGuard<'a, BufPage>,
    my_buf: &'a RwLock<BufPage>,
) -> ((u32, u16), (u32, u16), RwLockWriteGuard<'a, BufPage>)
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
        drop(buf);
        buf = get_buf_mut(file, my_blockno, &my_buf);
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

    return (start, (my_blockno, my_offset), buf);
}

// macro_rules! read_buf (
//     ( $buf:ident, $typ:ty ) => {
//         {
//             let (it, tail_data) = $buf.split_at(size_of::<$typ>());
//             let (head, data_ptr, tail) = unsafe { it.align_to_mut::<$typ>() };
//
//             debug_assert_eq!(head.len(), 0);
//             debug_assert_eq!(tail.len(), 0);
//             debug_assert_eq!(data_ptr.len(), 1);
//             $buf = tail_data;
//             &data_ptr[0]
//         }
//     }
//     ( $buf:ident, $typ:ty, $n:literal ) => {
//         {
//             debug_assert!($buf.len() >= size_of::<$typ>() * $n);
//
//             let (it, tail_data) = $buf.split_at(size_of::<$typ>() * $n:literal);
//             let (head, data_ptr, tail) = unsafe { it.align_to_mut::<$typ>() };
//
//             debug_assert_eq!(head.len(), 0);
//             debug_assert_eq!(tail.len(), 0);
//             debug_assert_eq!(data_ptr.len(), $n);
//             $buf = tail_data;
//             data_ptr
//         }
//     }
// );

#[macro_export]
macro_rules! val_to_write (
    ( $val:ident ) => {
        {
            let tmp = core::slice::from_ref(& $val );
            let (_, bytes, _) = unsafe { tmp.align_to::<u8>() };
            std::io::IoSlice::new(bytes)
        }
    };
    ( $drop:pat => $val:ident) => {
        {
            let tmp = & $val [..];
            let (_, bytes, _) = unsafe { tmp.align_to::<u8>() };
            std::io::IoSlice::new(bytes)
        }
    };
);

#[macro_export]
macro_rules! write_to_file (
    ( $writer:ident, $offset:ident, $( $val:ident $( $p:pat )? ),+ ) => {
        {
            let mut bufs = vec![ ( $(
                {
                    val_to_write!( $( $p => )? $val )
                }
            ),+ )];

            let mut bufs_ref: &mut [std::io::IoSlice] = &mut bufs[..];

            $writer.seek(std::io::SeekFrom::Start($offset))?;

            loop {
                let mut len_written = std::io::Write::write_vectored($writer, bufs_ref)?;
                $offset += len_written as u64;

                while len_written != 0 {
                    if bufs_ref[0].len() < len_written {
                        len_written -= bufs_ref[0].len();
                        bufs_ref = &mut bufs_ref[1..];
                    } else {
                        bufs_ref[0] = std::io::IoSlice::new(bufs_ref[0].split_at(len_written).1);
                        break;
                    }
                }

                if bufs_ref.len() == 0 {
                    break;
                }
            }

            Ok($offset)
        }
    }
);

#[macro_export]
macro_rules! val_to_read (
    ( $val:ident ) => {
        {
            let tmp = core::slice::from_mut(& $val );
            let (_, bytes, _) = unsafe { tmp.align_to::<u8>() };
            std::io::IoSliceMut::new(bytes)
        }
    };
    ( $drop:pat => $val:ident) => {
        {
            let tmp = &mut $val [..];
            let (_, bytes, _) = unsafe { tmp.align_to_mut::<u8>() };
            std::io::IoSliceMut::new(bytes)
        }
    };
);


#[macro_export]
macro_rules! read_from_file (
    ( $writer:expr, $offset:ident, $( $val:ident $( $p:pat )? ),+ ) => {
        {
            let mut bufs = vec![ $(
                {
                    val_to_read!( $( $p => )? $val )
                }
            ),+ ];
            let mut bufs_ref: &mut [std::io::IoSliceMut] = &mut bufs[..];

            std::io::Seek::seek($writer, std::io::SeekFrom::Start($offset))?;

            loop {
                let mut len_read = std::io::Read::read_vectored($writer, bufs_ref)?;
                $offset += len_read as u64;

                while len_read != 0 {
                    if bufs_ref[0].len() <= len_read {
                        len_read -= bufs_ref[0].len();
                        bufs_ref = &mut bufs_ref[1..];
                    } else {
                        // We now have fully processed the read components.
                        bufs_ref[0] = std::io::IoSliceMut::new(bufs_ref[0].split_at_mut(len_read).1);
                        break;
                    }
                }

                while bufs_ref.len() >= 0 && bufs_ref[0].len() == 0 {
                    bufs_ref = &mut bufs_ref[1..];
                }

                if bufs_ref.len() == 0 {
                    break;
                }
            }

            Ok($offset)
        }
    }
);


pub fn read_item_from_blocky_file<'a, T>(
    file: &File,
    cur_blockno: u32,
    cur_offset: u16,
    data: &mut T,
    mut buf: RwLockReadGuard<'a, BufPage>,
    my_buf: &'a RwLock<BufPage>,
) -> ((u32, u16), RwLockReadGuard<'a, BufPage>)
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
        drop(buf);
        buf = get_buf(file, my_blockno, &my_buf);
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

    return ((my_blockno, my_offset), buf);
}

#[must_use]
pub fn write_slice_to_blocky_file<'a, T>(
    file: &mut File,
    cur_blockno: u32,
    cur_offset: u16,
    data: &[T],
    mut buf: RwLockWriteGuard<'a, BufPage>,
    my_buf: &'a RwLock<BufPage>,
) -> ((u32, u16), RwLockWriteGuard<'a, BufPage>)
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
            drop(buf);
            buf = get_buf_mut(file, my_blockno, my_buf);
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

    return ((my_blockno, my_offset), buf);
}

#[must_use]
pub fn read_slice_from_blocky_file<'a, T>(
    file: &File,
    cur_blockno: u32,
    cur_offset: u16,
    buffer: &mut [T],
    mut buf: RwLockReadGuard<'a, BufPage>,
    my_buf: &'a RwLock<BufPage>,
) -> ((u32, u16), RwLockReadGuard<'a, BufPage>)
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
            drop(buf);
            buf = get_buf(file, my_blockno, my_buf);
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

    return ((my_blockno, my_offset), buf);
}
