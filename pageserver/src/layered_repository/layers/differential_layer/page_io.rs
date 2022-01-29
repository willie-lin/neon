use std::io::{Read, Write};
use bytes::Bytes;
use anyhow::{Result};
use varuint::{ReadVarint, WriteVarint};

pub fn read_page<T>(from: &mut T) -> Result<Bytes>
    where
        T: Read
{
    let mut page_length: u32 = ReadVarint::<u32>::read_varint(from)?;
    let mut vec: Vec<u8> = Vec::with_capacity(page_length as usize);

    from.take(page_length as u64).read_to_end(&mut vec)?;

    Ok(Bytes::from(vec))
}

pub fn write_page<T>(to: &mut T, page: &Bytes) -> Result<()>
    where
        T: Write
{
    let page_length = page.len() as u32;
    to.write_varint(page_length)?;
    to.write_all(&page[..])?;

    Ok(())
}
