use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::num::NonZeroU32;

use anyhow::{Result};
use byteorder::{ByteOrder, ReadBytesExt};
use bytes::{Buf, Bytes};
use varuint::*;
use crate::repository::WALRecord;


const MAX_REC_HEADER_SIZE: usize = 5 /* length = 2**32 */ + 5 /* main_data_offset = 2**32 */;


pub struct WalStream<T> {
    buffer: T,
    bytes_consumed: u64,
}

impl<T> WalStream<T> {
    pub fn new(buffer: T) -> WalStream<T> {
        WalStream {
            buffer,
            bytes_consumed: 0,
        }
    }
}

impl<T> WalStream<T>
    where
        T: Read
{
    fn read_record(&mut self) -> Result<WALRecord> {

        let mut rec_length: u32 = ReadVarint::<u32>::read_varint(&mut self.buffer)?;
        let rec_mdo: u32 = ReadVarint::<u32>::read_varint(&mut self.buffer)?;

        let mut vec: Vec<u8> = Vec::with_capacity(rec_length as usize);

        self.buffer.take(rec_length as u64).read_to_end(&mut vec)?;

        self.bytes_consumed += rec_length as u64 + rec_length.varint_size() + rec_mdo.varint_size();

        Ok(WALRecord {
            will_init: false,
            main_data_offset: NonZeroU32::new(0)?,
            rec: Bytes::from(vec),
        })
    }
}

impl<T> Iterator for WalStream<T>
    where
        T: Read
{
    type Item = WALRecord;

    fn next(&mut self) -> Option<Self::Item> {
        return self.read_record().ok();
    }
}

impl<T> WalStream<T>
    where
        T: Write
{
    pub fn write_page(&mut self, page: &Bytes) -> Result<()> {
        let page_size: u32 = page.len() as u32;

        self.buffer.write_varint(page_size)?;
        self.buffer.write_all(&page[..])?;
        self.bytes_consumed += page_size.varint_size() + page_size as u64;

        Ok(())
    }

    pub fn write_record(&mut self, record: &WALRecord) -> Result<()> {
        let rec_length: u32 = record.rec.len() as u32;

        self.buffer.write_varint(rec_length)?;
        self.buffer.write_all(&record.rec[..]);

        self.bytes_consumed += rec_length as u64 + rec_length.varint_size();

        Ok(())
    }
}
