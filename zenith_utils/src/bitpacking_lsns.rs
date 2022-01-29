use std::io::{Cursor, Read, SeekFrom, Write};
use bytes::Buf;
use tantivy_bitpacker::{BitPacker, BitUnpacker};
use tokio::io::AsyncSeekExt;
use varuint::{ReadVarint, WriteVarint};
use crate::lsn::Lsn;

/// A bitpacker for LSNs.
/// Stores LSNs in a compact, incremental manner.
///
/// Access times:
///     Random: O(n)
///     First: O(1) (amortized)
///     Next: O(1) (amortized)
///
/// Overhead: 1 byte per block of 32 LSNs for storing the number of significant bits stored,
///  plus 1 varuint per 32 LSNs for storing the minimum increment in those 32 LSNs.
/// The 1 byte per 32 LSNs overhead can be reused for up to 3 more sequential blocks of 32 LSNs.
///
/// Implementation taken and adapted from tantivy_bitpacker::BlockedBitpacker; under MIT licence.
///
/// Alterations from the original:
///  - Block size from 128 to 32
///     LSNs are highly variable; smaller blocks should be more efficient.
///  - We do differential encoding instead of absolute encoding.
///      O(1) random access is not important; it is more important to keep numbers small.
///  - We manually / unpack the Metadata entries into the 'offset_and_bits' field (here: 'bits').
///  - We've applied an optimization utilizing two unused bits in the num_bits_block metadata variable.
///  - We truncate the 2 unused LSN bits (WAL records are always aligned to 4 bytes; we don't need
///     those last 2 bits of precision).


const COMPRESS_BLOCK_SIZE: usize = 32;
/// Alignment is to MAXALIGN, which is 8 bytes,
/// which means we can round up Lsns to the next multiple of 8.
const IGNORE_LSN_BITS: usize = 3;


pub struct LsnPacker {
    /// base LSN of the packer, used for delta-encoding.
    base_lsn: Lsn,
    /// number of items packed
    n_items: usize,
    /// buffer of uncompressed deltas
    buffer: Vec<u64>,
    /// last lsn that was packed
    latest_value: u64,
    /// compressed blocks
    compressed_blocks: Vec<u8>,
    /// bit-lengths and min deltas of compressed blocks.
    bits: Cursor<Vec<u8>>,
    /// index of the last bit-length in the bits field
    bitsindex: usize,
}

pub struct LsnUnPacker {
    /// the base LSN in the delta-encoded stream.
    base_lsn: Lsn,
    /// total number of items in the stream.
    n_remaining: usize,
    /// remaining number of items in the stream.
    last_value: u64,
    /// last decompressed LSN, used for delta-encodings.
    buffer: Vec<u64>,
    /// buffer of decompressed deltas
    next_buf_block_offset: usize,
    /// offset of the next block in the buffer.
    data: Vec<u8>,
    /// compressed blocks
    bits: Cursor<Vec<u8>>,
    /// bit-lengths of compressed blocks.
    n_items: usize,
}

impl LsnPacker {
    pub fn new(base_lsn: Lsn) -> Self {
        Self {
            base_lsn,
            n_items: 0,
            latest_value: base_lsn.0,
            buffer: Vec::new(),
            compressed_blocks: vec![0; 8],
            bits: Cursor::new(Vec::new()),
            bitsindex: 0
        }
    }

    pub fn expected_length(num_items: usize, meta: &[u8]) -> usize {
        let mut cursor = Cursor::new(meta.to_vec());
        let mut total_length = 0usize;
        let mut items_remaining = num_items;
        while items_remaining > 0 {
            let blocks_and_size = cursor.get_u8();
            let mut num_blocks = (blocks_and_size >> 6) as usize + 1;
            let num_bits = blocks_and_size & 0b0011_1111 as usize;
            while num_blocks > 0 {
                let _ = ReadVarint::<u32>::read_varint(&mut cursor).unwrap();
                let num_items_in_block = items_remaining.min(COMPRESS_BLOCK_SIZE);
                items_remaining -= num_items_in_block;
                num_blocks -= 1;
                total_length += ((num_items_in_block * num_bits as usize) + 7) / 8;
            }
        }
        return total_length;
    }

    /// Return the inner parts of this packer. Data must be flushed.
    pub fn finish(self) -> (Lsn, usize, Vec<u8>, Vec<u8>) {
        assert!(self.buffer.is_empty());
        (
            self.base_lsn,
            self.n_items,
            self.bits.into_inner(),
            self.compressed_blocks,
        )
    }

    #[inline]
    pub fn add(&mut self, lsn: Lsn) {
        let value = lsn.0;
        self.buffer.push((value - self.latest_value) >> IGNORE_LSN_BITS);
        self.n_items += 1;
        self.latest_value = value;
        if self.buffer.len() == COMPRESS_BLOCK_SIZE as usize {
            self.flush();
        }

        debug_assert!(self.buffer.len() == self.n_items % COMPRESS_BLOCK_SIZE);
    }

    pub fn flush(&mut self) {
        if let Some((min_value, max_value)) = tantivy_bitpacker::minmax(self.buffer.iter()) {
            let mut bit_packer = BitPacker::new();
            let num_bits_block = tantivy_bitpacker::compute_num_bits(*max_value - min_value);
            // todo performance: the padding handling could be done better, e.g. use a slice and
            // return num_bytes written from bitpacker
            self.compressed_blocks
                .resize(self.compressed_blocks.len() - 8, 0); // remove padding for bitpacker
            let offset = self.compressed_blocks.len() as u64;
            // todo performance: for some bit_width we
            // can encode multiple vals into the
            // mini_buffer before checking to flush
            // (to be done in BitPacker)
            for val in self.buffer.iter() {
                bit_packer
                    .write(
                        *val - min_value,
                        num_bits_block,
                        &mut self.compressed_blocks,
                    )
                    .expect("cannot write bitpacking to output"); // write to in memory can't fail
            }
            bit_packer.flush(&mut self.compressed_blocks).unwrap();

            // If we do not yet have any data in the bits field, we can't apply the following
            // optimization.
            let skip_push = if self.bits.get_ref().len() > 0 {
                // Optimization: As the num_bits_block value is always between 0 and 62 (= 64 -
                // IGNORE_LSN_BITS), we can commandeer the top 2 bits of our
                // 'bits of precision in next BLOCKSZ ints' u8 value to store how many sequential
                // blocks use this number of bits. This saves ~ 1 byte per sequential block of this
                // prefix, so in a worst case this encoding is equal to that of a native
                // delta-encoded blocked_bitpacker, but does much better when the deltas vary
                // significantly between blocks.
                // It does have some extra overhead though, in that we need to store 4x as many
                // min_values as opposed to the tantivy_bitpacker::BlockedBitpacker implementation.
                let val = self.bits.get_ref()[self.bitsindex];
                let res = (val & 0b11_1111 == num_bits_block) && (val >> 6 != 0b11);
                res
            } else {
                false
            };

            // If we use the same number of bits as the previous block and the previous block info
            // still has bits left, we've updated that block. In all other cases, we just add a
            // new block here.
            if skip_push {
                self.bits.get_mut()[self.bitsindex] += 1 << 6;
            } else {
                assert!(num_bits_block <= 0b0011_1111);
                self.bitsindex = self.bits.position() as usize;
                self.bits
                    .write(&[num_bits_block; 1])
                    .expect("cannot write to memory?");
            }

            self.bits.write_varint(*min_value).expect("Cannot write varint");

            self.buffer.clear();
            self.compressed_blocks
                .resize(self.compressed_blocks.len() + 8, 0); // add padding for bitpacker
        }
    }
}

impl LsnUnPacker {
    pub fn new(base_lsn: Lsn, n_items: usize, data: &[u8], bits: &[u8]) -> Self {
        let mut data_vec = data.to_vec();
        data_vec.extend_from_slice(&[0; 7]);

        Self {
            base_lsn,
            n_items,
            n_remaining: n_items,
            last_value: base_lsn.0,
            buffer: Vec::with_capacity(COMPRESS_BLOCK_SIZE),
            next_buf_block_offset: 0,
            data: data_vec,
            bits: Cursor::new(bits.to_vec()),
        }
    }

    fn unpack_next_block(&mut self) -> Option<()> {
        if self.bits.get_ref().len() == 0 {
            return None;
        }
        let blocks_and_bits = {
            let mut it = [0u8; 1];
            self.bits.read(&mut it[..]).ok()?;
            it[0]
        };

        let num_blocks = 1 + (blocks_and_bits >> 6) as usize;
        let num_bits = blocks_and_bits & 0b0011_1111;

        self.buffer.resize(self.n_remaining.min(num_blocks as usize * COMPRESS_BLOCK_SIZE), 0);
        let mut bit_unpacker = BitUnpacker::new(num_bits);
        let mut base_offsets = vec![0u64; num_blocks as usize];

        for idx in 0..num_blocks {
            base_offsets[idx] = self.bits.read_varint().ok()?;
        }

        for idx in 0..self.buffer.len() {
            self.buffer[idx] = base_offsets[idx / COMPRESS_BLOCK_SIZE] + bit_unpacker.get(
                idx as u64,
                &mut self.data[self.next_buf_block_offset as usize..],
            );
        }

        // Add to the offset. All but the last block are guaranteed to be a multiple of
        // BLOCK_SIZE in size, thus always divisible by 8.
        self.next_buf_block_offset += (self.buffer.len() * num_bits as usize) / 8;

        Some(())
    }

}

impl Iterator for LsnUnPacker {
    type Item = Lsn;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.is_empty() {
            self.unpack_next_block()?;
        }

        let delta = self.buffer.remove(0) << IGNORE_LSN_BITS;


        self.last_value += delta;
        return Some(Lsn(self.last_value));
    }
}

#[cfg(test)]
mod test {
    use rand::{random, Rng};
    use crate::bitpacking_lsns::IGNORE_LSN_BITS;
    use crate::lsn::Lsn;
    use super::{LsnPacker, LsnUnPacker};

    fn test_packer_util(data: Vec<Lsn>) {
        let first = data.first().unwrap().clone();

        let mut packer = LsnPacker::new(first);
        for it in data.iter() {
            packer.add(*it);
        }
        let (base_lsn, n_items, data, bits) = packer.finish();

        let unpacked = LsnUnPacker::new(base_lsn, n_items, &data, &bits);
        let unpacked = unpacked.collect::<Vec<Lsn>>();
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_valid_lsns() {
        let mut rng = rand::thread_rng();

        let data = {
            let mut data = vec![Lsn(0); 128];
            let mut my_lsn = Lsn(0);
            for i in data.iter_mut() {
            let delta: u32 = rng.gen();
            * i = my_lsn;
            my_lsn = Lsn(my_lsn.0 + (delta as u64) << IGNORE_LSN_BITS);
            }
            data
        };

        test_packer_util(data);
        test_packer_util(vec![Lsn(0), Lsn(128),  Lsn(0xffff_fff0)]);
    }
}
