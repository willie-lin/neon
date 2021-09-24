use crate::layered_repository::RELISH_SEG_SIZE;
use crate::layered_repository::blob::BlobRange;

use serde::{Deserialize, Serialize};
use zenith_utils::lsn::Lsn;

//
//
//  map of blocks
//
// 0: offset
// 1:
// 2:
// 3: 
//
// LSN1 offset
// LSN2
// LSN3
// LSN4
// LSN1
// LSN2
// LSN3
//

#[derive(Serialize, Deserialize)]
pub struct BlockLsnMap {
    block_indexes: Vec<usize>,
    page_versions: Vec<(Lsn, BlobRange)>,
}

impl BlockLsnMap {

    pub fn new() -> Self {
        BlockLsnMap {
            block_indexes: Vec::new(),
            page_versions: Vec::new(),
        }
    }

    pub fn push(&mut self, blknum: u32, lsn: Lsn, blob_range: BlobRange) {
        let i = self.page_versions.len();

        // This function must be called with increasing block numbers
        assert!(blknum as usize + 1 >= self.block_indexes.len());

        // If this is a new block that we haven't seen yet, store the index where
        // its page versions start. 'resize' will fill any gap between the last
        // seen block number and the new one.
        self.block_indexes.resize((blknum + 1) as usize, i);

        // TODO: assert that the lsn is greater than any previous lsn for this block
        self.page_versions.push((lsn, blob_range));
    }

    pub fn get_page_reconstruct_versions(&self, blknum: u32, lsn: Lsn) -> &[(Lsn, BlobRange)] {
        // Figure out the range of elements in the 'page_versions' array that we're interested
        // in.
        let start_index = self.block_indexes[blknum as usize] as usize;
        let end_index = if blknum == RELISH_SEG_SIZE - 1 {
            self.page_versions.len()
        } else {
            self.block_indexes[(blknum + 1) as usize] as usize
        };

        let mut i = start_index;
        while i < end_index {
            if self.page_versions[i].0 > lsn {
                break;
            }
            i += 1;
        }
        &self.page_versions[start_index..i]
    }
}
