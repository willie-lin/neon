//!
//! A DifferentialLayer represents a collection of WAL records or page images
//! in a range of LSNs, for one segment. It is stored on a file on disk.
//!
//! Usually a delta layer only contains differences - in the form of WAL
//! records against a base LSN. However, if a segment is newly created, by
//! creating a new relation or extending an old one, there might be no base image. In that case, all the entries in
//! the delta layer must be page images or WAL records with the 'will_init' flag set, so
//! that they can be replayed without referring to an older page version. Also in some
//! circumstances, the predecessor layer might actually be another delta layer. That
//! can happen when you create a new branch in the middle of a delta layer, and the WAL
//! records on the new branch are put in a new delta layer.
//!
//! When a delta file needs to be accessed, we slurp the metadata and relsize chapters
//! into memory, into the DeltaLayerInner struct. See load() and unload() functions.
//! To access a page/WAL record, we search `page_version_metas` for the block # and LSN.
//! The byte ranges in the metadata can be used to find the page/WAL record in
//! PAGE_VERSIONS_CHAPTER.
//!
//! On disk, the delta files are stored in timelines/<timelineid> directory.
//! Currently, there are no subdirectories, and each delta file is named like this:
//!
//!    <spcnode>_<dbnode>_<relnode>_<forknum>_<segno>_<start LSN>_<end LSN>
//!
//! For example:
//!
//!    1663_13990_2609_0_5_000000000169C348_000000000169C349
//!
//! If a relation is dropped, we add a '_DROPPED' to the end of the filename to indicate that.
//! So the above example would become:
//!
//!    1663_13990_2609_0_5_000000000169C348_000000000169C349_DROPPED
//!
//! The end LSN indicates when it was dropped in that case, we don't store it in the
//! file contents in any way.
//!
//! A delta file is constructed using the 'bookfile' crate. Each file consists of two
//! parts: the page versions and the relation sizes. They are stored as separate chapters.
//!
pub mod file_format;
mod iter;
mod serdes;

use crate::layered_repository::filename::{DeltaFileName, DeltaLayerType, PathOrConf};
use crate::layered_repository::storage_layer::{
    Layer, PageReconstructData, PageReconstructResult, PageVersion, SegmentTag, RELISH_SEG_SIZE,
};
use crate::PageServerConf;
use crate::{ZTenantId, ZTimelineId};
use anyhow::{ensure, Result};
use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use serde::{Deserialize, Serialize};
use static_assertions;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::mem::size_of;
use std::ops::Bound::Included;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use futures::FutureExt;
use zenith_utils::vec_map::VecMap;

use zenith_utils::lsn::Lsn;
use zenith_utils::multi_bitmap::LayeredBitmap;

use crate::layered_repository::filebufutils::{get_buf, BufPage};
use crate::layered_repository::inmemory_layer::InMemoryLayer;
use crate::layered_repository::layers::differential_layer::file_format::{InfoPageV1, InfoPageVersions, LineageInfo, UnalignFileOffset};
use crate::layered_repository::layers::differential_layer::serdes::{read_lineage, read_page_versions_map, write_latest_metadata, write_old_page_versions_section, write_page_lineages_section, write_page_versions_map};
use crate::layered_repository::layers::meta::differential_meta::{
    AsLatestLsn, DifferentialMetadata,
};
use crate::layered_repository::layers::meta::{LayerMetadata, MutLayerMetadata, PageFindResult};
use crate::layered_repository::LayeredTimeline;
use crate::repository::Timeline;

// Magic constant to identify a Zenith Differential file
pub const DELTA_FILE_MAGIC: u32 = 0x7A6E7444; // zntD

/// Contains multi-layer bitmap of (local) PageNo -> PageRevision[] for
/// contained changed pages
///
/// Size:
/// ~128 bytes static for the first level of the bitmap, plus vec overhead
/// then 128 bytes bytes + vec overhead for each section of 1024 blocks that had a change,
/// then 8 bytes for each block changed.
///
/// Each changed block receives a record of the following form:
/// { nrecords: u16, offset_hi: u16, offset_lo: u32 }. See PageChangeRecordsPointer
///
/// Improvements could be something along the lines of packing the data more efficiently
/// by wasting less bits, but that shouldn't really be an issue right now.
static PAGE_CONTAINED_INFO_CHAPTER: u64 = 1;

/// PAGE_LSNS_* contains map of PageNo -> LSN for pages changed since last
/// snapshot that are not stored in this Layer.
///
/// Size:
/// 128 bytes + Vec overhead for the first level of the bitmap,
/// then 128 bytes + Vec overhead for each section of 1024 blocks with non-local changes
/// then 8 bytes for each block with non-local changes.
static PAGE_LSNS_MAP_CHAPTER: u64 = 2;

/// Per-page mapping of lsn to pageVersion: Pageversion being either PageImage or
/// WalRecord.
///
/// Changes are stored, per page, as follows:
/// a compact array of PageVersionRecords with {nrecords} entries, followed by
/// the data of these PageVersion records.
///
/// Note that WAL records that [are larger than {THRESHOLD} and/or touch multiple pages]
/// are stored seperately in OUT_OF_LINE_STORAGE;
static PAGE_CHANGES_CHAPTER: u64 = 3;
static OUT_OF_LINE_PAGE_VERSIONS_CHAPTER: u64 = 4;

/// Contains the [`Summary`] struct
static SUMMARY_CHAPTER: u64 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct LineagePointer(UnalignFileOffset);

impl AsLatestLsn for LineagePointer {
    type Arg = Arc<File>;

    fn as_latest_lsn(&self, arg: &Self::Arg) -> Lsn {
        let blockno = self.0.blockno();
        let offset = self.0.offset();

        let mut my_buf: RwLock<BufPage> = RwLock::new(BufPage::default());
        let mut buf = get_buf(arg.deref(), blockno, &my_buf);

        let (_, start) = buf.data().split_at(offset as usize);
        let (header_data, _) = start.split_at(size_of::<LineageInfo>());

        let lineage_info = {
            let (align, lineage_header, alig) = unsafe { header_data.align_to::<LineageInfo>() };
            debug_assert_eq!(align.len(), 0);
            debug_assert_eq!(alig.len(), 0);
            debug_assert_eq!(lineage_header.len(), 1);
            &lineage_header[0]
        };

        return lineage_info.latest_lsn;
    }
}

const NONE_PAYLOAD: u8 = 0;
const PAGE_PAYLOAD: u8 = 1;
const WALRECORD_PAYLOAD: u8 = 2;
const SHARED_WALRECORED_PAYLOAD: u8 = 3;

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Summary {
    tenantid: ZTenantId,
    timelineid: ZTimelineId,
    seg: SegmentTag,

    start_lsn: Lsn,
    // inclusive
    end_lsn: Lsn, // inclusive

    dropped: bool,
}

impl From<&DifferentialLayer> for Summary {
    fn from(layer: &DifferentialLayer) -> Self {
        Self {
            tenantid: layer.tenantid,
            timelineid: layer.timelineid,
            seg: layer.seg,

            start_lsn: layer.start_lsn,
            end_lsn: layer.end_lsn,

            dropped: layer.dropped,
        }
    }
}

///
/// DifferentialLayer is the in-memory data structure associated with an
/// on-disk differential file.  We keep a DifferentialLayer in memory for each
/// file, in the LayerMap. If a layer is in "loaded" state, we have a
/// copy of the file in memory, in 'inner'. Otherwise the struct is
/// just a placeholder for a file that exists on disk, and it needs to
/// be loaded before using it in queries.
///
pub struct DifferentialLayer {
    path_or_conf: PathOrConf,

    pub tenantid: ZTenantId,
    pub timelineid: ZTimelineId,
    pub seg: SegmentTag,

    //
    // This entry contains all the changes from 'start_lsn' to 'end_lsn'. The
    // start is inclusive, and end is exclusive.
    //
    pub start_lsn: Lsn,
    pub end_lsn: Lsn,

    dropped: bool,

    inner: RwLock<Option<Arc<DifferentialLayerInner>>>,
}

pub struct DifferentialLayerInner {
    /// All versions of all pages in the file are are kept here.
    /// Indexed by block number and LSN.
    metadata: DifferentialMetadata<LineagePointer>,
    last_snapshot: Lsn,
    metapage: InfoPageV1,
    file: Arc<File>,

    /// `relsizes` tracks the size of the relation at different points in time.
    relsizes: VecMap<Lsn, u32>,
}

impl Layer for DifferentialLayer {
    fn get_tenant_id(&self) -> ZTenantId {
        self.tenantid
    }

    fn get_timeline_id(&self) -> ZTimelineId {
        self.timelineid
    }

    fn get_seg_tag(&self) -> SegmentTag {
        self.seg
    }

    fn is_dropped(&self) -> bool {
        self.dropped
    }

    fn get_start_lsn(&self) -> Lsn {
        self.start_lsn
    }

    fn get_end_lsn(&self) -> Lsn {
        self.end_lsn
    }

    fn filename(&self) -> PathBuf {
        PathBuf::from(self.layer_name().to_string())
    }

    /// Look up given page in the cache.
    fn get_page_reconstruct_data(
        &self,
        blknum: u32,
        lsn: Lsn,
        cached_img_lsn: Option<Lsn>,
        reconstruct_data: &mut PageReconstructData,
    ) -> Result<PageReconstructResult> {
        let mut need_image = true;

        assert!(self.seg.blknum_in_seg(blknum));

        {
            let inner_locked = self.inner.read();
            let inner = inner_locked.as_deref().unwrap();

            // Determine if this layer contains the block;

            let contained = inner.metadata.query_page(blknum);

            let lineagepointer = match contained {
                PageFindResult::InThisLayer(data) => data,
                PageFindResult::InOtherLayer { last_modified } => {
                    return Ok(PageReconstructResult::Continue(last_modified));
                }
                PageFindResult::UsePreviousSnapshot => {
                    return Ok(PageReconstructResult::Continue(inner.last_snapshot));
                }
                PageFindResult::TryPreviousLayer => {
                    return Ok(PageReconstructResult::Continue(self.start_lsn));
                }
            };

            let res = read_lineage(
                &inner.file,
                lineagepointer.0.blockno(),
                lineagepointer.0.offset(),
                Some(lsn),
            );

            if res.is_none() {
                panic!("Lineage did not contain block; even though authorative data was expected");
            }

            let (mut versions, lsn) = res.unwrap();

            let (first, rest) = versions.split_first_mut().unwrap();

            rest.reverse();

            for (lsn, pageversion) in rest {
                match pageversion {
                    PageVersion::Wal(rec) => {
                        reconstruct_data.records.push((*lsn, rec.clone()));
                    }
                    PageVersion::Page(_) => {
                        unreachable!(
                            "PageVersion::Page can only appear as the first item in a lineage"
                        );
                    }
                }
            }

            match &first.1 {
                PageVersion::Page(bytes) => {
                    reconstruct_data.page_img = Some(bytes.clone());
                }
                PageVersion::Wal(rec) => {
                    reconstruct_data.page_img = None;
                    reconstruct_data.records.push((first.0, rec.clone()));
                }
            }

            need_image = false;
        }

        // If an older page image is needed to reconstruct the page, let the
        // caller know.
        if need_image {
            Ok(PageReconstructResult::Continue(self.start_lsn))
        } else {
            Ok(PageReconstructResult::Complete)
        }
    }

    /// Get size of the relation at given LSN
    fn get_seg_size(&self, lsn: Lsn) -> Result<u32> {
        assert!(lsn >= self.start_lsn && lsn <= self.end_lsn);
        ensure!(
            self.seg.rel.is_blocky(),
            "get_seg_size() called on a non-blocky rel"
        );

        let inner_locked = self.inner.read();
        let inner = inner_locked.as_deref().unwrap().clone();
        let slice = inner
            .relsizes
            .slice_range((Included(&Lsn(0)), Included(&lsn)));

        if let Some((_entry_lsn, entry)) = slice.last() {
            Ok(*entry)
        } else {
            Err(anyhow::anyhow!("could not find seg size in delta layer"))
        }
    }

    /// Does this segment exist at given LSN?
    fn get_seg_exists(&self, lsn: Lsn) -> Result<bool> {
        // Is the requested LSN after the rel was dropped?
        if self.dropped && lsn >= self.end_lsn {
            return Ok(false);
        }

        // Otherwise, it exists.
        Ok(true)
    }

    ///
    /// Release most of the memory used by this layer. If it's accessed again later,
    /// it will need to be loaded back.
    ///
    fn unload(&self) -> Result<()> {
        let mut inner = self.inner.write();
        *inner = None;
        Ok(())
    }

    fn delete(&self) -> Result<()> {
        // delete underlying file
        fs::remove_file(self.path())?;
        Ok(())
    }

    fn is_incremental(&self) -> bool {
        true
    }

    /// debugging function to print out the contents of the layer
    fn dump(&self) -> Result<()> {
        println!(
            "----- differential layer for ten {} tli {} seg {} {}-{} ----",
            self.tenantid, self.timelineid, self.seg, self.start_lsn, self.end_lsn
        );

        let locked_inner = self.inner.read();
        let inner = locked_inner.as_deref().unwrap().clone();
        println!("--- relsizes ---");
        for (k, v) in inner.relsizes.as_slice() {
            println!("  {}: {}", k, v);
        }
        println!("--- page lineages ---");
        for (blk, lineage_ptr) in inner.metadata.get_versions_map().iter() {
            println!(
                "  blck: {}, lsn: {}",
                blk,
                lineage_ptr.as_latest_lsn(&inner.file)
            );

            let iterator = lineage_ptr.iter(&inner.file);
            for branchinfo in iterator {
                println!(
                    "    branch {}: {:?}",
                    branchinfo.start_lsn, branchinfo.branch_ptr_data.typ
                );
                for (lsn, data) in branchinfo.iter(&inner.file) {
                    match data {
                        PageVersion::Page(bytes) => {
                            println!("      {}: img: {}B", lsn, bytes.len())
                        }
                        PageVersion::Wal(rec) => {
                            println!(
                                "      {}: wal: {}B, mdo={}",
                                lsn,
                                rec.rec.len(),
                                rec.main_data_offset
                            )
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn latest_page_versions_since_snapshot(&self) -> Option<Box<dyn Iterator<Item = (u32, Lsn)>>> {
        return None;
    }

    fn is_in_memory(&self) -> bool {
        return self.inner.read().is_some();
    }
}

impl DifferentialLayer {
    pub fn inner(&self) -> MappedRwLockReadGuard<Arc<DifferentialLayerInner>> {
        let mut read_locked = self.inner.read();
        if read_locked.is_none() {
            drop(read_locked);
            let write_locked = self.inner.write();

            *write_locked = Some(self.load_inner());

            read_locked = RwLockWriteGuard::downgrade(write_locked);
        }

        RwLockReadGuard::map(
            read_locked,
            |opt| opt.as_ref().expect("Failed to find info for"),
        )
    }

    pub fn load_inner(&self) -> Result<Arc<DifferentialLayerInner>> {
        let file = File::open(self.path())?;
        let mut reader = BufReader::new(file);

        let metapage: InfoPageVersions = InfoPageVersions::V1(InfoPageV1::default());

        let metapage = match metapage { InfoPageVersions::V1(it) => it };


        let mut pages_contained = LayeredBitmap::new(RELISH_SEG_SIZE as usize);

        read_page_versions_map(&mut reader, metapage.new_page_versions_map_start, &mut pages_contained)?;

        let mut inner = DifferentialLayerInner {
            metadata: DifferentialMetadata::new_with_maps(

            ),
            last_snapshot: Default::default(),
            metapage,
            file: Arc::new(file),
            relsizes: Default::default()
        };
        inner.load()?;
        Ok(Arc::new(inner))
    }

    pub fn new(
        conf: &'static PageServerConf,
        timelineid: ZTimelineId,
        tenantid: ZTenantId,
        filename: &DeltaFileName,
    ) -> Self {
        DifferentialLayer {
            path_or_conf: PathOrConf::Conf(conf),
            timelineid,
            tenantid,
            seg: filename.seg,
            start_lsn: filename.start_lsn,
            end_lsn: filename.end_lsn,
            dropped: filename.dropped,
            inner: RwLock::new(None),
        }
    }

    fn path_for(
        path_or_conf: &PathOrConf,
        timelineid: ZTimelineId,
        tenantid: ZTenantId,
        fname: &DeltaFileName,
    ) -> PathBuf {
        match path_or_conf {
            PathOrConf::Path(path) => path.clone(),
            PathOrConf::Conf(conf) => conf
                .timeline_path(&timelineid, &tenantid)
                .join(fname.to_string()),
        }
    }

    pub fn create_from_src(
        conf: &'static PageServerConf,
        timeline: &LayeredTimeline,
        dropped: bool,
        inmem: &InMemoryLayer,
    ) -> Result<Box<DifferentialLayer>> {
        let path_or_conf = PathOrConf::Conf(conf);
        let filename = DeltaFileName {
            seg: inmem.get_seg_tag(),
            start_lsn: inmem.get_start_lsn(),
            end_lsn: inmem.get_end_lsn(),
            dropped,
            kind: DeltaLayerType::Differential,
        };
        let mut meta = InfoPageV1 {
            old_page_versions_map_start: 0,
            new_page_versions_map_start: 0,
            page_images_start: 0,
            page_lineages_start: 0,
            seg_lengths_start: 0,
        };
        let mut my_buffer: RwLock<BufPage> = RwLock::new(BufPage::default());

        let inmem_inner = inmem.inner.read().unwrap();
        let mut relsizes = inmem_inner.get_segsizes().clone();

        let buf = Self::path_for(
            &path_or_conf,
            inmem.get_timeline_id(),
            inmem.get_tenant_id(),
            &filename,
        );

        let mut bufwriter = BufWriter::new(OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&buf)?);
        let mut offset = 0usize;

        offset = write_latest_metadata(&mut bufwriter, offset, &meta)?;

        meta.old_page_versions_map_start = offset;

        offset = write_old_page_versions_section(
            &mut bufwriter,
            offset,
            inmem_inner.deref().changed_since_snapshot()
        )?;

        meta.page_lineages_start = offset;

        let mut lineage_block_offsets =
            LayeredBitmap::<LineagePointer>::new(RELISH_SEG_SIZE as usize);

        offset = write_page_lineages_section(
            &mut bufwriter,
            offset,
            inmem_inner.deref(),
            &mut lineage_block_offsets,
        )?;

        meta.new_page_versions_map_start = offset;

        offset = write_page_versions_map(
            &mut bufwriter,
            offset,
            &lineage_block_offsets,
        )?;

        meta.page_images_start = offset;

        let file = bufwriter.into_inner()?;

        let arc_file = Arc::new(file);

        Ok(Box::new(DifferentialLayer {
            path_or_conf: PathOrConf::Conf(conf),
            inner: RwLock::new(Some(Arc::new(DifferentialLayerInner {
                metadata: DifferentialMetadata::new_with_maps(
                    lineage_block_offsets,
                    inmem_inner.changed_since_snapshot().clone(),
                    arc_file.clone(),
                ),
                last_snapshot: Lsn(0),
                metapage: meta,
                file: arc_file,
                relsizes,
            }))),
            dropped,
            timelineid: inmem.get_timeline_id(),
            tenantid: inmem.get_tenant_id(),
            seg: inmem.get_seg_tag(),
            start_lsn: inmem.get_start_lsn(),
            end_lsn: inmem.get_end_lsn(),
        }))
    }

    fn layer_name(&self) -> DeltaFileName {
        DeltaFileName {
            seg: self.seg,
            start_lsn: self.start_lsn,
            end_lsn: self.end_lsn,
            dropped: self.dropped,
            kind: DeltaLayerType::Differential,
        }
    }

    /// Path to the layer file in pageserver workdir.
    pub fn path(&self) -> PathBuf {
        Self::path_for(
            &self.path_or_conf,
            self.timelineid,
            self.tenantid,
            &self.layer_name(),
        )
    }
}
