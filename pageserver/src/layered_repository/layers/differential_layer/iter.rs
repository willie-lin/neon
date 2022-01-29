/// Iterator types on the differential layer's data types.

use crate::layered_repository::filebufutils::{get_buf, read_slice_from_blocky_file, BufPage};
use crate::layered_repository::layers::differential_layer::file_format::{
    BranchPtrType, LineageBranchInfo, LineageInfo, PageInitiatedLineageBranchHeader,
    PageOnlyLineageBranchHeader, VersionInfo, WalInitiatedLineageBranchHeader,
    WalOnlyLineageBranchHeader,
};
use crate::layered_repository::layers::differential_layer::LineagePointer;
use crate::layered_repository::layers::LAYER_PAGE_SIZE;
use crate::layered_repository::storage_layer::PageVersion;
use crate::repository::WALRecord;
use anyhow::{Result};
use bytes::Bytes;
use parking_lot::RwLock;
use std::fs::File;
use core::mem::{align_of, size_of};
use core::num::NonZeroU32;
use std::cmp::Ordering;
use hyper::Version;
use zenith_utils::lsn::Lsn;
use crate::{read_from_file, val_to_read};

const VERSIONS_PER_PAGE: usize = LAYER_PAGE_SIZE / size_of::<VersionInfo>();
const BRANCHINFOS_PER_PAGE: usize = LAYER_PAGE_SIZE / size_of::<LineageBranchInfo>();

impl LineageBranchInfo {
    pub fn iter<'a>(&self, file: &'a mut File) -> LineageBranchVersionIterator<'a> {
        return LineageBranchVersionIterator::new(file, *self);
    }
}

pub struct LineageBranchVersionIterator<'a> {
    /// The file we're reading from
    file: &'a mut File,
    /// the branch descriptor that we're iterating over
    branch: LineageBranchInfo,
    /// The last returned index of the iterator. None if none have been returned yet.
    versionarray_index: Option<usize>,
    /// The current offset of the start of the data buffer in the file;
    current_file_data_offset: usize,
    /// The offset of the end of the version infos buffer in the file.
    next_version_infos_file_offset: usize,
    /// Buffer holding version infos.
    /// NB: Length is unrelated to n_total_versions; this buffer might be larger than that.
    version_infos_buf: Vec<VersionInfo>,
    /// The index of the first version info in the buffer; i.e. how many version infos of
    /// the branch are located before the first one in the buffer.
    version_infos_buf_first_version_offset: usize,
    /// Buffer holding data bytes for the versioninfos.
    data_buf: Vec<u8>,
    /// The index of the first not-yet-used data byte.
    data_buf_first_unused: usize,
    /// The total number of versions in this branch.
    n_total_versions: usize,
}

impl<'a> LineageBranchVersionIterator<'a> {
    fn new(file: &'a mut File, branch: LineageBranchInfo) -> Self {
        LineageBranchVersionIterator {
            file,
            branch,
            next_version_infos_file_offset: branch.branch_ptr_data.item_ptr.full_offset(),
            versionarray_index: None,
            current_file_data_offset: 0usize,
            version_infos_buf: vec![],
            version_infos_buf_first_version_offset: 0,
            data_buf: vec![],
            data_buf_first_unused: 0,
            n_total_versions: 0,
        }
    }

    fn update_version_infos_buffer(&mut self) -> Result<()> {
        self.version_infos_buf_first_version_offset += self.version_infos_buf.len();
        let n_local_recs = VERSIONS_PER_PAGE.min(self.n_total_versions - self.version_infos_buf_first_version_offset);
        self.version_infos_buf.resize(n_local_recs, VersionInfo::default());

        {
            let file = self.file;
            let read_buf = &mut self.version_infos_buf[..];
            let mut offset = self.next_version_infos_file_offset as u64;
            read_from_file!(file, offset, read_buf []);
        }

        self.next_version_infos_file_offset += 1;
        Ok(())
    }

    /// Returns the LSN of the lsn that will be returned at the next next() call.
    /// Note: This modifies the version_infos_buf and version_infos_buf_start to include the next
    /// version info, if any. Do not rely on previous contents of that buffer.
    pub fn peek_lsn(&mut self) -> Option<Lsn> {
        return if let Some(index) = self.versionarray_index {
            // there
            if index >= self.n_total_versions {
                return None;
            }
            let local_offset = index - self.version_infos_buf_first_version_offset;

            if local_offset >= self.version_infos_buf.len() {
                self.update_version_infos_buffer();
            }

            let local_offset = index - self.version_infos_buf_first_version_offset;

            Some(Lsn(u64::from_ne_bytes(
                self.version_infos_buf[local_offset].lsn,
            )))
        } else {
            Some(self.branch.start_lsn)
        };
    }
}

impl<'a> Iterator for LineageBranchVersionIterator<'a> {
    type Item = (Lsn, PageVersion);

    fn next(&mut self) -> Option<Self::Item> {
        enum InitializedBranch {
            Page(PageInitiatedLineageBranchHeader),
            Wal(WalInitiatedLineageBranchHeader),
        }

        impl InitializedBranch {
            fn num_items(&self) -> u32 {
                match self {
                    InitializedBranch::Page(it) => it.num_entries,
                    InitializedBranch::Wal(it) => it.num_entries,
                }
            }

            fn data_len(&self) -> NonZeroU32 {
                match self {
                    InitializedBranch::Page(it) => it.head.length,
                    InitializedBranch::Wal(it) => it.head.length,
                }
            }
        }

        let mut data_buf = vec![0u8; LAYER_PAGE_SIZE];
        let mut data_buf_mut = &mut data_buf[..];

        match self.versionarray_index {
            // Initializing the iterator, returning the oldest record of the branch
            None => {
                self.versionarray_index = Some(0);
                let branch_hdr_ptr = self.branch.branch_ptr_data.item_ptr;

                let typ: InitializedBranch;
                let mut offset = branch_hdr_ptr.full_offset();

                match self.branch.branch_ptr_data.typ {
                    BranchPtrType::PageImage => {
                        let mut header = PageOnlyLineageBranchHeader::default();

                        {
                            let res = read_from_file!(self.file, offset, header, data_buf_mut []);
                            offset = res?;
                        }

                        if header.length as usize <= LAYER_PAGE_SIZE {
                            return Some((
                                self.branch.start_lsn,
                                PageVersion::Page(Bytes::from(Box::new(data_buf[..header.length as usize]))),
                            ))
                        }

                        data_buf.resize(header.length as usize, 0u8);
                        let data_buf_mut = &mut data_buf[LAYER_PAGE_SIZE..];
                        {
                            let res = read_from_file!(self.file, offset, data_buf_mut []);
                            offset = res?;
                        }
                        return Some((
                            self.branch.start_lsn,
                            PageVersion::Page(Bytes::from(data_buf)),
                        ));
                    }
                    BranchPtrType::WalRecord => {
                        let mut header = WalOnlyLineageBranchHeader::default();

                        {
                            let res = read_from_file!(self.file, offset, header, data_buf_mut []);
                            offset = res?;
                        }
                        if header.length <= LAYER_PAGE_SIZE {
                            return Some((
                                self.branch.start_lsn,
                                PageVersion::Wal(WALRecord {
                                    will_init: true,
                                    rec: Bytes::from(Box::new(data_buf[..header.length as usize])),
                                    main_data_offset: header.main_data_offset
                                }),
                            ))
                        }
                        data_buf.resize(header.length as usize, 0u8);
                        let data_buf_mut = &mut data_buf[LAYER_PAGE_SIZE..];
                        {
                            let res = read_from_file!(self.file, offset, data_buf_mut []);
                            offset = res?;
                        }
                        return Some((
                            self.branch.start_lsn,
                            PageVersion::Wal(WALRecord {
                                will_init: true,
                                rec: Bytes::from(data_buf),
                                main_data_offset: header.main_data_offset
                            }),
                        ));
                    }
                    BranchPtrType::PageInitiatedBranch => {
                        let mut header = PageInitiatedLineageBranchHeader::default();

                        {
                            let res = read_from_file!(self.file, offset, header, data_buf_mut []);
                            offset = res?;
                        }

                        self.n_total_versions = header.num_entries as usize;

                        let n_local_versions = self.n_total_versions.min(VERSIONS_PER_PAGE);
                        let (entries, tail) = data_buf_mut.split_at_mut(n_local_versions * LAYER_PAGE_SIZE);
                        data_buf_mut = tail;

                        self.next_version_infos_file_offset = branch_hdr_ptr.full_offset()
                            + size_of::<PageInitiatedLineageBranchHeader>()
                            + entries.len();

                        let (head, data, tail) = unsafe { entries.align_to::<VersionInfo>() };

                        debug_assert_eq!(head.len(), 0);
                        debug_assert_eq!(tail.len(), 0);
                        debug_assert_eq!(data.len(), n_local_versions);

                        self.version_infos_buf.extend_from_slice(data);

                        self.current_file_data_offset = self.branch.branch_ptr_data.item_ptr.full_offset()
                            + size_of::<PageInitiatedLineageBranchHeader>()
                            + self.n_total_versions * size_of::<VersionInfo>();

                        typ = InitializedBranch::Page(header);
                    }
                    BranchPtrType::WalInitiatedBranch => {
                        let mut header = WalInitiatedLineageBranchHeader::default();

                        {
                            let res = read_from_file!(self.file, offset, header, data_buf_mut []);
                            offset = res?;
                        }

                        self.n_total_versions = header.num_entries as usize;

                        let n_local_versions = self.n_total_versions.min(VERSIONS_PER_PAGE);
                        let (entries, tail) = data_buf_mut.split_at_mut(n_local_versions * LAYER_PAGE_SIZE);
                        data_buf_mut = tail;
                        self.next_version_infos_file_offset = branch_hdr_ptr.full_offset()
                            + size_of::<WalInitiatedLineageBranchHeader>()
                            + entries.len();
                        let (head, data, tail) = unsafe { entries.align_to::<VersionInfo>() };

                        debug_assert_eq!(head.len(), 0);
                        debug_assert_eq!(tail.len(), 0);
                        debug_assert_eq!(data.len(), n_local_versions);

                        self.version_infos_buf.extend_from_slice(data);

                        self.current_file_data_offset = self.branch.branch_ptr_data.item_ptr.full_offset()
                            + size_of::<WalInitiatedLineageBranchHeader>()
                            + self.n_total_versions * size_of::<VersionInfo>();

                        typ = InitializedBranch::Wal(header);
                    }
                }

                self.data_buf = data_buf;

                // We've initialized the iterator fields, now all we need to do is return the first value

                // If the first value is already buffered, we can simply return it.
                if typ.data_len() <= data_buf_mut.len() && self.n_total_versions == n_local_versions {
                    self.data_buf_first_unused = typ.data_len() as usize;
                    let mut bytes = Bytes::from(Vec::from(&data_buf_mut[..typ.data_len() as usize]));

                    return Some((
                        self.branch.start_lsn,
                        match typ {
                            InitializedBranch::Page(_) => PageVersion::Page(bytes),
                            InitializedBranch::Wal(rec) =>
                                PageVersion::Wal(WALRecord {
                                    will_init: true,
                                    rec: bytes,
                                    main_data_offset: rec.head.main_data_offset,
                                })
                        }
                    ))
                }

                // The buf is emptied and resized to contain the record data, plus
                // potentially some extra space for subsequent record data (if available)
                //
                // Note that this can read past the end of this branch's data, and thus past the end of the file.
                self.data_buf.resize(LAYER_PAGE_SIZE.max(typ.data_len() as usize), 0u8);

                {
                    offset = self.current_file_data_offset;
                    let data_buf_mut = &mut self.data_buf[..];
                    let res = read_from_file!(self.file, offset, data_buf_mut []);
                    self.current_file_data_offset = res?;
                }
                self.data_buf_first_unused = typ.data_len() as usize;
                let bytes = Bytes::from(Vec::from(&self.data_buf[..typ.data_len() as usize]));

                return Some((
                    self.branch.start_lsn,
                    match typ {
                        InitializedBranch::Page(_) => PageVersion::Page(bytes),
                        InitializedBranch::Wal(rec) =>
                            PageVersion::Wal(WALRecord {
                                will_init: true,
                                rec: bytes,
                                main_data_offset: rec.head.main_data_offset,
                            })
                    }
                ));
            }
            Some(index) => {
                // If we're at the end of the versions of this branch, we are done.
                if index >= self.n_total_versions {
                    return None;
                }

                let local_index = index - self.version_infos_buf_first_version_offset;

                // If the local version info buffer is all used up, we need to read more.
                if local_index >= self.version_infos_buf.len() {
                    self.update_version_infos_buffer();
                }

                // recalculate in case we updated the buffer
                let local_index = index - self.version_infos_buf_first_version_offset;
                let version_info = &self.version_infos_buf[local_index];

                if version_info.length > self.data_buf.len() - self.data_buf_first_unused {
                    self.data_buf.resize(LAYER_PAGE_SIZE.max(version_info.length as usize), 0u8);

                    {
                        let offset = self.current_file_data_offset;
                        let data_buf_mut = &mut self.data_buf[..];

                        let res = read_from_file!(self.file, offset, data_buf_mut []);
                        res?;
                    }

                    self.data_buf_first_unused = 0usize;
                }

                let bytes = Bytes::from(Box::new(self.data_buf[self.data_buf_first_unused..(self.data_buf_first_unused + version_info.length as usize)]));
                self.data_buf_first_unused += version_info.length as usize;
                self.current_file_data_offset += version_info.length as usize;

                self.versionarray_index = Some(index + 1);

                return Some((
                    Lsn(u64::from_ne_bytes(version_info.lsn)),
                    PageVersion::Wal(WALRecord {
                        will_init: false,
                        rec: bytes,
                        main_data_offset: version_info.main_data_offset,
                    }),
                ));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.versionarray_index {
            None => (1, None),
            Some(_) => (
                self.n_total_versions + 1,
                Some(self.n_total_versions + 1),
            ),
        }
    }
}

pub struct LineageBranchIterator<'a> {
    /// The file we're reading from.
    file: &'a File,
    /// The lineage pointer we're iterating over
    start_ptr: LineagePointer,
    /// The index of the last returned branch [in the lineage]
    branchversions_index: Option<usize>,
    /// The file offset of the end of the branch_infos_buf
    next_branches_offset: usize,
    /// A buffer holding a section of lineage's branch pointers.
    branch_infos_buf: Vec<LineageBranchInfo>,
    /// The start of the buffers' contents in the lineage's branches array
    branch_infos_buf_start: usize,
    /// The total number of branches in the lineage
    n_total_branches: usize,
}

impl LineagePointer {
    pub(crate) fn iter<'a>(&self, file: &'a File) -> LineageBranchIterator<'a> {
        LineageBranchIterator {
            file,
            start_ptr: self.clone(),
            branchversions_index: None,
            next_branches_offset: 0,   // filled at first iteration
            branch_infos_buf: Vec::new(), // empty by default, no extra allocation
            branch_infos_buf_start: 0,
            n_total_branches: 0,
        }
    }
}

impl<'a> LineageBranchIterator<'a> {
    fn new(file: &'a File, start_ptr: LineagePointer) -> LineageBranchIterator<'a> {
        LineageBranchIterator {
            file,
            start_ptr,
            branchversions_index: None,
            next_branches_offset: start_ptr.0.full_offset(),
            branch_infos_buf: Vec::new(),
            branch_infos_buf_start: 0,
            n_total_branches: 0,
        }
    }

    fn buffer_next_section(&mut self) -> Result<()> {
        self.branch_infos_buf_start += self.branch_infos_buf.len();

        let remaining_items = self.n_total_branches - self.branch_infos_buf_start;

        self.branch_infos_buf.resize(
            remaining_items.min(LAYER_PAGE_SIZE / size_of::<LineageBranchInfo>()),
            LineageBranchInfo::default()
        );
        let offset = self.next_branches_offset;

        let infos_buf = &mut self.branch_infos_buf[..];

        let res = read_from_file!(self.file, offset, infos_buf []);
        res?;

        self.next_branches_offset += self.branch_infos_buf.len();
        Ok(())
    }

    fn peek_lsn(&mut self) -> Option<Lsn> {
        if self.branchversions_index.is_none() {
            return None;
        }

        let index = self.branchversions_index.unwrap();

        let local_offset = index - self.branch_infos_buf_start;

        if local_offset >= self.branch_infos_buf.len() {
            self.buffer_next_section();
        }

        let local_offset = index - self.branch_infos_buf_start;

        return Some(self.branch_infos_buf[local_offset].start_lsn);
    }
}

impl<'a> Iterator for LineageBranchIterator<'a> {
    type Item = LineageBranchInfo;

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.branchversions_index {
            None => {
                (1, None)
            }
            Some(_) => {
                (self.n_total_branches, Some(self.n_total_branches))
            }
        }
    }

    fn next(&mut self) -> Option<Self::Item> {
        match self.branchversions_index {
            None => {
                self.branchversions_index = Some(0);

                let offset = self.start_ptr.0.full_offset();
                let lineage_info = LineageInfo::default();
                let lineage_infos_buf = vec![LineageBranchInfo::default(); LAYER_PAGE_SIZE / size_of::<LineageBranchInfo>()];

                let res = read_from_file!(self.file, offset, lineage_info, lineage_infos_buf []);

                self.next_branches_offset = res?;

                self.n_total_branches = lineage_info.num_branches as usize;
                let lineage_infos = &lineage_infos_buf[..self.n_total_branches.min(LAYER_PAGE_SIZE / size_of::<LineageBranchInfo>())];
                self.branch_infos_buf.clear();
                self.branch_infos_buf.extend_from_slice(lineage_infos);

                Some(self.branch_infos_buf[0])
            }
            Some(index) => {
                if index >= self.n_total_branches {
                    return None;
                }
                self.branchversions_index = Some(index + 1);

                let local_offset = index - self.branch_infos_buf_start;
                if local_offset >= self.branch_infos_buf.len() {
                    self.buffer_next_section();
                }
                let local_offset = index - self.branch_infos_buf_start;

                return Some(self.branch_infos_buf[local_offset]);
            }
        }
    }
}
