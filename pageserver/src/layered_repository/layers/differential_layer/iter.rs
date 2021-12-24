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
use hyper::Version;
use zenith_utils::lsn::Lsn;
use crate::read_from_file;

const VERSIONS_PER_PAGE: usize = LAYER_PAGE_SIZE / size_of::<VersionInfo>();
const BRANCHINFOS_PER_PAGE: usize = LAYER_PAGE_SIZE / size_of::<LineageBranchInfo>();

impl LineageBranchInfo {
    pub fn iter<'a>(&self, file: &'a File) -> LineageBranchVersionIterator<'a> {
        return LineageBranchVersionIterator::new(file, *self);
    }
}

pub struct LineageBranchVersionIterator<'a> {
    file: &'a File,
    branch: LineageBranchInfo,
    versionarray_index: Option<usize>,
    current_data_file_offset: usize,
    next_version_infos_file_offset: usize,
    version_infos_buf: Vec<VersionInfo>,
    version_infos_buf_first_version_offset: usize,
    data_buf: Vec<u8>,
    data_buf_first_unused: usize,
    n_total_versions: usize,
}

impl<'a> LineageBranchVersionIterator<'a> {
    fn new(file: &'a File, branch: LineageBranchInfo) -> Self {
        LineageBranchVersionIterator {
            file,
            branch,
            next_version_infos_file_offset: branch.branch_ptr_data.item_ptr.full_offset(),
            versionarray_index: None,
            current_data_file_offset: 0usize,
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
            read_from_file!(file, self.next_version_infos_file_offset, read_buf []);
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
            fn num_items(&self) -> NonZeroU32 {
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

                match branch.branch_ptr_data.typ {
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

                        self.current_data_file_offset = self.branch.branch_ptr_data.item_ptr.full_offset()
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

                        self.current_data_file_offset = self.branch.branch_ptr_data.item_ptr.full_offset()
                            + size_of::<WalInitiatedLineageBranchHeader>()
                            + self.n_total_versions * size_of::<VersionInfo>();

                        typ = InitializedBranch::Wal(header);
                    }
                }

                if (typ.data_len() as usize) < data_buf_mut.len() && self.n_total_versions == self.version_infos_buf.len() {
                    let bytes = Bytes::from(data_buf_mut.to_vec());
                    return Some((
                        self.branch.start_lsn,
                        match typ {
                            InitializedBranch::Page(_) => {
                                PageVersion::Page(bytes)
                            }
                            InitializedBranch::Wal(header) => {
                                PageVersion::Wal(WALRecord {
                                    will_init: true,
                                    rec: bytes,
                                    main_data_offset: header.head.main_data_offset,
                                })
                            }
                        },
                    ));
                }

                self.data_buf = data_buf;

                if typ.data_len() <= data_buf_mut.len() && self.n_total_versions == n_local_versions {
                    return Some((
                        self.branch.start_lsn,
                        PageVersion::Page(Bytes::from(Box::new(data_buf_mut[..header.length as usize]))),
                    ))
                }

                // The buf is emptied and resized to contain the record data, plus
                // potentially some extra space for subsequent record data (if available)
                //
                // Note that this can read past the end of this branch's data, and thus past the end of the file.
                self.data_buf.resize(LAYER_PAGE_SIZE.max(typ.data_len() as usize), 0u8);

                {
                    offset = self.current_data_file_offset;
                    let data_buf_mut = &mut self.data_buf[..];
                    let res = read_from_file!(self.file, offset, data_buf_mut []);
                    self.current_data_file_offset = res?;
                }
                self.data_buf_first_unused = typ.data_len() as usize;

                return Some((
                    self.branch.start_lsn,
                    PageVersion::Page(Bytes::from(Box::new(self.data_buf[..typ.data_len()]))),
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
                        let offset = self.current_data_file_offset;
                        let data_buf_mut = &mut self.data_buf[..];

                        let res = read_from_file!(self.file, offset, data_buf_mut []);
                        res?;
                    }

                    self.data_buf_first_unused = 0usize;
                }

                let bytes = Bytes::from(Box::new(self.data_buf[self.data_buf_first_unused..(self.data_buf_first_unused + version_info.length as usize)]));
                self.data_buf_first_unused += version_info.length as usize;
                self.current_data_file_offset += version_info.length as u64;

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
    file: &'a File,
    start_ptr: LineagePointer,
    branchversions_index: Option<usize>,
    next_branches_block: u32,
    branch_infos_buf: Vec<LineageBranchInfo>,
    branch_infos_buf_start: usize,
    n_total_branches: usize,
}

impl LineagePointer {
    pub(crate) fn iter<'a>(&self, file: &'a File) -> LineageBranchIterator<'a> {
        LineageBranchIterator {
            file,
            start_ptr: *self,
            branchversions_index: None,
            next_branches_block: 0,   // filled at first iteration
            branch_infos_buf: vec![], // empty by default, no extra allocation
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
            next_branches_block: 0,
            branch_infos_buf: Vec::new(),
            branch_infos_buf_start: 0,
            n_total_branches: 0,
        }
    }

    fn buffer_next_section(&mut self) {
        let my_buf: RwLock<BufPage> = RwLock::new(BufPage::default());
        self.branch_infos_buf_start += self.branch_infos_buf.len();
        let buf = get_buf(self.file, self.next_branches_block, &my_buf);

        let n_local_branches =
            VERSIONS_PER_PAGE.min(self.n_total_branches - self.branch_infos_buf_start);
        let branches = {
            let (versions_data, _) = buf
                .data()
                .split_at(n_local_branches * size_of::<LineageBranchInfo>());
            let (head, arr, tail) = unsafe { versions_data.align_to::<LineageBranchInfo>() };
            assert!(head.is_empty());
            assert!(tail.is_empty());
            assert_eq!(arr.len(), n_local_branches);
            arr
        };

        self.branch_infos_buf.clear();
        self.branch_infos_buf.extend_from_slice(branches);
        self.next_branches_block += 1;
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

    fn next(&mut self) -> Option<Self::Item> {
        let my_buf = RwLock::new(BufPage::default());

        match self.branchversions_index {
            None => {
                // setup iteration
                self.branchversions_index = Some(0);
                let blockno = self.start_ptr.0.blockno();
                let offset = self.start_ptr.0.offset();

                let mut buf = get_buf(self.file, blockno, &my_buf);

                let (header, branchinfos) =
                    buf.data()[(offset as usize)..].split_at(size_of::<LineageInfo>());
                let header = {
                    let (lead, data, tail) = unsafe { header.align_to::<LineageInfo>() };
                    debug_assert!(lead.is_empty());
                    debug_assert!(tail.is_empty());
                    debug_assert_eq!(data.len(), 1);
                    &data[0]
                };

                let local_branchinfos = {
                    let (lead, data, tail) = unsafe { branchinfos.align_to::<LineageBranchInfo>() };
                    debug_assert!(lead.is_empty());
                    debug_assert!(tail.is_empty());
                    debug_assert_eq!(data.len(), 1);
                    data
                };
                let n_local_branches = local_branchinfos.len();

                self.n_total_branches = header.num_branches as usize;
                self.branch_infos_buf.clear();
                self.branch_infos_buf.reserve(
                    self.n_total_branches
                        .min(n_local_branches.max(self.n_total_branches - n_local_branches))
                        .min(BRANCHINFOS_PER_PAGE),
                );
                self.branch_infos_buf_start = 0;
                self.branch_infos_buf.extend_from_slice(local_branchinfos);
                self.next_branches_block = blockno + 1;

                if n_local_branches == 0 {
                    self.buffer_next_section();
                }

                return Some(self.branch_infos_buf[0]);
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
