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
use bytes::Bytes;
use parking_lot::RwLock;
use std::fs::File;
use std::mem::{align_of, size_of};
use std::num::NonZeroU32;
use zenith_utils::lsn::Lsn;

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
    current_data_offset: (u32, u16),
    next_version_infos_block: u32,
    version_infos_buf: Vec<VersionInfo>,
    version_infos_buf_start: usize,
    n_total_versions: usize,
}

impl<'a> LineageBranchVersionIterator<'a> {
    fn new(file: &'a File, branch: LineageBranchInfo) -> Self {
        LineageBranchVersionIterator {
            file,
            branch,
            next_version_infos_block: 0,
            versionarray_index: None,
            current_data_offset: (0, 0),
            version_infos_buf: vec![],
            version_infos_buf_start: 0,
            n_total_versions: 0,
        }
    }

    fn update_version_infos_buffer(&mut self) {
        let my_buf: RwLock<BufPage> = RwLock::new(BufPage::default());
        self.version_infos_buf_start += self.version_infos_buf.len();
        let buf = get_buf(self.file, self.next_version_infos_block, &my_buf);

        let n_local_recs =
            VERSIONS_PER_PAGE.min(self.n_total_versions - self.version_infos_buf_start);
        let versions_arr = {
            let (versions_data, _) = buf.data().split_at(n_local_recs * size_of::<VersionInfo>());
            let (head, arr, tail) = unsafe { versions_data.align_to::<VersionInfo>() };
            assert!(head.is_empty());
            assert!(tail.is_empty());
            assert_eq!(arr.len(), n_local_recs);
            arr
        };

        self.version_infos_buf.clear();
        self.version_infos_buf.extend_from_slice(versions_arr);
        self.next_version_infos_block += 1;
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
            let local_offset = index - self.version_infos_buf_start;

            if local_offset >= self.version_infos_buf.len() {
                self.update_version_infos_buffer();
            }

            let local_offset = index - self.version_infos_buf_start;

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

        let mut my_buf: RwLock<BufPage> = RwLock::new(BufPage::default());

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

        match self.versionarray_index {
            // Initializing the iterator, returning the oldest record of the branch
            None => {
                self.versionarray_index = Some(0);
                let branch_hdr_ptr = self.branch.branch_ptr_data.item_ptr;

                let mut blockno = branch_hdr_ptr.blockno();
                let mut offset = branch_hdr_ptr.offset();

                let mut buf = get_buf(self.file, blockno, &my_buf);
                let (_, my_data) = buf.data().split_at(branch_hdr_ptr.offset() as usize);

                let typ: InitializedBranch;
                let versions_arr: &[u8];

                match self.branch.branch_ptr_data.typ {
                    BranchPtrType::PageImage => {
                        let (hdr, data) =
                            my_data.split_at(size_of::<PageOnlyLineageBranchHeader>());
                        let hdr = {
                            let (head, arr, tail) =
                                unsafe { hdr.align_to::<PageOnlyLineageBranchHeader>() };
                            assert!(head.is_empty());
                            assert!(tail.is_empty());
                            assert_eq!(arr.len(), 1);
                            arr[0]
                        };

                        offset += size_of::<PageOnlyLineageBranchHeader>() as u16;
                        let mut bytes = vec![0u8; u32::from(hdr.length) as usize];

                        let _ = read_slice_from_blocky_file(
                            self.file, blockno, offset, &mut bytes, &mut buf, &my_buf,
                        );
                        return Some((
                            self.branch.start_lsn,
                            PageVersion::Page(Bytes::from(bytes)),
                        ));
                    }
                    BranchPtrType::WalRecord => {
                        let (hdr, data) = my_data.split_at(size_of::<WalOnlyLineageBranchHeader>());

                        let hdr = {
                            let (head, arr, tail) =
                                unsafe { hdr.align_to::<WalOnlyLineageBranchHeader>() };
                            assert!(head.is_empty());
                            assert!(tail.is_empty());
                            assert_eq!(arr.len(), 1);
                            arr[0]
                        };

                        debug_assert!(hdr.length >= hdr.main_data_offset);

                        offset += size_of::<WalOnlyLineageBranchHeader>() as u16;
                        let mut bytes = vec![0u8; u32::from(hdr.length) as usize];

                        let _ = read_slice_from_blocky_file(
                            self.file,
                            blockno,
                            offset,
                            &mut bytes[..],
                            &mut buf,
                            &my_buf,
                        );
                        return Some((
                            self.branch.start_lsn,
                            PageVersion::Wal(WALRecord {
                                will_init: true,
                                rec: Bytes::from(bytes),
                                main_data_offset: hdr.main_data_offset,
                            }),
                        ));
                    }
                    BranchPtrType::PageInitiatedBranch => {
                        let (hdr, data) =
                            my_data.split_at(size_of::<PageInitiatedLineageBranchHeader>());
                        let hdr = {
                            let (head, arr, tail) =
                                unsafe { hdr.align_to::<PageInitiatedLineageBranchHeader>() };
                            assert!(head.is_empty());
                            assert!(tail.is_empty());
                            assert_eq!(arr.len(), 1);
                            arr[0]
                        };

                        offset += size_of::<PageInitiatedLineageBranchHeader>() as u16;
                        self.next_version_infos_block = blockno;
                        self.n_total_versions = u32::from(hdr.num_entries) as usize;

                        typ = InitializedBranch::Page(hdr);
                        versions_arr = data;
                    }
                    BranchPtrType::WalInitiatedBranch => {
                        let (hdr, data) =
                            my_data.split_at(size_of::<WalInitiatedLineageBranchHeader>());
                        let hdr = {
                            let (head, arr, tail) =
                                unsafe { hdr.align_to::<WalInitiatedLineageBranchHeader>() };
                            assert!(head.is_empty());
                            assert!(tail.is_empty());
                            assert_eq!(arr.len(), 1);
                            arr[0]
                        };

                        debug_assert!(hdr.head.length >= hdr.head.main_data_offset);

                        offset += size_of::<WalInitiatedLineageBranchHeader>() as u16;
                        self.next_version_infos_block = blockno;
                        self.n_total_versions = u32::from(hdr.num_entries) as usize;

                        typ = InitializedBranch::Wal(hdr);
                        versions_arr = data;
                    }
                };

                // Initialize the local version infos buffer.
                {
                    self.version_infos_buf_start = 0;

                    self.next_version_infos_block = blockno + 1;

                    let n_local_versions =
                        (LAYER_PAGE_SIZE - offset as usize) / size_of::<VersionInfo>();

                    self.version_infos_buf.reserve(
                        self.n_total_versions
                            .min(n_local_versions.max(self.n_total_versions - n_local_versions))
                            .min(VERSIONS_PER_PAGE),
                    );

                    assert_eq!(versions_arr.len() % align_of::<VersionInfo>(), 0);

                    let (versions_data, tail) =
                        versions_arr.split_at(n_local_versions * size_of::<VersionInfo>());

                    let versions_arr = {
                        let (head, arr, tail) = unsafe { versions_data.align_to::<VersionInfo>() };
                        assert!(head.is_empty());
                        assert!(tail.is_empty());
                        assert_eq!(arr.len(), n_local_versions);
                        arr
                    };

                    self.version_infos_buf.clear();
                    self.version_infos_buf.extend_from_slice(versions_arr);

                    let n_non_local = self.n_total_versions - self.version_infos_buf.len();

                    if n_non_local > 0 {
                        let pages = n_non_local / VERSIONS_PER_PAGE;
                        blockno = self.next_version_infos_block + (pages as u32);
                        offset =
                            ((n_non_local % VERSIONS_PER_PAGE) * size_of::<VersionInfo>()) as u16;
                        self.current_data_offset = (blockno, offset);
                    } else {
                        self.current_data_offset = (blockno, (LAYER_PAGE_SIZE - tail.len()) as u16);
                    }
                }

                let mut buf = get_buf(self.file, self.current_data_offset.0, &my_buf);
                let mut record_data = vec![0u8; u32::from(typ.data_len()) as usize];

                self.current_data_offset = read_slice_from_blocky_file(
                    self.file,
                    self.current_data_offset.0,
                    self.current_data_offset.1,
                    &mut record_data,
                    &mut buf,
                    &my_buf,
                );

                let version = match typ {
                    InitializedBranch::Page(_) => PageVersion::Page(Bytes::from(record_data)),
                    InitializedBranch::Wal(hdr) => PageVersion::Wal(WALRecord {
                        will_init: true,
                        rec: Bytes::from(record_data),
                        main_data_offset: hdr.head.main_data_offset,
                    }),
                };

                return Some((self.branch.start_lsn, version));
            }
            Some(index) => {
                // If we're at the end of the versions of this branch, we are done.
                if index >= self.n_total_versions {
                    return None;
                }

                let local_index = index - self.version_infos_buf_start;

                // If the local version info buffer is all used up, we need to read more.
                if local_index >= self.version_infos_buf.len() {
                    self.update_version_infos_buffer();
                }

                // recalculate in case we updated the buffer
                let local_index = index - self.version_infos_buf_start;
                let version_info = &self.version_infos_buf[local_index];
                let mut buf = get_buf(self.file, self.current_data_offset.0, &my_buf);
                let mut record_data = vec![0u8; u32::from(version_info.length) as usize];
                self.current_data_offset = read_slice_from_blocky_file(
                    self.file,
                    self.current_data_offset.0,
                    self.current_data_offset.1,
                    &mut record_data,
                    &mut buf,
                    &my_buf,
                );

                self.versionarray_index = Some(index + 1);
                return Some((
                    Lsn(u64::from_ne_bytes(version_info.lsn)),
                    PageVersion::Wal(WALRecord {
                        will_init: false,
                        rec: Bytes::from(record_data),
                        main_data_offset: version_info.main_data_offset,
                    }),
                ));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.versionarray_index {
            None => (self.n_total_versions + 1, Some(self.n_total_versions + 1)),
            Some(index) => (
                self.n_total_versions - index,
                Some(self.n_total_versions - index),
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
