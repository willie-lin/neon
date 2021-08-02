//! Acceptor part of proposer-acceptor consensus algorithm.

use anyhow::{anyhow, bail, Result};
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use byteorder::WriteBytesExt;
use bytes::Buf;
use bytes::Bytes;
use log::*;
use postgres_ffi::xlog_utils::TimeLineID;
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::io;
use std::io::Read;

use crate::replication::HotStandbyFeedback;
use postgres_ffi::xlog_utils::MAX_SEND_SIZE;
use zenith_utils::bin_ser::LeSer;
use zenith_utils::lsn::Lsn;
use zenith_utils::pq_proto::SystemId;
use zenith_utils::zid::{ZTenantId, ZTimelineId};

pub const SK_MAGIC: u32 = 0xcafeceefu32;
pub const SK_FORMAT_VERSION: u32 = 1;
const SK_PROTOCOL_VERSION: u32 = 1;
const UNKNOWN_SERVER_VERSION: u32 = 0;

/// Consensus logical timestamp.
type Term = u64;

/// Unique id of proposer. Not needed for correctness, used for monitoring.
type PgUuid = [u8; 16];

/// Persistent consensus state of the acceptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptorState {
    /// acceptor's last term it voted for (advanced in 1 phase)
    pub term: Term,
    /// acceptor's epoch (advanced, i.e. bumped to 'term' when VCL is reached).
    pub epoch: Term,
}

/// Information about Postgres. Safekeeper gets it once and then verifies
/// all further connections from computes match.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Postgres server version
    pub pg_version: u32,
    pub system_id: SystemId,
    /// Zenith timelineid
    pub ztli: ZTimelineId,
    pub tli: TimeLineID,
    pub wal_seg_size: u32,
}

/// Persistent information stored on safekeeper node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeKeeperState {
    /// magic for verifying content the control file
    pub magic: u32,
    /// safekeeper format version
    pub format_version: u32,
    /// persistent acceptor state
    pub acceptor_state: AcceptorState,
    /// information about server
    pub server: ServerInfo,
    /// Unique id of the last *elected* proposer we dealed with. Not needed
    /// correctness, exists for monitoring purposes.
    pub proposer_uuid: PgUuid,
    /// part of WAL acknowledged by quorum
    pub commit_lsn: Lsn,
    /// minimal LSN which may be needed for recovery of some safekeeper: min(commit_lsn) for all safekeepers
    pub restart_lsn: Lsn,
}

impl SafeKeeperState {
    pub fn new() -> SafeKeeperState {
        SafeKeeperState {
            magic: SK_MAGIC,
            format_version: SK_FORMAT_VERSION,
            acceptor_state: AcceptorState { term: 0, epoch: 0 },
            server: ServerInfo {
                pg_version: UNKNOWN_SERVER_VERSION, /* Postgres server version */
                system_id: 0,                       /* Postgres system identifier */
                ztli: ZTimelineId::from([0u8; 16]),
                tli: 0,
                wal_seg_size: 0,
            },
            proposer_uuid: [0; 16],
            commit_lsn: Lsn(0),  /* part of WAL acknowledged by quorum */
            restart_lsn: Lsn(0), /* minimal LSN which may be needed for recovery of some safekeeper */
        }
    }
}

// protocol messages

/// Initial Proposer -> Acceptor message
#[derive(Debug, Serialize, Deserialize)]
pub struct ProposerGreeting {
    /// proposer-acceptor protocol version
    pub protocol_version: u32,
    /// Postgres server version
    pub pg_version: u32,
    pub proposer_id: PgUuid,
    pub system_id: SystemId,
    /// Zenith timelineid
    pub ztli: ZTimelineId,
    pub tenant_id: ZTenantId,
    pub tli: TimeLineID,
    pub wal_seg_size: u32,
}

/// Acceptor -> Proposer initial response: the highest term known to me
/// (acceptor voted for).
#[derive(Debug, Serialize, Deserialize)]
pub struct AcceptorGreeting {
    term: u64,
}

/// Vote request sent from proposer to safekeepers
#[derive(Debug, Serialize, Deserialize)]
pub struct VoteRequest {
    term: Term,
}

/// Vote itself, sent from safekeeper to proposer
#[derive(Debug, Serialize, Deserialize)]
pub struct VoteResponse {
    term: Term,      // not really needed, just a sanity check
    vote_given: u64, // fixme u64 due to padding
    /// Safekeeper's log position, to let proposer choose the most advanced one
    epoch: Term,
    flush_lsn: Lsn,
    restart_lsn: Lsn,
}

/// Request with WAL message sent from proposer to safekeeper. Along the way it
/// announces 1) successful election (with VCL); 2) commit_lsn.
#[derive(Debug, Serialize, Deserialize)]
pub struct AppendRequest {
    h: AppendRequestHeader,
    wal_data: Bytes,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendRequestHeader {
    term: Term,
    /// volume commit LSN
    vcl: Lsn,
    /// start position of message in WAL
    begin_lsn: Lsn,
    /// end position of message in WAL
    end_lsn: Lsn,
    /// LSN committed by quorum of safekeepers
    commit_lsn: Lsn,
    /// restart LSN position  (minimal LSN which may be needed by proposer to perform recovery)
    restart_lsn: Lsn,
    // only for logging/debugging
    proposer_uuid: PgUuid,
}

/// Report safekeeper state to proposer
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct AppendResponse {
    // Current term of the safekeeper; if it is higher than proposer's, the
    // compute is out of date.
    pub term: Term,
    pub epoch: Term,
    // NOTE: this is physical end of wal on safekeeper; currently it doesn't
    // make much sense without taking epoch into account, as history can be
    // diverged.
    pub flush_lsn: Lsn,
    pub hs_feedback: HotStandbyFeedback,
}

/// Proposer -> Acceptor messages
#[derive(Debug)]
pub enum ProposerAcceptorMessage {
    Greeting(ProposerGreeting),
    VoteRequest(VoteRequest),
    AppendRequest(AppendRequest),
}

impl ProposerAcceptorMessage {
    /// Parse proposer message.
    pub fn parse(msg: Bytes) -> Result<ProposerAcceptorMessage> {
        // xxx using Reader is inefficient but easy to work with bincode
        let mut stream = msg.reader();
        // u64 is here to avoid padding; it will be removed once we stop packing C structs into the wire as is
        let tag = stream.read_u64::<LittleEndian>()? as u8 as char;
        match tag {
            'g' => {
                let msg = ProposerGreeting::des_from(&mut stream)?;
                Ok(ProposerAcceptorMessage::Greeting(msg))
            }
            'v' => {
                let msg = VoteRequest::des_from(&mut stream)?;
                Ok(ProposerAcceptorMessage::VoteRequest(msg))
            }
            'a' => {
                // read header followed by wal data
                let hdr = AppendRequestHeader::des_from(&mut stream)?;
                let rec_size = hdr
                    .end_lsn
                    .checked_sub(hdr.begin_lsn)
                    .ok_or(anyhow!("begin_lsn > end_lsn in AppendRequest"))?
                    .0 as usize;
                if rec_size > MAX_SEND_SIZE {
                    bail!(
                        "AppendRequest is longer than MAX_SEND_SIZE ({})",
                        MAX_SEND_SIZE
                    );
                }

                let mut wal_data_vec: Vec<u8> = vec![0; rec_size];
                stream.read_exact(&mut wal_data_vec)?;
                let wal_data = Bytes::from(wal_data_vec);
                let msg = AppendRequest {
                    h: hdr,
                    wal_data: wal_data,
                };

                Ok(ProposerAcceptorMessage::AppendRequest(msg))
            }
            _ => Err(anyhow!("unknown proposer-acceptor message tag: {}", tag,)),
        }
    }
}

/// Acceptor -> Proposer messages
#[derive(Debug)]
pub enum AcceptorProposerMessage {
    Greeting(AcceptorGreeting),
    VoteResponse(VoteResponse),
    AppendResponse(AppendResponse),
}

impl AcceptorProposerMessage {
    /// Serialize acceptor -> proposer message.
    pub fn serialize(&self, stream: &mut impl io::Write) -> Result<()> {
        match self {
            AcceptorProposerMessage::Greeting(msg) => {
                stream.write_u64::<LittleEndian>('g' as u64)?;
                msg.ser_into(stream)?;
            }
            AcceptorProposerMessage::VoteResponse(msg) => {
                stream.write_u64::<LittleEndian>('v' as u64)?;
                msg.ser_into(stream)?;
            }
            AcceptorProposerMessage::AppendResponse(msg) => {
                stream.write_u64::<LittleEndian>('a' as u64)?;
                msg.ser_into(stream)?;
            }
        }

        Ok(())
    }
}

pub trait Storage {
    /// Persist safekeeper state on disk, optionally syncing it.
    fn persist(&mut self, s: &SafeKeeperState, sync: bool) -> Result<()>;
    /// Write piece of wal in buf to disk.
    fn write_wal(&mut self, s: &SafeKeeperState, startpos: Lsn, buf: &[u8]) -> Result<()>;
}

/// SafeKeeper which consumes events (messages from compute) and provides
/// replies.
#[derive(Debug)]
pub struct SafeKeeper<ST: Storage> {
    /// Locally flushed part of WAL (end_lsn of last record). Established by
    /// reading wal.
    pub flush_lsn: Lsn,
    pub tli: u32,
    pub flushed_restart_lsn: Lsn,
    pub storage: ST,
    pub s: SafeKeeperState,          // persistent part
    pub elected_proposer_term: Term, // for monitoring/debugging
}

impl<ST> SafeKeeper<ST>
where
    ST: Storage,
{
    // constructor
    pub fn new(flush_lsn: Lsn, tli: u32, storage: ST, state: SafeKeeperState) -> SafeKeeper<ST> {
        SafeKeeper {
            flush_lsn,
            tli,
            flushed_restart_lsn: Lsn(0),
            storage,
            s: state,
            elected_proposer_term: 0,
        }
    }

    /// Process message from proposer and possibly form reply. Concurrent
    /// callers must exclude each other.
    pub fn process_msg(
        &mut self,
        msg: &ProposerAcceptorMessage,
    ) -> Result<AcceptorProposerMessage> {
        match msg {
            ProposerAcceptorMessage::Greeting(msg) => self.handle_greeting(msg),
            ProposerAcceptorMessage::VoteRequest(msg) => self.handle_vote_request(msg),
            ProposerAcceptorMessage::AppendRequest(msg) => self.handle_append_request(msg),
        }
    }

    /// Handle initial message from proposer: check its sanity and send my
    /// current term.
    fn handle_greeting(&mut self, msg: &ProposerGreeting) -> Result<AcceptorProposerMessage> {
        /* Check protocol compatibility */
        if msg.protocol_version != SK_PROTOCOL_VERSION {
            bail!(
                "incompatible protocol version {}, expected {}",
                msg.protocol_version,
                SK_PROTOCOL_VERSION
            );
        }
        if self.s.server.system_id != 0 && self.s.server.system_id != msg.system_id {
            bail!(
                "system identifier changed: got {}, expected {}",
                msg.system_id,
                self.s.server.system_id,
            );
        }
        /* Postgres upgrade is not treated as fatal error */
        if msg.pg_version != self.s.server.pg_version
            && self.s.server.pg_version != UNKNOWN_SERVER_VERSION
        {
            info!(
                "incompatible server version {}, expected {}",
                msg.pg_version, self.s.server.pg_version
            );
        }

        // set basic info about server, if not yet
        self.s.server.system_id = msg.system_id;
        self.s.server.ztli = msg.ztli;
        self.s.server.tli = msg.tli;
        self.s.server.wal_seg_size = msg.wal_seg_size;
        self.s.proposer_uuid = msg.proposer_id;
        self.storage.persist(&self.s, true)?;

        info!(
            "processed greeting from proposer {:?}, sending term {:?}",
            msg.proposer_id, self.s.acceptor_state.term
        );
        Ok(AcceptorProposerMessage::Greeting(AcceptorGreeting {
            term: self.s.acceptor_state.term,
        }))
    }

    /// Give vote for the given term, if we haven't done that previously.
    fn handle_vote_request(&mut self, msg: &VoteRequest) -> Result<AcceptorProposerMessage> {
        // initialize with refusal
        let mut resp = VoteResponse {
            term: msg.term,
            vote_given: false as u64,
            epoch: 0,
            flush_lsn: Lsn(0),
            restart_lsn: Lsn(0),
        };
        if self.s.acceptor_state.term < msg.term {
            self.s.acceptor_state.term = msg.term;
            // persist vote before sending it out
            self.storage.persist(&self.s, true)?;
            resp.vote_given = true as u64;
            resp.epoch = self.s.acceptor_state.epoch;
            resp.flush_lsn = self.flush_lsn;
            resp.restart_lsn = self.s.restart_lsn;
        }
        info!("processed VoteRequest for term {}: {:?}", msg.term, &resp);
        Ok(AcceptorProposerMessage::VoteResponse(resp))
    }

    /// Handle request to append WAL.
    fn handle_append_request(&mut self, msg: &AppendRequest) -> Result<AcceptorProposerMessage> {
        // log first AppendRequest from this proposer
        if self.elected_proposer_term < msg.h.term {
            info!(
                "start receiving WAL from timeline {} term {}",
                self.s.server.ztli, msg.h.term,
            );
            self.elected_proposer_term = msg.h.term;
        }

        // If our term is lower than elected proposer one, bump it.
        if self.s.acceptor_state.term < msg.h.term {
            self.s.acceptor_state.term = msg.h.term;
            self.storage.persist(&self.s, true)?;
        }
        // OTOH, if it is higher, immediately refuse the message.
        else if self.s.acceptor_state.term > msg.h.term {
            let resp = AppendResponse {
                term: self.s.acceptor_state.term,
                epoch: self.s.acceptor_state.epoch,
                flush_lsn: Lsn(0),
                hs_feedback: HotStandbyFeedback::empty(),
            };
            return Ok(AcceptorProposerMessage::AppendResponse(resp));
        }

        // do the job
        self.storage
            .write_wal(&self.s, msg.h.begin_lsn, &msg.wal_data)?;
        let mut sync_control_file = false;
        /*
         * Epoch switch happen when written WAL record cross the boundary.
         * The boundary is maximum of last WAL position at this node (FlushLSN) and global
         * maximum (vcl) determined by WAL proposer during handshake.
         * Switching epoch means that node completes recovery and start writing in the WAL new data.
         * XXX: this is wrong, we must actively truncate not matching part of log.
         */
        if self.s.acceptor_state.epoch < msg.h.term
            && msg.h.end_lsn > max(self.flush_lsn, msg.h.vcl)
        {
            info!("switched to new epoch {}", msg.h.term);
            self.s.acceptor_state.epoch = msg.h.term; /* bump epoch */
            sync_control_file = true;
        }
        if msg.h.end_lsn > self.flush_lsn {
            self.flush_lsn = msg.h.end_lsn;
        }

        self.s.proposer_uuid = msg.h.proposer_uuid;
        self.s.commit_lsn = msg.h.commit_lsn;
        self.s.restart_lsn = msg.h.restart_lsn;

        /*
         * Update restart LSN in control file.
         * To avoid negative impact on performance of extra fsync, do it only
         * when restart_lsn delta exceeds WAL segment size.
         */
        sync_control_file |=
            self.flushed_restart_lsn + (self.s.server.wal_seg_size as u64) < self.s.restart_lsn;
        self.storage.persist(&self.s, sync_control_file)?;
        if sync_control_file {
            self.flushed_restart_lsn = self.s.restart_lsn;
        }

        let resp = AppendResponse {
            term: self.s.acceptor_state.term,
            epoch: self.s.acceptor_state.epoch,
            flush_lsn: self.flush_lsn,
            // will be filled by caller code to avoid bothering safekeeper
            hs_feedback: HotStandbyFeedback::empty(),
        };
        trace!(
            "processed AppendRequest of len {}, flush_lsn={:X}/{:>08X}, resp {:?}",
            msg.wal_data.len(),
            (self.flush_lsn.0 >> 32) as u32,
            self.flush_lsn.0 as u32,
            &resp,
        );
        Ok(AcceptorProposerMessage::AppendResponse(resp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // fake storage for tests
    struct InMemoryStorage {
        persisted_state: SafeKeeperState,
    }

    impl Storage for InMemoryStorage {
        fn persist(&mut self, s: &SafeKeeperState, _sync: bool) -> Result<()> {
            self.persisted_state = s.clone();
            Ok(())
        }

        fn write_wal(&mut self, _s: &SafeKeeperState, _startpos: Lsn, _buf: &[u8]) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_voting() {
        let storage = InMemoryStorage {
            persisted_state: SafeKeeperState::new(),
        };
        let mut sk = SafeKeeper::new(Lsn(0), 0, storage, SafeKeeperState::new());

        // check voting for 1 is ok
        let vote_request = ProposerAcceptorMessage::VoteRequest(VoteRequest { term: 1 });
        let mut vote_resp = sk.process_msg(&vote_request);
        match vote_resp.unwrap() {
            AcceptorProposerMessage::VoteResponse(resp) => assert!(resp.vote_given != 0),
            _ => assert!(false),
        }

        // reboot...
        let state = sk.storage.persisted_state.clone();
        let storage = InMemoryStorage {
            persisted_state: state.clone(),
        };
        sk = SafeKeeper::new(Lsn(0), 0, storage, state);

        // and ensure voting second time for 1 is not ok
        vote_resp = sk.process_msg(&vote_request);
        match vote_resp.unwrap() {
            AcceptorProposerMessage::VoteResponse(resp) => assert!(resp.vote_given == 0),
            _ => assert!(false),
        }
    }

    #[test]
    fn test_epoch_switch() {
        let storage = InMemoryStorage {
            persisted_state: SafeKeeperState::new(),
        };
        let mut sk = SafeKeeper::new(Lsn(0), 0, storage, SafeKeeperState::new());

        let mut ar_hdr = AppendRequestHeader {
            term: 1,
            vcl: Lsn(2),
            begin_lsn: Lsn(1),
            end_lsn: Lsn(2),
            commit_lsn: Lsn(0),
            restart_lsn: Lsn(0),
            proposer_uuid: [0; 16],
        };
        let mut append_request = AppendRequest {
            h: ar_hdr.clone(),
            wal_data: Bytes::from_static(b"b"),
        };

        // check that AppendRequest before VCL doesn't switch epoch
        let resp = sk.process_msg(&ProposerAcceptorMessage::AppendRequest(append_request));
        assert!(resp.is_ok());
        assert!(sk.storage.persisted_state.acceptor_state.epoch == 0);

        // but record after VCL does the switch
        ar_hdr.begin_lsn = Lsn(2);
        ar_hdr.end_lsn = Lsn(3);
        append_request = AppendRequest {
            h: ar_hdr.clone(),
            wal_data: Bytes::from_static(b"b"),
        };
        let resp = sk.process_msg(&ProposerAcceptorMessage::AppendRequest(append_request));
        assert!(resp.is_ok());
        assert!(sk.storage.persisted_state.acceptor_state.epoch == 1);
    }
}