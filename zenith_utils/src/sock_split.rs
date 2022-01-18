use std::{
    io::{self, BufReader},
    net::{Shutdown, TcpStream},
    sync::Arc,
};

use rustls::Session;

/// Wrapper supporting reads of a shared stream.
#[repr(transparent)]
pub struct ArcStream<S>(Arc<S>);

impl<S> ArcStream<S> {
    pub fn new(stream: S) -> Self {
        Self(Arc::new(stream))
    }
}

impl<S> io::Read for ArcStream<S>
where
    for<'a> &'a <Arc<S> as std::ops::Deref>::Target: io::Read,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (&*self.0).read(buf)
    }
}

impl<S> io::Write for ArcStream<S>
where
    for<'a> &'a <Arc<S> as std::ops::Deref>::Target: io::Write,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        (&*self.0).write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        (&*self.0).flush()
    }
}

impl<S> std::ops::Deref for ArcStream<S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<S> Clone for ArcStream<S> {
    fn clone(&self) -> Self {
        ArcStream(Arc::clone(&self.0))
    }
}

/// Wrapper which supports both writes and buffered reads.
/// This should've been just an [`io::Write`] impl for [`BufReader`].
/// TODO: maybe it should implement everything [`BufReader`] implements,
/// e.g. [`std::io::BufRead`] and [`std::io::Seek`].
#[repr(transparent)]
pub struct BufStream<S>(BufReader<S>);

impl<S: io::Read> BufStream<S> {
    pub fn new(stream: S) -> Self {
        Self(BufReader::new(stream))
    }
}

impl<S> BufStream<S> {
    /// Unwrap into the internal [`BufReader`].
    fn into_reader(self) -> BufReader<S> {
        self.0
    }

    /// Returns an shared reference to the underlying stream.
    fn get_ref(&self) -> &S {
        self.0.get_ref()
    }

    /// Returns an exclusive reference to the underlying stream.
    fn get_mut(&mut self) -> &mut S {
        self.0.get_mut()
    }
}
impl<S: io::Read> io::Read for BufStream<S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }
}

impl<S: io::Write> io::Write for BufStream<S> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.get_mut().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.get_mut().flush()
    }
}

type BufArcStream<T> = BufStream<ArcStream<T>>;

type TlsStream<T> = rustls::StreamOwned<rustls::ServerSession, T>;

pub trait CanRead {}
pub trait CanWrite {}

pub trait StreamFamily<S> {
    type Raw;
    type Tls;
}

pub enum Stream<F: StreamFamily<S>, S> {
    Raw(F::Raw),
    Tls(F::Tls),
}

pub enum SRead {}

impl CanRead for SRead {}

impl<S> StreamFamily<S> for SRead {
    type Raw = BufArcStream<S>;
    type Tls = rustls_split::ReadHalf<rustls::ServerSession>;
}

pub enum SWrite {}

impl CanWrite for SWrite {}

impl<S> StreamFamily<S> for SWrite {
    type Raw = ArcStream<S>;
    type Tls = rustls_split::WriteHalf<rustls::ServerSession>;
}

pub enum SBidi {}

impl CanRead for SBidi {}
impl CanWrite for SBidi {}

impl<S> StreamFamily<S> for SBidi
where
    S: io::Read + io::Write,
    for<'a> &'a S: io::Read + io::Write,
{
    type Raw = BufArcStream<S>;
    type Tls = Box<TlsStream<BufArcStream<S>>>;
}

pub type ReadStream<S> = Stream<SRead, S>;
pub type WriteStream<S> = Stream<SWrite, S>;
pub type BidiStream<S> = Stream<SBidi, S>;

impl ReadStream<TcpStream> {
    pub fn shutdown(&mut self, how: Shutdown) -> io::Result<()> {
        match self {
            Self::Raw(stream) => stream.get_ref().shutdown(how),
            Self::Tls(stream) => stream.shutdown(how),
        }
    }
}

impl WriteStream<TcpStream> {
    pub fn shutdown(&mut self, how: Shutdown) -> io::Result<()> {
        match self {
            Self::Raw(stream) => stream.shutdown(how),
            Self::Tls(stream) => stream.shutdown(how),
        }
    }
}

impl BidiStream<TcpStream> {
    pub fn shutdown(&mut self, how: Shutdown) -> io::Result<()> {
        use std::io::Write;

        match self {
            Self::Raw(stream) => stream.get_ref().shutdown(how),
            Self::Tls(stream) => {
                if how == Shutdown::Read {
                    stream.sock.get_ref().shutdown(how)
                } else {
                    stream.sess.send_close_notify();
                    let res = stream.flush();
                    stream.sock.get_ref().shutdown(how)?;
                    res
                }
            }
        }
    }
}

impl BidiStream<TcpStream> {
    /// Split the bi-directional stream into two owned read and write halves.
    pub fn split(self) -> (ReadStream<TcpStream>, WriteStream<TcpStream>) {
        match self {
            Self::Raw(stream) => {
                let writer = ArcStream::clone(stream.get_ref());
                (ReadStream::Raw(stream), WriteStream::Raw(writer))
            }
            Self::Tls(stream) => {
                let reader = stream.sock.into_reader();
                let buffer_data = reader.buffer().to_owned();

                let read_buf_cfg = rustls_split::BufCfg::with_data(buffer_data, 8192);
                let write_buf_cfg = rustls_split::BufCfg::with_capacity(8192);

                // TODO: make rustls_split work with any Read + Write impl
                let (read_half, write_half) = rustls_split::split(
                    Arc::try_unwrap(reader.into_inner().0).unwrap(),
                    stream.sess,
                    read_buf_cfg,
                    write_buf_cfg,
                );

                (ReadStream::Tls(read_half), WriteStream::Tls(write_half))
            }
        }
    }
}

impl<S: io::Read + io::Write> BidiStream<S>
where
    for<'a> &'a S: io::Read + io::Write,
{
    pub fn from_raw(stream: S) -> Self {
        Self::Raw(BufStream::new(ArcStream::new(stream)))
    }

    pub fn start_tls(self, mut session: rustls::ServerSession) -> io::Result<Self> {
        match self {
            Self::Raw(mut stream) => {
                session.complete_io(&mut stream)?;
                assert!(!session.is_handshaking());
                Ok(Self::Tls(Box::new(TlsStream::new(session, stream))))
            }
            Self::Tls { .. } => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "TLS is already started on this stream",
            )),
        }
    }
}

impl<F, S> io::Read for Stream<F, S>
where
    F: StreamFamily<S> + CanRead,
    F::Raw: io::Read,
    F::Tls: io::Read,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Raw(stream) => stream.read(buf),
            Self::Tls(stream) => stream.read(buf),
        }
    }
}

impl<F, S> io::Write for Stream<F, S>
where
    F: StreamFamily<S> + CanWrite,
    F::Raw: io::Write,
    F::Tls: io::Write,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Raw(stream) => stream.write(buf),
            Self::Tls(stream) => stream.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Raw(stream) => stream.flush(),
            Self::Tls(stream) => stream.flush(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    #[test]
    fn test_read_write() -> anyhow::Result<()> {
        let (tx, rx) = std::os::unix::net::UnixStream::pair()?;
        let mut tx = BidiStream::from_raw(tx);
        let mut rx = BidiStream::from_raw(rx);

        let mut bytes = [1, 2, 3, 4];
        tx.write(&bytes)?;
        rx.read(&mut bytes)?;
        assert_eq!(bytes, [1, 2, 3, 4]);

        let (tx, rx) = std::os::unix::net::UnixStream::pair()?;
        let mut tx = WriteStream::Raw(ArcStream::new(tx));
        let mut rx = ReadStream::Raw(BufStream::new(ArcStream::new(rx)));

        let mut bytes = [1, 2, 3, 4];
        tx.write(&bytes)?;
        rx.read(&mut bytes)?;
        assert_eq!(bytes, [1, 2, 3, 4]);

        Ok(())
    }
}
