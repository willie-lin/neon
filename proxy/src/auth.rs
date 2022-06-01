//! Client authentication mechanisms.

pub mod backend;
pub use backend::DatabaseInfo;

mod credentials;
pub use credentials::ClientCredentials;

mod flow;
pub use flow::*;

use crate::error::UserFacingError;
use std::io;
use thiserror::Error;

/// Convenience wrapper for the authentication error.
pub type Result<T> = std::result::Result<T, AuthError>;

/// Common authentication error.
#[derive(Debug, Error)]
pub enum AuthErrorImpl {
    // This will be dropped in the future.
    #[error(transparent)]
    Legacy(#[from] backend::LegacyAuthError),

    #[error(transparent)]
    Link(#[from] backend::LinkAuthError),

    #[error(transparent)]
    GetAuthInfo(#[from] backend::GetAuthInfoError),

    #[error(transparent)]
    WakeCompute(#[from] backend::WakeComputeError),

    /// SASL protocol errors (includes [SCRAM](crate::scram)).
    #[error(transparent)]
    Sasl(#[from] crate::sasl::Error),

    /// Happens whenever we couldn't extract the project (aka cluster) name.
    #[error(transparent)]
    BadProjectName(#[from] credentials::ProjectNameError),

    #[error("Unsupported authentication method: {0}")]
    BadAuthMethod(String),

    #[error("Malformed password message")]
    MalformedPassword,

    /// Errors produced by [`crate::stream::PqStream`].
    #[error(transparent)]
    Io(#[from] io::Error),
}

#[derive(Debug, Error)]
#[error(transparent)]
pub struct AuthError(Box<AuthErrorImpl>);

impl AuthError {
    pub fn bad_auth_method(name: impl Into<String>) -> Self {
        AuthErrorImpl::BadAuthMethod(name.into()).into()
    }
}

impl<T> From<T> for AuthError
where
    AuthErrorImpl: From<T>,
{
    fn from(e: T) -> Self {
        Self(Box::new(e.into()))
    }
}

impl UserFacingError for AuthError {
    fn to_string_client(&self) -> String {
        use AuthErrorImpl::*;
        match self.0.as_ref() {
            Legacy(e) => e.to_string_client(),
            Link(e) => e.to_string_client(),
            Sasl(e) => e.to_string_client(),
            BadProjectName(e) => e.to_string_client(),
            BadAuthMethod(_) => self.to_string(),
            MalformedPassword => self.to_string(),
            _ => "Internal error".to_string(),
        }
    }
}

/// Upcast (almost) any error into an opaque [`io::Error`].
fn io_error(e: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e)
}
