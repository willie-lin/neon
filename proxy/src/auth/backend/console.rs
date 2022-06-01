//! Cloud API V2.

use crate::{
    auth::{self, io_error, AuthFlow, AuthError, ClientCredentials, DatabaseInfo},
    compute, scram,
    stream::PqStream,
    url::ApiUrl,
};
use serde::{Deserialize, Serialize};
use std::future::Future;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncWrite};
use utils::pq_proto::{BeMessage as Be, BeParameterStatusMessage};

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("Console responded with a malformed JSON: {0}")]
    BadResponse(#[from] serde_json::Error),

    /// HTTP status (other than 200) returned by the console.
    #[error("Console responded with an HTTP status: {0}")]
    HttpStatus(reqwest::StatusCode),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl From<reqwest::Error> for TransportError {
    fn from(e: reqwest::Error) -> Self {
        io_error(e).into()
    }
}

#[derive(Debug, Error)]
pub enum GetAuthInfoError {
    // We shouldn't include the actual secret here.
    #[error("Console responded with a malformed auth secret")]
    BadSecret,

    #[error(transparent)]
    Transport(TransportError),
}

impl<E> From<E> for GetAuthInfoError
where
    TransportError: From<E>,
{
    fn from(e: E) -> Self {
        Self::Transport(e.into())
    }
}

#[derive(Debug, Error)]
pub enum WakeComputeError {
    #[error("Console responded with a malformed compute address: {0}")]
    BadComputeAddress(String),

    #[error(transparent)]
    Transport(TransportError),
}

impl<E> From<E> for WakeComputeError
where
    TransportError: From<E>,
{
    fn from(e: E) -> Self {
        Self::Transport(e.into())
    }
}

// TODO: convert into an enum with "error"
#[derive(Serialize, Deserialize, Debug)]
struct GetRoleSecretResponse {
    role_secret: String,
}

// TODO: convert into an enum with "error"
#[derive(Serialize, Deserialize, Debug)]
struct GetWakeComputeResponse {
    address: String,
}

/// Auth secret which is managed by the cloud.
pub enum AuthInfo {
    /// Md5 hash of user's password.
    Md5([u8; 16]),

    /// [SCRAM](crate::scram) authentication info.
    Scram(scram::ServerSecret),
}

#[must_use]
pub(super) struct Api<'a> {
    endpoint: &'a ApiUrl,
    creds: &'a ClientCredentials,
    /// Cache project name, since we'll need it several times.
    project: &'a str,
}

impl<'a> Api<'a> {
    /// Construct an API object containing the auth parameters.
    pub(super) fn new(endpoint: &'a ApiUrl, creds: &'a ClientCredentials) -> auth::Result<Self> {
        Ok(Self {
            endpoint,
            creds,
            project: creds.project_name()?,
        })
    }

    /// Authenticate the existing user or throw an error.
    pub(super) async fn handle_user(
        self,
        client: &mut PqStream<impl AsyncRead + AsyncWrite + Unpin + Send>,
    ) -> auth::Result<compute::NodeInfo> {
        handle_user(client, &self, Self::get_auth_info, Self::wake_compute).await
    }

    async fn get_auth_info(&self) -> Result<AuthInfo, GetAuthInfoError> {
        let mut url = self.endpoint.clone();
        url.path_segments_mut().push("proxy_get_role_secret");
        url.query_pairs_mut()
            .append_pair("project", self.project)
            .append_pair("role", &self.creds.user);

        // TODO: use a proper logger
        println!("cplane request: {url}");

        let resp = reqwest::get(url.into_inner()).await?;
        if !resp.status().is_success() {
            return Err(TransportError::HttpStatus(resp.status()).into());
        }

        let response: GetRoleSecretResponse = serde_json::from_str(&resp.text().await?)?;

        scram::ServerSecret::parse(response.role_secret.as_str())
            .map(AuthInfo::Scram)
            .ok_or(GetAuthInfoError::BadSecret)
    }

    /// Wake up the compute node and return the corresponding connection info.
    async fn wake_compute(&self) -> Result<DatabaseInfo, WakeComputeError> {
        let mut url = self.endpoint.clone();
        url.path_segments_mut().push("proxy_wake_compute");
        url.query_pairs_mut().append_pair("project", self.project);

        // TODO: use a proper logger
        println!("cplane request: {url}");

        let resp = reqwest::get(url.into_inner()).await?;
        if !resp.status().is_success() {
            return Err(TransportError::HttpStatus(resp.status()).into());
        }

        let response: GetWakeComputeResponse = serde_json::from_str(&resp.text().await?)?;

        let (host, port) = parse_host_port(&response.address)
            .ok_or(WakeComputeError::BadComputeAddress(response.address))?;

        Ok(DatabaseInfo {
            host,
            port,
            dbname: self.creds.dbname.to_owned(),
            user: self.creds.user.to_owned(),
            password: None,
        })
    }
}

/// Common logic for user handling in API V2.
/// We reuse this for a mock API implementation in [`super::postgres`].
pub(super) async fn handle_user<'a, Endpoint, GetAuthInfo, WakeCompute>(
    client: &mut PqStream<impl AsyncRead + AsyncWrite + Unpin>,
    endpoint: &'a Endpoint,
    get_auth_info: impl FnOnce(&'a Endpoint) -> GetAuthInfo,
    wake_compute: impl FnOnce(&'a Endpoint) -> WakeCompute,
) -> auth::Result<compute::NodeInfo>
where
    GetAuthInfo: Future<Output = Result<AuthInfo, GetAuthInfoError>>,
    WakeCompute: Future<Output = Result<DatabaseInfo, WakeComputeError>>,
{
    let auth_info = get_auth_info(endpoint).await?;

    let flow = AuthFlow::new(client);
    let scram_keys = match auth_info {
        AuthInfo::Md5(_) => {
            // TODO: decide if we should support MD5 in api v2
            return Err(AuthError::bad_auth_method("MD5"));
        }
        AuthInfo::Scram(secret) => {
            let scram = auth::Scram(&secret);
            Some(compute::ScramKeys {
                client_key: flow.begin(scram).await?.authenticate().await?.as_bytes(),
                server_key: secret.server_key.as_bytes(),
            })
        }
    };

    client
        .write_message_noflush(&Be::AuthenticationOk)?
        .write_message_noflush(&BeParameterStatusMessage::encoding())?;

    Ok(compute::NodeInfo {
        db_info: wake_compute(endpoint).await?,
        scram_keys,
    })
}

fn parse_host_port(input: &str) -> Option<(String, u16)> {
    let (host, port) = input.split_once(':')?;
    Some((host.to_owned(), port.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_db_info() -> anyhow::Result<()> {
        let _: DatabaseInfo = serde_json::from_value(json!({
            "host": "localhost",
            "port": 5432,
            "dbname": "postgres",
            "user": "john_doe",
            "password": "password",
        }))?;

        let _: DatabaseInfo = serde_json::from_value(json!({
            "host": "localhost",
            "port": 5432,
            "dbname": "postgres",
            "user": "john_doe",
        }))?;

        Ok(())
    }
}
