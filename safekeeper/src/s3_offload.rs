//
// Offload old WAL segments to S3 and remove them locally
//

use anyhow::Context;
use aws_sdk_s3::types::ByteStream;
use aws_sdk_s3::{config, Client, Credentials, Endpoint, Region};
use postgres_ffi::xlog_utils::*;
use std::borrow::Cow;
use std::collections::HashSet;
use std::env;
use std::path::Path;
use std::time::SystemTime;
use tokio::fs;
use tokio::runtime;
use tokio::time::sleep;
use tracing::*;
use walkdir::WalkDir;

use crate::SafeKeeperConf;

pub fn thread_main(conf: SafeKeeperConf) {
    // Create a new thread pool
    //
    // FIXME: keep it single-threaded for now, make it easier to debug with gdb,
    // and we're not concerned with performance yet.
    //let runtime = runtime::Runtime::new().unwrap();
    let runtime = runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    info!("Starting S3 offload task");

    runtime.block_on(async {
        main_loop(&conf).await.unwrap();
    });
}

async fn offload_files(
    client: &Client,
    bucket_name: &str,
    listing: &HashSet<String>,
    dir_path: &Path,
    conf: &SafeKeeperConf,
) -> anyhow::Result<u64> {
    let horizon = SystemTime::now() - conf.ttl.unwrap();
    let mut n: u64 = 0;
    for entry in WalkDir::new(dir_path) {
        let entry = entry?;
        let path = entry.path();

        if path.is_file()
            && IsXLogFileName(entry.file_name().to_str().unwrap())
            && entry.metadata().unwrap().created().unwrap() <= horizon
        {
            let relpath = path.strip_prefix(&conf.workdir).unwrap();
            let s3path = String::from("walarchive/") + relpath.to_str().unwrap();
            if !listing.contains(&s3path) {
                client
                    .put_object()
                    .bucket(bucket_name)
                    .key(&s3path)
                    .body(ByteStream::from_path(&path).await?)
                    .send()
                    .await?;

                fs::remove_file(&path).await?;
                n += 1;
            }
        }
    }
    Ok(n)
}

async fn main_loop(conf: &SafeKeeperConf) -> anyhow::Result<()> {
    let s3_region = env::var("S3_REGION").context("S3_REGION env var is not set")?;
    let s3_endpoint = env::var("S3_ENDPOINT").context("S3_ENDPOINT env var is not set")?;
    let bucket_name = "zenith-testbucket";

    let config = config::Builder::new()
        .region(Region::new(Cow::Owned(s3_region)))
        .endpoint_resolver(Endpoint::immutable(s3_endpoint.parse().with_context(
            || format!("Failed to parse endpoint '{}' as url", s3_endpoint),
        )?))
        .credentials_provider(Credentials::new(
            env::var("S3_ACCESSKEY").context("S3_ACCESSKEY env var is not set")?,
            env::var("S3_SECRET").context("S3_SECRET env var is not set")?,
            None,
            None,
            "safekeeper static credentials",
        ))
        .build();

    let client = Client::from_conf(config);

    loop {
        let listing = gather_wal_entries(&client, bucket_name).await?;
        let n = offload_files(&client, bucket_name, &listing, &conf.workdir, conf).await?;
        info!("Offload {} files to S3", n);
        sleep(conf.ttl.unwrap()).await;
    }
}

async fn gather_wal_entries(client: &Client, bucket_name: &str) -> anyhow::Result<HashSet<String>> {
    let mut document_keys = HashSet::new();

    let mut continuation_token = None::<String>;
    loop {
        let mut request = client
            .list_objects_v2()
            .bucket(bucket_name)
            .prefix("walarchive/");
        if let Some(token) = &continuation_token {
            request = request.continuation_token(token);
        }

        let response = request.send().await?;
        document_keys.extend(
            response
                .contents
                .unwrap_or_default()
                .into_iter()
                .filter_map(|o| o.key),
        );

        continuation_token = response.continuation_token;
        if continuation_token.is_none() {
            break;
        }
    }
    Ok(document_keys)
}
