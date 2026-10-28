#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! M33/M34 regression tests for the shared data ops:
//! `forget` honours `opts.tenant` instead of silently resolving in the
//! handle's tenant, and `update` refuses a dataset owned by someone else
//! when no ACL is in play.

use std::path::Path;
use std::sync::Arc;

use cognee::add::generate_dataset_id;
use cognee::config::Settings;
use cognee::models::Dataset;
use cognee_bindings_common::HandleState;
use cognee_bindings_common::SdkError;
use cognee_bindings_common::ops::{data, datasets};
use cognee_database::IngestDb;
use serde_json::json;
use uuid::Uuid;

/// A handle with every store rooted under `dir` — same hermetic shape as
/// `handle_close.rs`: sqlite file, mock embedding, dummy LLM key (no
/// network, no `.env` pickup).
fn handle_under(dir: &Path) -> Arc<HandleState> {
    let db_path = dir.join("cognee.db");
    let settings = Settings {
        sections: [
            (
                "relational.url".to_string(),
                json!(format!("sqlite://{}?mode=rwc", db_path.display())),
            ),
            ("embedding.provider".to_string(), json!("mock".to_owned())),
            ("llm.api_key".to_string(), json!("sk-test".to_owned())),
        ]
        .into(),
        data_root_directory: dir.join("data").to_string_lossy().into_owned(),
        system_root_directory: dir.join("sys").to_string_lossy().into_owned(),
        ..Settings::default()
    };
    Arc::new(HandleState::from_settings(settings))
}

/// Create `"x"` owned by `owner` in `tenant`, under the deterministic id
/// every name-based resolution derives for it.
async fn create_dataset(state: &HandleState, owner: Uuid, tenant: Option<Uuid>) -> Uuid {
    let id = generate_dataset_id("x", owner, tenant);
    let svc = state.services().await.unwrap();
    IngestDb::create_dataset(
        svc.database.as_ref(),
        Dataset::new("x".into(), owner, tenant, id),
    )
    .await
    .unwrap();
    id
}

async fn dataset_exists(state: &HandleState, id: Uuid) -> bool {
    let svc = state.services().await.unwrap();
    svc.database.get_dataset(id).await.unwrap().is_some()
}

fn forget_dataset_by_name(name: &str) -> serde_json::Value {
    json!({"kind": "dataset", "dataset": {"name": name}})
}

/// M33: `{"tenant": T}` routes the by-name resolution to the owner's
/// dataset *in T* — the tenant-less same-named dataset is untouched.
#[tokio::test]
async fn forget_with_tenant_deletes_only_that_tenants_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let state = handle_under(dir.path());
    let owner = state.owner_id().await.unwrap();
    let tenant = Uuid::new_v4();
    let tenanted = create_dataset(&state, owner, Some(tenant)).await;
    let tenantless = create_dataset(&state, owner, None).await;

    data::forget(
        &state,
        forget_dataset_by_name("x"),
        &json!({"tenant": tenant.to_string()}),
    )
    .await
    .expect("forget in tenant T succeeds");

    assert!(
        !dataset_exists(&state, tenanted).await,
        "T's dataset is gone"
    );
    assert!(
        dataset_exists(&state, tenantless).await,
        "the tenant-less dataset is untouched"
    );
}

/// Without opts the handle's tenant (None) is used — the old behaviour.
#[tokio::test]
async fn forget_without_opts_deletes_the_tenantless_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let state = handle_under(dir.path());
    let owner = state.owner_id().await.unwrap();
    let tenant = Uuid::new_v4();
    let tenanted = create_dataset(&state, owner, Some(tenant)).await;
    let tenantless = create_dataset(&state, owner, None).await;

    data::forget(
        &state,
        forget_dataset_by_name("x"),
        &serde_json::Value::Null,
    )
    .await
    .expect("forget without opts succeeds");

    assert!(
        !dataset_exists(&state, tenantless).await,
        "the tenant-less dataset is gone"
    );
    assert!(
        dataset_exists(&state, tenanted).await,
        "T's dataset is untouched"
    );
}

/// A malformed `tenant` is a Validation error raised before anything is
/// deleted.
#[tokio::test]
async fn a_malformed_tenant_is_a_validation_error() {
    let dir = tempfile::tempdir().unwrap();
    let state = handle_under(dir.path());
    let owner = state.owner_id().await.unwrap();
    let tenantless = create_dataset(&state, owner, None).await;

    let err = data::forget(
        &state,
        forget_dataset_by_name("x"),
        &json!({"tenant": "not-a-uuid"}),
    )
    .await
    .expect_err("a non-UUID tenant is rejected");
    assert!(matches!(err, SdkError::Validation(_)), "got {err:?}");
    assert!(
        dataset_exists(&state, tenantless).await,
        "nothing was deleted"
    );
}

/// M34: `opts.datasetId` is caller-chosen and the bindings call `update`
/// without an ACL — the lib's ownership gate must refuse a dataset owned
/// by someone else, before any data is attached or detached.
#[tokio::test]
async fn update_against_another_owners_dataset_is_denied() {
    let dir = tempfile::tempdir().unwrap();
    let state = handle_under(dir.path());
    let foreign = create_dataset(&state, Uuid::new_v4(), None).await;

    let err = data::update(
        &state,
        &Uuid::new_v4().to_string(),
        json!({"type": "text", "text": "replacement"}),
        "x",
        &json!({"datasetId": foreign.to_string()}),
    )
    .await
    .expect_err("another owner's dataset is denied");
    assert!(
        err.to_string().contains("denied"),
        "expected a permission denial, got {err:?}"
    );
    // And the foreign dataset was not touched.
    assert!(dataset_exists(&state, foreign).await);
    let items = datasets::list_data(&state, &foreign.to_string()).await;
    // list_data is owner-scoped so it errors for a foreign dataset — either
    // way no items were attached through the update.
    assert!(items.is_err() || items.unwrap().as_array().unwrap().is_empty());
}
