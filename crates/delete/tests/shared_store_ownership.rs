//! M03/M04/M05 — shared-store delete ownership: the same physical graph node,
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test fixtures panic on failure — unwrapping is the assertion"
)]

//! vector point or EdgeType text may be claimed by several provenance rows
//! (one `Entity` extracted in two datasets). In the shared store a delete may
//! only remove artifacts no surviving `(dataset, data)` pair still claims —
//! Python's `get_global_*_related_*` rule — and the hard-mode orphan sweep
//! must not take a node another dataset's ledger still claims.

use std::collections::HashMap;
use std::sync::Arc;

use cognee_database::{
    DatabaseConnection, DeleteDb, GraphEdge, GraphNode, connect, initialize,
    ops::{self, graph_storage},
};
use cognee_delete::{DeleteMode, DeleteRequest, DeleteScope, DeleteService};
use cognee_graph::{GraphDBTrait, MockGraphDB};
use cognee_models::{Data, Dataset, EdgeType, Triplet};
use cognee_storage::{MockStorage, StorageTrait};
use cognee_test_utils::PerDatasetMocks;
use cognee_vector::{MockVectorDB, VectorDB, VectorPoint};
use serde_json::json;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Relational dataset + data + link, exactly as cognify leaves them.
struct Seed {
    db: Arc<DatabaseConnection>,
    storage: Arc<MockStorage>,
    owner: Uuid,
}

async fn seed() -> Seed {
    let db = connect("sqlite::memory:").await.unwrap();
    initialize(&db).await.unwrap();
    Seed {
        db: Arc::new(db),
        storage: Arc::new(MockStorage::new()),
        owner: Uuid::new_v4(),
    }
}

impl Seed {
    async fn dataset(&self, name: &str) -> Uuid {
        ops::datasets::create_dataset(
            &self.db,
            Dataset::new(name.to_string(), self.owner, None, Uuid::new_v4()),
        )
        .await
        .unwrap()
        .id
    }

    async fn item(&self, dataset: Uuid) -> Uuid {
        let location = self.storage.store(b"content", "item.txt").await.unwrap();
        let data_id = Uuid::new_v4();
        ops::data::create_data(
            &self.db,
            Data::builder(
                data_id,
                "item.txt",
                &location,
                "file://item.txt",
                "txt",
                "text/plain",
                format!("hash-{data_id}"),
                self.owner,
            )
            .build(),
        )
        .await
        .unwrap();
        ops::datasets::attach_data_to_dataset(&self.db, dataset, data_id)
            .await
            .unwrap();
        data_id
    }

    fn node_row(&self, dataset: Uuid, data: Uuid, node_id: Uuid, node_type: &str) -> GraphNode {
        GraphNode {
            id: Uuid::new_v4(),
            slug: node_id,
            user_id: self.owner,
            data_id: data,
            dataset_id: dataset,
            pipeline_run_id: None,
            label: Some(node_type.to_string()),
            node_type: node_type.to_string(),
            indexed_fields: json!(["name"]),
            attributes: None,
            created_at: chrono::Utc::now(),
        }
    }

    fn edge_row(
        &self,
        dataset: Uuid,
        data: Uuid,
        slug: Uuid,
        src: Uuid,
        dst: Uuid,
        rel: &str,
    ) -> GraphEdge {
        GraphEdge {
            id: Uuid::new_v4(),
            slug,
            user_id: self.owner,
            data_id: data,
            dataset_id: dataset,
            pipeline_run_id: None,
            source_node_id: src,
            destination_node_id: dst,
            relationship_name: rel.to_string(),
            label: Some(rel.to_string()),
            attributes: None,
            created_at: chrono::Utc::now(),
        }
    }

    async fn prov_node_count(&self, dataset: Uuid) -> usize {
        self.db.get_nodes_by_dataset(dataset).await.unwrap().len()
    }

    async fn data_exists(&self, id: Uuid) -> bool {
        ops::data::get_data(&self.db, id).await.unwrap().is_some()
    }

    async fn graph_node(graph: &MockGraphDB, id: Uuid) -> bool {
        graph
            .get_graph_data()
            .await
            .unwrap()
            .0
            .iter()
            .any(|(nid, _)| *nid == id.to_string())
    }

    async fn point(vector: &MockVectorDB, ty: &str, field: &str, id: Uuid) -> bool {
        vector
            .retrieve(ty, field, &[id])
            .await
            .map(|hits| !hits.is_empty())
            .unwrap_or(false)
    }
}

fn node(id: Uuid, node_type: &str) -> serde_json::Value {
    json!({"id": id.to_string(), "type": node_type})
}

fn edge(
    src: Uuid,
    dst: Uuid,
    rel: &str,
) -> (
    String,
    String,
    String,
    HashMap<std::borrow::Cow<'static, str>, serde_json::Value>,
) {
    (
        src.to_string(),
        dst.to_string(),
        rel.to_string(),
        HashMap::new(),
    )
}

fn shared_service(
    seed: &Seed,
    graph: Arc<MockGraphDB>,
    vector: Arc<MockVectorDB>,
) -> DeleteService {
    DeleteService::new(
        seed.storage.clone() as Arc<dyn StorageTrait>,
        seed.db.clone() as Arc<dyn DeleteDb>,
    )
    .with_graph_db(graph as Arc<dyn GraphDBTrait>)
    .with_vector_db(vector as Arc<dyn VectorDB>)
}

async fn point_to(vector: &MockVectorDB, ty: &str, field: &str, id: Uuid) {
    if !vector.has_collection(ty, field).await.unwrap_or(false) {
        vector.create_collection(ty, field, 4).await.unwrap();
    }
    vector
        .index_points(ty, field, &[VectorPoint::new(id, vec![1.0; 4])])
        .await
        .unwrap();
}

// ---------------------------------------------------------------------------
// M05 — dataset deletion in the shared store
// ---------------------------------------------------------------------------

/// Shared store: datasets A and B both claim node slug S (a graph node + an
/// Entity vector point) and an edge whose EdgeType text both use. Deleting A
/// must keep S's node, its point and the shared EdgeType point, delete the
/// A-only artifacts, and remove only A's ledger rows.
#[tokio::test]
async fn a_dataset_delete_keeps_artifacts_another_dataset_claims() {
    let seed = seed().await;
    let a = seed.dataset("a").await;
    let b = seed.dataset("b").await;
    let xa = seed.item(a).await;
    let xb = seed.item(b).await;

    let graph = Arc::new(MockGraphDB::new());
    let vector = Arc::new(MockVectorDB::new());
    let s = Uuid::new_v4();
    let sa = Uuid::new_v4();
    let (n1, n2) = (Uuid::new_v4(), Uuid::new_v4());
    graph
        .add_nodes_raw(vec![
            node(s, "Entity"),
            node(sa, "Entity"),
            node(n1, "Entity"),
            node(n2, "Entity"),
        ])
        .await
        .unwrap();
    point_to(&vector, "Entity", "name", s).await;
    point_to(&vector, "Entity", "name", sa).await;
    let et = EdgeType::deterministic_id("is_a");
    point_to(&vector, "EdgeType", "relationship_name", et).await;

    // Ledger: S claimed by A and B; Sa only by A; an "is_a" edge in each.
    let e_a = Uuid::new_v4();
    let e_b = Uuid::new_v4();
    graph_storage::upsert_nodes(
        &seed.db,
        &[
            seed.node_row(a, xa, s, "Entity"),
            seed.node_row(b, xb, s, "Entity"),
            seed.node_row(a, xa, sa, "Entity"),
        ],
    )
    .await
    .unwrap();
    graph_storage::upsert_edges(
        &seed.db,
        &[
            seed.edge_row(a, xa, e_a, n1, n2, "is_a"),
            seed.edge_row(b, xb, e_b, n1, n2, "is_a"),
        ],
    )
    .await
    .unwrap();

    shared_service(&seed, graph.clone(), vector.clone())
        .execute(&DeleteRequest {
            scope: DeleteScope::DatasetById { dataset_id: a },
            mode: DeleteMode::Soft,
            memory_only: false,
        })
        .await
        .unwrap();

    // The shared node, its point and the shared EdgeType point all stay —
    // B still claims them; the A-only node and its point are gone.
    assert!(
        Seed::graph_node(&graph, s).await,
        "shared node must survive"
    );
    assert!(!Seed::graph_node(&graph, sa).await, "A-only node must go");
    assert!(Seed::point(&vector, "Entity", "name", s).await);
    assert!(!Seed::point(&vector, "Entity", "name", sa).await);
    assert!(
        Seed::point(&vector, "EdgeType", "relationship_name", et).await,
        "EdgeType still claimed by B's edge"
    );

    // Ledger: A's rows gone, B's intact.
    assert_eq!(seed.prov_node_count(a).await, 0);
    assert_eq!(seed.prov_node_count(b).await, 1);
}

/// Shared store, item scope: item X in A and item Y in B both claim slug S.
/// Deleting X from A keeps S and removes X-only artifacts.
#[tokio::test]
async fn an_item_delete_keeps_artifacts_another_pair_claims() {
    let seed = seed().await;
    let a = seed.dataset("a").await;
    let b = seed.dataset("b").await;
    let x = seed.item(a).await;
    let y = seed.item(b).await;

    let graph = Arc::new(MockGraphDB::new());
    let vector = Arc::new(MockVectorDB::new());
    let s = Uuid::new_v4();
    let sax = Uuid::new_v4();
    graph
        .add_nodes_raw(vec![node(s, "Entity"), node(sax, "Entity")])
        .await
        .unwrap();
    point_to(&vector, "Entity", "name", s).await;
    point_to(&vector, "Entity", "name", sax).await;
    graph_storage::upsert_nodes(
        &seed.db,
        &[
            seed.node_row(a, x, s, "Entity"),
            seed.node_row(b, y, s, "Entity"),
            seed.node_row(a, x, sax, "Entity"),
        ],
    )
    .await
    .unwrap();

    shared_service(&seed, graph.clone(), vector.clone())
        .execute(&DeleteRequest {
            scope: DeleteScope::DataInDataset {
                dataset_id: a,
                data_id: x,
                delete_dataset_if_empty: false,
            },
            mode: DeleteMode::Soft,
            memory_only: false,
        })
        .await
        .unwrap();

    assert!(
        Seed::graph_node(&graph, s).await,
        "claimed slug must survive"
    );
    assert!(!Seed::graph_node(&graph, sax).await, "X-only slug must go");
    assert!(Seed::point(&vector, "Entity", "name", s).await);
    assert!(!Seed::point(&vector, "Entity", "name", sax).await);
    assert_eq!(seed.prov_node_count(a).await, 0);
    assert_eq!(seed.prov_node_count(b).await, 1);
}

// ---------------------------------------------------------------------------
// M03 — cleanup runs for every removed (dataset, data) link
// ---------------------------------------------------------------------------

/// Shared store: X linked to A and B. Deleting X from A removes A's ledger
/// rows for X but keeps its artifacts — (B, X) still claims them — and the
/// data row. Deleting X from B then removes everything.
#[tokio::test]
async fn an_item_delete_cleans_every_removed_link() {
    let seed = seed().await;
    let a = seed.dataset("a").await;
    let b = seed.dataset("b").await;
    let location = seed.storage.store(b"content", "x.txt").await.unwrap();
    let x = Uuid::new_v4();
    ops::data::create_data(
        &seed.db,
        Data::builder(
            x,
            "x.txt",
            &location,
            "file://x.txt",
            "txt",
            "text/plain",
            "hash-x".to_string(),
            seed.owner,
        )
        .build(),
    )
    .await
    .unwrap();
    for ds in [a, b] {
        ops::datasets::attach_data_to_dataset(&seed.db, ds, x)
            .await
            .unwrap();
    }

    let graph = Arc::new(MockGraphDB::new());
    let vector = Arc::new(MockVectorDB::new());
    let s = Uuid::new_v4();
    graph.add_nodes_raw(vec![node(s, "Entity")]).await.unwrap();
    point_to(&vector, "Entity", "name", s).await;
    graph_storage::upsert_nodes(
        &seed.db,
        &[
            seed.node_row(a, x, s, "Entity"),
            seed.node_row(b, x, s, "Entity"),
        ],
    )
    .await
    .unwrap();

    let svc = shared_service(&seed, graph.clone(), vector.clone());
    for (ds, want_node, want_data) in [(a, true, true), (b, false, false)] {
        svc.execute(&DeleteRequest {
            scope: DeleteScope::DataInDataset {
                dataset_id: ds,
                data_id: x,
                delete_dataset_if_empty: false,
            },
            mode: DeleteMode::Soft,
            memory_only: false,
        })
        .await
        .unwrap();
        assert_eq!(Seed::graph_node(&graph, s).await, want_node, "dataset {ds}");
        assert_eq!(Seed::point(&vector, "Entity", "name", s).await, want_node);
        assert_eq!(seed.data_exists(x).await, want_data, "dataset {ds}");
    }
}

/// Per-dataset storage: the detached dataset's own databases lose the item's
/// artifacts while the surviving dataset's store and the data row stay.
#[tokio::test]
async fn per_dataset_storage_cleans_the_detached_datasets_store() {
    let seed = seed().await;
    let a = seed.dataset("a").await;
    let b = seed.dataset("b").await;
    let x = seed.item(a).await;
    ops::datasets::attach_data_to_dataset(&seed.db, b, x)
        .await
        .unwrap();

    let pdm = PerDatasetMocks::new();
    let sa = Uuid::new_v4();
    let sb = Uuid::new_v4();
    pdm.graph(a)
        .add_nodes_raw(vec![node(sa, "Entity")])
        .await
        .unwrap();
    pdm.graph(b)
        .add_nodes_raw(vec![node(sb, "Entity")])
        .await
        .unwrap();
    pdm.vector(a)
        .create_collection("Entity", "name", 4)
        .await
        .unwrap();
    pdm.vector(a)
        .index_points("Entity", "name", &[VectorPoint::new(sa, vec![1.0; 4])])
        .await
        .unwrap();
    pdm.vector(b)
        .create_collection("Entity", "name", 4)
        .await
        .unwrap();
    pdm.vector(b)
        .index_points("Entity", "name", &[VectorPoint::new(sb, vec![1.0; 4])])
        .await
        .unwrap();
    graph_storage::upsert_nodes(
        &seed.db,
        &[
            seed.node_row(a, x, sa, "Entity"),
            seed.node_row(b, x, sb, "Entity"),
        ],
    )
    .await
    .unwrap();

    let svc = DeleteService::new(
        seed.storage.clone() as Arc<dyn StorageTrait>,
        seed.db.clone() as Arc<dyn DeleteDb>,
    )
    .with_graph_db(pdm.graph_db())
    .with_vector_db(pdm.vector_db())
    .with_dataset_storage(pdm.clone());

    svc.execute(&DeleteRequest {
        scope: DeleteScope::DataInDataset {
            dataset_id: a,
            data_id: x,
            delete_dataset_if_empty: false,
        },
        mode: DeleteMode::Soft,
        memory_only: false,
    })
    .await
    .unwrap();

    // A's own store and A's ledger rows for X are gone; B's store and rows
    // are untouched; the data row survives (B still links it).
    assert!(
        !Seed::graph_node(&pdm.graph(a), sa).await,
        "A's node must go"
    );
    assert!(
        Seed::graph_node(&pdm.graph(b), sb).await,
        "B's node must stay"
    );
    assert!(!Seed::point(&pdm.vector(a), "Entity", "name", sa).await);
    assert!(Seed::point(&pdm.vector(b), "Entity", "name", sb).await);
    assert_eq!(seed.prov_node_count(a).await, 0);
    assert_eq!(seed.prov_node_count(b).await, 1);
    assert!(seed.data_exists(x).await, "B still links the data row");
}

// ---------------------------------------------------------------------------
// M04 — the hard sweep keeps nodes a surviving ledger row claims
// ---------------------------------------------------------------------------

/// Shared store, Hard mode: a degree-one EntityType node claimed by dataset
/// B survives the sweep (graph node + vector point); an unclaimed degree-one
/// Entity is still swept.
#[tokio::test]
async fn a_hard_sweep_keeps_degree_one_nodes_still_claimed() {
    let seed = seed().await;
    let a = seed.dataset("a").await;
    let b = seed.dataset("b").await;
    let xa = seed.item(a).await;
    let xb = seed.item(b).await;
    let _ = (xa, xb);

    let graph = Arc::new(MockGraphDB::new());
    let vector = Arc::new(MockVectorDB::new());
    let t = Uuid::new_v4(); // claimed degree-one EntityType
    let u = Uuid::new_v4(); // unclaimed degree-one Entity
    let (n1, n2) = (Uuid::new_v4(), Uuid::new_v4());
    graph
        .add_nodes_raw(vec![
            node(t, "EntityType"),
            node(u, "Entity"),
            node(n1, "Entity"),
            node(n2, "Entity"),
        ])
        .await
        .unwrap();
    graph
        .add_edges(&[edge(n1, t, "is_a"), edge(n2, u, "mentions")])
        .await
        .unwrap();
    point_to(&vector, "EntityType", "name", t).await;
    point_to(&vector, "Entity", "name", u).await;

    // B's ledger claims T (n1 is B's node, so T's edge is a live claim target);
    // nothing claims U.
    graph_storage::upsert_nodes(
        &seed.db,
        &[
            seed.node_row(b, xb, t, "EntityType"),
            seed.node_row(b, xb, n1, "Entity"),
        ],
    )
    .await
    .unwrap();

    shared_service(&seed, graph.clone(), vector.clone())
        .execute(&DeleteRequest {
            scope: DeleteScope::DatasetById { dataset_id: a },
            mode: DeleteMode::Hard,
            memory_only: false,
        })
        .await
        .unwrap();

    assert!(Seed::graph_node(&graph, t).await, "claimed type node stays");
    assert!(
        Seed::point(&vector, "EntityType", "name", t).await,
        "claimed point stays"
    );
    assert!(
        !Seed::graph_node(&graph, u).await,
        "unclaimed node is swept"
    );
    assert!(
        !Seed::point(&vector, "Entity", "name", u).await,
        "unclaimed point is swept"
    );
}

// ---------------------------------------------------------------------------
// Preview == execution (R-bot feedback)
// ---------------------------------------------------------------------------

/// (a) Item scope, shared store: X linked to A and B; deleting X from A
/// previews (and executes) the pair's ledger rows — the artifacts stay
/// claimed by (B, X), so both preview and result count zero graph/vector
/// deletions and one provenance row.
#[tokio::test]
async fn preview_counts_the_per_pair_cleanup_for_item_deletes() {
    let seed = seed().await;
    let a = seed.dataset("a").await;
    let b = seed.dataset("b").await;
    let location = seed.storage.store(b"content", "x.txt").await.unwrap();
    let x = Uuid::new_v4();
    ops::data::create_data(
        &seed.db,
        Data::builder(
            x,
            "x.txt",
            &location,
            "file://x.txt",
            "txt",
            "text/plain",
            "hash-x".to_string(),
            seed.owner,
        )
        .build(),
    )
    .await
    .unwrap();
    for ds in [a, b] {
        ops::datasets::attach_data_to_dataset(&seed.db, ds, x)
            .await
            .unwrap();
    }

    let graph = Arc::new(MockGraphDB::new());
    let vector = Arc::new(MockVectorDB::new());
    let s = Uuid::new_v4();
    graph.add_nodes_raw(vec![node(s, "Entity")]).await.unwrap();
    point_to(&vector, "Entity", "name", s).await;
    graph_storage::upsert_nodes(
        &seed.db,
        &[
            seed.node_row(a, x, s, "Entity"),
            seed.node_row(b, x, s, "Entity"),
        ],
    )
    .await
    .unwrap();

    let svc = shared_service(&seed, graph, vector);
    let request = DeleteRequest {
        scope: DeleteScope::DataInDataset {
            dataset_id: a,
            data_id: x,
            delete_dataset_if_empty: false,
        },
        mode: DeleteMode::Soft,
        memory_only: false,
    };
    let preview = svc.preview(&request).await.unwrap();
    let result = svc.execute(&request).await.unwrap();

    assert_eq!(
        (
            preview.graph_nodes_to_delete,
            preview.vector_points_to_delete,
            preview.provenance_nodes_to_delete,
            preview.provenance_edges_to_delete
        ),
        (
            result.deleted_graph_nodes,
            result.deleted_vector_points,
            result.deleted_provenance_nodes,
            result.deleted_provenance_edges
        ),
        "preview must count exactly what execute deletes"
    );
    assert_eq!(result.deleted_provenance_nodes, 1, "the (A,X) row goes");
    assert_eq!(result.deleted_graph_nodes, 0, "(B,X) keeps the shared slug");
}

/// (b) Shared store, EdgeType filtering: A and B each claim an edge with the
/// same retrieval text — deleting A previews no EdgeType point for it (the
/// text is claimed outside A), matching execution.
#[tokio::test]
async fn preview_filters_edge_type_points_claimed_elsewhere() {
    let seed = seed().await;
    let a = seed.dataset("a").await;
    let b = seed.dataset("b").await;
    let xa = seed.item(a).await;
    let xb = seed.item(b).await;

    let graph = Arc::new(MockGraphDB::new());
    let vector = Arc::new(MockVectorDB::new());
    let (n1, n2) = (Uuid::new_v4(), Uuid::new_v4());
    let et = EdgeType::deterministic_id("is_a");
    point_to(&vector, "EdgeType", "relationship_name", et).await;
    // A's edge produces a Triplet point — the one point this delete removes.
    let triplet = Triplet::new(n1, n2, "is_a".to_string(), String::new()).id;
    point_to(&vector, "Triplet", "text", triplet).await;
    let ea = Uuid::new_v4();
    let eb = Uuid::new_v4();
    graph_storage::upsert_edges(
        &seed.db,
        &[
            seed.edge_row(a, xa, ea, n1, n2, "is_a"),
            seed.edge_row(b, xb, eb, n1, n2, "is_a"),
        ],
    )
    .await
    .unwrap();

    let svc = shared_service(&seed, graph, vector);
    let request = DeleteRequest {
        scope: DeleteScope::DatasetById { dataset_id: a },
        mode: DeleteMode::Soft,
        memory_only: false,
    };
    let preview = svc.preview(&request).await.unwrap();
    let result = svc.execute(&request).await.unwrap();

    assert_eq!(
        (
            preview.graph_nodes_to_delete,
            preview.vector_points_to_delete,
            preview.provenance_nodes_to_delete,
            preview.provenance_edges_to_delete
        ),
        (
            result.deleted_graph_nodes,
            result.deleted_vector_points,
            result.deleted_provenance_nodes,
            result.deleted_provenance_edges
        ),
        "preview must count exactly what execute deletes"
    );
    // A's edge slug is unique to A → its triplet point goes, but the shared
    // "is_a" EdgeType stays claimed by B — 1 vector point total.
    assert_eq!(
        result.deleted_vector_points, 1,
        "triplet only — EdgeType stays"
    );
}

/// (c) Multi-dataset, shared store: A and B both claim slug S — deleting
/// both in one request deletes S once (by the second cleanup), and preview
/// counts it once too.
#[tokio::test]
async fn preview_counts_a_slug_deleted_once_across_a_multi_dataset_request() {
    let seed = seed().await;
    let a = seed.dataset("a").await;
    let b = seed.dataset("b").await;
    let xa = seed.item(a).await;
    let xb = seed.item(b).await;

    let graph = Arc::new(MockGraphDB::new());
    let vector = Arc::new(MockVectorDB::new());
    let s = Uuid::new_v4();
    graph.add_nodes_raw(vec![node(s, "Entity")]).await.unwrap();
    point_to(&vector, "Entity", "name", s).await;
    graph_storage::upsert_nodes(
        &seed.db,
        &[
            seed.node_row(a, xa, s, "Entity"),
            seed.node_row(b, xb, s, "Entity"),
        ],
    )
    .await
    .unwrap();

    let svc = shared_service(&seed, graph, vector);
    let preview = svc.preview_datasets_by_id(&[a, b]).await.unwrap();
    let result = svc
        .execute_datasets_by_id(&[a, b], DeleteMode::Soft, false)
        .await
        .unwrap();

    assert_eq!(
        (
            preview.graph_nodes_to_delete,
            preview.vector_points_to_delete,
            preview.provenance_nodes_to_delete,
            preview.provenance_edges_to_delete
        ),
        (
            result.deleted_graph_nodes,
            result.deleted_vector_points,
            result.deleted_provenance_nodes,
            result.deleted_provenance_edges
        ),
        "preview must count exactly what execute deletes"
    );
    assert_eq!(
        result.deleted_graph_nodes, 1,
        "S goes once, not twice or zero"
    );
}

/// (d) Structural edges: `is_part_of` rows carry a nil `data_id`, so no
/// pair-scoped provenance count can see them — yet they die with the pair
/// whose endpoints are exclusive to it. Preview must count the row and its
/// `EdgeType` point, and execution must actually remove both.
#[tokio::test]
async fn structural_edges_die_with_their_pair_and_preview_counts_them() {
    let seed = seed().await;
    let a = seed.dataset("a").await;
    let xa = seed.item(a).await;

    let graph = Arc::new(MockGraphDB::new());
    let vector = Arc::new(MockVectorDB::new());
    let (chunk, doc) = (Uuid::new_v4(), Uuid::new_v4());
    graph
        .add_nodes_raw(vec![node(chunk, "TextChunk"), node(doc, "Document")])
        .await
        .unwrap();
    point_to(&vector, "TextChunk", "name", chunk).await;
    point_to(&vector, "Document", "name", doc).await;
    let et = EdgeType::deterministic_id("is_part_of");
    point_to(&vector, "EdgeType", "relationship_name", et).await;
    graph_storage::upsert_nodes(
        &seed.db,
        &[
            seed.node_row(a, xa, chunk, "TextChunk"),
            seed.node_row(a, xa, doc, "Document"),
        ],
    )
    .await
    .unwrap();
    // The structural row: nil `data_id`, both endpoints exclusive to (A,Xa).
    graph_storage::upsert_edges(
        &seed.db,
        &[seed.edge_row(a, Uuid::nil(), Uuid::new_v4(), chunk, doc, "is_part_of")],
    )
    .await
    .unwrap();

    let svc = shared_service(&seed, graph, vector.clone());
    let request = DeleteRequest {
        scope: DeleteScope::DataInDataset {
            dataset_id: a,
            data_id: xa,
            delete_dataset_if_empty: false,
        },
        mode: DeleteMode::Soft,
        memory_only: false,
    };
    let preview = svc.preview(&request).await.unwrap();
    let result = svc.execute(&request).await.unwrap();

    assert_eq!(
        (
            preview.graph_nodes_to_delete,
            preview.vector_points_to_delete,
            preview.provenance_nodes_to_delete,
            preview.provenance_edges_to_delete
        ),
        (
            result.deleted_graph_nodes,
            result.deleted_vector_points,
            result.deleted_provenance_nodes,
            result.deleted_provenance_edges
        ),
        "preview must count exactly what execute deletes"
    );
    assert_eq!(
        result.deleted_provenance_edges, 1,
        "the nil-data_id row dies with the pair"
    );
    assert_eq!(
        result.deleted_vector_points, 3,
        "two node points and the is_part_of EdgeType point"
    );
    assert!(
        !Seed::point(&vector, "EdgeType", "relationship_name", et).await,
        "the structural EdgeType point is gone"
    );
}

/// (e) Two datasets deleted in one request claim one slug with different
/// `indexed_fields`: A's row indexes `name` and `extra`, B's only `name`.
/// The slug dies with the second cleanup — but its artifacts are the union
/// every dying row named, so both points go and preview counts both.
#[tokio::test]
async fn a_dying_slug_takes_the_union_of_its_rows_indexed_fields() {
    let seed = seed().await;
    let a = seed.dataset("a").await;
    let b = seed.dataset("b").await;
    let xa = seed.item(a).await;
    let xb = seed.item(b).await;

    let graph = Arc::new(MockGraphDB::new());
    let vector = Arc::new(MockVectorDB::new());
    let s = Uuid::new_v4();
    graph.add_nodes_raw(vec![node(s, "Entity")]).await.unwrap();
    point_to(&vector, "Entity", "name", s).await;
    point_to(&vector, "Entity", "extra", s).await;
    graph_storage::upsert_nodes(
        &seed.db,
        &[
            GraphNode {
                indexed_fields: json!(["name", "extra"]),
                ..seed.node_row(a, xa, s, "Entity")
            },
            GraphNode {
                indexed_fields: json!(["name"]),
                ..seed.node_row(b, xb, s, "Entity")
            },
        ],
    )
    .await
    .unwrap();

    let svc = shared_service(&seed, graph, vector.clone());
    let preview = svc.preview_datasets_by_id(&[a, b]).await.unwrap();
    let result = svc
        .execute_datasets_by_id(&[a, b], DeleteMode::Soft, false)
        .await
        .unwrap();

    assert_eq!(
        (
            preview.graph_nodes_to_delete,
            preview.vector_points_to_delete,
            preview.provenance_nodes_to_delete,
            preview.provenance_edges_to_delete
        ),
        (
            result.deleted_graph_nodes,
            result.deleted_vector_points,
            result.deleted_provenance_nodes,
            result.deleted_provenance_edges
        ),
        "preview must count exactly what execute deletes"
    );
    assert_eq!(
        result.deleted_vector_points, 2,
        "the union of indexed fields, not the last row's"
    );
    assert!(
        !Seed::point(&vector, "Entity", "extra", s).await,
        "the field only the earlier row indexed is gone too"
    );
}

/// (f) Two datasets deleted together hold edges with different slugs but
/// the same retrieval text — one `EdgeType` point. Each row is unique to
/// its dataset: the point goes exactly once, and preview counts once.
#[tokio::test]
async fn a_retrieval_text_shared_across_dying_slugs_goes_once() {
    let seed = seed().await;
    let a = seed.dataset("a").await;
    let b = seed.dataset("b").await;
    let xa = seed.item(a).await;
    let xb = seed.item(b).await;

    let graph = Arc::new(MockGraphDB::new());
    let vector = Arc::new(MockVectorDB::new());
    let (n1, n2) = (Uuid::new_v4(), Uuid::new_v4());
    let et = EdgeType::deterministic_id("is_a");
    point_to(&vector, "EdgeType", "relationship_name", et).await;
    for (src, dst) in [(n1, n2), (n2, n1)] {
        let t = Triplet::new(src, dst, "is_a".to_string(), String::new()).id;
        point_to(&vector, "Triplet", "text", t).await;
    }
    // Different edge slugs, same retrieval text → one EdgeType point.
    graph_storage::upsert_edges(
        &seed.db,
        &[
            seed.edge_row(a, xa, Uuid::new_v4(), n1, n2, "is_a"),
            seed.edge_row(b, xb, Uuid::new_v4(), n2, n1, "is_a"),
        ],
    )
    .await
    .unwrap();

    let svc = shared_service(&seed, graph, vector.clone());
    let preview = svc.preview_datasets_by_id(&[a, b]).await.unwrap();
    let result = svc
        .execute_datasets_by_id(&[a, b], DeleteMode::Soft, false)
        .await
        .unwrap();

    assert_eq!(
        (
            preview.graph_nodes_to_delete,
            preview.vector_points_to_delete,
            preview.provenance_nodes_to_delete,
            preview.provenance_edges_to_delete
        ),
        (
            result.deleted_graph_nodes,
            result.deleted_vector_points,
            result.deleted_provenance_nodes,
            result.deleted_provenance_edges
        ),
        "preview must count exactly what execute deletes"
    );
    assert_eq!(
        result.deleted_vector_points, 3,
        "two triplets and the shared EdgeType point once"
    );
    assert!(
        !Seed::point(&vector, "EdgeType", "relationship_name", et).await,
        "a text whose claims all died is gone — a surviving row in a dying \
         pair is not a claim"
    );
}
