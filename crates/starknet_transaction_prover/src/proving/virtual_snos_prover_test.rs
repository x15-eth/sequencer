//! Integration tests for the VirtualSnosProver (full prove_transaction flow).
//!
//! `test_prove_transfer_transaction` exercises the full prover pipeline against Sepolia
//! (transaction extraction, OS execution, and proof generation) and supports three modes
//! (see [`crate::running::rpc_records`] and [`crate::test_utils::resolve_test_mode`]):
//!
//! - **Live mode** (default): runs against a real node (requires `NODE_URL`).
//! - **Recording mode** (`RECORD_RPC_RECORDS=1`): runs against a real node through a recording
//!   proxy and saves all RPC interactions to a records file.
//! - **Offline mode** (records file present): replays pre-recorded interactions from a mock server.
//!
//! `test_augment_with_delete_siblings_fetches_orphan_preimage` is self-contained — it
//! constructs the problematic edge case in code and feeds it through a `MockRpcServer`, so it
//! runs in CI without any chain dependency.

use std::collections::HashSet;

use blockifier::state::cached_state::StateMaps;
use blockifier_reexecution::state_reader::rpc_objects::BlockId;
use blockifier_test_utils::calldata::create_calldata;
use indexmap::IndexMap;
use rstest::rstest;
use serde_json::json;
use starknet_api::block::BlockNumber;
use starknet_api::core::ContractAddress;
use starknet_api::state::StorageKey;
use starknet_api::{contract_address, felt};
use starknet_proof_verifier::verify_proof;
use starknet_rust_core::types::{
    BinaryNode,
    ContractLeafData,
    ContractStorageKeys,
    ContractsProof,
    EdgeNode,
    GlobalRoots,
    MerkleNode,
    StorageProof as RpcStorageProof,
};
use starknet_types_core::felt::Felt;
use url::Url;

use crate::proving::virtual_snos_prover::VirtualSnosProver;
use crate::running::rpc_records::{MockRpcServer, RpcInteraction, RpcRecords};
use crate::running::storage_proofs::{RpcStorageProofsProvider, RpcStorageProofsQuery};
use crate::running::virtual_block_executor::{BaseBlockInfo, VirtualBlockExecutionData};
use crate::test_utils::{
    build_client_side_rpc_invoke,
    resolve_test_mode,
    runner_factory,
    DUMMY_ACCOUNT_ADDRESS,
    STRK_TOKEN_ADDRESS_SEPOLIA,
};

/// Integration test for the full prover pipeline with a STRK `transfer` transaction.
/// Runs on a Sepolia environment; in live/recording mode requires a Sepolia RPC node via
/// `NODE_URL`.
#[rstest]
#[tokio::test(flavor = "multi_thread")]
#[ignore] // Run with --ignored; supports live, recording, and offline modes.
async fn test_prove_transfer_transaction() {
    let test_mode = resolve_test_mode("test_prove_transfer_transaction").await;

    let strk_token = ContractAddress::try_from(STRK_TOKEN_ADDRESS_SEPOLIA).unwrap();
    let account = ContractAddress::try_from(DUMMY_ACCOUNT_ADDRESS).unwrap();
    let recipient = contract_address!("0x123");

    let amount_low = felt!("1");
    let amount_high = felt!("0");
    let calldata =
        create_calldata(strk_token, "transfer", &[recipient.into(), amount_low, amount_high]);
    let rpc_tx = build_client_side_rpc_invoke(account, calldata);

    let factory = runner_factory(&test_mode.rpc_url());
    let prover = VirtualSnosProver::from_runner(factory);

    let result = prover.prove_transaction(BlockId::Latest, rpc_tx).await;
    test_mode.finalize();

    let output = result.expect("prove_transaction should succeed");
    let proof_facts = output.proof_facts.clone();
    let proof = output.proof.clone();
    tokio::task::spawn_blocking(move || verify_proof(proof_facts, proof))
        .await
        .expect("proof verification task panicked")
        .expect("proof verification should succeed");
}

/// Verifies the wiring added by the storage-delete support: when the first-round storage proof
/// is missing the preimage of a sibling subtree on a deletion path, the provider issues a
/// SECOND `starknet_getStorageProof` whose response carries that preimage, then merges the
/// supplement into the proof.
///
/// The Patricia tree, modeled on the edge case the PDF described:
///
/// ```text
/// A (edge, path=0, length=3) → B (binary)
///                              ├─ C (edge, path=0, length=247) → leaf
///                              └─ D (orphan — preimage missing from the first proof)
/// ```
///
/// Deleting the leaf under `C` (key = `Felt::ZERO`) forces the committer to canonicalize `B`
/// by promoting `D`. That needs `D`'s preimage, which the first proof doesn't have. The
/// supplement fetch crafts a key whose top 4 bits are `0001` so the RPC walks through `D`'s
/// subtree and returns its preimage.
#[tokio::test(flavor = "multi_thread")]
async fn test_augment_with_delete_siblings_fetches_orphan_preimage() {
    // Tree shape: contract addr 100, storage trie rooted at A.
    let addr = ContractAddress::try_from(Felt::from(100u64)).unwrap();
    let (a, b, c, d, leaf, d_child) = (
        Felt::from(1u64),
        Felt::from(2u64),
        Felt::from(3u64),
        Felt::from(4u64),
        Felt::from(5u64),
        Felt::from(7u64),
    );

    // First-round proof: nodes for A, B, C only. D is referenced by B but not in the map.
    let mut first_round_nodes = IndexMap::default();
    first_round_nodes
        .insert(a, MerkleNode::EdgeNode(EdgeNode { path: Felt::ZERO, length: 3, child: b }));
    first_round_nodes.insert(b, MerkleNode::BinaryNode(BinaryNode { left: c, right: d }));
    first_round_nodes
        .insert(c, MerkleNode::EdgeNode(EdgeNode { path: Felt::ZERO, length: 247, child: leaf }));

    let first_round_proof = synthetic_storage_proof(/* storage_root */ a, first_round_nodes);

    // Supplement that the RPC should return when queried with the crafted sibling key:
    // it contains D's preimage (and a child to keep the proof well-formed).
    let mut supplement_nodes = IndexMap::default();
    supplement_nodes.insert(
        d,
        MerkleNode::EdgeNode(EdgeNode { path: Felt::ZERO, length: 247, child: d_child }),
    );

    let supplement_proof = synthetic_storage_proof(/* storage_root */ a, supplement_nodes);

    // Wire a MockRpcServer to return `supplement_proof` for any `starknet_getStorageProof` call
    // whose params include the crafted sibling key (`1 << 247`).
    let crafted_sibling_key = {
        let mut bytes = [0u8; 32];
        bytes[1] = 0x80; // bit 247 from LSB, the top-4-bits = 0001 pattern
        Felt::from_bytes_be(&bytes)
    };
    let addr_hex = format!("{:#x}", addr.0.key());
    let crafted_hex = format!("{crafted_sibling_key:#x}");
    let records = RpcRecords {
        interactions: vec![RpcInteraction {
            method: "starknet_getStorageProof".to_string(),
            sorted_params: json!({
                "block_id": {"block_number": 1},
                "class_hashes": [],
                "contract_addresses": [addr_hex.clone()],
                "contracts_storage_keys": [{
                    "contract_address": addr_hex,
                    "storage_keys": [crafted_hex],
                }],
            }),
            response: json!({
                "jsonrpc": "2.0",
                "id": 0,
                "result": rpc_storage_proof_to_json(&supplement_proof),
            }),
        }],
    };
    let mock_server = MockRpcServer::new(&records).await;

    let provider = RpcStorageProofsProvider::new(Url::parse(&mock_server.url()).unwrap());

    // Build the inputs to augment_with_delete_siblings.
    let query = RpcStorageProofsQuery {
        class_hashes: vec![],
        contract_addresses: vec![addr],
        contract_storage_keys: vec![ContractStorageKeys {
            contract_address: *addr.0.key(),
            storage_keys: vec![Felt::ZERO],
        }],
    };
    let mut state_diff = StateMaps::default();
    state_diff.storage.insert((addr, StorageKey::from(0u32)), Felt::ZERO);

    let execution_data = VirtualBlockExecutionData {
        execution_outputs: vec![],
        l2_to_l1_messages: vec![],
        base_block_info: BaseBlockInfo {
            block_context: blockifier::context::BlockContext::create_for_account_testing(),
            base_block_hash: starknet_api::block::BlockHash::default(),
            prev_base_block_hash: starknet_api::block::BlockHash::default(),
            base_block_header_commitments: Default::default(),
        },
        initial_reads: StateMaps::default(),
        state_diff,
        executed_class_hashes: HashSet::new(),
    };

    // Sanity: D is NOT in the first-round proof.
    let storage_nodes_before = &first_round_proof.contracts_storage_proofs[0];
    assert!(
        !storage_nodes_before.contains_key(&d),
        "test setup error: D's preimage must be missing from the first-round proof",
    );

    let augmented = provider
        .augment_with_delete_siblings(BlockNumber(1), &query, first_round_proof, &execution_data)
        .await
        .expect("augment_with_delete_siblings should succeed");

    // D's preimage is now present — the supplement fetch fired and merged correctly.
    let storage_nodes_after = &augmented.contracts_storage_proofs[0];
    assert!(
        storage_nodes_after.contains_key(&d),
        "augment_with_delete_siblings did not fetch D's preimage: nodes={:?}",
        storage_nodes_after.keys().collect::<Vec<_>>(),
    );
}

/// Builds a minimal `RpcStorageProof` for a single contract whose storage trie root is
/// `storage_root` and whose inner-node map is `storage_nodes`.
fn synthetic_storage_proof(
    storage_root: Felt,
    storage_nodes: IndexMap<Felt, MerkleNode>,
) -> RpcStorageProof {
    RpcStorageProof {
        classes_proof: IndexMap::default(),
        contracts_proof: ContractsProof {
            nodes: IndexMap::default(),
            contract_leaves_data: vec![ContractLeafData {
                nonce: Felt::ZERO,
                class_hash: Felt::ZERO,
                storage_root: Some(storage_root),
            }],
        },
        contracts_storage_proofs: vec![storage_nodes],
        global_roots: GlobalRoots {
            contracts_tree_root: Felt::ZERO,
            classes_tree_root: Felt::ZERO,
            block_hash: Felt::ZERO,
        },
    }
}

/// Serializes an `RpcStorageProof` into the JSON shape returned by `starknet_getStorageProof`.
fn rpc_storage_proof_to_json(proof: &RpcStorageProof) -> serde_json::Value {
    serde_json::to_value(proof).expect("RpcStorageProof serializes to JSON")
}
