//! `dnapool` — a complete toolkit for preparing microbiome / DNA sequence data
//! and interaction graphs for Rust machine-learning frameworks (Burn, AxonML,
//! Flodl).
//!
//! **Everything is a [`Module`]**: a small struct that holds its configuration
//! and exposes a single [`forward`](Module::forward) method, exactly like
//! [`PaddedTensor`]. There are no free functions. Everything is re-exported
//! from the crate root, so one import is enough:
//!
//! ```no_run
//! use dnapool::*;
//!
//! let padded  = PaddedTensor::new().forward("sequences.csv")?;
//! let tensors = Tensor::new().forward(&padded.sequences)?;
//! let graphs  = AdjUndirTensor::new().forward("edges.csv")?;
//! graphs.print()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Modules
//!
//! | Module | Input | Output |
//! | --- | --- | --- |
//! | [`PaddedTensor`] | CSV path | [`PatchResult`] |
//! | [`Tensor`] | `[String]` | `Vec<Vec<f32>>` |
//! | [`LoadInteractions`] | edge-list path | `Vec<`[`Interaction`]`>` |
//! | [`AdjacencyTensor`] | edge-list path | [`AdjacencyResult`] (directed graph, weighted + binary matrices) |
//! | [`AdjUndirTensor`] | edge-list path | [`AdjUndirResult`] (directed + undirected graphs and matrices) |
//! | [`BuildGraph`] | `[Interaction]` | [`Directed`] graph + name lookup |
//! | [`BuildNodeIndex`] | `[Interaction]` | [`Nodes`] |
//! | [`BuildDirected`] / [`BuildUndirected`] | `[Interaction]` | [`Directed`] / [`Undirected`] |
//! | [`WeightedAdjacency`] / [`BinaryAdjacency`] | [`Directed`] | `Vec<Vec<f64>>` / `Vec<Vec<u8>>` |
//! | [`DirectedAdjacency`] / [`UndirectedAdjacency`] | [`Directed`] / [`Undirected`] | `Vec<Vec<f64>>` |
//! | [`PrintMatrix`] | `[Vec<f64>]` | prints a labelled matrix |
//!
//! The one-call modules ([`PaddedTensor`], [`AdjacencyTensor`],
//! [`AdjUndirTensor`]) are built from the finer-grained ones below them, which
//! stay available for step-by-step use.

pub mod adjacency;
pub mod adjundir;
pub mod interaction;
pub mod module;
pub mod pad;
pub mod tensor;

// core trait
pub use module::{Module, ModuleResult};

// sequences: padding + one-hot encoding
pub use pad::{PaddedTensor, PatchResult};
pub use tensor::Tensor;

// shared graph types: interactions, loading, printing
pub use interaction::{Directed, Interaction, LoadInteractions, PrintMatrix, Undirected};

// directed graphs
pub use adjacency::{
    AdjacencyResult, AdjacencyTensor, BinaryAdjacency, BuildGraph, WeightedAdjacency,
};

// directed + undirected graphs
pub use adjundir::{
    AdjUndirResult, AdjUndirTensor, BuildDirected, BuildNodeIndex, BuildUndirected,
    DirectedAdjacency, Nodes, UndirectedAdjacency,
};

#[cfg(test)]
mod tests {
    //! Guards the flat `use dnapool::*` API: every module must resolve from the
    //! crate root and implement [`Module`].
    use crate::*;

    fn is_module<I: ?Sized, M: Module<I>>(_: &M) {}

    #[test]
    fn every_module_is_reachable_and_implements_module() {
        is_module::<str, _>(&PaddedTensor::new());
        is_module::<[String], _>(&Tensor::new());
        is_module::<str, _>(&LoadInteractions::new());
        is_module::<str, _>(&AdjacencyTensor::new());
        is_module::<str, _>(&AdjUndirTensor::new());
        is_module::<[Interaction], _>(&BuildGraph::new());
        is_module::<[Interaction], _>(&BuildNodeIndex::new());
        is_module::<[Interaction], _>(&BuildDirected::new(Nodes::default()));
        is_module::<[Interaction], _>(&BuildUndirected::new(Nodes::default()));
        is_module::<Directed, _>(&WeightedAdjacency::new());
        is_module::<Directed, _>(&BinaryAdjacency::new());
        is_module::<Directed, _>(&DirectedAdjacency::new());
        is_module::<Undirected, _>(&UndirectedAdjacency::new());
        is_module::<[Vec<f64>], _>(&PrintMatrix::new(&[]));
    }

    #[test]
    fn end_to_end_pipeline() {
        let padded = PaddedTensor::new()
            .pad_sequences(&["ACGT".to_string(), "AC".to_string()])
            .unwrap();
        let t = Tensor::new().forward(&padded).unwrap();
        assert_eq!(t[0].len(), t[1].len());

        let e = |s: &str, t: &str, w: f64| Interaction { source: s.into(), target: t.into(), weight: w };
        let r = AdjUndirTensor::new()
            .from_interactions(&[e("A", "B", 1.0), e("B", "A", 2.0)])
            .unwrap();
        assert_eq!(r.directed_matrix[0][1], 1.0);
        assert_eq!(r.undirected_matrix[0][1], 3.0);
    }
}
