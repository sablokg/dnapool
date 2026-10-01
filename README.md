# dnapool

- Making Rust Machine and Graph learning microbiome DNA automated
- Padding and masking for Burn, AxonML and Flodl. 
- For CNN1D, Transformers and Autoencoders.

```
cargo build
```

## Usage

Add it as a path/git dependency (not yet published to crates.io):

```toml
[dependencies]
dnapool = { path = "../dnapool" }
```

Everything in the crate is a **module**: a struct with `new()` and a single
`forward(&input)` method (the `Module` trait), like `PaddedTensor`. There are no
free functions. Everything is exported from the crate root:

```rust
use dnapool::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // sequences.csv: lines of `SEQUENCE,label`
    let padded  = PaddedTensor::new().forward("sequences.csv")?;
    let tensors = Tensor::new().forward(&padded.sequences)?;
    println!("{} sequences, {} labels", tensors.len(), padded.labels.len());

    // edges.csv: lines of `source,target[,weight]`
    let graphs = AdjUndirTensor::new().forward("edges.csv")?;
    graphs.print()?;                          // both matrices, labelled
    let m = &graphs.undirected_matrix;        // row i == graphs.nodes.labels[i]
    Ok(())
}
```

## Modules

| Module | Input | Output |
| --- | --- | --- |
| `PaddedTensor` | CSV path | `PatchResult` |
| `Tensor` | `[String]` | `Vec<Vec<f32>>` |
| `LoadInteractions` | edge-list path | `Vec<Interaction>` |
| `AdjacencyTensor` | edge-list path | `AdjacencyResult` (directed graph, weighted + binary matrices) |
| `AdjUndirTensor` | edge-list path | `AdjUndirResult` (directed + undirected graphs and matrices) |
| `BuildGraph` | `[Interaction]` | `Directed` graph + name lookup |
| `BuildNodeIndex` | `[Interaction]` | `Nodes` |
| `BuildDirected` / `BuildUndirected` | `[Interaction]` | `Directed` / `Undirected` |
| `WeightedAdjacency` / `BinaryAdjacency` | `Directed` | `Vec<Vec<f64>>` / `Vec<Vec<u8>>` |
| `DirectedAdjacency` / `UndirectedAdjacency` | `Directed` / `Undirected` | `Vec<Vec<f64>>` |
| `PrintMatrix` | `[Vec<f64>]` | prints a labelled matrix |

The three one-call modules (`PaddedTensor`, `AdjacencyTensor`, `AdjUndirTensor`)
are built from the finer-grained ones, which stay available for step-by-step use:

```rust
use dnapool::*;

let ints  = LoadInteractions::new().forward("edges.csv")?;
let nodes = BuildNodeIndex::new().forward(&ints)?;
let g     = BuildUndirected::new(nodes.clone()).forward(&ints)?;
let m     = UndirectedAdjacency::new().forward(&g)?;
PrintMatrix::new(&nodes.labels).forward(&m)?;
```

## Changelog

**0.2.0** (breaking): all free functions (`patcher`, `tensorconvert`,
`load_interactions`, `build_graph`, `build_directed`, `build_undirected`,
`build_node_index`, `*_adjacency_matrix`, `print_matrix`) were replaced by
modules. `adjacency` and `adjundir` now share one `Interaction` (fields
`source`, `target`, `weight`), one `LoadInteractions` and one `PrintMatrix`.

Gaurav Sablok \
gsablok@proton.me
