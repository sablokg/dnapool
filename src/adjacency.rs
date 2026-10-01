use crate::interaction::{Directed, Interaction, LoadInteractions, PrintMatrix};
use crate::module::{Module, ModuleResult};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// Graph-builder module. `forward` builds a directed petgraph graph from a
/// list of interactions. Proteins are deduplicated into nodes as they're
/// encountered; repeated edges between the same pair accumulate their
/// weights. Also returns the name -> `NodeIndex` lookup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BuildGraph;

impl BuildGraph {
    pub fn new() -> Self {
        Self
    }
}

impl Module<[Interaction]> for BuildGraph {
    type Output = (Directed, HashMap<String, NodeIndex>);

    fn forward(&self, interactions: &[Interaction]) -> ModuleResult<Self::Output> {
        let mut graph: Directed = DiGraph::new();
        let mut index: HashMap<String, NodeIndex> = HashMap::new();

        let mut get_or_add = |graph: &mut Directed, name: &str| -> NodeIndex {
            *index
                .entry(name.to_string())
                .or_insert_with(|| graph.add_node(name.to_string()))
        };

        for edge in interactions {
            let src = get_or_add(&mut graph, &edge.source);
            let dst = get_or_add(&mut graph, &edge.target);

            if let Some(existing) = graph.find_edge(src, dst) {
                // Same directed pair seen again: accumulate weight rather
                // than adding a parallel edge.
                if let Some(w) = graph.edge_weight_mut(existing) {
                    *w += edge.weight;
                }
            } else {
                graph.add_edge(src, dst, edge.weight);
            }
        }

        Ok((graph, index))
    }
}

/// Dense weighted adjacency module. `forward` orders nodes by petgraph's
/// internal `NodeIndex` (i.e. insertion order). `matrix[i][j]` is the edge
/// weight from node i to node j, or 0.0 if no edge exists.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WeightedAdjacency;

impl WeightedAdjacency {
    pub fn new() -> Self {
        Self
    }
}

impl Module<Directed> for WeightedAdjacency {
    type Output = Vec<Vec<f64>>;

    fn forward(&self, graph: &Directed) -> ModuleResult<Self::Output> {
        let n = graph.node_count();
        let mut matrix = vec![vec![0.0_f64; n]; n];
        for edge in graph.edge_references() {
            matrix[edge.source().index()][edge.target().index()] = *edge.weight();
        }
        Ok(matrix)
    }
}

/// Same as [`WeightedAdjacency`] but a 0/1 binary matrix, ignoring edge
/// weights - useful when the PPI network is unweighted or you only care
/// about connectivity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BinaryAdjacency;

impl BinaryAdjacency {
    pub fn new() -> Self {
        Self
    }
}

impl Module<Directed> for BinaryAdjacency {
    type Output = Vec<Vec<u8>>;

    fn forward(&self, graph: &Directed) -> ModuleResult<Self::Output> {
        let n = graph.node_count();
        let mut matrix = vec![vec![0u8; n]; n];
        for edge in graph.edge_references() {
            matrix[edge.source().index()][edge.target().index()] = 1;
        }
        Ok(matrix)
    }
}

/// Everything [`AdjacencyTensor`] produces in one go: the directed graph, its
/// node labels (in `NodeIndex` order, so `labels[i]` is row/column `i`), the
/// name -> `NodeIndex` lookup, and both dense adjacency matrices.
#[derive(Debug, Clone)]
pub struct AdjacencyResult {
    pub labels: Vec<String>,
    pub index: HashMap<String, NodeIndex>,
    pub graph: Directed,
    pub weighted: Vec<Vec<f64>>,
    pub binary: Vec<Vec<u8>>,
}

impl AdjacencyResult {
    /// Prints the weighted adjacency matrix with the node labels as headers.
    pub fn print(&self, decimals: bool) -> ModuleResult<()> {
        PrintMatrix::new(&self.labels)
            .with_decimals(decimals)
            .forward(&self.weighted)
    }
}

/// Directed graph + adjacency module (the graph counterpart of
/// [`crate::pad::PaddedTensor`]).
///
/// [`forward`](Module::forward) takes the path of an edge-list file, and
/// returns an [`AdjacencyResult`] holding the graph and its weighted and
/// binary adjacency matrices. Use [`from_interactions`](Self::from_interactions)
/// if the interactions are already loaded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdjacencyTensor;

impl AdjacencyTensor {
    pub fn new() -> Self {
        Self
    }

    /// Builds the graph and matrices from already-loaded interactions.
    pub fn from_interactions(&self, interactions: &[Interaction]) -> ModuleResult<AdjacencyResult> {
        let (graph, index) = BuildGraph::new().forward(interactions)?;
        let weighted = WeightedAdjacency::new().forward(&graph)?;
        let binary = BinaryAdjacency::new().forward(&graph)?;
        let labels = graph.node_weights().cloned().collect();
        Ok(AdjacencyResult { labels, index, graph, weighted, binary })
    }
}

impl Module<str> for AdjacencyTensor {
    type Output = AdjacencyResult;

    fn forward(&self, path: &str) -> ModuleResult<Self::Output> {
        let interactions = LoadInteractions::new().forward(path)?;
        self.from_interactions(&interactions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges() -> Vec<Interaction> {
        let e = |s: &str, t: &str, w: f64| Interaction {
            source: s.into(),
            target: t.into(),
            weight: w,
        };
        vec![e("A", "B", 1.0), e("B", "C", 2.0), e("A", "B", 0.5)]
    }

    #[test]
    fn build_and_matrices() {
        let (g, idx) = BuildGraph::new().forward(&edges()).unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(idx["C"].index(), 2);
        let w = WeightedAdjacency::new().forward(&g).unwrap();
        assert_eq!(w[0][1], 1.5); // repeated edge accumulated
        assert_eq!(w[1][0], 0.0); // directed
        let b = BinaryAdjacency::new().forward(&g).unwrap();
        assert_eq!(b[1][2], 1);
    }

    #[test]
    fn adjacency_tensor_from_file() {
        let p = std::env::temp_dir().join("dnapool_adj.csv");
        std::fs::write(&p, "source,target,weight\nA,B,1\nB,C,2\nA,B,0.5\n").unwrap();
        let r = AdjacencyTensor::new().forward(p.to_str().unwrap()).unwrap();
        assert_eq!(r.labels, vec!["A", "B", "C"]);
        assert_eq!(r.weighted[0][1], 1.5);
        assert_eq!(r.binary[1][2], 1);
        assert_eq!(r.graph.node_count(), 3);
        r.print(true).unwrap();
    }
}
