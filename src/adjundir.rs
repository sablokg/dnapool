use crate::interaction::{Directed, Interaction, LoadInteractions, PrintMatrix, Undirected};
use crate::module::{Module, ModuleResult};
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// A shared, ordered node list: `labels[i]` is node `i`, and `index` maps a
/// name back to `i`. Both graph builders use it so the directed and undirected
/// matrices are directly comparable row-for-row.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Nodes {
    pub labels: Vec<String>,
    pub index: HashMap<String, usize>,
}

/// Node-index module. `forward` collects every distinct protein into
/// [`Nodes`], in order of first appearance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BuildNodeIndex;

impl BuildNodeIndex {
    pub fn new() -> Self {
        Self
    }
}

impl Module<[Interaction]> for BuildNodeIndex {
    type Output = Nodes;

    fn forward(&self, interactions: &[Interaction]) -> ModuleResult<Self::Output> {
        let mut nodes = Nodes::default();
        for edge in interactions {
            for name in [&edge.source, &edge.target] {
                if !nodes.index.contains_key(name) {
                    nodes.index.insert(name.clone(), nodes.labels.len());
                    nodes.labels.push(name.clone());
                }
            }
        }
        Ok(nodes)
    }
}

/// Adds every node in `nodes` order to a graph, so `NodeIndex(i)` == `labels[i]`.
fn resolve(nodes: &Nodes, node_ids: &[NodeIndex], name: &str) -> ModuleResult<NodeIndex> {
    nodes
        .index
        .get(name)
        .map(|&i| node_ids[i])
        .ok_or_else(|| format!("node '{name}' is missing from the node index").into())
}

/// Directed graph-builder module: source -> target only, never the reverse.
/// Repeated edges between the same ordered pair accumulate their weights.
/// Nodes are added in [`Nodes`] order so petgraph's `NodeIndex(i)` always
/// equals `labels[i]`. Errors if an interaction names an unknown node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildDirected {
    pub nodes: Nodes,
}

impl BuildDirected {
    pub fn new(nodes: Nodes) -> Self {
        Self { nodes }
    }
}

impl Module<[Interaction]> for BuildDirected {
    type Output = Directed;

    fn forward(&self, interactions: &[Interaction]) -> ModuleResult<Self::Output> {
        let mut graph: Directed = DiGraph::new();
        let ids: Vec<NodeIndex> = self.nodes.labels.iter().map(|n| graph.add_node(n.clone())).collect();
        for edge in interactions {
            let src = resolve(&self.nodes, &ids, &edge.source)?;
            let dst = resolve(&self.nodes, &ids, &edge.target)?;
            if let Some(existing) = graph.find_edge(src, dst) {
                *graph.edge_weight_mut(existing).unwrap() += edge.weight;
            } else {
                graph.add_edge(src, dst, edge.weight);
            }
        }
        Ok(graph)
    }
}

/// Undirected graph-builder module: source <-> target, so a single edge covers
/// both directions. Repeated edges between the same unordered pair accumulate.
/// Nodes are added in [`Nodes`] order, same as [`BuildDirected`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildUndirected {
    pub nodes: Nodes,
}

impl BuildUndirected {
    pub fn new(nodes: Nodes) -> Self {
        Self { nodes }
    }
}

impl Module<[Interaction]> for BuildUndirected {
    type Output = Undirected;

    fn forward(&self, interactions: &[Interaction]) -> ModuleResult<Self::Output> {
        let mut graph: Undirected = UnGraph::new_undirected();
        let ids: Vec<NodeIndex> = self.nodes.labels.iter().map(|n| graph.add_node(n.clone())).collect();
        for edge in interactions {
            let a = resolve(&self.nodes, &ids, &edge.source)?;
            let b = resolve(&self.nodes, &ids, &edge.target)?;
            if let Some(existing) = graph.find_edge(a, b) {
                *graph.edge_weight_mut(existing).unwrap() += edge.weight;
            } else {
                graph.add_edge(a, b, edge.weight);
            }
        }
        Ok(graph)
    }
}

/// Directed adjacency module: `matrix[i][j]` = weight of edge i -> j only.
/// Generally NOT symmetric.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DirectedAdjacency;

impl DirectedAdjacency {
    pub fn new() -> Self {
        Self
    }
}

impl Module<Directed> for DirectedAdjacency {
    type Output = Vec<Vec<f64>>;

    fn forward(&self, graph: &Directed) -> ModuleResult<Self::Output> {
        let n = graph.node_count();
        let mut matrix = vec![vec![0.0_f64; n]; n];
        for e in graph.edge_references() {
            matrix[e.source().index()][e.target().index()] = *e.weight();
        }
        Ok(matrix)
    }
}

/// Undirected adjacency module: `matrix[i][j] == matrix[j][i]` = weight of the
/// edge between i and j. Always symmetric.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UndirectedAdjacency;

impl UndirectedAdjacency {
    pub fn new() -> Self {
        Self
    }
}

impl Module<Undirected> for UndirectedAdjacency {
    type Output = Vec<Vec<f64>>;

    fn forward(&self, graph: &Undirected) -> ModuleResult<Self::Output> {
        let n = graph.node_count();
        let mut matrix = vec![vec![0.0_f64; n]; n];
        for e in graph.edge_references() {
            let (i, j) = (e.source().index(), e.target().index());
            matrix[i][j] = *e.weight();
            matrix[j][i] = *e.weight();
        }
        Ok(matrix)
    }
}

/// Everything [`AdjUndirTensor`] produces in one go: a shared node ordering
/// (`nodes.labels[i]` is row/column `i` of both matrices), the directed and
/// undirected graphs, and their adjacency matrices.
#[derive(Debug, Clone)]
pub struct AdjUndirResult {
    pub nodes: Nodes,
    pub directed: Directed,
    pub undirected: Undirected,
    pub directed_matrix: Vec<Vec<f64>>,
    pub undirected_matrix: Vec<Vec<f64>>,
}

impl AdjUndirResult {
    /// Prints both matrices with the node labels as headers.
    pub fn print(&self) -> ModuleResult<()> {
        let printer = PrintMatrix::new(&self.nodes.labels);
        println!("Directed:");
        printer.forward(&self.directed_matrix)?;
        println!("Undirected:");
        printer.forward(&self.undirected_matrix)
    }
}

/// Directed + undirected graph and adjacency module (the graph counterpart of
/// [`crate::pad::PaddedTensor`]).
///
/// [`forward`](Module::forward) takes the path of an edge-list file and returns
/// an [`AdjUndirResult`] with both graphs and both matrices on a shared node
/// ordering. Use [`from_interactions`](Self::from_interactions) if the
/// interactions are already loaded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdjUndirTensor;

impl AdjUndirTensor {
    pub fn new() -> Self {
        Self
    }

    /// Builds both graphs and matrices from already-loaded interactions.
    pub fn from_interactions(&self, interactions: &[Interaction]) -> ModuleResult<AdjUndirResult> {
        let nodes = BuildNodeIndex::new().forward(interactions)?;
        let directed = BuildDirected::new(nodes.clone()).forward(interactions)?;
        let undirected = BuildUndirected::new(nodes.clone()).forward(interactions)?;
        let directed_matrix = DirectedAdjacency::new().forward(&directed)?;
        let undirected_matrix = UndirectedAdjacency::new().forward(&undirected)?;
        Ok(AdjUndirResult { nodes, directed, undirected, directed_matrix, undirected_matrix })
    }
}

impl Module<str> for AdjUndirTensor {
    type Output = AdjUndirResult;

    fn forward(&self, path: &str) -> ModuleResult<Self::Output> {
        let interactions = LoadInteractions::new().forward(path)?;
        self.from_interactions(&interactions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges() -> Vec<Interaction> {
        let e = |a: &str, b: &str, w: f64| Interaction { source: a.into(), target: b.into(), weight: w };
        vec![e("A", "B", 1.0), e("B", "A", 2.0), e("B", "C", 3.0)]
    }

    #[test]
    fn directed_vs_undirected() {
        let ints = edges();
        let nodes = BuildNodeIndex::new().forward(&ints).unwrap();
        assert_eq!(nodes.labels, vec!["A", "B", "C"]);

        let d = BuildDirected::new(nodes.clone()).forward(&ints).unwrap();
        let dm = DirectedAdjacency::new().forward(&d).unwrap();
        assert_eq!((dm[0][1], dm[1][0]), (1.0, 2.0)); // not symmetric

        let u = BuildUndirected::new(nodes).forward(&ints).unwrap();
        let um = UndirectedAdjacency::new().forward(&u).unwrap();
        assert_eq!((um[0][1], um[1][0]), (3.0, 3.0)); // A-B and B-A merged
    }

    #[test]
    fn missing_node_is_an_error_not_a_panic() {
        let nodes = Nodes { labels: vec!["A".into()], index: [("A".to_string(), 0)].into() };
        assert!(BuildDirected::new(nodes.clone()).forward(&edges()).is_err());
        assert!(BuildUndirected::new(nodes).forward(&edges()).is_err());
    }

    #[test]
    fn adjundir_tensor_from_file() {
        let p = std::env::temp_dir().join("dnapool_adjundir.csv");
        std::fs::write(&p, "a,b,w\nA,B,1\nB,A,2\nB,C,3\n").unwrap();
        let r = AdjUndirTensor::new().forward(p.to_str().unwrap()).unwrap();
        assert_eq!(r.nodes.labels, vec!["A", "B", "C"]);
        assert_eq!((r.directed_matrix[0][1], r.directed_matrix[1][0]), (1.0, 2.0));
        assert_eq!((r.undirected_matrix[0][1], r.undirected_matrix[1][0]), (3.0, 3.0));
        r.print().unwrap();
    }
}
