use crate::module::{Module, ModuleResult};
use petgraph::graph::{DiGraph, UnGraph};
use std::fmt::Write as _;
use std::fs;

/// A single weighted interaction: `source` -> `target` (or `source <-> target`
/// when consumed by an undirected builder), with an interaction
/// confidence/weight (use 1.0 if the data is unweighted).
#[derive(Debug, Clone, PartialEq)]
pub struct Interaction {
    pub source: String,
    pub target: String,
    pub weight: f64,
}

/// Directed weighted graph used throughout the crate.
pub type Directed = DiGraph<String, f64>;
/// Undirected weighted graph used throughout the crate.
pub type Undirected = UnGraph<String, f64>;

/// Loader module. [`forward`](Module::forward) takes a file path and parses a
/// simple CSV/TSV edge list into [`Interaction`]s. Accepts commas or
/// whitespace as the delimiter, skips blank lines, and skips a header row if
/// the third column isn't numeric. Missing weight defaults to 1.0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadInteractions;

impl LoadInteractions {
    pub fn new() -> Self {
        Self
    }
}

impl Module<str> for LoadInteractions {
    type Output = Vec<Interaction>;

    fn forward(&self, path: &str) -> ModuleResult<Self::Output> {
        let content = fs::read_to_string(path).map_err(|e| format!("reading '{path}': {e}"))?;
        let mut out = Vec::new();
        for (i, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let cols: Vec<&str> = if line.contains(',') {
                line.split(',').map(str::trim).collect()
            } else {
                line.split_whitespace().collect()
            };
            if cols.len() < 2 {
                continue;
            }
            let weight: f64 = cols.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            if i == 0 && cols.get(2).is_some() && cols[2].parse::<f64>().is_err() {
                // looks like a header row (non-numeric 3rd column) - skip it
                continue;
            }
            out.push(Interaction {
                source: cols[0].to_string(),
                target: cols[1].to_string(),
                weight,
            });
        }
        Ok(out)
    }
}

/// Pretty-printer module. [`forward`](Module::forward) prints a matrix to
/// stdout with `labels` as row/column headers; `decimals` chooses `{:.2}`
/// (default) vs `{:.0}` cells. Errors if the matrix is not `labels.len()`
/// square. Use [`render`](Self::render) to get the text instead of printing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrintMatrix {
    pub labels: Vec<String>,
    pub decimals: bool,
}

impl PrintMatrix {
    pub fn new(labels: &[String]) -> Self {
        Self { labels: labels.to_vec(), decimals: true }
    }

    /// Chooses two decimals (`true`) or integers (`false`) per cell.
    pub fn with_decimals(mut self, decimals: bool) -> Self {
        self.decimals = decimals;
        self
    }

    /// Formats the labelled matrix as text.
    pub fn render(&self, matrix: &[Vec<f64>]) -> ModuleResult<String> {
        let n = self.labels.len();
        if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
            return Err(format!("matrix must be {n}x{n} to match the {n} labels").into());
        }
        let width = self.labels.iter().map(|l| l.len()).max().unwrap_or(4).max(6) + 2;
        let mut out = format!("{:width$}", "", width = width);
        for l in &self.labels {
            write!(out, "{:>width$}", l, width = width)?;
        }
        out.push('\n');
        for (label, row) in self.labels.iter().zip(matrix) {
            write!(out, "{:width$}", label, width = width)?;
            for v in row {
                if self.decimals {
                    write!(out, "{:>width$.2}", v, width = width)?;
                } else {
                    write!(out, "{:>width$.0}", v, width = width)?;
                }
            }
            out.push('\n');
        }
        Ok(out)
    }
}

impl Module<[Vec<f64>]> for PrintMatrix {
    type Output = ();

    fn forward(&self, matrix: &[Vec<f64>]) -> ModuleResult<Self::Output> {
        print!("{}", self.render(matrix)?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_skipping_header_and_defaulting_weight() {
        let p = std::env::temp_dir().join("dnapool_interactions.tsv");
        std::fs::write(&p, "a,b,w\nA,B,2\nB C\n\n").unwrap();
        let ints = LoadInteractions::new().forward(p.to_str().unwrap()).unwrap();
        assert_eq!(ints.len(), 2);
        assert_eq!(ints[0], Interaction { source: "A".into(), target: "B".into(), weight: 2.0 });
        assert_eq!(ints[1].weight, 1.0);
    }

    #[test]
    fn missing_file_errors() {
        assert!(LoadInteractions::new().forward("/no/such/file.csv").is_err());
    }

    #[test]
    fn renders_and_validates_shape() {
        let labels = vec!["A".to_string(), "B".to_string()];
        let m = vec![vec![0.0, 1.5], vec![0.0, 0.0]];
        let text = PrintMatrix::new(&labels).render(&m).unwrap();
        assert!(text.contains("1.50"));
        assert!(PrintMatrix::new(&labels).with_decimals(false).render(&m).unwrap().contains(" 2"));
        assert!(PrintMatrix::new(&labels).render(&[vec![0.0]]).is_err());
    }
}
