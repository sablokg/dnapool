use crate::module::{Module, ModuleResult};

/// One-hot tensor encoder module.
///
/// One-hot encodes each base of every sequence (A/T/G/C/N, case-sensitive;
/// any other character is skipped) and flattens each sequence into a single
/// `Vec<f32>` of 4 floats per base. `N` encodes to all zeros, which makes it
/// the natural padding symbol (see [`crate::pad::PaddedTensor`]).
///
/// ```
/// use dnapool::{Module, Tensor};
/// let out = Tensor::new().forward(&["AN".to_string()]).unwrap();
/// assert_eq!(out, vec![vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tensor;

impl Tensor {
    pub fn new() -> Self {
        Self
    }

    /// One-hot code for a single base, or `None` for unsupported characters.
    fn encode_base(base: char) -> Option<[f32; 4]> {
        match base {
            'A' => Some([1.0, 0.0, 0.0, 0.0]),
            'T' => Some([0.0, 1.0, 0.0, 0.0]),
            'G' => Some([0.0, 0.0, 1.0, 0.0]),
            'C' => Some([0.0, 0.0, 0.0, 1.0]),
            'N' => Some([0.0, 0.0, 0.0, 0.0]),
            _ => None,
        }
    }
}

impl Module<[String]> for Tensor {
    type Output = Vec<Vec<f32>>;

    fn forward(&self, input: &[String]) -> ModuleResult<Self::Output> {
        Ok(input
            .iter()
            .map(|seq| {
                seq.chars()
                    .filter_map(Self::encode_base)
                    .flatten()
                    .collect()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_and_skips_unknown() {
        let out = Tensor::new().forward(&["AT?".to_string()]).unwrap();
        assert_eq!(out, vec![vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]);
    }
}
