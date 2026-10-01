use crate::module::{Module, ModuleResult};
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
*/

/// The result of padding a set of variable-length sequences: the padded
/// sequences themselves (each padded with trailing `N` up to the length of
/// the longest sequence in the input) alongside their original labels, in
/// the same order as the input file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchResult {
    pub sequences: Vec<String>,
    pub labels: Vec<usize>,
}

/// Padding module.
///
/// [`forward`](Module::forward) takes the path of a CSV file of
/// `sequence,label` lines, pads every sequence with trailing `N` characters up
/// to the length of the longest sequence, and returns a [`PatchResult`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PaddedTensor;

impl PaddedTensor {
    /// The character used for padding (encodes to all zeros in
    /// [`crate::tensor::Tensor`]).
    pub const PAD: char = 'N';

    pub fn new() -> Self {
        Self
    }

    /// Pads already-loaded sequences to the length of the longest one.
    /// Errors if `sequences` is empty.
    pub fn pad_sequences(&self, sequences: &[String]) -> ModuleResult<Vec<String>> {
        let maxlen = sequences
            .iter()
            .map(|x| x.len())
            .max()
            .ok_or("Input file contains no sequences")?;

        Ok(sequences
            .iter()
            .map(|seq| {
                let mut padded = seq.clone();
                padded.extend(std::iter::repeat_n(Self::PAD, maxlen - seq.len()));
                padded
            })
            .collect())
    }
}

impl Module<str> for PaddedTensor {
    type Output = PatchResult;

    fn forward(&self, pathfile: &str) -> ModuleResult<Self::Output> {
        let file = File::open(pathfile)?;
        let fileread = BufReader::new(file);
        let mut sequence: Vec<String> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();
        for line in fileread.lines() {
            let line = line?;
            let linevec: Vec<&str> = line.split(',').collect();

            if linevec.len() < 2 {
                return Err(format!("Invalid line: {}", line).into());
            }
            sequence.push(linevec[0].to_string());
            labels.push(linevec[1].parse::<usize>()?);
        }

        Ok(PatchResult {
            sequences: self.pad_sequences(&sequence)?,
            labels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(name: &str, body: &str) -> String {
        let p = std::env::temp_dir().join(name);
        File::create(&p).unwrap().write_all(body.as_bytes()).unwrap();
        p.to_str().unwrap().to_string()
    }

    #[test]
    fn pads_to_longest() {
        let path = write_tmp("dnapool_pad_ok.csv", "ACGT,1\nAC,0\n");
        let r = PaddedTensor::new().forward(&path).unwrap();
        assert_eq!(r.sequences, vec!["ACGT", "ACNN"]);
        assert_eq!(r.labels, vec![1, 0]);
    }

    #[test]
    fn empty_file_errors() {
        let path = write_tmp("dnapool_pad_empty.csv", "");
        assert!(PaddedTensor::new().forward(&path).is_err());
    }

    #[test]
    fn bad_line_errors() {
        let path = write_tmp("dnapool_pad_bad.csv", "ACGT\n");
        assert!(PaddedTensor::new().forward(&path).is_err());
    }
}
