use std::error::Error;

/// Result type returned by every [`Module::forward`].
pub type ModuleResult<T> = Result<T, Box<dyn Error>>;

/// A composable processing step, in the spirit of `nn.Module` / Burn's
/// `Module`: configuration lives in the struct, data goes through
/// [`forward`](Module::forward).
///
/// `I` is the (possibly unsized) input type, so a module can take `str`,
/// `[String]`, `[Interaction]`, a graph, etc. by reference.
pub trait Module<I: ?Sized> {
    /// What `forward` produces on success.
    type Output;

    /// Runs the module on `input`.
    fn forward(&self, input: &I) -> ModuleResult<Self::Output>;
}
