pub use ndarray as nd;
#[allow(unused_imports)]
pub use nd::{prelude::{Array, Array1, Array2, Array3, ArrayView, ArrayView1, ArrayView2, ArrayView3,
                   Axis, Dim, Ix1, Ix2, Ix3,
                   Dimension, ShapeBuilder,NdFloat,AsArray}, s};
pub use anyhow::{Context as _, Result as Res, anyhow};
pub use std::convert::TryInto;

