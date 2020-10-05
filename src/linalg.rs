use ndarray::*;
use num_traits::{Float, FromPrimitive};
use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ShiftDirect {
    Left,
    Right,
}

#[derive(Clone)]
pub struct Padded2D<T: Float> {
    pub data: Array2<T>,
    pub raw_size: (usize, usize),
    pub padding_size: usize,
}

impl<T: Float> Padded2D<T> {
    pub fn from_arr(a: &Array2<T>, padding_size: usize) -> Self {
        let shape = a.shape();
        let raw_size = (shape[0], shape[1]);
        let data = a.padding_zero(padding_size);
        Self {
            data,
            raw_size,
            padding_size,
        }
    }

    pub fn raw_view(&self) -> ArrayView2<T> {
        self.shift_view(0, ShiftDirect::Left).unwrap()
    }

    pub fn shift_view(&self, l: usize, shift_dir: ShiftDirect) -> Result<ArrayView2<T>, &str> {
        let padding_size = self.padding_size;
        if padding_size < l {
            return Err("Shitf size l is larger than padding size!");
        }
        let raw_size = self.raw_size;
        let start = match shift_dir {
            ShiftDirect::Left => padding_size + l,
            ShiftDirect::Right => padding_size - l,
        };
        Ok(self.data.slice(s![.., start..start + raw_size.1]))
    }
}

pub trait Padding2D {
    fn padding_zero(&self, padding_size: usize) -> Self;
    fn into_padding_zero(&mut self, padding_size: usize);
}

impl<T> Padding2D for Array2<T>
where
    T: Float,
{
    // padding 2D array
    fn padding_zero(&self, padding_size: usize) -> Self {
        let mut a = self.clone();
        a.into_padding_zero(padding_size);
        a
    }

    fn into_padding_zero(&mut self, padding_size: usize) {
        let shape = self.shape();
        let mut a: Array2<T> = Array2::zeros((shape[0], shape[1] + padding_size * 2));
        let mut a_smut = a.slice_mut(s![.., padding_size..padding_size + shape[1]]);
        Zip::from(&mut a_smut).and(&self.to_owned()).apply(|a, &b| {
            *a = b;
        });
        *self = a;
    }
}

pub trait Tensor3D<T>
where
    T: Float + std::ops::AddAssign + 'static,
{
    fn convolution(&self, rhs: &Padded2D<T>) -> Array2<T>;
    fn transpose_convolution(&self, rhs: &Padded2D<T>) -> Array2<T>;
}

impl<T> Tensor3D<T> for Array3<T>
where
    T: Float + std::ops::AddAssign + std::ops::MulAssign + 'static,
{
    fn convolution(&self, rhs: &Padded2D<T>) -> Array2<T> {
        let shape3d = self.shape();
        let n = shape3d[0];
        let l = shape3d[2];
        let t = rhs.raw_size.1;
        let mut arr = Array2::zeros((n, t));
        for i in 0..l {
            let brr = self
                .slice(s![.., .., i])
                .dot(&rhs.shift_view(i, ShiftDirect::Right).unwrap());
            Zip::from(&mut arr).and(&brr).apply(|a, &b| {
                *a += b;
            });
        }
        arr
    }

    fn transpose_convolution(&self, rhs: &Padded2D<T>) -> Array2<T> {
        let shape3d = self.shape();
        let k = shape3d[1];
        let l = shape3d[2];
        let t = rhs.raw_size.1;
        let mut arr = Array2::zeros((k, t));
        for i in 0..l {
            let brr = self
                .slice(s![.., .., i])
                .t()
                .dot(&rhs.shift_view(i, ShiftDirect::Left).unwrap());
            Zip::from(&mut arr).and(&brr).apply(|a, &b| {
                *a += b;
            });
        }
        for i in 1..l {
            Zip::from(&mut arr.slice_mut(s![.., t - i])).apply(|a| {
                *a *= T::from(l).unwrap() / T::from(i).unwrap();
            })
        }
        arr
    }
}

pub trait Smooth {
    fn smooth(&self, l: usize) -> Self;
    fn smooth_x(rhs: &Self, l: usize) -> Self;
}

impl<T> Smooth for Array2<T>
where
    T: Float + Clone,
{
    fn smooth(&self, l: usize) -> Self {
        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];
        let mut arr = Array2::zeros((rows, cols));
        let mut brr: Array1<T> = Array1::zeros(rows);
        for i in 0..l - 1 {
            brr = brr + self.slice(s![.., i]);
        }
        for i in 0..cols {
            let j = i + l - 1;
            if j < cols {
                brr = brr + self.slice(s![.., j]);
            }
            if i >= l {
                brr = brr - self.slice(s![.., i - l]);
            }
            Zip::from(&mut arr.slice_mut(s![.., i]))
                .and(&brr)
                .apply(|a, &b| {
                    *a = b;
                });
        }
        arr
    }

    fn smooth_x(rhs: &Self, l: usize) -> Self {
        let shape = rhs.shape();
        let rows = shape[0];
        let cols = shape[1];
        let mut arr = Array2::zeros((rows, cols));
        let mut brr: Array1<T> = Array1::zeros(cols);
        for i in 0..l - 1 {
            brr = brr + rhs.slice(s![i, ..]);
        }
        for i in 0..rows {
            let j = i + l - 1;
            if j < rows {
                brr = brr + rhs.slice(s![j, ..]);
            }
            if i >= l {
                brr = brr - rhs.slice(s![i - l, ..]);
            }
            Zip::from(&mut arr.slice_mut(s![i, ..]))
                .and(&brr)
                .apply(|a, &b| {
                    *a = b;
                });
        }
        arr
    }
}

pub trait RandomArray {
    type ShapeTuple;
    fn random(shape: Self::ShapeTuple, random_seed: u8) -> Self;
    fn shuffle(&self, random_seed: u8) -> Self;
}

impl<T: Float + SampleUniform> RandomArray for Array2<T> {
    type ShapeTuple = (usize, usize);

    fn random(shape: Self::ShapeTuple, random_seed: u8) -> Self {
        let mut arr = Array2::zeros(shape);
        let mut rng = StdRng::from_seed([random_seed; 32]);
        let ud = Uniform::new::<T, T>(T::zero(), T::one());
        Zip::from(&mut arr).apply(|a| {
            *a = ud.sample(&mut rng);
        });
        arr
    }

    fn shuffle(&self, random_seed: u8) -> Self {
        let shape = self.shape();
        let y_size = shape[0];
        let x_size = shape[1];
        let mut rng = StdRng::from_seed([random_seed; 32]);
        let ud = Uniform::new::<usize, usize>(0, x_size);
        let mut arr = Array2::zeros((y_size, x_size));
        for y in 0..y_size {
            let i = ud.sample(&mut rng);
            Zip::from(&mut arr.slice_mut(s![y, i..]))
                .and(&self.slice(s![y, ..x_size - i]))
                .apply(|a, &b| {
                    *a = b;
                });
            Zip::from(&mut arr.slice_mut(s![y, ..i]))
                .and(&self.slice(s![y, x_size - i..]))
                .apply(|a, &b| {
                    *a = b;
                });
        }
        arr
    }
}

impl<T: Float + SampleUniform> RandomArray for Array3<T> {
    type ShapeTuple = (usize, usize, usize);

    fn random(shape: Self::ShapeTuple, random_seed: u8) -> Self {
        let mut arr = Array3::zeros(shape);
        let mut rng = StdRng::from_seed([random_seed; 32]);
        let ud = Uniform::new::<T, T>(T::zero(), T::one());
        Zip::from(&mut arr).apply(|a| {
            *a = ud.sample(&mut rng);
        });
        arr
    }

    fn shuffle(&self, random_seed: u8) -> Self {
        let shape = self.shape();
        let n = shape[0];
        let k = shape[1];
        let l = shape[2];
        let mut rng = StdRng::from_seed([random_seed; 32]);
        let ud = Uniform::new::<usize, usize>(0, l);
        let mut arr = Array3::zeros((n, k, l));
        for ki in 0..k {
            for ni in 0..n {
                let i = ud.sample(&mut rng);
                Zip::from(&mut arr.slice_mut(s![ni, ki, i..]))
                    .and(&self.slice(s![ni, ki, ..l - i]))
                    .apply(|a, &b| {
                        *a = b;
                    });
                Zip::from(&mut arr.slice_mut(s![ni, ki, ..i]))
                    .and(&self.slice(s![ni, ki, l - i..]))
                    .apply(|a, &b| {
                        *a = b;
                    });
            }
        }
        arr
    }
}

pub fn frobenius_norm_2<T: Float>(a: &Array2<T>) -> T {
    a.fold(T::zero(), |m, &x| m + x * x)
}

pub fn norm_1st_ij<T: Float>(a: &Array2<T>) -> T {
    a.sum() - a.diag().sum()
}

pub trait Skewness<T> {
    fn skewness_core(&self) -> Array1<T>;
}

impl<T: Float + FromPrimitive> Skewness<T> for Array2<T> {
    fn skewness_core(&self) -> Array1<T> {
        let n = self.shape()[0];
        let mu = self.mean_axis(Axis(1)).unwrap().into_shape((n, 1)).unwrap();
        let sigma = self.std_axis(Axis(1), T::one()).into_shape((n, 1)).unwrap();
        ((self - &mu) / sigma).fold_axis(Axis(1), T::zero(), |&m, &x| m + x * x * x)
    }
}

pub trait OrdArray<T: Clone, D: RemoveAxis> {
    fn max_axis(&self, axis: Axis) -> Array<T, D::Smaller>;
    fn min_axis(&self, axis: Axis) -> Array<T, D::Smaller>;
}

impl<T: Float, D: RemoveAxis> OrdArray<T, D> for Array<T, D> {
    fn max_axis(&self, axis: Axis) -> Array<T, D::Smaller> {
        self.fold_axis(axis, T::neg_infinity(), |&m, &x| if m < x { x } else { m })
    }

    fn min_axis(&self, axis: Axis) -> Array<T, D::Smaller> {
        self.fold_axis(axis, T::infinity(), |&m, &x| if m > x { x } else { m })
    }
}

pub fn reconstruction_cost<T: Float>(x_bar: &Array2<T>, x: &Array2<T>) -> T {
    frobenius_norm_2(&(x_bar - x))
}

pub fn xortho_cost<T: Float + 'static>(wtx: &Array2<T>, h: &Array2<T>, l: usize) -> T {
    norm_1st_ij(&wtx.smooth(l).dot(&h.t()))
}
