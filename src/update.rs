use ndarray::*;
use num_traits::Float;

use crate::linalg::*;

pub fn update_wh<T>(
    k: usize,
    l: usize,
    lambda: T,
    w: &mut Array3<T>,
    h: &mut Array2<T>,
    x_bar: &mut Array2<T>,
    wtx: &mut Array2<T>,
    x: &Array2<T>,
    x_pad: &Padded2D<T>,
    cost: &mut Vec<T>,
    epsilon: T,
    update_w: bool,
    update_h: bool,
) where
    T: Float + std::ops::AddAssign + ScalarOperand + std::ops::MulAssign + 'static,
{
    if !update_w && !update_h {
        return;
    }
    let one = T::one();
    let ten = T::from(10).unwrap();
    let nine_ten = (ten - one) / ten;

    let mut w_new = w.to_owned();
    let mut h_new = h.to_owned();

    let h_pad = Padded2D::from_arr(h, l);
    let wtx_bar = w.transpose_convolution(&Padded2D::from_arr(x_bar, l));

    let one_eye = Array2::ones((k, k)) - Array2::eye(k);
    let l_sh_one_eye = Array2::smooth_x(&h.t().dot(&one_eye), l) * lambda;
    if update_w {
        for i in 0..l {
            let hlt = h_pad
                .shift_view(i, ShiftDirect::Right)
                .unwrap()
                .t()
                .to_owned();
            let n0 = x.dot(&hlt);
            let d0 = x_bar.dot(&hlt);
            let d1 = x_pad
                .shift_view(i, ShiftDirect::Right)
                .unwrap()
                .dot(&l_sh_one_eye);
            let dw1 = n0 / (d0 + d1 + epsilon);
            Zip::from(&mut w_new.slice_mut(s![.., .., i]))
                .and(&dw1)
                .apply(|a, &b| {
                    *a *= b;
                });
        }
    }
    if update_h {
        let dh = wtx.to_owned() / (wtx_bar + one_eye.dot(&wtx.smooth(l)) * lambda + epsilon);
        Zip::from(&mut h_new).and(&dh).apply(|a, &b| {
            *a *= b;
        });
    }

    let cost_last = cost[cost.len() - 1];
    let w_old_nine_ten = w.to_owned() * nine_ten;
    let h_old_nine_ten = h.to_owned() * nine_ten;
    for i in 0..11 {
        *x_bar = w_new.convolution(&Padded2D::from_arr(&h_new, l));
        let c = reconstruction_cost(x_bar, x);
        if c < cost_last || i == 10 {
            cost.push(c);
            break;
        } else {
            if update_w {
                w_new = &w_new / ten + &w_old_nine_ten;
            }
            if update_h {
                h_new = &h_new / ten + &h_old_nine_ten;
            }
        }
    }
    if update_w {
        *w = w_new;
    }
    if update_h {
        *h = h_new;
    }
}
