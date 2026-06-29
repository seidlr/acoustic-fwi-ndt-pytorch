//! Native 2D acoustic forward solver + adjoint-state gradient.
//!
//! Bit-faithful port of `fwi/forward.py` (4th/2nd-order zero-padded stencil, Stoermer/
//! Verlet leapfrog, additive dt^2 source injection, trace = phi AFTER the update) and
//! `fwi/adjoint.py` (re-run forward storing the bare Laplacian, adjoint solve with the
//! time-reversed receiver residual as sources, correlate). The kernel returned by
//! `gradient` equals torch autograd's dJ/d(alpha2) to machine precision.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// 4th-order central 2nd-derivative stencil: (offset, coefficient), denominator 12.
const C4: [(i64, f64); 5] = [(-2, -1.0), (-1, 16.0), (0, -30.0), (1, 16.0), (2, -1.0)];
// 2nd-order: (offset, coefficient), denominator 1.
const C2: [(i64, f64); 3] = [(-1, 1.0), (0, -2.0), (1, 1.0)];

/// Bare Laplacian d2phi/dx2 + d2phi/dy2 into `lap` (zero outside the grid = Dirichlet).
#[inline]
fn laplacian(phi: &[f64], lap: &mut [f64], ni: usize, nj: usize, dx2: f64, dy2: f64, order: usize) {
    let (cx, denom): (&[(i64, f64)], f64) = if order == 4 { (&C4, 12.0) } else { (&C2, 1.0) };
    let invx = 1.0 / (denom * dx2);
    let invy = 1.0 / (denom * dy2);
    for i in 0..ni {
        let row = i * nj;
        for j in 0..nj {
            let mut sx = 0.0;
            let mut sy = 0.0;
            for &(off, c) in cx {
                let jj = j as i64 + off;
                if jj >= 0 && (jj as usize) < nj {
                    sx += c * phi[row + jj as usize];
                }
                let ii = i as i64 + off;
                if ii >= 0 && (ii as usize) < ni {
                    sy += c * phi[(ii as usize) * nj + j];
                }
            }
            lap[row + j] = sx * invx + sy * invy;
        }
    }
}

/// One leapfrog step: phi_new = 2 phi - phi_old + alpha2 * lap * dt2, then inject sources.
/// Rotates the three buffers so that after the call phi=new, phi_old=prev, phi_new reusable.
#[allow(clippy::too_many_arguments)]
#[inline]
fn step(
    alpha2: &[f64],
    phi: &mut Vec<f64>,
    phi_old: &mut Vec<f64>,
    phi_new: &mut Vec<f64>,
    lap: &mut [f64],
    ni: usize,
    nj: usize,
    dx2: f64,
    dy2: f64,
    dt2: f64,
    order: usize,
    src_i: &[i64],
    src_j: &[i64],
    src_val: &[f64], // one value per source for this timestep
) {
    laplacian(phi, lap, ni, nj, dx2, dy2, order);
    for idx in 0..ni * nj {
        phi_new[idx] = 2.0 * phi[idx] - phi_old[idx] + alpha2[idx] * lap[idx] * dt2;
    }
    for s in 0..src_i.len() {
        let cell = (src_i[s] as usize) * nj + (src_j[s] as usize);
        phi_new[cell] += src_val[s] * dt2; // accumulate handles coincident sources
    }
    // rotate: phi_old <- phi (prev), phi <- phi_new; old phi_old buffer becomes scratch
    std::mem::swap(phi, phi_old); // phi_old = prev phi, phi = prev phi_old
    std::mem::swap(phi, phi_new); // phi = phi_new, phi_new = prev phi_old (scratch)
}

/// Forward solve. Returns traces (n_rec*nt, row-major) and, if `store_nabla`, the bare
/// Laplacian stack (nt*ni*nj).
#[allow(clippy::too_many_arguments)]
fn run_forward(
    alpha2: &[f64],
    ni: usize,
    nj: usize,
    src_sig: &[f64], // (n_src, nt) row-major
    n_src: usize,
    nt: usize,
    src_i: &[i64],
    src_j: &[i64],
    rec_i: &[i64],
    rec_j: &[i64],
    dx2: f64,
    dy2: f64,
    dt2: f64,
    order: usize,
    store_nabla: bool,
) -> (Vec<f64>, Vec<f64>) {
    let n = ni * nj;
    let n_rec = rec_i.len();
    let mut phi = vec![0.0; n];
    let mut phi_old = vec![0.0; n];
    let mut phi_new = vec![0.0; n];
    let mut lap = vec![0.0; n];
    let mut traces = vec![0.0; n_rec * nt];
    let mut nabla = if store_nabla { vec![0.0; nt * n] } else { Vec::new() };
    let mut src_val = vec![0.0; n_src];
    for t in 0..nt {
        for s in 0..n_src {
            src_val[s] = src_sig[s * nt + t];
        }
        step(
            alpha2, &mut phi, &mut phi_old, &mut phi_new, &mut lap, ni, nj, dx2, dy2, dt2,
            order, src_i, src_j, &src_val,
        );
        for r in 0..n_rec {
            traces[r * nt + t] = phi[(rec_i[r] as usize) * nj + (rec_j[r] as usize)];
        }
        if store_nabla {
            nabla[t * n..(t + 1) * n].copy_from_slice(&lap);
        }
    }
    (traces, nabla)
}

/// Forward simulation exposed to Python. `src_sig` is (n_src, nt); returns (n_rec, nt).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn forward<'py>(
    py: Python<'py>,
    alpha2: PyReadonlyArray2<'py, f64>,
    src_sig: PyReadonlyArray2<'py, f64>,
    src_i: PyReadonlyArray1<'py, i64>,
    src_j: PyReadonlyArray1<'py, i64>,
    rec_i: PyReadonlyArray1<'py, i64>,
    rec_j: PyReadonlyArray1<'py, i64>,
    dx_m: f64,
    dy_m: f64,
    dt: f64,
    nt: usize,
    order: usize,
) -> Bound<'py, PyArray2<f64>> {
    let a = alpha2.as_array();
    let (ni, nj) = (a.shape()[0], a.shape()[1]);
    let a_slice = a.as_slice().unwrap();
    let ss = src_sig.as_array();
    let n_src = ss.shape()[0];
    let ss_slice = ss.as_slice().unwrap();
    let si = src_i.as_slice().unwrap();
    let sj = src_j.as_slice().unwrap();
    let ri = rec_i.as_slice().unwrap();
    let rj = rec_j.as_slice().unwrap();
    let n_rec = ri.len();
    let (traces, _) = run_forward(
        a_slice, ni, nj, ss_slice, n_src, nt, si, sj, ri, rj, dx_m * dx_m, dy_m * dy_m,
        dt * dt, order, false,
    );
    Array2::from_shape_vec((n_rec, nt), traces)
        .unwrap()
        .into_pyarray_bound(py)
}

/// Adjoint-state gradient dJ/d(alpha2). `grad_traces` (n_rec, nt) is the upstream
/// dL/d(traces); the adjoint source is its time reversal. Returns (ni, nj).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gradient<'py>(
    py: Python<'py>,
    alpha2: PyReadonlyArray2<'py, f64>,
    src_sig: PyReadonlyArray2<'py, f64>,
    src_i: PyReadonlyArray1<'py, i64>,
    src_j: PyReadonlyArray1<'py, i64>,
    rec_i: PyReadonlyArray1<'py, i64>,
    rec_j: PyReadonlyArray1<'py, i64>,
    grad_traces: PyReadonlyArray2<'py, f64>,
    dx_m: f64,
    dy_m: f64,
    dt: f64,
    nt: usize,
    order: usize,
    cutoff: usize,
) -> Bound<'py, PyArray2<f64>> {
    let a = alpha2.as_array();
    let (ni, nj) = (a.shape()[0], a.shape()[1]);
    let n = ni * nj;
    let a_slice = a.as_slice().unwrap();
    let ss = src_sig.as_array();
    let n_src = ss.shape()[0];
    let ss_slice = ss.as_slice().unwrap();
    let si = src_i.as_slice().unwrap();
    let sj = src_j.as_slice().unwrap();
    let ri = rec_i.as_slice().unwrap();
    let rj = rec_j.as_slice().unwrap();
    let n_rec = ri.len();
    let dx2 = dx_m * dx_m;
    let dy2 = dy_m * dy_m;
    let dt2 = dt * dt;

    // 1) forward storing the bare Laplacian per step
    let (_traces, nabla) = run_forward(
        a_slice, ni, nj, ss_slice, n_src, nt, si, sj, ri, rj, dx2, dy2, dt2, order, true,
    );

    // 2) adjoint source = time-reversed upstream gradient, injected at the receivers
    let gt_view = grad_traces.as_array();
    let gt = gt_view.as_slice().unwrap(); // (n_rec, nt)
    let mut adj = vec![0.0; n_rec * nt];
    for r in 0..n_rec {
        for t in 0..nt {
            adj[r * nt + t] = gt[r * nt + (nt - 1 - t)];
        }
    }

    // 3) adjoint solve (same leapfrog, sources at receivers), correlate on the fly:
    //    kernel += lambda(t) * nabla2u(nt-1-t) for t in 0..(nt-cutoff). Scale +1 makes
    //    this equal dJ/d(alpha2) (grad_traces already carries the misfit's dt).
    let mut phi = vec![0.0; n];
    let mut phi_old = vec![0.0; n];
    let mut phi_new = vec![0.0; n];
    let mut lap = vec![0.0; n];
    let mut kernel = vec![0.0; n];
    let mut src_val = vec![0.0; n_rec];
    let t_max = nt.saturating_sub(cutoff);
    for t in 0..nt {
        for r in 0..n_rec {
            src_val[r] = adj[r * nt + t];
        }
        step(
            a_slice, &mut phi, &mut phi_old, &mut phi_new, &mut lap, ni, nj, dx2, dy2,
            dt2, order, ri, rj, &src_val,
        );
        if t < t_max {
            let base = (nt - 1 - t) * n;
            for idx in 0..n {
                kernel[idx] += phi[idx] * nabla[base + idx];
            }
        }
    }

    Array2::from_shape_vec((ni, nj), kernel)
        .unwrap()
        .into_pyarray_bound(py)
}

#[pymodule]
fn fwi_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward, m)?)?;
    m.add_function(wrap_pyfunction!(gradient, m)?)?;
    Ok(())
}
