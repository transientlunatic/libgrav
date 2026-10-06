//! PyO3 extension module.
//!
//! All public functions accept and return 1-D numpy arrays of `f64` in SI
//! units (kg for mass, radians for angles, dimensionless otherwise).  The
//! Python wrapper layer in `libgrav/units.py` is responsible for accepting
//! `astropy.units.Quantity` / `pint.Quantity` inputs and converting them to
//! plain SI arrays before calling these functions.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use grav::binary;
use pyo3::prelude::*;
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

/// Convert a numpy array (kg) into a Vec of uom Mass values.
fn array_to_masses(arr: &PyReadonlyArray1<f64>) -> Vec<Mass> {
    arr.as_array()
        .iter()
        .map(|&v| Mass::new::<kilogram>(v))
        .collect()
}

// ── binary module ─────────────────────────────────────────────────────────────

/// Total mass $M = m_1 + m_2$ (kg).
#[pyfunction]
fn total_mass<'py>(
    py: Python<'py>,
    m1: PyReadonlyArray1<'py, f64>,
    m2: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let masses1 = array_to_masses(&m1);
    let masses2 = array_to_masses(&m2);
    let result: Vec<f64> = masses1
        .into_iter()
        .zip(masses2)
        .map(|(a, b)| binary::total_mass(a, b).get::<kilogram>())
        .collect();
    Ok(result.into_pyarray_bound(py))
}

/// Mass ratio $q = m_2 / m_1$.
#[pyfunction]
fn mass_ratio<'py>(
    py: Python<'py>,
    m1: PyReadonlyArray1<'py, f64>,
    m2: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let masses1 = array_to_masses(&m1);
    let masses2 = array_to_masses(&m2);
    let result: Vec<f64> = masses1
        .into_iter()
        .zip(masses2)
        .map(|(a, b)| binary::mass_ratio(a, b))
        .collect();
    Ok(result.into_pyarray_bound(py))
}

/// Symmetric mass ratio $\eta = m_1 m_2 / M^2$.
#[pyfunction]
fn symmetric_mass_ratio<'py>(
    py: Python<'py>,
    m1: PyReadonlyArray1<'py, f64>,
    m2: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let masses1 = array_to_masses(&m1);
    let masses2 = array_to_masses(&m2);
    let result: Vec<f64> = masses1
        .into_iter()
        .zip(masses2)
        .map(|(a, b)| binary::symmetric_mass_ratio(a, b))
        .collect();
    Ok(result.into_pyarray_bound(py))
}

/// Chirp mass $\mathcal{M}$ (kg).
#[pyfunction]
fn chirp_mass<'py>(
    py: Python<'py>,
    m1: PyReadonlyArray1<'py, f64>,
    m2: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let masses1 = array_to_masses(&m1);
    let masses2 = array_to_masses(&m2);
    let result: Vec<f64> = masses1
        .into_iter()
        .zip(masses2)
        .map(|(a, b)| binary::chirp_mass(a, b).get::<kilogram>())
        .collect();
    Ok(result.into_pyarray_bound(py))
}

/// Component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and mass ratio $q$.
///
/// Returns a tuple of two arrays `(m1, m2)` in kilograms.
/// Requires `q` ∈ (0, 1].
#[pyfunction]
fn masses_from_chirp_mass_q<'py>(
    py: Python<'py>,
    mc: PyReadonlyArray1<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let mc_masses = array_to_masses(&mc);
    let q_arr = q.as_array();
    let (m1s, m2s): (Vec<f64>, Vec<f64>) = mc_masses
        .into_iter()
        .zip(q_arr.iter())
        .map(|(mc_val, &q_val)| {
            let (m1, m2) = binary::masses_from_chirp_mass_q(mc_val, q_val);
            (m1.get::<kilogram>(), m2.get::<kilogram>())
        })
        .unzip();
    Ok((m1s.into_pyarray_bound(py), m2s.into_pyarray_bound(py)))
}

/// Component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and symmetric
/// mass ratio $\eta$.
///
/// Returns a tuple of two arrays `(m1, m2)` in kilograms.
/// Requires `eta` ∈ (0, 0.25].
#[pyfunction]
fn masses_from_chirp_mass_eta<'py>(
    py: Python<'py>,
    mc: PyReadonlyArray1<'py, f64>,
    eta: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let mc_masses = array_to_masses(&mc);
    let eta_arr = eta.as_array();
    let (m1s, m2s): (Vec<f64>, Vec<f64>) = mc_masses
        .into_iter()
        .zip(eta_arr.iter())
        .map(|(mc_val, &eta_val)| {
            let (m1, m2) = binary::masses_from_chirp_mass_eta(mc_val, eta_val);
            (m1.get::<kilogram>(), m2.get::<kilogram>())
        })
        .unzip();
    Ok((m1s.into_pyarray_bound(py), m2s.into_pyarray_bound(py)))
}

/// Effective inspiral spin $\chi_\mathrm{eff}$.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn chi_eff<'py>(
    py: Python<'py>,
    m1: PyReadonlyArray1<'py, f64>,
    m2: PyReadonlyArray1<'py, f64>,
    a1: PyReadonlyArray1<'py, f64>,
    a2: PyReadonlyArray1<'py, f64>,
    tilt1: PyReadonlyArray1<'py, f64>,
    tilt2: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n = m1.len()?;
    let m1 = array_to_masses(&m1);
    let m2 = array_to_masses(&m2);
    let a1 = a1.as_array();
    let a2 = a2.as_array();
    let t1 = tilt1.as_array();
    let t2 = tilt2.as_array();
    let result: Vec<f64> = (0..n)
        .map(|i| binary::chi_eff(m1[i], m2[i], a1[i], a2[i], t1[i], t2[i]))
        .collect();
    Ok(result.into_pyarray_bound(py))
}

/// Effective precession spin $\chi_p$.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn chi_p<'py>(
    py: Python<'py>,
    m1: PyReadonlyArray1<'py, f64>,
    m2: PyReadonlyArray1<'py, f64>,
    a1: PyReadonlyArray1<'py, f64>,
    a2: PyReadonlyArray1<'py, f64>,
    tilt1: PyReadonlyArray1<'py, f64>,
    tilt2: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n = m1.len()?;
    let m1 = array_to_masses(&m1);
    let m2 = array_to_masses(&m2);
    let a1 = a1.as_array();
    let a2 = a2.as_array();
    let t1 = tilt1.as_array();
    let t2 = tilt2.as_array();
    let result: Vec<f64> = (0..n)
        .map(|i| binary::chi_p(m1[i], m2[i], a1[i], a2[i], t1[i], t2[i]))
        .collect();
    Ok(result.into_pyarray_bound(py))
}

/// Cartesian spin components in the L-frame.
///
/// Returns a 6-tuple ``(S1x, S1y, S1z, S2x, S2y, S2z)`` of float64 arrays,
/// each the same length as the inputs.  All inputs and outputs are
/// dimensionless (spin magnitudes) or plain radians (angles).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn spin_components<'py>(
    py: Python<'py>,
    a1: PyReadonlyArray1<'py, f64>,
    a2: PyReadonlyArray1<'py, f64>,
    tilt1: PyReadonlyArray1<'py, f64>,
    tilt2: PyReadonlyArray1<'py, f64>,
    phi12: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let n = a1.len()?;
    let a1 = a1.as_array();
    let a2 = a2.as_array();
    let t1 = tilt1.as_array();
    let t2 = tilt2.as_array();
    let phi = phi12.as_array();
    let mut s1x = Vec::with_capacity(n);
    let mut s1y = Vec::with_capacity(n);
    let mut s1z = Vec::with_capacity(n);
    let mut s2x = Vec::with_capacity(n);
    let mut s2y = Vec::with_capacity(n);
    let mut s2z = Vec::with_capacity(n);
    for i in 0..n {
        let (x1, y1, z1, x2, y2, z2) =
            binary::spin_components(a1[i], a2[i], t1[i], t2[i], phi[i]);
        s1x.push(x1);
        s1y.push(y1);
        s1z.push(z1);
        s2x.push(x2);
        s2y.push(y2);
        s2z.push(z2);
    }
    Ok((
        s1x.into_pyarray_bound(py),
        s1y.into_pyarray_bound(py),
        s1z.into_pyarray_bound(py),
        s2x.into_pyarray_bound(py),
        s2y.into_pyarray_bound(py),
        s2z.into_pyarray_bound(py),
    ))
}

/// Newtonian orbital angular momentum magnitude (kg m² s⁻¹).
#[pyfunction]
fn orbital_angular_momentum<'py>(
    py: Python<'py>,
    m1: PyReadonlyArray1<'py, f64>,
    m2: PyReadonlyArray1<'py, f64>,
    f_ref: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let masses1 = array_to_masses(&m1);
    let masses2 = array_to_masses(&m2);
    let f_arr = f_ref.as_array();
    let result: Vec<f64> = masses1
        .into_iter()
        .zip(masses2)
        .zip(f_arr.iter())
        .map(|((m1, m2), &f)| binary::orbital_angular_momentum(m1, m2, f))
        .collect();
    Ok(result.into_pyarray_bound(py))
}

/// Transform bilby/LALInference precessing-spin parameters into the
/// Cartesian spin components and inclination LALSimulation waveform
/// generators expect.
///
/// Returns a 7-tuple ``(iota, S1x, S1y, S1z, S2x, S2y, S2z)`` of float64
/// arrays, each the same length as the inputs.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn transform_precessing_spins<'py>(
    py: Python<'py>,
    theta_jn: PyReadonlyArray1<'py, f64>,
    phi_jl: PyReadonlyArray1<'py, f64>,
    tilt1: PyReadonlyArray1<'py, f64>,
    tilt2: PyReadonlyArray1<'py, f64>,
    phi12: PyReadonlyArray1<'py, f64>,
    a1: PyReadonlyArray1<'py, f64>,
    a2: PyReadonlyArray1<'py, f64>,
    m1: PyReadonlyArray1<'py, f64>,
    m2: PyReadonlyArray1<'py, f64>,
    f_ref: PyReadonlyArray1<'py, f64>,
    phase: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let n = theta_jn.len()?;
    let theta_jn = theta_jn.as_array();
    let phi_jl = phi_jl.as_array();
    let t1 = tilt1.as_array();
    let t2 = tilt2.as_array();
    let phi12 = phi12.as_array();
    let a1 = a1.as_array();
    let a2 = a2.as_array();
    let masses1 = array_to_masses(&m1);
    let masses2 = array_to_masses(&m2);
    let f_ref = f_ref.as_array();
    let phase = phase.as_array();

    let mut iota = Vec::with_capacity(n);
    let mut s1x = Vec::with_capacity(n);
    let mut s1y = Vec::with_capacity(n);
    let mut s1z = Vec::with_capacity(n);
    let mut s2x = Vec::with_capacity(n);
    let mut s2y = Vec::with_capacity(n);
    let mut s2z = Vec::with_capacity(n);
    for i in 0..n {
        let (i0, x1, y1, z1, x2, y2, z2) = binary::transform_precessing_spins(
            theta_jn[i],
            phi_jl[i],
            t1[i],
            t2[i],
            phi12[i],
            a1[i],
            a2[i],
            masses1[i],
            masses2[i],
            f_ref[i],
            phase[i],
        );
        iota.push(i0);
        s1x.push(x1);
        s1y.push(y1);
        s1z.push(z1);
        s2x.push(x2);
        s2y.push(y2);
        s2z.push(z2);
    }
    Ok((
        iota.into_pyarray_bound(py),
        s1x.into_pyarray_bound(py),
        s1y.into_pyarray_bound(py),
        s1z.into_pyarray_bound(py),
        s2x.into_pyarray_bound(py),
        s2y.into_pyarray_bound(py),
        s2z.into_pyarray_bound(py),
    ))
}

// ── module registration ───────────────────────────────────────────────────────

#[pymodule]
fn _libgrav(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(total_mass, m)?)?;
    m.add_function(wrap_pyfunction!(mass_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(symmetric_mass_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(chirp_mass, m)?)?;
    m.add_function(wrap_pyfunction!(masses_from_chirp_mass_q, m)?)?;
    m.add_function(wrap_pyfunction!(masses_from_chirp_mass_eta, m)?)?;
    m.add_function(wrap_pyfunction!(chi_eff, m)?)?;
    m.add_function(wrap_pyfunction!(chi_p, m)?)?;
    m.add_function(wrap_pyfunction!(spin_components, m)?)?;
    m.add_function(wrap_pyfunction!(orbital_angular_momentum, m)?)?;
    m.add_function(wrap_pyfunction!(transform_precessing_spins, m)?)?;
    Ok(())
}
