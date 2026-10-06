/// Integration tests for the `binary` module — exercises the public API only.
use grav::binary::{chi_eff, chi_p, chirp_mass, mass_ratio, symmetric_mass_ratio, total_mass};
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

const MSUN_KG: f64 = 1.988_416e30;

fn sol(m: f64) -> Mass {
    Mass::new::<kilogram>(m * MSUN_KG)
}

#[test]
fn gw150914_like_parameters() {
    // Rough parameters for GW150914 (Mchirp ~ 28.3 Msun, q ~ 0.82)
    let m1 = sol(35.6);
    let m2 = sol(30.6);

    let mc = chirp_mass(m1, m2).get::<kilogram>() / MSUN_KG;
    assert!(
        (mc - 28.3).abs() < 0.5,
        "chirp mass {mc:.2} outside expected range"
    );

    let q = mass_ratio(m1, m2);
    assert!(
        (q - 0.86).abs() < 0.05,
        "mass ratio {q:.3} outside expected range"
    );

    let eta = symmetric_mass_ratio(m1, m2);
    assert!(eta > 0.0 && eta <= 0.25);

    let mt = total_mass(m1, m2).get::<kilogram>() / MSUN_KG;
    assert!((mt - 66.2).abs() < 0.1);
}

#[test]
fn spin_parameters_non_spinning() {
    let m1 = sol(30.0);
    let m2 = sol(20.0);

    let xeff = chi_eff(m1, m2, 0.0, 0.0, 0.0, 0.0);
    assert_eq!(xeff, 0.0);

    let xp = chi_p(m1, m2, 0.0, 0.0, 0.0, 0.0);
    assert_eq!(xp, 0.0);
}
