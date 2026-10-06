use clap::{ArgAction, Parser, Subcommand};
use grav::binary::{
    chi_eff, chi_p, chirp_mass, mass_ratio, masses_from_chirp_mass_eta, masses_from_chirp_mass_q,
    symmetric_mass_ratio, total_mass,
};
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

const MSUN: f64 = 1.988_416e30;

/// Gravitational-wave binary parameter calculator.
///
/// Masses are in solar masses (M☉) unless --si is given, in which case they
/// are in kilograms.  Spin magnitudes are dimensionless (0–1).  Tilt angles
/// are in radians.
///
/// Output is a single number on stdout, suitable for shell pipelines.
/// Use --verbose to include a label and units.
#[derive(Parser)]
#[command(name = "grav", version, author)]
struct Cli {
    /// Accept and emit mass values in kilograms instead of solar masses
    #[arg(long, global = true, action = ArgAction::SetTrue)]
    si: bool,

    /// Print a label and units alongside the result
    #[arg(long, global = true, short = 'v', action = ArgAction::SetTrue)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Total mass M = m1 + m2
    #[command(alias = "m")]
    TotalMass {
        /// Primary mass (heavier component, m1 ≥ m2)
        m1: f64,
        /// Secondary mass
        m2: f64,
    },

    /// Mass ratio q = m2/m1  (0 < q ≤ 1, requires m1 ≥ m2)
    #[command(alias = "q")]
    MassRatio {
        /// Primary mass (heavier component, m1 ≥ m2)
        m1: f64,
        /// Secondary mass
        m2: f64,
    },

    /// Symmetric mass ratio η = m1·m2/(m1+m2)²  (0 < η ≤ 0.25)
    #[command(alias = "eta")]
    SymMassRatio {
        /// Primary mass (heavier component, m1 ≥ m2)
        m1: f64,
        /// Secondary mass
        m2: f64,
    },

    /// Chirp mass ℳ = (m1·m2)^(3/5) / M^(1/5)
    #[command(alias = "mc")]
    ChirpMass {
        /// Primary mass (heavier component)
        m1: f64,
        /// Secondary mass
        m2: f64,
    },

    /// Effective aligned spin χ_eff = (m1·a1·cos θ1 + m2·a2·cos θ2) / M
    #[command(alias = "chieff")]
    ChiEff {
        /// Primary mass
        m1: f64,
        /// Secondary mass
        m2: f64,
        /// Primary spin magnitude a1 ∈ [0, 1]
        a1: f64,
        /// Secondary spin magnitude a2 ∈ [0, 1]
        a2: f64,
        /// Primary tilt angle θ1 (radians, 0 = aligned)
        tilt1: f64,
        /// Secondary tilt angle θ2 (radians, 0 = aligned)
        tilt2: f64,
    },

    /// Precession spin parameter χ_p  (requires m1 ≥ m2)
    #[command(alias = "chip")]
    ChiP {
        /// Primary mass (heavier component, m1 ≥ m2)
        m1: f64,
        /// Secondary mass
        m2: f64,
        /// Primary spin magnitude a1 ∈ [0, 1]
        a1: f64,
        /// Secondary spin magnitude a2 ∈ [0, 1]
        a2: f64,
        /// Primary tilt angle θ1 (radians)
        tilt1: f64,
        /// Secondary tilt angle θ2 (radians)
        tilt2: f64,
    },

    /// Component masses (m1, m2) from chirp mass ℳ and mass ratio q
    ///
    /// Prints m1 on the first line and m2 on the second.
    /// Requires q ∈ (0, 1].
    #[command(name = "masses-from-mc-q", alias = "inv-q")]
    MassesFromMcQ {
        /// Chirp mass ℳ
        mc: f64,
        /// Mass ratio q = m2/m1 ∈ (0, 1]
        q: f64,
    },

    /// Component masses (m1, m2) from chirp mass ℳ and symmetric mass ratio η
    ///
    /// Prints m1 on the first line and m2 on the second.
    /// Requires η ∈ (0, 0.25].
    #[command(name = "masses-from-mc-eta", alias = "inv-eta")]
    MassesFromMcEta {
        /// Chirp mass ℳ
        mc: f64,
        /// Symmetric mass ratio η ∈ (0, 0.25]
        eta: f64,
    },
}

fn to_kg(val: f64, si: bool) -> f64 {
    if si {
        val
    } else {
        val * MSUN
    }
}

fn from_kg(val: f64, si: bool) -> f64 {
    if si {
        val
    } else {
        val / MSUN
    }
}

fn mass(val: f64, si: bool) -> Mass {
    Mass::new::<kilogram>(to_kg(val, si))
}

/// Print a single scalar result.
fn print_scalar(value: f64, label: &str, unit: &str, verbose: bool) {
    if verbose {
        if unit.is_empty() {
            println!("{label} = {value}");
        } else {
            println!("{label} = {value} {unit}");
        }
    } else {
        println!("{value}");
    }
}

/// Print a pair of masses (m1, m2) — one per line.
fn print_pair(m1: f64, m2: f64, unit: &str, verbose: bool) {
    if verbose {
        if unit.is_empty() {
            println!("m1 = {m1}");
            println!("m2 = {m2}");
        } else {
            println!("m1 = {m1} {unit}");
            println!("m2 = {m2} {unit}");
        }
    } else {
        println!("{m1}");
        println!("{m2}");
    }
}

fn main() {
    let cli = Cli::parse();
    let si = cli.si;
    let verbose = cli.verbose;
    let mass_unit = if si { "kg" } else { "Msun" };

    match cli.command {
        Command::TotalMass { m1, m2 } => {
            let r = total_mass(mass(m1, si), mass(m2, si));
            print_scalar(
                from_kg(r.get::<kilogram>(), si),
                "total_mass",
                mass_unit,
                verbose,
            );
        }
        Command::MassRatio { m1, m2 } => {
            let r = mass_ratio(mass(m1, si), mass(m2, si));
            print_scalar(r, "mass_ratio", "", verbose);
        }
        Command::SymMassRatio { m1, m2 } => {
            let r = symmetric_mass_ratio(mass(m1, si), mass(m2, si));
            print_scalar(r, "eta", "", verbose);
        }
        Command::ChirpMass { m1, m2 } => {
            let r = chirp_mass(mass(m1, si), mass(m2, si));
            print_scalar(
                from_kg(r.get::<kilogram>(), si),
                "chirp_mass",
                mass_unit,
                verbose,
            );
        }
        Command::ChiEff {
            m1,
            m2,
            a1,
            a2,
            tilt1,
            tilt2,
        } => {
            let r = chi_eff(mass(m1, si), mass(m2, si), a1, a2, tilt1, tilt2);
            print_scalar(r, "chi_eff", "", verbose);
        }
        Command::ChiP {
            m1,
            m2,
            a1,
            a2,
            tilt1,
            tilt2,
        } => {
            let r = chi_p(mass(m1, si), mass(m2, si), a1, a2, tilt1, tilt2);
            print_scalar(r, "chi_p", "", verbose);
        }
        Command::MassesFromMcQ { mc, q } => {
            if q <= 0.0 || q > 1.0 {
                eprintln!("error: q must be in (0, 1], got {q}");
                std::process::exit(1);
            }
            let (r1, r2) = masses_from_chirp_mass_q(mass(mc, si), q);
            print_pair(
                from_kg(r1.get::<kilogram>(), si),
                from_kg(r2.get::<kilogram>(), si),
                mass_unit,
                verbose,
            );
        }
        Command::MassesFromMcEta { mc, eta } => {
            if eta <= 0.0 || eta > 0.25 {
                eprintln!("error: eta must be in (0, 0.25], got {eta}");
                std::process::exit(1);
            }
            let (r1, r2) = masses_from_chirp_mass_eta(mass(mc, si), eta);
            print_pair(
                from_kg(r1.get::<kilogram>(), si),
                from_kg(r2.get::<kilogram>(), si),
                mass_unit,
                verbose,
            );
        }
    }
}
