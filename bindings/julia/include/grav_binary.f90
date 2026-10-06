! grav_binary.f90 — Fortran module for compact binary parameter functions.
!
! Groups all binary system parameter interfaces under a single Fortran module,
! mirroring the domain-organised namespace in Python (libgrav.binary) and
! Julia (Grav.Binary).
!
! Usage:
!
!   use grav_binary
!   real(c_double), parameter :: MSUN = grav_binary_MSUN
!   print *, chirp_mass(30*MSUN, 30*MSUN) / MSUN   ! ~26.1
!
! Build:
!   gfortran -c grav_binary.f90   (produces grav_binary.mod)
!   Then link with -L<lib_dir> -lgrav

module grav_binary
    use iso_c_binding, only: c_double
    implicit none

    !> Solar mass in kilograms.
    real(c_double), parameter :: grav_binary_MSUN = 1.988416e30_c_double

    ! ── C interface declarations ─────────────────────────────────────────────

    interface

        !> Total mass M = m1 + m2 (kg).
        function total_mass(m1, m2) bind(C, name="grav_total_mass")
            import c_double
            real(c_double), value :: m1, m2
            real(c_double)        :: total_mass
        end function

        !> Mass ratio q = m2 / m1 (dimensionless).  Requires m1 >= m2.
        function mass_ratio(m1, m2) bind(C, name="grav_mass_ratio")
            import c_double
            real(c_double), value :: m1, m2
            real(c_double)        :: mass_ratio
        end function

        !> Symmetric mass ratio eta = m1*m2 / M^2, in (0, 0.25].
        function symmetric_mass_ratio(m1, m2) &
                bind(C, name="grav_symmetric_mass_ratio")
            import c_double
            real(c_double), value :: m1, m2
            real(c_double)        :: symmetric_mass_ratio
        end function

        !> Chirp mass Mc = (m1*m2)^(3/5) / M^(1/5) (kg).
        function chirp_mass(m1, m2) bind(C, name="grav_chirp_mass")
            import c_double
            real(c_double), value :: m1, m2
            real(c_double)        :: chirp_mass
        end function

        !> Primary mass m1 (kg) from chirp mass and mass ratio q = m2/m1.
        function m1_from_chirp_mass_q(mc, q) bind(C, name="grav_m1_from_mc_q")
            import c_double
            real(c_double), value :: mc, q
            real(c_double)        :: m1_from_chirp_mass_q
        end function

        !> Secondary mass m2 (kg) from chirp mass and mass ratio q = m2/m1.
        function m2_from_chirp_mass_q(mc, q) bind(C, name="grav_m2_from_mc_q")
            import c_double
            real(c_double), value :: mc, q
            real(c_double)        :: m2_from_chirp_mass_q
        end function

        !> Primary mass m1 (kg) from chirp mass and symmetric mass ratio eta.
        function m1_from_chirp_mass_eta(mc, eta) &
                bind(C, name="grav_m1_from_mc_eta")
            import c_double
            real(c_double), value :: mc, eta
            real(c_double)        :: m1_from_chirp_mass_eta
        end function

        !> Secondary mass m2 (kg) from chirp mass and symmetric mass ratio eta.
        function m2_from_chirp_mass_eta(mc, eta) &
                bind(C, name="grav_m2_from_mc_eta")
            import c_double
            real(c_double), value :: mc, eta
            real(c_double)        :: m2_from_chirp_mass_eta
        end function

        !> Effective inspiral spin chi_eff in [-1, 1].
        function chi_eff(m1, m2, a1, a2, tilt1, tilt2) &
                bind(C, name="grav_chi_eff")
            import c_double
            real(c_double), value :: m1, m2, a1, a2, tilt1, tilt2
            real(c_double)        :: chi_eff
        end function

        !> Effective precession spin chi_p in [0, 1].  Requires m1 >= m2.
        function chi_p(m1, m2, a1, a2, tilt1, tilt2) &
                bind(C, name="grav_chi_p")
            import c_double
            real(c_double), value :: m1, m2, a1, a2, tilt1, tilt2
            real(c_double)        :: chi_p
        end function

    end interface

end module grav_binary
