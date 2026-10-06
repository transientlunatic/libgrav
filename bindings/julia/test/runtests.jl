using Test
using Grav

const MSUN = Grav.MSUN

@testset "Grav.jl" begin

    @testset "total_mass" begin
        @test total_mass(30.0 * MSUN, 30.0 * MSUN) ≈ 60.0 * MSUN  rtol=1e-10
        @test total_mass(10.0 * MSUN,  5.0 * MSUN) ≈ 15.0 * MSUN  rtol=1e-10
    end

    @testset "mass_ratio" begin
        @test mass_ratio(30.0 * MSUN, 30.0 * MSUN) ≈ 1.0   rtol=1e-10
        @test mass_ratio(30.0 * MSUN, 10.0 * MSUN) ≈ 1/3   rtol=1e-10
    end

    @testset "symmetric_mass_ratio" begin
        @test symmetric_mass_ratio(30.0 * MSUN, 30.0 * MSUN) ≈ 0.25  rtol=1e-10
        # η ≤ 1/4 for all mass ratios
        @test symmetric_mass_ratio(30.0 * MSUN, 10.0 * MSUN) ≤ 0.25
    end

    @testset "chirp_mass" begin
        # For equal masses m: Mc = 2m * (1/4)^(3/5)
        m = 30.0 * MSUN
        mc_expected = 2.0 * m * 0.25^(3/5)
        @test chirp_mass(m, m) ≈ mc_expected  rtol=1e-10
    end

    @testset "masses_from_chirp_mass_q roundtrip" begin
        m1 = 30.0 * MSUN
        m2 = 20.0 * MSUN
        mc = chirp_mass(m1, m2)
        q  = mass_ratio(m1, m2)
        (r1, r2) = masses_from_chirp_mass_q(mc, q)
        @test r1 ≈ m1  rtol=1e-10
        @test r2 ≈ m2  rtol=1e-10
    end

    @testset "masses_from_chirp_mass_q equal masses" begin
        m = 30.0 * MSUN
        mc = chirp_mass(m, m)
        (r1, r2) = masses_from_chirp_mass_q(mc, 1.0)
        @test r1 ≈ m  rtol=1e-10
        @test r2 ≈ m  rtol=1e-10
    end

    @testset "masses_from_chirp_mass_eta roundtrip" begin
        m1 = 30.0 * MSUN
        m2 = 20.0 * MSUN
        mc  = chirp_mass(m1, m2)
        eta = symmetric_mass_ratio(m1, m2)
        (r1, r2) = masses_from_chirp_mass_eta(mc, eta)
        @test r1 ≈ m1  rtol=1e-10
        @test r2 ≈ m2  rtol=1e-10
    end

    @testset "masses_from_chirp_mass_eta equal masses" begin
        m = 30.0 * MSUN
        mc = chirp_mass(m, m)
        (r1, r2) = masses_from_chirp_mass_eta(mc, 0.25)
        @test r1 ≈ m  rtol=1e-10
        @test r2 ≈ m  rtol=1e-10
    end

    @testset "chi_eff — non-spinning" begin
        m = 30.0 * MSUN
        @test chi_eff(m, m, 0.0, 0.0, 0.0, 0.0) ≈ 0.0  atol=1e-15
    end

    @testset "chi_eff — aligned spins" begin
        m = 30.0 * MSUN
        # Both spins fully aligned: chi_eff = (m1*a1 + m2*a2) / M = 0.5
        @test chi_eff(m, m, 0.5, 0.5, 0.0, 0.0) ≈ 0.5  rtol=1e-10
    end

    @testset "chi_p — non-spinning" begin
        m = 30.0 * MSUN
        @test chi_p(m, m, 0.0, 0.0, 0.0, 0.0) ≈ 0.0  atol=1e-15
    end

    @testset "broadcasting" begin
        # Julia broadcasting should work transparently over scalar ccall wrappers
        m1 = [30.0, 10.0, 5.0] .* MSUN
        m2 = [30.0,  5.0, 5.0] .* MSUN
        mc = chirp_mass.(m1, m2)
        @test length(mc) == 3
        @test all(mc .> 0.0)
    end

    @testset "spin_components — aligned" begin
        (s1x, s1y, s1z, s2x, s2y, s2z) = spin_components(0.5, 0.3, 0.0, 0.0, 0.0)
        @test s1x ≈ 0.0  atol=1e-14
        @test s1y ≈ 0.0  atol=1e-14
        @test s1z ≈ 0.5  rtol=1e-12
        @test s2z ≈ 0.3  rtol=1e-12
    end

    @testset "spin_components — in-plane" begin
        # tilt1 = pi/2 -> S1x = a1
        (s1x, _, s1z, _, _, _) = spin_components(0.8, 0.0, pi/2, 0.0, 0.0)
        @test s1x ≈ 0.8  rtol=1e-12
        @test s1z ≈ 0.0  atol=1e-14
    end

    @testset "orbital_angular_momentum — Newtonian formula" begin
        m1 = 30.0 * MSUN
        m2 = 20.0 * MSUN
        f = 20.0
        G = 6.674_30e-11
        m = m1 + m2
        mu = m1 * m2 / m
        expected = mu * (G * m)^(2/3) / (pi * f)^(1/3)
        @test orbital_angular_momentum(m1, m2, f) ≈ expected  rtol=1e-10
    end

    @testset "transform_precessing_spins — matches real LALSimulation output" begin
        # Reference values generated with the actual LALSimulation C library
        # (lalsimulation.SimInspiralTransformPrecessingNewInitialConditions),
        # not transcribed from documentation — see crates/grav/src/binary.rs
        # for the same cross-check in the Rust core.
        LAL_MSUN = 1.988_409_870_698_050_731_911_960_804_878_414_216e30
        (iota, s1x, s1y, s1z, s2x, s2y, s2z) = transform_precessing_spins(
            0.4, 0.3, 0.5, 0.3, 1.2, 0.6, 0.4,
            30.0 * LAL_MSUN, 20.0 * LAL_MSUN, 20.0, 0.0,
        )
        expected = (
            0.383072563130684, -0.28710948916277035, 0.01771231708241292,
            0.5265495371342236, -0.04953631276943223, -0.10732802301558862,
            0.38213459565024244,
        )
        got = (iota, s1x, s1y, s1z, s2x, s2y, s2z)
        for (g, e) in zip(got, expected)
            @test g ≈ e  atol=1e-9
        end
    end

    @testset "transform_precessing_spins — aligned reduces to theta_jn" begin
        m1 = 30.0 * MSUN
        m2 = 20.0 * MSUN
        (iota, s1x, s1y, s1z, s2x, s2y, s2z) = transform_precessing_spins(
            0.4, 0.3, 0.0, 0.0, 1.2, 0.6, 0.4, m1, m2, 20.0, 0.0,
        )
        @test iota ≈ 0.4  atol=1e-12
        @test s1x ≈ 0.0  atol=1e-12
        @test s1y ≈ 0.0  atol=1e-12
        @test s1z ≈ 0.6  atol=1e-12
        @test s2z ≈ 0.4  atol=1e-12
    end

    # ── Grav.Binary submodule ───────────────────────────────────────────────

    @testset "Binary submodule accessible" begin
        @test isdefined(Grav, :Binary)
        @test Grav.Binary isa Module
    end

    @testset "Binary.chirp_mass matches top-level" begin
        m = 30.0 * MSUN
        @test Grav.Binary.chirp_mass(m, m) ≈ chirp_mass(m, m)  rtol=1e-10
    end

    @testset "Binary.total_mass" begin
        @test Grav.Binary.total_mass(30.0 * MSUN, 30.0 * MSUN) ≈ 60.0 * MSUN  rtol=1e-10
    end

    @testset "Binary.mass_ratio" begin
        @test Grav.Binary.mass_ratio(30.0 * MSUN, 15.0 * MSUN) ≈ 0.5  rtol=1e-10
    end

    @testset "Binary.symmetric_mass_ratio" begin
        @test Grav.Binary.symmetric_mass_ratio(30.0 * MSUN, 30.0 * MSUN) ≈ 0.25  rtol=1e-10
    end

    @testset "Binary.masses_from_chirp_mass_q roundtrip" begin
        m1 = 30.0 * MSUN
        m2 = 20.0 * MSUN
        mc = Grav.Binary.chirp_mass(m1, m2)
        q  = Grav.Binary.mass_ratio(m1, m2)
        (r1, r2) = Grav.Binary.masses_from_chirp_mass_q(mc, q)
        @test r1 ≈ m1  rtol=1e-10
        @test r2 ≈ m2  rtol=1e-10
    end

    @testset "Binary.masses_from_chirp_mass_eta roundtrip" begin
        m1 = 30.0 * MSUN
        m2 = 20.0 * MSUN
        mc  = Grav.Binary.chirp_mass(m1, m2)
        eta = Grav.Binary.symmetric_mass_ratio(m1, m2)
        (r1, r2) = Grav.Binary.masses_from_chirp_mass_eta(mc, eta)
        @test r1 ≈ m1  rtol=1e-10
        @test r2 ≈ m2  rtol=1e-10
    end

    @testset "Binary.chi_eff" begin
        m = 30.0 * MSUN
        @test Grav.Binary.chi_eff(m, m, 0.5, 0.5, 0.0, 0.0) ≈ 0.5  rtol=1e-10
    end

    @testset "Binary.chi_p" begin
        m = 30.0 * MSUN
        @test Grav.Binary.chi_p(m, m, 0.0, 0.0, 0.0, 0.0) ≈ 0.0  atol=1e-15
    end

    @testset "Binary broadcasting" begin
        m1 = [30.0, 10.0] .* MSUN
        m2 = [30.0,  5.0] .* MSUN
        mc = Grav.Binary.chirp_mass.(m1, m2)
        @test length(mc) == 2
        @test all(mc .> 0.0)
    end

end
