test_that("total_mass returns sum of component masses", {
  expect_equal(total_mass(30 * MSUN, 30 * MSUN), 60 * MSUN, tolerance = 1e-10)
  expect_equal(total_mass(10 * MSUN,  5 * MSUN), 15 * MSUN, tolerance = 1e-10)
})

test_that("mass_ratio is m2/m1", {
  expect_equal(mass_ratio(30 * MSUN, 30 * MSUN), 1.0,  tolerance = 1e-10)
  expect_equal(mass_ratio(30 * MSUN, 10 * MSUN), 1/3,  tolerance = 1e-10)
})

test_that("symmetric_mass_ratio is 0.25 for equal masses", {
  expect_equal(symmetric_mass_ratio(30 * MSUN, 30 * MSUN), 0.25, tolerance = 1e-10)
  expect_lte(symmetric_mass_ratio(30 * MSUN, 10 * MSUN), 0.25)
})

test_that("chirp_mass matches analytic formula for equal masses", {
  m <- 30 * MSUN
  mc_expected <- 2 * m * 0.25^(3/5)
  expect_equal(chirp_mass(m, m), mc_expected, tolerance = 1e-10)
})

test_that("chi_eff is zero for non-spinning binary", {
  m <- 30 * MSUN
  expect_equal(chi_eff(m, m, 0, 0, 0, 0), 0, tolerance = 1e-15)
})

test_that("chi_eff is 0.5 for equal aligned spins a=0.5", {
  m <- 30 * MSUN
  expect_equal(chi_eff(m, m, 0.5, 0.5, 0, 0), 0.5, tolerance = 1e-10)
})

test_that("chi_p is zero for non-spinning binary", {
  m <- 30 * MSUN
  expect_equal(chi_p(m, m, 0, 0, 0, 0), 0, tolerance = 1e-15)
})

test_that("masses_from_chirp_mass_q roundtrip", {
  mc  <- chirp_mass(30 * MSUN, 20 * MSUN)
  q   <- mass_ratio(30 * MSUN, 20 * MSUN)
  res <- masses_from_chirp_mass_q(mc, q)
  expect_equal(res$m1, 30 * MSUN, tolerance = 1e-8)
  expect_equal(res$m2, 20 * MSUN, tolerance = 1e-8)
})

test_that("masses_from_chirp_mass_q equal masses", {
  mc  <- chirp_mass(30 * MSUN, 30 * MSUN)
  res <- masses_from_chirp_mass_q(mc, 1.0)
  expect_equal(res$m1, 30 * MSUN, tolerance = 1e-8)
  expect_equal(res$m2, 30 * MSUN, tolerance = 1e-8)
})

test_that("masses_from_chirp_mass_eta roundtrip", {
  mc  <- chirp_mass(30 * MSUN, 20 * MSUN)
  eta <- symmetric_mass_ratio(30 * MSUN, 20 * MSUN)
  res <- masses_from_chirp_mass_eta(mc, eta)
  expect_equal(res$m1, 30 * MSUN, tolerance = 1e-8)
  expect_equal(res$m2, 20 * MSUN, tolerance = 1e-8)
})

test_that("masses_from_chirp_mass_eta equal masses", {
  mc  <- chirp_mass(30 * MSUN, 30 * MSUN)
  res <- masses_from_chirp_mass_eta(mc, 0.25)
  expect_equal(res$m1, 30 * MSUN, tolerance = 1e-8)
  expect_equal(res$m2, 30 * MSUN, tolerance = 1e-8)
})

test_that("masses_from_chirp_mass_q vectorised", {
  mc  <- chirp_mass(c(30, 15) * MSUN, c(20, 10) * MSUN)
  q   <- mass_ratio(c(30, 15) * MSUN, c(20, 10) * MSUN)
  res <- masses_from_chirp_mass_q(mc, q)
  expect_length(res$m1, 2L)
  expect_length(res$m2, 2L)
})

test_that("functions are vectorised", {
  m1 <- c(30, 20, 10) * MSUN
  m2 <- c(30, 20,  5) * MSUN
  mc <- chirp_mass(m1, m2)
  expect_length(mc, 3L)
  expect_true(all(mc > 0))
})

# ── binary environment tests ──────────────────────────────────────────────────

test_that("binary environment is accessible", {
  expect_true(exists("binary"))
  expect_true(is.environment(binary))
})

test_that("binary$chirp_mass matches top-level chirp_mass", {
  m <- 30 * MSUN
  expect_equal(binary$chirp_mass(m, m), chirp_mass(m, m), tolerance = 1e-10)
})

test_that("binary$total_mass", {
  expect_equal(binary$total_mass(30 * MSUN, 30 * MSUN), 60 * MSUN, tolerance = 1e-10)
})

test_that("binary$mass_ratio", {
  expect_equal(binary$mass_ratio(30 * MSUN, 15 * MSUN), 0.5, tolerance = 1e-10)
})

test_that("binary$symmetric_mass_ratio", {
  expect_equal(binary$symmetric_mass_ratio(30 * MSUN, 30 * MSUN), 0.25, tolerance = 1e-10)
})

test_that("binary$masses_from_chirp_mass_q roundtrip", {
  mc  <- binary$chirp_mass(30 * MSUN, 20 * MSUN)
  q   <- binary$mass_ratio(30 * MSUN, 20 * MSUN)
  res <- binary$masses_from_chirp_mass_q(mc, q)
  expect_equal(res$m1, 30 * MSUN, tolerance = 1e-8)
  expect_equal(res$m2, 20 * MSUN, tolerance = 1e-8)
})

test_that("binary$masses_from_chirp_mass_eta roundtrip", {
  mc  <- binary$chirp_mass(30 * MSUN, 20 * MSUN)
  eta <- binary$symmetric_mass_ratio(30 * MSUN, 20 * MSUN)
  res <- binary$masses_from_chirp_mass_eta(mc, eta)
  expect_equal(res$m1, 30 * MSUN, tolerance = 1e-8)
  expect_equal(res$m2, 20 * MSUN, tolerance = 1e-8)
})

test_that("binary$chi_eff aligned spins", {
  m <- 30 * MSUN
  expect_equal(binary$chi_eff(m, m, 0.5, 0.5, 0, 0), 0.5, tolerance = 1e-10)
})

test_that("binary$chi_p non-spinning", {
  m <- 30 * MSUN
  expect_equal(binary$chi_p(m, m, 0, 0, 0, 0), 0, tolerance = 1e-15)
})

test_that("binary$MSUN constant", {
  expect_equal(binary$MSUN, MSUN)
})
