% examples/matlab/example.m
%
% Demonstrates calling the libgrav C API from MATLAB, using a domain-organised
% binary struct that mirrors libgrav.binary in Python and Grav.Binary in Julia.
%
% Prerequisites:
%   1. Build the shared library from the repo root:
%        cargo build --release -p grav-capi
%   2. Run this script from the repo root, or update REPO_ROOT below.
%
% MATLAB version: R2019b or newer (calllib with pointer aliasing).

% ── locate the library and header ────────────────────────────────────────────
repo_root   = fileparts(fileparts(fileparts(mfilename('fullpath'))));
lib_dir     = fullfile(repo_root, 'target', 'release');
header_file = fullfile(repo_root, 'bindings', 'julia', 'include', 'grav.h');

if ispc
    lib_file = fullfile(lib_dir, 'grav.dll');
elseif ismac
    lib_file = fullfile(lib_dir, 'libgrav.dylib');
else
    lib_file = fullfile(lib_dir, 'libgrav.so');
end

if ~libisloaded('grav')
    loadlibrary(lib_file, header_file, 'alias', 'grav');
end

% ── domain-organised binary struct ───────────────────────────────────────────
%
% Wraps all compact binary parameter functions in a struct so they can be
% accessed as binary.chirp_mass(m1, m2), mirroring libgrav.binary in Python
% and Grav.Binary in Julia.

MSUN = 1.988416e30;

binary.MSUN = MSUN;
binary.total_mass           = @(m1,m2) calllib('grav','grav_total_mass',m1,m2);
binary.mass_ratio           = @(m1,m2) calllib('grav','grav_mass_ratio',m1,m2);
binary.symmetric_mass_ratio = @(m1,m2) calllib('grav','grav_symmetric_mass_ratio',m1,m2);
binary.chirp_mass           = @(m1,m2) calllib('grav','grav_chirp_mass',m1,m2);
binary.m1_from_chirp_mass_q = @(mc,q)  calllib('grav','grav_m1_from_mc_q',mc,q);
binary.m2_from_chirp_mass_q = @(mc,q)  calllib('grav','grav_m2_from_mc_q',mc,q);
binary.m1_from_chirp_mass_eta = @(mc,eta) calllib('grav','grav_m1_from_mc_eta',mc,eta);
binary.m2_from_chirp_mass_eta = @(mc,eta) calllib('grav','grav_m2_from_mc_eta',mc,eta);
binary.chi_eff = @(m1,m2,a1,a2,t1,t2) calllib('grav','grav_chi_eff',m1,m2,a1,a2,t1,t2);
binary.chi_p   = @(m1,m2,a1,a2,t1,t2) calllib('grav','grav_chi_p',  m1,m2,a1,a2,t1,t2);

% ── use the binary struct ─────────────────────────────────────────────────────
m1 = 30.0 * MSUN;
m2 = 30.0 * MSUN;

fprintf('=== GW150914-like binary (30+30 Msun) ===\n');
fprintf('Total mass      : %.4f Msun\n', binary.total_mass(m1, m2) / MSUN);
fprintf('Mass ratio      : %.4f\n',       binary.mass_ratio(m1, m2));
fprintf('Sym. mass ratio : %.4f\n',       binary.symmetric_mass_ratio(m1, m2));
fprintf('Chirp mass      : %.4f Msun\n',  binary.chirp_mass(m1, m2) / MSUN);

% Inverse transforms
mc  = binary.chirp_mass(m1, m2);
q   = binary.mass_ratio(m1, m2);
eta = binary.symmetric_mass_ratio(m1, m2);
fprintf('\n=== Inverse transforms ===\n');
fprintf('m1 from (Mc, q)   : %.4f Msun\n', binary.m1_from_chirp_mass_q(mc, q)   / MSUN);
fprintf('m2 from (Mc, q)   : %.4f Msun\n', binary.m2_from_chirp_mass_q(mc, q)   / MSUN);
fprintf('m1 from (Mc, eta) : %.4f Msun\n', binary.m1_from_chirp_mass_eta(mc, eta) / MSUN);
fprintf('m2 from (Mc, eta) : %.4f Msun\n', binary.m2_from_chirp_mass_eta(mc, eta) / MSUN);

a1    = 0.3;   a2    = 0.2;
tilt1 = 0.5236; tilt2 = 1.0472;   % 30 deg, 60 deg in radians

fprintf('chi_eff         : %.4f\n', binary.chi_eff(m1, m2, a1, a2, tilt1, tilt2));
fprintf('chi_p           : %.4f\n', binary.chi_p(  m1, m2, a1, a2, tilt1, tilt2));

% ── vectorised example using arrayfun ─────────────────────────────────────────
masses_sun = 10:10:50;  % [10 20 30 40 50] Msun equal-mass binaries
mc_sun = arrayfun(@(m) binary.chirp_mass(m*MSUN, m*MSUN) / MSUN, masses_sun);

fprintf('\n=== Chirp masses for equal-mass binaries ===\n');
fprintf('  m1=m2 (Msun)  Mc (Msun)\n');
fprintf('  %10.1f  %9.4f\n', [masses_sun; mc_sun]);

unloadlibrary('grav');
