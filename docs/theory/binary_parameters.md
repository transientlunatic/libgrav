# Binary system parameters

A compact binary system is characterised by the masses and spins of its two
components.  This page documents the parameterisations implemented in the
`grav::binary` module and the conventions used throughout the library.

## Mass parameters

Given component masses $m_1 \geq m_2 > 0$, the derived mass parameters are:

**Total mass**

$$M = m_1 + m_2$$

**Mass ratio**

$$q = \frac{m_2}{m_1} \in (0, 1]$$

**Symmetric mass ratio**

$$\eta = \frac{m_1 m_2}{M^2} \in \left(0, \tfrac{1}{4}\right]$$

with $\eta = 1/4$ for equal-mass systems.

**Chirp mass**

$$\mathcal{M} = \frac{(m_1 m_2)^{3/5}}{M^{1/5}} = M \eta^{3/5}$$

The chirp mass governs the leading-order phase evolution during inspiral:

$$\dot{f} \propto \mathcal{M}^{5/3} f^{11/3}$$

It is always true that $\mathcal{M} \leq M$.

## Spin parameters

Each black hole carries a dimensionless spin $\mathbf{a}_i$ with magnitude
$a_i = |\mathbf{a}_i| \in [0,1]$ (normalised to the Kerr maximum $GM^2/c$)
and tilt angle $\theta_i \in [0, \pi]$ measured from the orbital angular
momentum axis $\hat{L}$.

**Effective inspiral spin**

$$\chi_\mathrm{eff} = \frac{m_1 a_1 \cos\theta_1 + m_2 a_2 \cos\theta_2}{M}
\in [-1, 1]$$

$\chi_\mathrm{eff}$ is approximately conserved through inspiral at 1.5
post-Newtonian order {cite}`Racine2008` and is the dominant spin observable
from the inspiral phase of a gravitational-wave signal.

**Effective precession spin**

$$\chi_p = \max\!\left(
    a_1 \sin\theta_1,\;
    \frac{3 + 4q}{4(1 + q)}\, q\, a_2 \sin\theta_2
\right) \in [0, 1]$$

as defined in {cite}`Hannam2014`.  $\chi_p$ characterises the degree of
orbital-plane precession driven by in-plane spin components.
