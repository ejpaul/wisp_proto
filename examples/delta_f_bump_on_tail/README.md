# WISP Beam-Plasma Simulation: Delta-f Methods and Operator Splitting

Implementation of full-f and delta-f particle methods for the bump-on-tail instability, built on the [WISP](https://github.com/ejpaul/wisp_proto) framework. Developed as part of the Spring 2026 undergraduate research project at the Columbia Plasma Fusion Research Center under Dr. Elizabeth Paul.

---

## Overview

This codebase simulates a 1D electrostatic beam-plasma system where a warm electron beam (density fraction $n_b/n_e \ll 1$, drift velocity $u_b$, thermal width $v_b$) drives an unstable Langmuir wave. The simulation tracks a single Fourier mode of the electric field,

$$E(x,t) = E_c(t)\cos(x - t) + E_s(t)\sin(x - t),$$

in normalized units ($\omega_p = k = 1$, phase velocity $v_\varphi = 1$). The code validates against the Berk-Breizman linear growth rate and the nonlinear saturation amplitude $E_\text{sat} = (3.2)^2 \gamma^2$.

---

## Repository Structure

| File | Description |
|------|-------------|
| `core.py` | Simulation engine with Numba parallelization. Supports linear/nonlinear delta-f, full-f, general marker distribution $g$, Strang and Trotter splitting. |
| `distribution_diagnostics.py` | Runs delta-f and full-f side by side, reconstructs $f_0 + \delta f$ from particle weights, produces the 4-panel comparison figure. |
| `compare_linear_nonlinear.py` | Compares linear vs nonlinear delta-f to demonstrate why the $(1 - \delta w)$ factor is necessary for capturing saturation. |
| `saturation_and_scans.py` | Verifies the saturation amplitude prediction and performs convergence scans over particle count and time step. |
| `splitting_comparison.py` | Convergence study comparing Trotter, Strang, Strang+Verlet, and Yoshida 4th-order operator splitting methods. |
| `parallel_scaling.py` | Thread scaling study measuring speedup vs `NUMBA_NUM_THREADS`. |
| `diagnostics.py` | Utility functions: analytic growth rate, velocity distribution plotting, gamma fitting. |

---

## Methods

### Delta-f Particle Method

The delta-f method decomposes the distribution as $f = f_0 + \delta f$ and represents the perturbation through particle weights $\delta w_i = \delta f / g$, where $g$ is the marker distribution from which particles are sampled. The general weight equation (Paul 2026) is

$$\frac{d(\delta w_i)}{dt} = -\left(\frac{f}{g} - \delta w_i\right) \frac{d \ln f_0}{dt}.$$

Our implementation specializes to $g = f_0$ with $f/g \approx 1$, giving the nonlinear weight equation

$$\frac{d(\delta w_i)}{dt} = (1 - \delta w_i)\,\dot{v}_i\,\frac{2(v_i - u_b)}{v_b^2}.$$

The code also supports a general marker distribution $g \neq f_0$ (e.g., uniform velocity sampling) through the `compute_f_over_g()` function and the generalized reconstruction $f(v) = f_0(v) + \langle \delta w \rangle(v) \cdot g(v)$.

### Distribution Reconstruction

The full distribution is recovered from delta-f data by velocity-space binning: particles are sorted into velocity bins, the mean weight per bin is computed, and the result is multiplied by the marker distribution $g$. For $g = f_0$, this simplifies to $f(v) = f_0(v)[1 + \langle \delta w \rangle(v)]$. The mean weight (rather than the sum) is used because the sum would scale with the number of particles per bin, reflecting sampling density rather than the physical perturbation.

### Full-f vs Delta-f: Noise and Saturation Timing

A key observation in our results is that the full-f simulation saturates much earlier ($t \approx 200$) than the delta-f simulation ($t \approx 7000$), despite identical physics parameters. This occurs because the two methods have fundamentally different effective noise floors:

- **Full-f** initializes all particle weights at $w_i = 1$, so the statistical fluctuation from finite sampling acts as a large initial perturbation. With $N = 500{,}000$ particles, this effective noise is on the order of $1/\sqrt{N} \sim 10^{-3}$, which is nine orders of magnitude above the physical seed $E_0 \sim 10^{-12}$. The instability therefore begins growing from this much larger effective amplitude, reaching saturation correspondingly earlier.

- **Delta-f** initializes all weights at $\delta w_i = 0$, producing zero initial current and no sampling noise. The instability grows from the true physical seed $E_0$, requiring many more e-foldings (and therefore much more time) to reach the saturation level.

Both simulations produce physically correct results once they saturate. The difference in saturation time reflects the different effective initial conditions, not an error in either method. This is precisely why delta-f is preferred for studying weak instabilities: it allows the clean linear growth phase to be resolved over the full dynamic range from $E_0$ to $E_\text{sat}$.

### Linear vs Nonlinear Delta-f: Weight-Driven vs Physical Redistribution

The linear delta-f reconstruction shows perturbation structure near $v_\varphi$ that arises entirely from the weights rather than from actual particle velocity redistribution. This distinction is important:

- In **linear delta-f**, particle velocities are never updated (unperturbed orbits). The weights grow according to $d(\delta w_i)/dt = \dot{v}_i \cdot 2(v_i - u_b)/v_b^2$ with no feedback mechanism. The reconstructed $f_0 + \delta f$ shows structure because the weights encode where the wave is depositing energy in velocity space, but the particles themselves have not moved. If run long enough, the weights grow without bound and the reconstruction becomes unphysical ($\delta f \gg f_0$).

- In **nonlinear delta-f**, velocities are updated and the $(1 - \delta w_i)$ factor suppresses weight growth as $\delta w_i \to 1$. The distribution changes reflect actual wave-particle trapping: particles near $v_\varphi$ are physically redistributed in velocity, forming the quasilinear plateau that is the saturation mechanism.

The `compare_linear_nonlinear.py` script demonstrates this contrast directly, showing the linear method's catastrophic breakdown at $t = 10{,}000$ (weights exceeding $f_0$ by a factor of $\sim 600$) alongside the nonlinear method's healthy saturation.

---

## Operator Splitting Methods

The coupled particle-field system is advanced using operator splitting. Four methods are implemented and compared:

| Method | Order | Cost/step | Description |
|--------|-------|-----------|-------------|
| Lie-Trotter | 1 | 1x | Full particles, then full field |
| Strang | 2 | ~1x | Half field, full particles, half field (midpoint evaluation) |
| Strang + Verlet | 2 | ~1x | Strang splitting with velocity Verlet particle integrator |
| Yoshida | 4 | 3x | Composition of 3 Strang sub-steps with coefficients $c_1 \approx 1.35$, $c_2 \approx -1.70$ |

Convergence is measured by computing the growth rate error $|\gamma_\text{fit} - \gamma_\text{ref}|$ as a function of $\Delta t$, where $\gamma_\text{ref}$ is the numerical reference at the finest time step (not the analytic value, to isolate splitting error from particle noise).

---

## Usage

### Quick test runs
```bash
python distribution_diagnostics.py --quick
python saturation_and_scans.py --quick
python compare_linear_nonlinear.py --quick
python splitting_comparison.py --quick
```

### Full production runs
```bash
python distribution_diagnostics.py
python saturation_and_scans.py
python compare_linear_nonlinear.py
python splitting_comparison.py
```

### Parallel scaling study
```bash
python parallel_scaling.py --n-particles 500000 --n-steps 3000
```

All delta-f scripts set `NUMBA_NUM_THREADS=8` before imports for optimal performance on multi-core machines.

---

## Key Results

- **Saturation amplitude:** Measured $E_\text{sat}$ matches $(3.2)^2 \gamma^2$ with a ratio of 0.96
- **Growth rate:** Fitted $\gamma$ converges to the Berk-Breizman prediction as $N$ and $\Delta t$ are refined
- **Quasilinear plateau:** Visible in the full-f velocity distribution at $v_\varphi = 1.0$
- **Parallel speedup:** ~6x at 8 threads with Numba `prange` parallelization
- **Splitting convergence:** Slopes of ~1 (Trotter), ~2 (Strang, Strang+Verlet), and ~4 (Yoshida) confirmed on log-log plots

---

## Dependencies

- Python 3.10+
- NumPy
- Matplotlib
- Numba
- tqdm

```bash
pip install numpy matplotlib numba tqdm
```

---

## References

1. H. L. Berk, B. N. Breizman, and M. Pekker, "Numerical simulation of bump-on-tail instability with source and sink," *Physics of Plasmas*, 2(8):3007-3016, 1995.
2. C. Liu et al., "Hybrid simulation of energetic particles interacting with magnetohydrodynamics using a slow manifold algorithm and GPU acceleration," *Computer Physics Communications*, 275:108313, 2022.
3. E. J. Paul, "Vlasov-Poisson bump-on-tail model," WISP documentation (unpublished), 2026.
4. H. Yoshida, "Construction of higher order symplectic integrators," *Physics Letters A*, 150(5-7):262-268, 1990.
5. S. Blanes, F. Casas, and A. Murua, "Splitting methods for differential equations," *Acta Numerica*, arXiv:2401.01722v3, 2024.
