# ASPiRe-3D — Governing Equations

All equations are solved on a cell-centred structured finite-volume grid masked
to a cylindrical core. Discretization is conservative; the pressure and implicit
transport systems are sparse.

## 1. Single-phase Darcy flow

```
u = -(k/mu) grad P ,        div(u) = q
div( (k/mu) grad P ) = -q
```

Face transmissibilities use the harmonic mean of cell permeabilities (correctly
choking plugged cells); the pressure-Poisson system is solved implicitly
(sparse direct/iterative).

## 2. Multi-species advection–dispersion–reaction transport

For each species `C_s` (mobility flag `m_s`, dispersion `D_s`, reaction `r_s`):

```
phi dC_s/dt + m_s [ div(u C_s) - div(phi D_s grad C_s) ] = r_s
D_s = D_m,s + alpha_L |v| ,   v = u/phi
```

First-order upwind advection (monotone) + central dispersion; implicit
backward-Euler giving a non-symmetric M-matrix (bounded, stable at any dt).
Immobile species (`m_s = 0`) evolve by reaction only.

## 3. Reaction kinetics (validated models)

Precipitation/dissolution (saturation index `SI = (alkali/alk_ref)(sal/sal_ref)`):
```
R_p = f_T [ k_p (SI-1)_+ - k_d (1-SI)_+ H(c_p) ] ,  f_T = Arrhenius(T)
```
Fines migration (critical velocity `v_c`):
```
R_det = k_det (|v|/v_c - 1)_+ sigma ,   R_dep = k_dep |v| c_s
```

## 4. Adsorption & retention

Surfactant adsorption via linear-driving-force toward an isotherm:
```
dq/dt = k_a ( q_eq(C) - q )
Langmuir:   q_eq = q_max K_L C / (1 + K_L C)
Freundlich: q_eq = K_F C^(1/n_F)
salinity/pH modifiers: q_eq *= (1 + beta_s sal)(1 - beta_pH alk)_+
```
Retardation `R = 1 + (1/phi) dq_eq/dC` emerges from the operator-split coupling.

Polymer retention + Residual Resistance Factor:
```
d(sigma_p)/dt = k_r ( sigma_max (k/k0)^(-gamma) - sigma_p )_+
RRF = 1 + (RRF_max - 1) min(sigma_p/sigma_ref, 1)
k = f_phi(phi) / RRF
```

## 5. Formation-damage coupling

```
phi = clip( phi0 - sum_i (mass_i / rho_i) , phi_min , 1 )
k   = f(phi) / RRF ,   f in { Kozeny-Carman, power-law, exponential }
Kozeny-Carman: k/k0 = (phi/phi0)^3 ((1-phi0)/(1-phi))^2
```

## 6. Numerical coupling

Sequential (Lie) operator splitting: transport (implicit) → reaction (explicit,
stiffness-controlled sub-stepping) → damage update → pressure/velocity re-solve
when permeability drifts past a tolerance. Conservative, monotone, sparse.
