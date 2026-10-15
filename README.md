# MPS-GPLHR

Core matrix-product-state GPLHR code and numerical data accompanying
*Matrix-product-state GPLHR for non-Hermitian many-body eigenproblems*.

This repository provides the solver source and the data used in the paper's
figures. Model-specific experiment drivers, plotting scripts, and cluster job
scripts are not included. The data can be inspected directly; reproducing the
computations requires constructing the corresponding models and initial states.

## Repository structure

```text
GPLHR_MPS/
    native/
        native_gplhr.py          # Block GPLHR iteration
        native_tensor.py         # Symmetry-resolved MPS/MPO operations and SRC
        small_dense.py           # Projected dense eigenproblems
        amen_preconditioner.py   # Auxiliary preconditioning routines
projects/
    feast_mps_hubbard_nonhermitian/
                                # Linear-system solvers used by preconditioners
renormalizer/                   # Bundled tensor-network dependency
data/                          # Paper figure data (NPZ)
requirements.txt
LICENSE
```

## Installation

Use Python 3.10 or 3.11 as a starting environment: the source uses Python 3.10
syntax, while the retained NumPy/SciPy pins predate newer Python versions.
The dependency list has not yet been validated as a complete installation or
as the exact environment used for the paper. In particular, older pinned
packages may require additional build dependencies on some platforms.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run Python from the repository root so that `GPLHR_MPS`, `projects`, and the
bundled `renormalizer` are importable. Alternatively, add the repository root
to `PYTHONPATH`. This repository does not currently provide package-installation
metadata for `pip install .`.

## Solver interface

```python
from GPLHR_MPS.native import (
    NativeGPLHRConfig,
    NativeGPLHRResult,
    solve_native_gplhr,
)
from GPLHR_MPS.native.native_tensor import NativeMPO, NativeMPS
```

The following is a calling template, not a standalone model example. Supply
`operator`, `parameters`, and `initial_states` before executing it:

```python
config = NativeGPLHRConfig(
    active_roots=5,
    retained_roots=5,
    target=20 + 0.15j,
    max_bond=200,
    max_iterations=500,
    residual_tolerance=1e-6,
    preconditioner="direct_gmres",
)

result = solve_native_gplhr(
    operator,                    # NativeMPO constructed for your model
    parameters,                  # Parameter object required by this code path
    initial_states=initial_states,  # Five compatible NativeMPS objects
    config=config,
)

print(result.values)
print(result.residual_norms)
print(result.converged)
```

These numerical settings illustrate the interface; they do not reconstruct a
paper experiment without its model and initialization. The returned result also
contains `states`, `actions`, `iterations`, `history`, and `diagnostics`.

The current native tensor representation uses a four-state local basis and
U(1) x U(1) charges, with local charge order
`(0,0), (0,1), (1,0), (1,1)`. Inputs must follow these conventions; arbitrary MPO
objects from other libraries cannot be passed directly. The parameter object
must provide `n_up` and `n_down` for the sector check, with further attributes
required by initialization and some preconditioners. Supply an explicit initial
block for a custom model: otherwise the solver uses its built-in Hubbard
initialization. Some retained preconditioner options also contain Hubbard-specific
construction and must be checked before use with another model.

## Figure data

All files below are in `data/`. NPZ is a NumPy archive containing named arrays.

| File | Paper figure | Contents |
|---|---|---|
| `l8_real_residual_spectrum_insets.npz` | Fig. 1(a,b) | L=8 real-spectrum Hubbard residual and spectrum comparisons |
| `l8_complex_residual_spectrum_insets.npz` | Fig. 1(c,d) | L=8 complex-spectrum Hubbard residual and spectrum comparisons |
| `gplhr_residual_convergence_hubbard20.npz` | Fig. 2(a) | L=20 Hubbard convergence histories |
| `gplhr_residual_convergence_hubbard30.npz` | Fig. 2(b) | L=30 Hubbard convergence histories |
| `l20_rank1_20_D256_final_energy.npz` | Fig. 3(a) | Final energy estimates for 20 states |
| `l20_rank1_20_D256_final_residual_squared.npz` | Fig. 3(b) | Final squared residuals for those states |
| `gplhr_residual_convergence_ssh128.npz` | Fig. 4 | L=128 spinless interacting SSH convergence histories |
| `ssh20_spinful_chi200_spectrum_residuals.npz` | Fig. 5 | L=20 spinful interacting SSH spectrum and residual data |
| `liouvillian_spectrum_L5_L50.npz` | Fig. 6 | Liouvillian spectra, gaps, and residual diagnostics for L=5,...,50 |

Inspect an archive from the repository root:

```python
import numpy as np

with np.load("data/ssh20_spinful_chi200_spectrum_residuals.npz",
             allow_pickle=False) as data:
    for name in data.files:
        array = data[name]
        print(name, array.shape, array.dtype)
```

Array names and shapes vary between archives. Distinguish residual norms from
squared residuals, and final estimates from tolerance-certified results. Inclusion
of a result in an archive does not imply that every eigenpair met the stopping
tolerance. Consult the stored diagnostics and the paper when interpreting results.

## Third-party source and license

The bundled `renormalizer/` code comes from the
[Renormalizer project](https://github.com/shuaigroup/Renormalizer), a Python
tensor-network package. Its upstream package metadata identifies version 0.0.11;
this bundled snapshot may include local changes and should not be assumed
identical to an upstream release.

The accompanying `LICENSE` contains the Apache License 2.0 text supplied with
that dependency. Preserve its source copyright notices and attribution when
redistributing it. The original upstream authorship metadata describes
Renormalizer, not the authorship of the MPS-GPLHR solver.
