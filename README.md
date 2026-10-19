# Mechanistic bridges: code for the master-equation mean-field review

Code accompanying the preprint

> Yannaël Bossard, Lahna Bekri, Alain Destexhe.
> *Mechanistic bridges from receptors to whole-brain dynamics: promise and limits of master-equation mean-field models.*
> arXiv:2608.00306, <https://arxiv.org/abs/2608.00306>

## Context

The review follows a lineage of mean-field models, from the master equation of El Boustani & Destexhe (2009) to receptor-aware whole-brain models. Two parts of the paper rest on computations, and this repository reproduces them.

- **Covariance closure** (Supplementary, *Covariance Matrix* and *Benchmarks and validity domain of the covariance closures*). Re-deriving the covariance equation of the master equation, we found that the published one is a mean-value closure, which misses a term of the same order as the terms it keeps. We derive a hierarchy of second-order equations (complete, corrected, stationary) and test them, with the published one, against exact solutions of the master equation. No sampling is used: every reference value is a closed form, an explicit finite sum, or the exact evolution of the probability distribution.
- **Computational cost** (main text, *Algorithmic work and array accesses as benchmark dimensions*; Supplementary, *Algorithmic cost analysis for neural simulation*). Work and memory accesses per biological second for whole-brain model classes, from mean fields to spiking networks and a data-driven fMRI encoder.

## Files

In the descriptions below, Δ is the gain (slope) of the transfer function at the fixed point.

### `covariance_closure_tables_v3.py`: stationary benchmark

One population of 4000 neurons, with an affine and a sigmoid transfer function. For these two functions, the exact stationary variance of the master equation is known without sampling: a closed form for the affine function, and an explicit finite sum over the stationary law for the sigmoid. The script prints the two Supplementary tables (as LaTeX rows) and every number quoted in that subsection. A sparse solve of the full 4001-state chain serves as a check.

- **Why:** to judge the covariance equations against exact values rather than against simulations.
- **Shows:** for |Δ| ≤ 0.9, the corrected equation is within 1% of the exact variance. The published one is off by the factor (1+Δ)/2, so it is 20 times too small for an inhibitory feedback Δ = −0.9. Beyond |Δ| = 1, the divergence of the corrected equation correctly signals that the fixed point is no longer a stationary state. The published equation instead stays finite and wrong: 0.57 Hz² against 4334 Hz² at Δ = −1.2.

### `transient_check_v3.py`: transient benchmark

Propagates the exact probability distribution of the master equation after a step of the input (matrix exponential, no sampling). It integrates the five covariance equations from the same initial state.

- **Why:** at stationary states the corrected and stationary equations coincide, so only a transient can test their source terms, and hence the choice of the equation boxed in the main text.
- **Shows:** the complete and corrected equations follow the exact variance within 0.3% of its peak, and differ from each other by less than 0.05 Hz². Dropping term (iii) more than doubles the error. The stationary equation misses the transient, and the published one is off by up to 65% of the peak.

### `make_landscape_figure.py`: sigmoid landscape figure

Landscape V(ξ) = −ln π / N of the exact stationary law of the sigmoid benchmark, for |Δ| = 0.5, 1 and 1.2 (Supplementary figure).

- **Shows:** there is one valley below the critical gain |Δ| = 1, a flat bottom at |Δ| = 1 and two valleys above it. This is why a single second-order (Gaussian) description fails at the critical gain, and why beyond it the closures only hold within one valley.

### `Plot_WBM_comparison.ipynb`: computational-cost figures

Estimates FLOP-equivalent work and source-level memory accesses per biological second, using the accounting formulas of the Supplementary, for Montbrió–Pazó–Roxin, Wong–Wang, the first-order adaptive mean field and its second-order extension (local or global covariances), spiking networks (LIF, AdEx, Hodgkin–Huxley) and TRIBE v2. It saves the main-text figure (`cost_compute_P2`) and the three Supplementary figures (`cost_compute_P10`, `memory_traffic_P2`, `memory_traffic_P10`) in `figures/`, and prints the numbers quoted in the text.

- **Why:** to compare model classes under an explicit convention that does not depend on hardware.
- **Shows:** keeping covariances local to each node doubles the node cost, but keeps the O(K + ρK²) scaling of first-order models. A dense global-covariance closure instead scales as O(P³K³) and quickly dominates. These are accounting scenarios, not runtime benchmarks.

## Running the code

The code requires Python 3 with NumPy, SciPy and Matplotlib. It was tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.

```bash
pip install -r requirements.txt
python covariance_closure_tables_v3.py
python transient_check_v3.py
python make_landscape_figure.py
jupyter notebook Plot_WBM_comparison.ipynb    # needs Jupyter, or open it in Google Colab
```

## Authors

- Yannaël Bossard, <yannael.bossard@ens-paris-saclay.fr>
- Claude Opus 4.8 and Claude Opus 5.5 (Anthropic AI models), co-authors of the code.

## How to cite

```bibtex
@misc{bossard2026mechanistic,
  title         = {Mechanistic bridges from receptors to whole-brain dynamics: promise and limits of master-equation mean-field models},
  author        = {Bossard, Yanna{\"e}l and Bekri, Lahna and Destexhe, Alain},
  year          = {2026},
  eprint        = {2608.00306},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2608.00306}
}
```

## License

MIT, see [`LICENSE`](LICENSE).
