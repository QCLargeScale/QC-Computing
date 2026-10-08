# VQLS benchmark — local and global cost in Qiskit

`vqls_cost.py` solves a linear system **A x = b** with the Variational Quantum Linear Solver (VQLS) and checks every step against plain linear algebra. It is meant for benchmarking how VQLS behaves on different matrices, sizes, ansätze and optimizers.

Reference: C. Bravo-Prieto, R. LaRose, M. Cerezo, Y. Subasi, L. Cincio, P. J. Coles,
*Variational Quantum Linear Solver*, Quantum **7**, 1188 (2023) — [arXiv:1909.05820](https://arxiv.org/abs/1909.05820), [Quantum journal](https://quantum-journal.org/papers/q-2023-11-22-1188/).

---

## 1. Installation

```bash
pip install qiskit numpy scipy matplotlib
```

Tested with Qiskit 2.5. Everything runs on the exact statevector simulator; no hardware account is needed.

## 2. Quick start

```bash
# 1. sanity check: 5 built-in matrices, every line should say PASS
python vqls_cost.py --demo

# 2. solve one system and plot the result
python vqls_cost.py --matrix tridiag --N 6 --b 1,2,3,4,5,6 --layers 3 \
                    --optimize COBYLA --maxiter 2000 --plot
```

---

## 3. The method

| Symbol | Meaning |
|---|---|
| `U` | circuit with `U|0> = |b>` |
| `V(θ)` | ansatz, `|x(θ)> = V(θ)|0>` (RY rotations + CX or CZ layers, real amplitudes) |
| `|ψ> = A|x>` | |
| `A = Σ_l c_l P_l` | Pauli (LCU) decomposition with `L` terms |

**Cost functions** (paper Eqs. 5–8, both normalised by `<ψ|ψ>`):

```
Global  C_G = 1 - |<b|ψ>|² / <ψ|ψ>
Local   C_L = 1/2 - 1/(2n) Σ_j <ψ| U Z_j U† |ψ> / <ψ|ψ>
        C_L ≤ C_G ≤ n·C_L          both are 0 exactly at the solution
```

Training uses `C_L` by default: for larger `n` the gradient of `C_G` vanishes much faster (barren plateau).

**What a small cost guarantees** (paper Eq. 9). With `ε` the trace distance between the VQLS state and the exact solution (for pure states `ε = √(1 − F)`, `F` = fidelity):

```
C_G ≥ ε² / κ²          C_L ≥ ε² / (n κ²)      →      ε ≤ κ · √(n · C_L)
```

`κ` is the condition number of the scaled, padded `A`. A tiny cost is therefore **not** a small error for an ill-conditioned matrix. The paper's termination threshold for a target `ε` is `γ = ε² / (n κ²)`.

**Measuring the cost with circuits.** Each term is a Hadamard test on one ancilla (real part, and imaginary part with an extra `S†`):

| Term | Quantity | Used for |
|---|---|---|
| `β_ll'` | `<x| P_l' P_l |x>` | `<ψ|ψ>` |
| `δ^j_ll'` | `<x| P_l' U Z_j U† P_l |x>` | local cost |
| `γ_l` | `<0| U† P_l V |0>` | global cost |

Note: here `γ_l` is a single amplitude measured with **controlled V and controlled U†**. The paper's `γ_ll'` (Eq. 17) is a product of two such amplitudes measured with the *Hadamard-overlap test*, which needs no controlled V or U. Both give the same cost; the paper's version is much cheaper on hardware.

Circuits per cost evaluation grow like `n · L²`, so circuit evaluation is off by default above `n = 4` (force with `--circuits`).

---

## 4. What the script does

Each step prints `[PASS]` / `[FAIL]` checks; `--pause` stops after each one.

| Step | Type | What happens |
|---|---|---|
| 1 | prep | Read `A` (dense, sparse, Pauli list, Ising or file) and `b` |
| 2 | prep | Scale `A` to spectral norm 1, pad to `2^n` as `[[A,0],[0,I]]`, `b → [b;0]`, normalise `b` |
| 3 | VQLS | Pauli decomposition `A = Σ c_l P_l` |
| 4 | VQLS | Build `U` with `U|0> = |b>` |
| 5 | VQLS | Build and draw the ansatz `V(θ)` |
| 6 | check | Exact costs with linear algebra (reference) |
| 7 | check | Single Hadamard-test terms vs numpy |
| 8 | VQLS | Full `C_L`, `C_G` from Hadamard tests — must equal step 6 |
| 9 | VQLS | Same with shot noise (`--shots`) |
| 10 | VQLS | Optimizer loop (`--optimize`), prints every θ tried |
| 11 | check | Result: best θ, costs, fidelity, VQLS `x` vs exact `x` |

"check" steps are possible only in simulation: in a real run the exact `x` is unknown, and reading all `2^n` entries of `x` would need exponentially many measurements.

Standard VQLS pipeline: **3 → 4 → 5 → (8 + 9 repeated inside 10)**. By default step 10 uses the exact cost (step 6) for speed; `--opt-circuits [--shots N]` uses the real circuit cost.

---

## 5. Choosing the number of layers

```
qubits n      = ceil(log2 N)
parameters    = n · (layers + 1)
min --layers  = ceil((2^n − 1) / n) − 1      so that parameters ≥ 2^n − 1
CNOTs         = layers · (n − 1)
```

| N | 5–8 | 9–16 | 17–32 | 33–64 | 65–128 |
|---|---|---|---|---|---|
| n | 3 | 4 | 5 | 6 | 7 |
| min layers | 2 | 3 | 6 | 10 | 18 |

Enough parameters is necessary, not sufficient: the circuit structure also decides which states are reachable.

---

## 6. Options

| Option | Values |
|---|---|
| `--matrix` | `tridiag` \| `randherm` \| `random` \| `ising` \| file (`.npy .npz .txt .csv`) |
| `--N` | matrix size, any integer (padded to `2^n`) |
| `--b` | `ones` \| `random` \| `e0`, `e1`, … \| `1,2,3,…` \| file |
| `--layers` | ansatz depth (default 2) |
| `--entangler` | `cx` (default) \| `cz` |
| `--optimize` | `COBYLA` \| `Nelder-Mead` \| `Powell` \| `BFGS` \| `L-BFGS-B` \| `SLSQP` |
| `--maxiter` | **maximum number of cost evaluations**, the same for every optimizer (default 500) |
| `--train-on` | `local` (default) \| `global` |
| `--init` | `small` (default, normal(0, 0.1)) \| `zeros` \| `random` (uniform 0–2π) |
| `--seed` | seed for the initial θ (default 11) |
| `--print-every` | print every k-th θ |
| `--circuits` / `--no-circuits` | force Hadamard-test circuits on / off |
| `--opt-circuits` | optimise with the circuit cost (slow) |
| `--shots` | shots per circuit (adds shot noise) |
| `--plot` | cost-landscape + size-scan plot (and the optimisation plot with `--optimize`) |
| `--opt-plot` | only the optimisation plot, no slow size scan |
| `--scan-N` | sizes for the size scan, e.g. `4,8,16,32,64` |
| `--theta-csv` | name of the θ-history CSV |
| `--demo` / `--format` / `--n` | built-in test problems |
| `--ask` | interactive input |
| `--pause` | stop after every step |

`python vqls_cost.py -h` lists everything.

### Matrix kinds

| Kind | Description | Solution |
|---|---|---|
| `tridiag` | 1-D Poisson-like, diagonal 2.5, off-diagonals −1 (sparse) | real |
| `randherm` | random Hermitian + shift | complex |
| `random` | random non-Hermitian + shift | complex |
| `ising` | paper Eq. 26: `(Σ X_j + J Σ Z_j Z_j+1 + η I)/ζ`, `J = 0.1`, `κ = 10`; `N` must be `2^n` | real |

The ansatz makes **real states only**. For `randherm` and `random` the fidelity cannot reach 1; use `tridiag`, `ising` or a real matrix of your own.

### Your own matrix

```bash
python vqls_cost.py --matrix my_A.npy --b my_b.txt --layers 3 --optimize COBYLA --plot
```

From Python:

```python
from vqls_cost import VQLSProblem, optimize
prob = VQLSProblem(A, b, layers=3)        # A: numpy, scipy.sparse, Pauli list or SparsePauliOp
CL, CG = prob.cost_exact(theta)
res = optimize(prob, method="COBYLA", maxiter=2000)
```

---

## 7. Outputs

| File | Content |
|---|---|
| `vqls_costs.png` | cost along θ₀ for your problem, mean cost and gradient variance vs size (`--plot`) |
| `vqls_optimization.png` | cost and true error per evaluation, VQLS `x` vs exact `x` (`--plot` or `--opt-plot` with `--optimize`) |
| `vqls_theta_history.csv` | every θ the optimizer tried, with `C_L` and `C_G` |

The CSV is written when the optimizer finishes; stopping with Ctrl+C before that loses it.

---

## 8. Reading the result

All of these describe how close the VQLS state is to the exact (normalised) solution:

| Quantity | Formula |
|---|---|
| fidelity | `F = |<x_exact|x>|²` |
| trace distance | `ε = √(1 − F)` |
| L2 error | `‖x − x_exact‖ ≈ √(2(1 − √F))` |

| Fidelity | L2 error | Verdict |
|---|---|---|
| < 0.95 | > 0.22 | not solved |
| 0.99 | ≈ 0.10 | close |
| 0.999 | ≈ 0.032 | good |
| ≥ 0.9999 | ≤ 0.010 | essentially exact |

A fidelity of 0.91 sounds high but is an angle of about 17° between the vectors: not solved.

If the fidelity stays below 0.99:

- cost still falling at the end → raise `--maxiter`
- cost flat but not near 0 → another `--seed`, or one more layer
- cost near 0 but error large → `A` is badly conditioned (see `cond(A)` in step 1 and the bound in section 3)
- complex matrix → the real RY ansatz cannot reach `x`

VQLS returns the **normalised** solution only. The original-scale `x` can be recovered classically from `x̂` as `x = s · x̂` with `s = <A x̂, b> / ‖A x̂‖²`.

---

## 9. Experiments

Each one changes **one** setting relative to experiment 3. Compare `C_L(best)`, fidelity and number of evaluations from step 11.

| # | Question | Command (prefix `python vqls_cost.py`) |
|---|---|---|
| 1 | Does everything work? | `--demo` |
| 2 | What happens in each step? | `--matrix tridiag --N 6 --b 1,2,3,4,5,6 --pause` |
| 3 | **Reference run** | `--matrix tridiag --N 6 --b 1,2,3,4,5,6 --layers 3 --optimize COBYLA --maxiter 2000 --plot` |
| 4 | Circuit too small | same as 3 with `--layers 1` (6 parameters < 7 needed) |
| 5 | Entangling gate | same as 3 with `--entangler cz` |
| 6 | Local vs global training | same as 3 with `--train-on global` |
| 7 | Starting point | `--matrix tridiag --N 16 --b random --layers 3 --optimize COBYLA --maxiter 5000 --print-every 250`, with and without `--init random` |
| 8 | Optimizer | same as 3 with `--optimize Nelder-Mead` or `--optimize L-BFGS-B` (same `--maxiter`) |
| 9 | Bigger problem | `--matrix tridiag --N 32 --b random --layers 6 --optimize COBYLA --maxiter 10000 --print-every 500 --plot` |
| 10 | Barren plateau | `--matrix tridiag --N 8 --b random --layers 2 --plot --scan-N 4,8,16,32,64,128` |
| 11 | Circuit cost, with/without shot noise | `--matrix tridiag --N 4 --b 1,2,3,4 --optimize COBYLA --opt-circuits --maxiter 200 --print-every 20` (then add `--shots 4000`) |

What to expect:

- **3** — `C_L ≈ 1e-7`, fidelity ≈ 1, the two `x` lines match.
- **4** — `C_L` stays above 0 and fidelity below 1 however long it runs: the ansatz cannot produce `x`.
- **5** — in our runs this layout got stuck (`C_L ≈ 0.067`, fidelity ≈ 0.96). Whether this is a reachability limit of this CZ layout or a local minimum is not established; note that the paper's own hardware-efficient ansatz uses RY + CZ.
- **6** — at 3 qubits about as good as 3; the flat `C_G` landscape appears only for larger `N` (repeat with `--N 32 --b random --layers 6 --maxiter 10000`).
- **7** — a random start often lands on a flat region; a start near 0 is easier.
- **8** — COBYLA / Nelder-Mead use cost values only; L-BFGS-B estimates gradients from extra evaluations (fewer steps on smooth problems, but more sensitive to shot noise).
- **9** — effort grows quickly with `N`: more layers, more parameters, many more evaluations.

---

## 10. Benchmarking notes

- **Fair optimizer comparison.** `--maxiter` counts cost evaluations for every optimizer (the budget is enforced inside the objective, not by scipy, whose `maxiter` means iterations for most methods). The optimizer message reports when the budget was reached.
- **Shot noise.** Every Hadamard test (and its real and imaginary circuit) gets its own seed derived from `--seed`, so noise is independent between circuits and runs are reproducible.
- **What drives the cost of a matrix.** The number of circuits scales with `L²`. VQLS assumes `L` grows only polynomially with `n`; a generic dense matrix has up to `4^n` Pauli terms, and identity padding can add terms. Report `L` (step 3) and `κ` (step 1) for every matrix you benchmark.
- **Paper-style metric.** The paper measures time-to-solution: the effort to reach the threshold `γ = ε²/(n κ²)` for a target `ε`, as a function of `κ`, `1/ε` and `n` (Figs. 4–7). The script currently stops on `--maxiter` or optimizer convergence; to compare with the paper, read off the evaluation at which `C_L` first drops below `γ` from the θ-history CSV.

---

## 11. Known limitations

- Real-amplitude ansatz only (complex solutions are not reachable).
- `γ` terms use controlled V and U† instead of the paper's Hadamard-overlap test.
- The ±π/2 "slope" in the text-only `--scan` is an approximation (the cost is a ratio, so the parameter-shift rule is not exact); the plot's size scan uses finite differences.
- Step 11 prints the real part of `x`; for complex matrices the imaginary part is not shown.
- The Pauli decomposition costs `4^n` inner products: fine up to `n ≈ 8–10`.
