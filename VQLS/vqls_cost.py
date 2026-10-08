"""
vqls_cost.py  --  Variational Quantum Linear Solver (VQLS): local and global cost, in Qiskit
=========================================================================================

GOAL
    Solve  A x = b  with a parameterised quantum circuit.  The circuit V(theta) makes a
    trial state |x(theta)>; an optimizer (COBYLA, ...) changes theta until the cost is ~0,
    then |x(theta*)> is the (normalised) solution.

        |b> = U|0>            U        : circuit that prepares b
        |x> = V(theta)|0>     V(theta) : the ansatz, RY angles + CX (or CZ) gates
        |psi> = A|x>

    Global cost   C_G = 1 - |<b|psi>|^2 / <psi|psi>
    Local cost    C_L = 1/2 - 1/(2n) * sum_j <psi| U Z_j U^dag |psi> / <psi|psi>
    Relation      C_L <= C_G <= n * C_L        (both are 0 exactly at the solution)

    Train on C_L: for larger N the gradient of C_G vanishes much faster (barren plateau).


WHAT THE CODE DOES  (each step prints [PASS]/[FAIL] checks; --pause stops after each)
    [VQLS]  = standard step of the VQLS algorithm (Bravo-Prieto et al. 2019)
    [prep]  = classical preparation, common practice
    [check] = verification only; possible in simulation, not in a real large run

    STEP 1   [prep]   Read the matrix A (dense, sparse, Pauli list, Ising or file) and b.
    STEP 2   [prep]   Pad to size 2^n (n = qubits) and normalise:
                      A -> [[A,0],[0,I]], b -> [b;0].  b must have length 1 (quantum
                      state). Padding is only needed when N is not a power of 2.
    STEP 3   [VQLS]   Write A as a sum of unitaries, here Pauli strings:
                      A = sum_l c_l P_l  (L terms). Fewer terms = fewer circuits.
    STEP 4   [VQLS]   Build U with U|0> = |b>.
    STEP 5   [VQLS]   Build the ansatz V(theta) (RY + CX layers) and draw it.
    STEP 6   [check]  Exact costs with plain linear algebra, as a reference.
    STEP 7   [check]  Single Hadamard-test terms (beta, delta, gamma) vs numpy.
    STEP 8   [VQLS]   Full C_L and C_G from Hadamard-test circuits. This is how the
                      cost is measured on a quantum computer; must equal STEP 6.
    STEP 9   [VQLS]   Same with shot noise (--shots), as on real hardware.
    STEP 10  [VQLS]   Optimizer loop (--optimize): prints every theta tried, C_L, C_G.
                      In a real run each cost here comes from STEP 8-9 circuits
                      (--opt-circuits --shots); by default the script uses the exact
                      cost (STEP 6) for speed.
    STEP 11  [check]  Result: best theta, costs, fidelity, x from VQLS vs exact x.
                      In a real run the exact x is unknown (that is the problem being
                      solved), and reading all 2^n entries of x would take
                      exponentially many measurements; one measures a few quantities
                      of x instead.

    Standard VQLS pipeline:  3 -> 4 -> 5 -> (8 + 9 repeated inside 10)
    PLOTS    vqls_costs.png         cost slice along theta_0 + size scan   (--plot)
             vqls_optimization.png  cost per evaluation + solution         (--plot --optimize)
    CSV      vqls_theta_history.csv every theta the optimizer tried, with C_L and C_G

    Circuit-faithful evaluation (Bravo-Prieto et al. 2019), each term a Hadamard test:
        beta_{l l'}    = <x| P_l' P_l |x>              -> <psi|psi>
        delta^j_{l l'} = <x| P_l' U Z_j U^dag P_l |x>  -> <psi|U Z_j U^dag|psi>   (local)
        gamma_l        = <0| U^dag P_l V |0>           -> <b|psi>                 (global)
    Circuits per cost evaluation ~ n * L^2, so they switch off above n = 4 (use --circuits).


CHOOSING THE NUMBER OF LAYERS   (see also: python layers_table.py)
    qubits n     = ceil(log2(N))
    parameters   = n * (layers + 1)               = number of theta angles (RY gates)
    min --layers = ceil((2^n - 1) / n) - 1        so that parameters >= 2^n - 1
    CNOTs (CX)   = layers * (n - 1)

        N        5-8   9-16   17-32   33-64   65-128
        n          3      4       5       6        7
        layers     2      3       6      10       18

EXPERIMENTS TO TRY, IN THIS ORDER
    Each experiment changes ONE setting compared with experiment 3 (the reference run).
    Compare three numbers from STEP 11:  C_L(best),  fidelity,  evaluations.
      fidelity ~ 1                 -> solved
      fidelity stuck below 1       -> the circuit cannot reach x (layers / entangler)
      cost still falling at the end -> not enough iterations (raise --maxiter)

    1. Does everything work?
         python vqls_cost.py --demo
       Runs 5 built-in test matrices through all checks.
       Expect : every line says PASS.
       Lesson : the circuits give the same cost as plain linear algebra. Run once.

    2. What happens in each step?
         python vqls_cost.py --matrix tridiag --N 6 --b 1,2,3,4,5,6 --pause
       Runs steps 1-8 for one problem and stops after each step (press Enter).
       Look at: STEP 2 (6 -> padded to 8, n = 3 qubits), STEP 3 (Pauli terms of A),
                STEP 5 (circuit drawing), STEP 8 (circuit cost = exact cost).
       Lesson : how a matrix becomes qubits, Pauli terms and circuits. No optimizer
                yet: theta is random, so the costs are large.

    3. Can VQLS solve the system?   <- REFERENCE RUN
         python vqls_cost.py --matrix tridiag --N 6 --b 1,2,3,4,5,6 --layers 3 --optimize COBYLA --maxiter 2000 --plot
       COBYLA searches for the best theta (12 parameters).
       Look at: STEP 11 -> fidelity, and the lines "x from VQLS" vs "x exact".
       Expect : C_L ~ 1e-7, fidelity ~ 1.000000, both x lines match.
       Lesson : a successful VQLS run. Compare every following experiment with this.

    4. What if the circuit is too small?        changed: --layers 1
         python vqls_cost.py --matrix tridiag --N 6 --b 1,2,3,4,5,6 --layers 1 --optimize COBYLA --maxiter 2000
       1 layer = 3 x 2 = 6 parameters, but x needs 2^3 - 1 = 7 free values.
       Expect : C_L stops above 0, fidelity stays below 1, however long it runs.
       Lesson : if the ansatz cannot produce x, no optimizer can find it.

    5. Does the entangling gate matter?         changed: --entangler cz
         python vqls_cost.py --matrix tridiag --N 6 --b 1,2,3,4,5,6 --layers 3 --entangler cz --optimize COBYLA --maxiter 2000
       Same depth (12 parameters >= 7 needed) but CZ gates instead of CX.
       Expect : stuck at C_L ~ 0.067, fidelity ~ 0.96.
       Lesson : enough parameters is not enough; the circuit STRUCTURE decides which
                states are reachable, and this CZ layout misses some.

    6. Local or global cost for training?       changed: --train-on global
         python vqls_cost.py --matrix tridiag --N 6 --b 1,2,3,4,5,6 --layers 3 --optimize COBYLA --maxiter 2000 --train-on global
       The optimizer minimises C_G instead of C_L.
       Expect : at N = 6 (3 qubits) it likely works about as well as experiment 3.
       Lesson : the flat landscape of C_G only shows for LARGER N. To see it, repeat
                with --N 32 --b random --layers 6 --maxiter 10000 --print-every 500.

    7. Does the starting guess matter?          changed: --init random  (and N = 16)
         python vqls_cost.py --matrix tridiag --N 16 --b random --layers 3 --optimize COBYLA --maxiter 5000 --init random --print-every 250
         python vqls_cost.py --matrix tridiag --N 16 --b random --layers 3 --optimize COBYLA --maxiter 5000 --print-every 250
       Run both: random angles vs the default small angles near 0.
       Look at: final C_L and evaluations needed in the two runs.
       Lesson : a random start often lands on a flat region; a start near 0 is easier.

    8. Which optimizer works best?              changed: --optimize
         python vqls_cost.py --matrix tridiag --N 6 --b 1,2,3,4,5,6 --layers 3 --optimize Nelder-Mead --maxiter 5000 --print-every 250
         python vqls_cost.py --matrix tridiag --N 6 --b 1,2,3,4,5,6 --layers 3 --optimize L-BFGS-B --print-every 50
       Look at: evaluations needed to reach a low C_L, compared with COBYLA (exp. 3).
       Lesson : COBYLA, Nelder-Mead : use cost values only; robust, can be slow.
                                      Nelder-Mead often struggles with many parameters.
                L-BFGS-B             : uses gradients (estimated from extra evaluations
                                      around each theta); fewer steps on smooth problems,
                                      but hurt more by shot noise on real hardware.

    9. What changes for a bigger problem?       changed: N = 32, 6 layers
         python vqls_cost.py --matrix tridiag --N 32 --b random --layers 6 --optimize COBYLA --maxiter 10000 --print-every 500 --plot
       n = 5 qubits, 6 layers = 35 parameters (the minimum from the layers table).
       Look at: how many evaluations it needs and how low C_L gets
                (an earlier run was still improving at ~1.3e-4 after 8300 evaluations).
       Lesson : effort grows fast with N: more layers, more parameters, many more
                evaluations. This is the practical limit of VQLS.

    Optional, after 1-9:
   10. Barren plateau: cost and gradient vs N (plot panels 2-3)
         python vqls_cost.py --matrix tridiag --N 8 --b random --layers 2 --plot --scan-N 4,8,16,32,64,128
   11. Real circuit evaluation, without and with shot noise (small N only, slow)
         python vqls_cost.py --matrix tridiag --N 4 --b 1,2,3,4 --optimize COBYLA --opt-circuits --maxiter 200 --print-every 20
         python vqls_cost.py --matrix tridiag --N 4 --b 1,2,3,4 --optimize COBYLA --opt-circuits --shots 4000 --maxiter 200 --print-every 20
   12. Your own matrix and b from files
         python vqls_cost.py --matrix my_A.npy --b my_b.txt --layers 3 --optimize COBYLA --plot


OPTIONS AT A GLANCE   (python vqls_cost.py -h for all)
    --matrix      tridiag | randherm | random | ising | file (.npy .npz .txt .csv)
    --N           matrix size, any integer (padded to 2^n)
    --b           ones | random | e0, e1, ... | 1,2,3,... | file
    --layers      ansatz depth                 --entangler   cx (default) | cz
    --optimize    COBYLA | Nelder-Mead | Powell | BFGS | L-BFGS-B | SLSQP
    --maxiter     max cost evaluations         --print-every  print every k-th theta
    --init        small | zeros | random       --seed         seed for the initial theta
    --train-on    local | global               --plot         make the plots
    --scan-N      sizes for plot panels 2-3    --circuits / --no-circuits / --opt-circuits / --shots
    --demo        built-in test problems       --pause        stop after every step

NOTES
    * randherm and random give COMPLEX solutions; the RY ansatz only makes real states,
      so the fidelity cannot reach 1 there. Use tridiag, ising or a real matrix to test.
    * The CSV is written when the optimizer finishes; Ctrl+C before that loses it.
    * Or import and call:  from vqls_cost import VQLSProblem
"""

import argparse
import time

import numpy as np
import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector

PAUSE = False
TOL = 1e-8


# ----------------------------------------------------------------------------
# small helpers for printing checks
# ----------------------------------------------------------------------------
def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"   ({detail})" if detail else ""))
    return ok


def pause():
    if PAUSE:
        input("  ... press Enter to continue ...")


# ----------------------------------------------------------------------------
# STEP 1: matrix input in different formats  ->  dense array
# ----------------------------------------------------------------------------
def to_dense(A):
    """Accept dense numpy, scipy.sparse, Pauli list [(c,'XZI'),...] or SparsePauliOp."""
    if isinstance(A, SparsePauliOp):
        return A.to_matrix(), "SparsePauliOp"
    if isinstance(A, list):
        return SparsePauliOp.from_list([(lab, c) for c, lab in A]).to_matrix(), "Pauli list"
    if sp.issparse(A):
        return A.toarray().astype(complex), "scipy.sparse"
    return np.asarray(A, dtype=complex), "dense numpy"
    print(A)

# ----------------------------------------------------------------------------
# STEP 2: pad to 2^n and normalise
# ----------------------------------------------------------------------------
def pad_and_normalise(A, b):
    print(A)
    print(b)
    """Pad A -> [[A,0],[0,I]] and b -> [b;0] up to the next power of two.
    The solution becomes [x;0], so nothing about the problem changes.
    A is scaled to spectral norm 1 first, so the identity block fits in."""
    N = A.shape[0]
    n = max(1, int(np.ceil(np.log2(N))))
    scale = np.linalg.norm(A, 2)
    A = A / scale
    if 2**n != N:
        Ap = np.eye(2**n, dtype=complex)
        Ap[:N, :N] = A
        bp = np.zeros(2**n, dtype=complex)
        bp[:N] = b
        A, b = Ap, bp
    b = b / np.linalg.norm(b)
    print(A)
    return A, b, n, scale


# ----------------------------------------------------------------------------
# STEP 3: Pauli (LCU) decomposition
# ----------------------------------------------------------------------------
def pauli_decompose(A, cutoff=1e-12):
    """A = sum_l c_l P_l.  Costs 4^n inner products -- fine up to n ~ 8-10."""
    op = SparsePauliOp.from_operator(A).simplify(atol=cutoff)
    return [(complex(c), p.to_label().lstrip("-i")) for c, p in zip(op.coeffs, op.paulis)]


# ----------------------------------------------------------------------------
# STEP 5: ansatz V(theta) -- hardware-efficient RY + CX ladder (or CZ)
# ----------------------------------------------------------------------------
def make_ansatz(n, layers, entangler="cx"):
    """Hardware-efficient real-amplitude ansatz: RY layer, then `layers` x (entangler + RY).
        cx : CNOT ladder (like Qiskit RealAmplitudes) -- can reach every real state with
             enough layers (n=3: 2 layers, n=4: 3, n=5: about 6)
        cz : CZ brick pattern -- NOTE: on 3 qubits it only reaches a 6-dim slice of the
             7-dim real sphere at ANY depth, so it cannot hit every solution
    Both make REAL states only: a complex solution x needs a different ansatz."""
    theta = ParameterVector("t", n * (layers + 1))
    qc = QuantumCircuit(n, name="V")
    k = 0
    for q in range(n):
        qc.ry(theta[k], q); k += 1
    for _ in range(layers):
        if entangler == "cz":
            for q in range(0, n - 1, 2):
                qc.cz(q, q + 1)
            for q in range(1, n - 1, 2):
                qc.cz(q, q + 1)
        else:
            for q in range(n - 1):
                qc.cx(q, q + 1)
        for q in range(n):
            qc.ry(theta[k], q); k += 1
    return qc


# ----------------------------------------------------------------------------
# Hadamard test machinery
# ----------------------------------------------------------------------------
def controlled_pauli(qc, label, anc, sys_qubits):
    """Controlled Pauli string. label[-1] acts on qubit 0 (Qiskit order)."""
    for q, ch in zip(sys_qubits, reversed(label)):
        if ch == "X":
            qc.cx(anc, q)
        elif ch == "Y":
            qc.cy(anc, q)
        elif ch == "Z":
            qc.cz(anc, q)


def ancilla_z(qc, shots=None, seed=None):
    """Return <Z_anc> = P0 - P1 for the ancilla (last qubit)."""
    anc = qc.num_qubits - 1
    if shots is None:
        probs = Statevector(qc).probabilities([anc])
        return probs[0] - probs[1]
    m = qc.copy()
    m.add_register(__import__("qiskit").ClassicalRegister(1, "c"))
    m.measure(anc, 0)
    res = StatevectorSampler(seed=seed).run([m], shots=shots).result()[0]
    counts = res.data.c.get_counts()
    return (counts.get("0", 0) - counts.get("1", 0)) / shots


def hadamard_test(build_body, n, shots=None, seed=None):
    """Estimate <W> = Re + i Im for the controlled block written by build_body(qc, anc, sys)."""
    vals = []
    rng = np.random.default_rng(seed)  # FIX: real and imag circuits get independent shot noise
    for imag in (False, True):
        qc = QuantumCircuit(n + 1)
        anc, sysq = n, list(range(n))
        qc.h(anc)
        if imag:
            qc.sdg(anc)        # with S^dag: P0 - P1 = Im<W>
        build_body(qc, anc, sysq)
        qc.h(anc)
        vals.append(ancilla_z(qc, shots, int(rng.integers(1 << 31)) if shots else None))
    return vals[0] + 1j * vals[1]


# ----------------------------------------------------------------------------
# The problem object: everything you need to call the costs
# ----------------------------------------------------------------------------
class VQLSProblem:
    def __init__(self, A, b=None, U=None, layers=2, verbose=True, decompose=True, entangler="cx"):
        """decompose=False skips the Pauli decomposition (4^n work). The exact
        cost still works; the circuit cost needs the decomposition."""
        self.v = verbose

        # ---- STEP 1 ----
        if self.v: header("STEP 1  Read matrix")
        Ad, fmt = to_dense(A)
        N = Ad.shape[0]
        if b is None:
            b = np.ones(N) / np.sqrt(N)
        b = np.asarray(b, dtype=complex)
        if self.v:
            print(f"  input format : {fmt}")
            print(f"  shape        : {Ad.shape}")
            herm = np.allclose(Ad, Ad.conj().T)
            print(f"  hermitian    : {herm}")
            print(f"  cond(A)      : {np.linalg.cond(Ad):.4g}")
            check("square matrix", Ad.shape[0] == Ad.shape[1])
            check("b has matching length", len(b) == N)
            pause()

        # ---- STEP 2 ----
        if self.v: header("STEP 2  Pad to 2^n and normalise")
        self.A, self.b, self.n, self.scale = pad_and_normalise(Ad, b)
        self.N_in = N   # original matrix size, before padding
        if self.v:
            print(f"  N = {N}  ->  2^n = {2**self.n}   (n = {self.n} qubits)")
            print(f"  A scaled by 1/{self.scale:.4g}")
            check("||A||_2 = 1", abs(np.linalg.norm(self.A, 2) - 1) < 1e-8)
            check("||b|| = 1", abs(np.linalg.norm(self.b) - 1) < 1e-10)
            pause()
            
        # ---- STEP 3 ----
        if not decompose:
            self.terms, self.L = None, None
            if self.v:
                header("STEP 3  Pauli decomposition SKIPPED (exact cost only)")
        else:
            self._decompose()

        self._build_rest(U, layers, entangler)

    def _decompose(self):
        if self.v: header("STEP 3  Pauli / LCU decomposition  A = sum_l c_l P_l")
        self.terms = pauli_decompose(self.A)
        self.L = len(self.terms)
        A_rec = SparsePauliOp.from_list([(p, c) for c, p in self.terms]).to_matrix()
        if self.v:
            print(f"  number of terms L = {self.L}")
            for c, p in self.terms[:8]:
                print(f"     {c.real:+.4f}{c.imag:+.4f}j   {p}")
            if self.L > 8:
                print(f"     ... ({self.L - 8} more)")
            err = np.linalg.norm(A_rec - self.A)
            check("sum c_l P_l reproduces A", err < 1e-10, f"||diff|| = {err:.2e}")
            pause()

    def _build_rest(self, U, layers, entangler="cx"):
        # ---- STEP 4 ----
        if self.v: header("STEP 4  State preparation U|0> = |b>")
        if U is None:
            U = QuantumCircuit(self.n, name="U")
            U.append(StatePreparation(self.b), range(self.n))
        self.U = U
        bU = Statevector(U).data
        if self.v:
            ov = abs(np.vdot(self.b, bU))
            check("|<b|U|0>| = 1", abs(ov - 1) < 1e-8, f"overlap = {ov:.10f}")
            pause()

        # ---- STEP 5 ----
        if self.v: header("STEP 5  Ansatz V(theta)")
        self.V = make_ansatz(self.n, layers, entangler)
        self.n_params = self.V.num_parameters
        if self.v:
            print(f"  entangler = {entangler}")
            # one-line summary of the circuit size for this run
            #   needed        = 2^n - 1                  free values of a real normalised x
            #   min layers    = ceil((2^n - 1) / n) - 1  so that parameters >= needed
            #   2-qubit gates = layers * (n - 1)         one CX ladder (or CZ brick) per layer
            n = self.n
            needed = 2**n - 1
            min_layers = -(-needed // n) - 1                 # ceil(needed / n) - 1
            gate = "CNOTs" if entangler == "cx" else "CZs"
            print(f"  N = {self.N_in} -> n = {n} qubits | needed {needed} | min --layers {min_layers}"
                  f" | your --layers {layers} | parameters {self.n_params}"
                  f" | {gate} {layers * (n - 1)}")
            if layers < min_layers:
                print(f"  WARNING: --layers {layers} is below the minimum {min_layers}: "
                      f"the ansatz cannot reach every solution (fidelity may stay below 1)")
            print(self.V.draw(output="text", fold=100))
            pause()

        # exact Z_j matrices for the reference cost
        self.Zj = [SparsePauliOp("I" * (self.n - 1 - j) + "Z" + "I" * j).to_matrix()
                   for j in range(self.n)]
        self.Umat = np.array(__import__("qiskit").quantum_info.Operator(U).data)

    # ------------------------------------------------------------------
    def x_state(self, theta):
        return Statevector(self.V.assign_parameters(theta)).data
    
    # ------------------------------------------------------------------
    # STEP 6: exact reference (plain linear algebra, no circuits)
    # ------------------------------------------------------------------
    def cost_exact(self, theta=None, x=None):
        if x is None:
            x = self.x_state(theta)
        psi = self.A @ x
        nrm = np.vdot(psi, psi).real
        CG = 1 - abs(np.vdot(self.b, psi)) ** 2 / nrm
        loc = sum(np.vdot(psi, self.Umat @ Z @ self.Umat.conj().T @ psi).real for Z in self.Zj)
        CL = 0.5 - loc / (2 * self.n * nrm)
        return CL, CG

    # ------------------------------------------------------------------
    # STEP 7: individual Hadamard-test terms
    # ------------------------------------------------------------------
    def _Vgate(self, theta):
        return self.V.assign_parameters(theta).to_gate(label="V")

    def beta(self, theta, l, lp, shots=None, seed=None):
        """<x| P_l' P_l |x>   (V is not controlled: it only prepares |x>)."""
        Vg = self._Vgate(theta)
        Pl, Plp = self.terms[l][1], self.terms[lp][1]

        def body(qc, anc, s):
            qc.append(Vg, s)
            controlled_pauli(qc, Pl, anc, s)
            controlled_pauli(qc, Plp, anc, s)

        # body puts V after the H on the ancilla; that is fine because V acts on system only
        return hadamard_test(body, self.n, shots, seed)

    def delta(self, theta, l, lp, j, shots=None, seed=None):
        """<x| P_l' U Z_j U^dag P_l |x>.  Only Z_j needs a control: U U^dag = I otherwise."""
        Vg = self._Vgate(theta)
        Ug, Udg = self.U.to_gate(label="U"), self.U.inverse().to_gate(label="Udg")
        Pl, Plp = self.terms[l][1], self.terms[lp][1]

        def body(qc, anc, s):
            qc.append(Vg, s)
            controlled_pauli(qc, Pl, anc, s)
            qc.append(Udg, s)
            qc.cz(anc, s[j])
            qc.append(Ug, s)
            controlled_pauli(qc, Plp, anc, s)

        return hadamard_test(body, self.n, shots, seed)

    def gamma(self, theta, l, shots=None, seed=None):
        """<0| U^dag P_l V |0>.  Here V and U^dag must both be controlled."""
        cV = self._Vgate(theta).control(1)
        cUdg = self.U.inverse().to_gate(label="Udg").control(1)
        Pl = self.terms[l][1]

        def body(qc, anc, s):
            qc.append(cV, [anc] + s)
            controlled_pauli(qc, Pl, anc, s)
            qc.append(cUdg, [anc] + s)

        return hadamard_test(body, self.n, shots, seed)

    # ------------------------------------------------------------------
    # STEP 8: assemble both costs from Hadamard tests
    # ------------------------------------------------------------------
    def cost_circuit(self, theta, shots=None, seed=None, which=("local", "global")):
        c = np.array([t[0] for t in self.terms])
        L, n = self.L, self.n
        ncirc = 0
        # FIX: one independent seed per Hadamard test (before, every circuit reused `seed`,
        # so all shot noise was correlated). Same `seed` still gives a reproducible run.
        rng = np.random.default_rng(seed)
        nxt = lambda: int(rng.integers(1 << 31)) if shots else None

        # beta matrix: hermitian, and beta_ll = 1 for Pauli strings -> only l < l'
        B = np.eye(L, dtype=complex)
        for l in range(L):
            for lp in range(l + 1, L):
                B[l, lp] = self.beta(theta, l, lp, shots, nxt()); ncirc += 2
                B[lp, l] = np.conj(B[l, lp])
        psi_norm = np.real(c.conj() @ B.T @ c)      # sum_{l,l'} c_l'^* c_l beta_{l l'}

        out = {"psi_norm": psi_norm}

        if "local" in which:
            loc = 0.0
            for j in range(n):
                D = np.zeros((L, L), dtype=complex)
                for l in range(L):
                    for lp in range(l, L):
                        D[l, lp] = self.delta(theta, l, lp, j, shots, nxt()); ncirc += 2
                        D[lp, l] = np.conj(D[l, lp])
                loc += np.real(c.conj() @ D.T @ c)
            out["CL"] = 0.5 - loc / (2 * n * psi_norm)

        if "global" in which:
            g = np.array([self.gamma(theta, l, shots, nxt()) for l in range(L)]); ncirc += 2 * L
            out["CG"] = 1 - abs(c @ g) ** 2 / psi_norm

        out["n_circuits"] = ncirc
        return out


# ----------------------------------------------------------------------------
# Example matrices in each format
# ----------------------------------------------------------------------------
def example_dense(n=2, seed=1):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
    A = (M + M.conj().T) / 2 + 3 * np.eye(2**n)          # hermitian, well conditioned
    return A, None


def example_sparse(N=6):
    """1-D Poisson (tridiagonal) matrix, N not a power of 2 -> tests padding."""
    A = sp.diags([-1, 2.5, -1], [-1, 0, 1], shape=(N, N), format="csr")
    return A, None


def example_nonhermitian(n=2, seed=5):
    """Non-hermitian dense matrix: complex Pauli coefficients, tests the imaginary parts."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
    return M + 4 * np.eye(2**n), None


def example_pauli():
    return [(1.0, "III"), (0.2, "XZI"), (0.15, "ZIX"), (0.1, "IYY")], None


def example_ising(n=3, J=0.1, kappa=10):
    """A = (1/zeta)(sum X_j + J sum Z_j Z_j+1 + eta I), b = H^n |0>  (Bravo-Prieto et al.)."""
    lab = []
    for j in range(n):
        lab.append(("I" * (n - 1 - j) + "X" + "I" * j, 1.0))
    for j in range(n - 1):
        lab.append(("I" * (n - 2 - j) + "ZZ" + "I" * j, J))
    H0 = SparsePauliOp.from_list(lab)
    ev = np.linalg.eigvalsh(H0.to_matrix())
    zeta = (ev[-1] - ev[0]) / (1 - 1 / kappa)
    eta = zeta - ev[-1]
    A = ((H0 + SparsePauliOp("I" * n, eta)) / zeta).simplify()
    U = QuantumCircuit(n, name="U")
    U.h(range(n))
    b = np.ones(2**n) / np.sqrt(2**n)
    return A, b, U


# ----------------------------------------------------------------------------
# Run all checks for one problem
# ----------------------------------------------------------------------------
def run_problem(name, A, b=None, U=None, layers=2, shots=None, seed=7, circuits=True, entangler="cx"):
    print("\n\n" + "#" * 72)
    print(f"#  PROBLEM: {name}")
    print("#" * 72)
    prob = VQLSProblem(A, b, U, layers=layers, decompose=circuits, entangler=entangler)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, prob.n_params)

    # ---- STEP 6 ----
    header("STEP 6  Exact reference costs (linear algebra, random theta)")
    CL, CG = prob.cost_exact(theta)
    n = prob.n
    print(f"  C_L (exact) = {CL:.8f}")
    print(f"  C_G (exact) = {CG:.8f}")
    check("C_L <= C_G", CL <= CG + TOL)
    check("C_G <= n * C_L", CG <= n * CL + TOL, f"n*C_L = {n*CL:.6f}")
    x_true = np.linalg.solve(prob.A, prob.b)
    x_true /= np.linalg.norm(x_true)
    CL0, CG0 = prob.cost_exact(x=x_true)
    check("both costs = 0 at the true solution", abs(CL0) < 1e-10 and abs(CG0) < 1e-10,
          f"C_L = {CL0:.1e}, C_G = {CG0:.1e}")
    pause()
    if not circuits:
        print("\n  (steps 7-9 skipped: circuits off -- use --circuits to turn them on)")
        return prob

    # ---- STEP 7 ----
    header("STEP 7  Single Hadamard-test terms vs numpy")
    x = prob.x_state(theta)
    P = [SparsePauliOp(p).to_matrix() for _, p in prob.terms]
    errs = [abs(np.vdot(x, P[q] @ P[p] @ x) - prob.beta(theta, p, q))
            for p in range(min(prob.L, 6)) for q in range(min(prob.L, 6))]
    check("beta, all pairs (first 6 terms), incl. imaginary parts", max(errs) < 1e-8,
          f"max err = {max(errs):.1e}")
    l, lp = 0, min(1, prob.L - 1)
    b_ref = np.vdot(x, P[lp] @ P[l] @ x)
    b_cir = prob.beta(theta, l, lp)
    check(f"beta[{l},{lp}]", abs(b_ref - b_cir) < 1e-8, f"numpy {b_ref:.5f}  circuit {b_cir:.5f}")
    d_ref = np.vdot(x, P[lp] @ prob.Umat @ prob.Zj[0] @ prob.Umat.conj().T @ P[l] @ x)
    d_cir = prob.delta(theta, l, lp, 0)
    check(f"delta^0[{l},{lp}]", abs(d_ref - d_cir) < 1e-8, f"numpy {d_ref:.5f}  circuit {d_cir:.5f}")
    g_ref = np.vdot(prob.b, P[l] @ x)
    g_cir = prob.gamma(theta, l)
    check(f"gamma[{l}]", abs(g_ref - g_cir) < 1e-8, f"numpy {g_ref:.5f}  circuit {g_cir:.5f}")
    pause()

    # ---- STEP 8 ----
    header("STEP 8  Full costs from Hadamard tests (exact statevector)")
    t0 = time.time()
    res = prob.cost_circuit(theta)
    dt = time.time() - t0
    print(f"  circuits run    : {res['n_circuits']}   ({dt:.1f} s)")
    print(f"  C_L  circuit = {res['CL']:.8f}   exact = {CL:.8f}")
    print(f"  C_G  circuit = {res['CG']:.8f}   exact = {CG:.8f}")
    check("C_L circuit == exact", abs(res["CL"] - CL) < 1e-8)
    check("C_G circuit == exact", abs(res["CG"] - CG) < 1e-8)
    pause()

    # ---- STEP 9 ----
    if shots:
        header(f"STEP 9  With shot noise ({shots} shots per circuit)")
        t0 = time.time()
        r = prob.cost_circuit(theta, shots=shots, seed=seed)
        print(f"  C_L  shots = {r['CL']:.5f}   exact = {CL:.5f}   |err| = {abs(r['CL']-CL):.2e}")
        print(f"  C_G  shots = {r['CG']:.5f}   exact = {CG:.5f}   |err| = {abs(r['CG']-CG):.2e}")
        print(f"  ({time.time()-t0:.1f} s; expected error scale ~ 1/sqrt(shots) = {1/np.sqrt(shots):.1e})")
        pause()
    return prob


# ----------------------------------------------------------------------------
# Size scan: how cost, terms and circuit count grow with n
# ----------------------------------------------------------------------------
def size_scan(n_values=(2, 3, 4, 5), layers=2, samples=200, seed=3):
    header("SIZE SCAN  Ising family, random theta, exact costs")
    print(f"  {'n':>2} {'N':>5} {'L':>4} {'#circ local':>12} {'#circ global':>13}"
          f" {'mean C_L':>10} {'mean C_G':>10} {'var dC_L':>10} {'var dC_G':>10}"
          "\n  (var dC = variance over random theta of the cost change along parameter 0)")
    rng = np.random.default_rng(seed)
    for n in n_values:
        A, b, U = example_ising(n)
        prob = VQLSProblem(A, b, U, layers=layers, verbose=False)
        L = prob.L
        nc_loc = L * (L - 1) + n * L * (L + 1)          # beta (off-diag) + delta, x2 for re/im already
        nc_glob = L * (L - 1) + 2 * L
        CLs, CGs, gL, gG = [], [], [], []
        for _ in range(samples):
            th = rng.uniform(0, 2 * np.pi, prob.n_params)
            cl, cg = prob.cost_exact(th)
            CLs.append(cl); CGs.append(cg)
            # slope along the first parameter (difference over a +-pi/2 shift)
            e = np.zeros_like(th); e[0] = np.pi / 2
            clp, cgp = prob.cost_exact(th + e)
            clm, cgm = prob.cost_exact(th - e)
            gL.append((clp - clm) / 2); gG.append((cgp - cgm) / 2)
        print(f"  {n:>2} {2**n:>5} {L:>4} {nc_loc:>12} {nc_glob:>13}"
              f" {np.mean(CLs):>10.4f} {np.mean(CGs):>10.4f} {np.var(gL):>10.2e} {np.var(gG):>10.2e}")


# ----------------------------------------------------------------------------
# YOUR INPUT: matrix of any size N and the vector b
# ----------------------------------------------------------------------------
MATRIX_KINDS = ["tridiag", "randherm", "random", "ising"]


def make_matrix(kind, N, seed=1):
    """Build an N x N matrix.  kind:
        tridiag  : 1-D Poisson-like tridiagonal, scipy.sparse CSR (any N)
        randherm : random hermitian + shift, dense numpy           (any N)
        random   : random non-hermitian + shift, dense numpy      (any N)
        ising    : Bravo-Prieto Ising family, Pauli form          (N must be 2^n)
        <file>   : .npy (dense), .npz (scipy sparse), .txt/.csv (dense, whitespace or comma)
    """
    rng = np.random.default_rng(seed)
    if kind == "tridiag":
        return sp.diags([-1, 2.5, -1], [-1, 0, 1], shape=(N, N), format="csr")
    if kind == "randherm":
        M = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
        return (M + M.conj().T) / 2 + 2 * np.sqrt(N) * np.eye(N)
    if kind == "random":
        M = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
        return M + 2 * np.sqrt(N) * np.eye(N)
    if kind == "ising":
        n = int(round(np.log2(N)))
        if 2**n != N:
            raise ValueError(f"ising needs N = 2^n, got N = {N}")
        return example_ising(n)[0]
    # otherwise: a file name
    if kind.endswith(".npy"):
        return np.load(kind)
    if kind.endswith(".npz"):
        return sp.load_npz(kind)
    delim = "," if kind.endswith(".csv") else None
    return np.loadtxt(kind, delimiter=delim)


def make_vector(spec, N, seed=2):
    """b from a spec:
        ones        : all ones (uniform superposition)
        random      : random real entries
        e0 / e3 ... : basis vector with a 1 at that index
        1,2,0.5,... : explicit comma-separated values (length must be N)
        file.npy / file.txt : load from file
    """
    if spec == "ones":
        b = np.ones(N)
    elif spec == "random":
        b = np.random.default_rng(seed).normal(size=N)
    elif spec.startswith("e") and spec[1:].isdigit():
        b = np.zeros(N); b[int(spec[1:])] = 1.0
    elif spec.endswith(".npy"):
        b = np.load(spec)
    elif spec.endswith((".txt", ".csv")):
        b = np.loadtxt(spec, delimiter="," if spec.endswith(".csv") else None)
    else:
        b = np.array([complex(v.replace(" ", "")) for v in spec.split(",")])
        if np.allclose(b.imag, 0):
            b = b.real
    b = np.asarray(b).ravel()
    if len(b) != N:
        raise ValueError(f"b has length {len(b)}, but the matrix is {N} x {N}")
    if np.linalg.norm(b) == 0:
        raise ValueError("b must not be the zero vector")
    return b


def ask_inputs():
    """Interactive prompts (python vqls_cost.py --ask)."""
    print("\nMatrix kinds:", ", ".join(MATRIX_KINDS), " -- or a file name (.npy/.npz/.txt/.csv)")
    kind = input("  matrix kind  [tridiag]: ").strip() or "tridiag"
    if kind in MATRIX_KINDS:
        N = int(input("  size N       [6]: ").strip() or 6)
    else:
        N = None
    A = make_matrix(kind, N if N else 0)
    N = A.shape[0]
    print(f"  matrix is {N} x {N}")
    print("Vector b: ones | random | e0, e1 ... | comma list like 1,2,3 | file")
    bspec = input("  b            [ones]: ").strip() or "ones"
    return kind, A, bspec, make_vector(bspec, N)


# ----------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------
# reference palette (light surface)
C_LOCAL, C_GLOBAL, C_BOUND = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
C_ERROR = "#1baf7a"   # error line in the optimisation plot


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def landscape_data(prob, k=0, npts=121, n_circ=5, seed=7, circuits=True):
    """Sweep parameter k over [0, 2pi], others fixed at random values."""
    rng = np.random.default_rng(seed)
    th0 = rng.uniform(0, 2 * np.pi, prob.n_params)
    ts = np.linspace(0, 2 * np.pi, npts)
    CL, CG = [], []
    for t in ts:
        th = th0.copy(); th[k] = t
        cl, cg = prob.cost_exact(th)
        CL.append(cl); CG.append(cg)
    circ = None
    if circuits and prob.terms is not None:
        tc = np.linspace(0, 2 * np.pi, n_circ)
        cl_c, cg_c = [], []
        for t in tc:
            th = th0.copy(); th[k] = t
            r = prob.cost_circuit(th)
            cl_c.append(r["CL"]); cg_c.append(r["CG"])
        circ = (tc, np.array(cl_c), np.array(cg_c))
    return ts, np.array(CL), np.array(CG), circ


def scan_data(kind, N_values, bspec, layers=2, samples=200, seed=3):
    """Mean cost and gradient variance vs size, exact costs, random theta."""
    rows = []
    rng = np.random.default_rng(seed)
    for N in N_values:
        try:
            A = make_matrix(kind, N)
            b = make_vector(bspec, N)
        except ValueError as e:
            print(f"  skip N={N}: {e}")
            continue
        if kind == "ising":
            _, b, U = example_ising(int(round(np.log2(N))))
        else:
            U = None
        prob = VQLSProblem(A, b, U, layers=layers, verbose=False, decompose=False)
        CL, CG, gL, gG = [], [], [], []
        eps = 1e-4
        for _ in range(samples):
            th = rng.uniform(0, 2 * np.pi, prob.n_params)
            cl, cg = prob.cost_exact(th)
            CL.append(cl); CG.append(cg)
            e = np.zeros_like(th); e[0] = eps
            clp, cgp = prob.cost_exact(th + e)
            clm, cgm = prob.cost_exact(th - e)
            gL.append((clp - clm) / (2 * eps)); gG.append((cgp - cgm) / (2 * eps))
        rows.append(dict(N=N, n=prob.n, CL=np.mean(CL), CG=np.mean(CG),
                         CLs=np.std(CL), CGs=np.std(CG), vL=np.var(gL), vG=np.var(gG)))
        print(f"  N={N:>4} (n={prob.n})  mean C_L={rows[-1]['CL']:.4f}  mean C_G={rows[-1]['CG']:.4f}"
              f"  Var dC_L={rows[-1]['vL']:.2e}  Var dC_G={rows[-1]['vG']:.2e}")
    return rows


def make_plot(prob, label, kind, bspec, N_values, layers, out="vqls_costs.png", circuits=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    header("PLOT  computing data")
    print("  panel 1: cost landscape along theta_0 for your problem")
    ts, CL, CG, circ = landscape_data(prob, circuits=circuits)
    check("C_L <= C_G along the whole sweep", np.all(CL <= CG + TOL))
    check("C_G <= n*C_L along the whole sweep", np.all(CG <= prob.n * CL + TOL))
    if circ is not None:
        tc, clc, cgc = circ
        idx = [np.argmin(abs(ts - t)) for t in tc]
        err = max(np.max(abs(clc - CL[idx])), np.max(abs(cgc - CG[idx])))
        check("circuit points lie on the exact curves", err < 1e-6, f"max err = {err:.1e}")
    print(f"  panels 2-3: size scan, kind = {kind}, N = {list(N_values)}")
    rows = scan_data(kind, N_values, bspec, layers=layers)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURF)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.80, bottom=0.21, wspace=0.28)
    fig.suptitle(f"VQLS cost functions  -  {label}", x=0.05, ha="left",
                 fontsize=13, color=INK, fontweight="bold")

    # panel 1: landscape
    ax = axes[0]
    _style(ax, f"Cost along θ₀  (N = {2**prob.n}, n = {prob.n}, other θ fixed)", "θ₀  [rad]", "cost")
    ax.plot(ts, prob.n * CL, color=C_BOUND, lw=1.5, ls="--", label="n · C_L  (upper bound on C_G)")
    ax.plot(ts, CG, color=C_GLOBAL, lw=2, label="C_G  global")
    ax.plot(ts, CL, color=C_LOCAL, lw=2, label="C_L  local")
    if circ is not None:
        ax.plot(tc, cgc, "o", ms=8, mfc=C_GLOBAL, mec=SURF, mew=2, label="Hadamard-test circuits")
        ax.plot(tc, clc, "o", ms=8, mfc=C_LOCAL, mec=SURF, mew=2)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, max(1.0, min(1.6, np.max(prob.n * CL) * 1.05)))
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])
    ax.legend(fontsize=8, frameon=False, loc="lower center", ncol=2, labelcolor=INK2)

    if rows:
        nn = np.array([r["n"] for r in rows])
        # panel 2: mean cost vs size
        ax = axes[1]
        _style(ax, "Mean cost at random θ", "qubits n   (matrix size N = 2ⁿ)", "mean cost")
        for key, col, name in (("CG", C_GLOBAL, "C_G  global"), ("CL", C_LOCAL, "C_L  local")):
            m = np.array([r[key] for r in rows]); s = np.array([r[key + "s"] for r in rows])
            ax.fill_between(nn, m - s, m + s, color=col, alpha=0.12, lw=0)
            ax.plot(nn, m, "-o", color=col, lw=2, ms=8, mec=SURF, mew=2, label=name)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(nn, [f"{n}\nN={2**n}" for n in nn])
        ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK2)

        # panel 3: gradient variance vs size
        ax = axes[2]
        _style(ax, "Gradient variance  Var[∂C/∂θ₀]  (log scale)", "qubits n   (matrix size N = 2ⁿ)",
               "variance")
        ax.semilogy(nn, [r["vG"] for r in rows], "-o", color=C_GLOBAL, lw=2, ms=8, mec=SURF, mew=2,
                    label="C_G  global")
        ax.semilogy(nn, [r["vL"] for r in rows], "-o", color=C_LOCAL, lw=2, ms=8, mec=SURF, mew=2,
                    label="C_L  local")
        ax.set_xticks(nn, [f"{n}\nN={2**n}" for n in nn])
        ax.legend(fontsize=8, frameon=False, loc="upper right", labelcolor=INK2)

    fig.text(0.05, 0.02, f"Ansatz: RY + entangler, {layers} layers.   Panel 1: your matrix and b, "
             f"dots = full Hadamard-test circuit evaluation.   Panels 2-3: kind = {kind}, b = {bspec}, "
             "200 random θ per size, exact statevector, band = ±1 std.",
             fontsize=8, color=INK2)
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(f"\n  plot saved to {out}")
    return out


# ----------------------------------------------------------------------------
# OPTIMIZER: the actual VQLS loop  theta -> cost -> optimizer -> new theta
# ----------------------------------------------------------------------------
def initial_theta(n_params, init="small", seed=11):
    """Starting guess theta_0.
        small  : normal(0, 0.1)   -- state starts near |0...0>, avoids the flat plateau
        zeros  : all 0
        random : uniform(0, 2pi)  -- a random point, often on the plateau for large n
    """
    rng = np.random.default_rng(seed)
    if init == "zeros":
        return np.zeros(n_params)
    if init == "random":
        return rng.uniform(0, 2 * np.pi, n_params)
    return rng.normal(0, 0.1, n_params)


def optimize(prob, method="COBYLA", maxiter=500, train_on="local", init="small", seed=11,
             use_circuits=False, shots=None, print_every=1, csv="vqls_theta_history.csv"):
    """Minimise C_L (or C_G) over theta with scipy.optimize.minimize.

    EVERY theta the optimizer asks for is printed (every `print_every`-th evaluation)
    and saved to `csv`, together with C_L and C_G at that theta.
    Gradient methods (BFGS, L-BFGS-B, SLSQP) also evaluate extra thetas for their
    finite-difference gradients; those show up in the log too.
    """
    from scipy.optimize import minimize

    header(f"STEP 10  Optimise theta with {method}  (training on C_{'L' if train_on == 'local' else 'G'})")
    th0 = initial_theta(prob.n_params, init, seed)
    mode = "Hadamard-test circuits" + (f", {shots} shots" if shots else "") if use_circuits else "exact statevector"
    print(f"  cost evaluated with : {mode}")
    print(f"  initial guess       : {init}   theta_0 = {np.array2string(th0, precision=4, separator=', ')}")
    print(f"  max iterations      : {maxiter}")
    print(f"  printing every      : {print_every} evaluation(s)\n")
    print(f"  {'eval':>5} {'C_L':>12} {'C_G':>12} {'best':>12}   theta")

    hist = []
    best = [np.inf]
    rng = np.random.default_rng(seed)

    # exact solution, ONLY used to record the error for the plot (VQLS never uses it)
    x_exact = np.linalg.solve(prob.A, prob.b); x_exact /= np.linalg.norm(x_exact)

    def error_of(th):
        x = prob.x_state(th)
        ph = np.exp(-1j * np.angle(np.vdot(x_exact, x)))   # remove the global sign / phase
        return np.linalg.norm(x * ph - x_exact)

    class _Budget(Exception):
        pass

    def objective(th):
        # FIX: --maxiter = max COST EVALUATIONS for every optimizer. scipy's own "maxiter"
        # counts iterations for L-BFGS-B / Nelder-Mead / BFGS (many evaluations each), so
        # the budget is enforced here instead -> fair comparison between optimizers.
        if len(hist) >= maxiter:
            raise _Budget
        if use_circuits:
            r = prob.cost_circuit(th, shots=shots, seed=int(rng.integers(1 << 30)) if shots else None)
            cl, cg = r["CL"], r["CG"]
        else:
            cl, cg = prob.cost_exact(th)
        val = cl if train_on == "local" else cg
        best[0] = min(best[0], val)
        hist.append((len(hist) + 1, cl, cg, th.copy(), error_of(th)))
        k = len(hist)
        if k == 1 or k % print_every == 0:
            ths = np.array2string(th, precision=4, separator=", ", max_line_width=10_000)
            print(f"  {k:>5} {cl:>12.6e} {cg:>12.6e} {best[0]:>12.6e}   {ths}")
        return val

    t0 = time.time()
    opts = {"maxiter": 10**9}          # no scipy limit; the budget is checked in objective()
    try:
        res = minimize(objective, th0, method=method, options=opts)
        msg = res.message
    except _Budget:
        res, msg = None, f"stopped: evaluation budget --maxiter {maxiter} reached"
    dt = time.time() - t0

    # best theta seen (not always res.x for every method)
    kbest = int(np.argmin([h[1] if train_on == "local" else h[2] for h in hist]))
    th_best = hist[kbest][3]
    CL, CG = prob.cost_exact(th_best)

    # compare with the exact solution (global phase removed)
    x = prob.x_state(th_best)
    x_true = np.linalg.solve(prob.A, prob.b); x_true /= np.linalg.norm(x_true)
    ov = np.vdot(x_true, x)
    x_aligned = x * np.exp(-1j * np.angle(ov))
    fid = abs(ov) ** 2

    header("STEP 11  Result")
    print(f"  optimizer message   : {msg}")
    print(f"  evaluations         : {len(hist)}   ({dt:.1f} s)")
    print(f"  best at evaluation  : {kbest + 1}")
    print(f"  best theta          : {np.array2string(th_best, precision=6, separator=', ')}")
    print(f"  C_L(best)  (exact)  : {CL:.3e}")
    print(f"  C_G(best)  (exact)  : {CG:.3e}")
    print(f"  fidelity |<x_true|x>|^2 = {fid:.6f}")
    m = min(len(x), 8)
    print(f"  x from VQLS (first {m}): {np.round(x_aligned[:m].real, 4)}")
    print(f"  x exact     (first {m}): {np.round(x_true[:m].real, 4)}")
    check("C_L <= C_G <= n*C_L at the optimum", CL <= CG + TOL and CG <= prob.n * CL + TOL)
    ic = 1 if train_on == "local" else 2   # FIX: check the cost that was trained (C_L or C_G)
    check("training reduced the cost", min(h[ic] for h in hist) < hist[0][ic])
    if fid < 0.99:
        tips = ["more layers (--layers 3 or 4)", "another start (--init random / --seed)",
                "more iterations (--maxiter)"]
        if not np.allclose(x_true.imag, 0, atol=1e-8):
            tips.insert(0, "the exact x is COMPLEX but the RY ansatz only makes real states")
        print("  note: fidelity < 0.99 -- try: " + "; ".join(tips))

    # CSV with every evaluation
    with open(csv, "w") as f:
        f.write("eval,C_L,C_G," + ",".join(f"theta_{i}" for i in range(prob.n_params)) + "\n")
        for k, cl, cg, th, _ in hist:
            f.write(f"{k},{cl:.12e},{cg:.12e}," + ",".join(f"{t:.10f}" for t in th) + "\n")
    print(f"\n  every theta + C_L + C_G saved to {csv}")
    return dict(hist=hist, theta=th_best, CL=CL, CG=CG, fidelity=fid,
                x=x_aligned, x_true=x_true, result=res)


# ----------------------------------------------------------------------------
# HOW TO READ THE RESULT: fidelity, overlap, angle, error
# ----------------------------------------------------------------------------
# All four numbers describe the SAME thing: how close the VQLS answer x is to
# the exact answer x_exact (both vectors normalised to length 1).
#
#   fidelity  F = |<x_exact | x>|^2        1 = identical, 0 = completely different
#   overlap     = sqrt(F)                  dot product of the two vectors
#   angle       = arccos(overlap)          0 deg = identical, 90 deg = perpendicular
#   error       = ||x - x_exact||_2        L2 (Euclidean) distance, 0 = identical
#                 ~ sqrt(2 * (1 - overlap))      (max sqrt(2) ~ 1.41)
#
#   Example: F = 0.9114 -> overlap 0.955 -> angle ~17 deg -> error ~0.30
#            -> NOT solved, although 91% sounds high.
#
# Why a high-looking fidelity can still be bad: F squares the overlap, so it
# stays high even for vectors that are visibly different.
#
#   fidelity        error      verdict
#   < 0.95          > 0.22     not solved
#   0.99            ~ 0.14     close
#   0.999           ~ 0.045    good
#   >= 0.9999       < 0.014    essentially exact
#
# The error is ONE number for the whole vector:
#   error = sqrt( (x_0 - x_exact_0)^2 + (x_1 - x_exact_1)^2 + ... )
#   typical gap per entry (RMS) ~ error / sqrt(N)
# Since ||x_exact|| = 1, this L2 error equals the relative error.
#
# Cost vs error: the optimizer only sees the cost (C_L); the error needs the
# exact answer, so it is known only in simulation. Roughly error ~ sqrt(cost),
# e.g. cost 4e-9 -> error ~ 1e-4.
#
# If fidelity < 0.99:
#   - cost still falling at the end  -> raise --maxiter
#   - cost flat but not near 0       -> try another --seed, or one more --layers
#   - cost near 0 but error large    -> A is badly conditioned (see cond(A), STEP 1)
#   - complex matrix (randherm, random) -> real RY ansatz cannot reach x
# ----------------------------------------------------------------------------

def make_opt_plot(prob, out_opt, label, method, train_on, out="vqls_optimization.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = out_opt["hist"]
    ev = np.array([h[0] for h in hist])
    CL = np.array([h[1] for h in hist]); CG = np.array([h[2] for h in hist])
    floor = 1e-16

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=SURF,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.2, wspace=0.22)
    fig.suptitle(f"VQLS optimisation  -  {label}", x=0.06, ha="left",
                 fontsize=13, color=INK, fontweight="bold")

    ax = axes[0]
    tr = "C_L" if train_on == "local" else "C_G"
    cost = CL if train_on == "local" else CG
    col = C_LOCAL if train_on == "local" else C_GLOBAL
    ERR = np.array([h[4] for h in hist])
    # index of the best theta so far, at every iteration
    ibest = np.zeros(len(cost), dtype=int)
    for i in range(1, len(cost)):
        ibest[i] = i if cost[i] < cost[ibest[i - 1]] else ibest[i - 1]

    def plain(v):  # 4.2e-07 -> 0.00000042
        return np.format_float_positional(v, precision=2, fractional=False, trim="-")

    _style(ax, f"Cost and error per iteration  ({method}, trained on {tr})",
           "iteration (cost evaluations)", "value")
    ax.plot(ev, cost, color=col, lw=1, alpha=0.25)
    ax.plot(ev, ERR, color=C_ERROR, lw=1, alpha=0.25)
    ax.plot(ev, cost[ibest], color=col, lw=2,
            label=f"cost {tr}  (what the optimizer sees)   final = {plain(cost[ibest][-1])}")
    ax.plot(ev, ERR[ibest], color=C_ERROR, lw=2,
            label=f"error ||x - x_exact||  (true distance)   final = {plain(ERR[ibest][-1])}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(0, max(np.max(cost[ibest]), np.max(ERR[ibest])) * 1.08)
    ax.set_xlim(0, ev[-1])
    ax.legend(fontsize=8, frameon=False, loc="upper right", labelcolor=INK2)

    ax = axes[1]
    x, xt = out_opt["x"].real, out_opt["x_true"].real
    idx = np.arange(len(x))
    _style(ax, f"Solution  (fidelity {out_opt['fidelity']:.4f})", "component i", "x_i  (normalised)")
    w = 0.4 if len(x) <= 32 else 0.8
    ax.bar(idx - w / 2 if len(x) <= 32 else idx, xt, width=w, color=C_BOUND, alpha=0.55,
           label="exact solution", zorder=2)
    ax.plot(idx + (w / 2 if len(x) <= 32 else 0), x, "o", ms=8 if len(x) <= 32 else 4,
            mfc=C_LOCAL, mec=SURF, mew=2, label="VQLS  x(θ*)", zorder=3)
    ax.axhline(0, color=GRID, lw=1)
    if len(x) <= 16:
        ax.set_xticks(idx)
    ax.legend(fontsize=8, frameon=False, loc="best", labelcolor=INK2)

    fig.text(0.06, 0.03, f"{len(hist)} cost evaluations.   Faint lines: value at every θ the optimizer "
             "tried.   Bold lines: at the best θ so far.   Every θ is in the CSV.",
             fontsize=8, color=INK2)
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(f"  optimisation plot saved to {out}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="VQLS local and global cost. Examples:\n"
        "  python vqls_cost.py --matrix tridiag --N 6 --b 1,2,3,4,5,6 --plot\n"
        "  python vqls_cost.py --matrix randherm --N 16 --b random --plot\n"
        "  python vqls_cost.py --matrix my_A.npy --b my_b.npy --plot\n"
        "  python vqls_cost.py --ask --plot\n"
        "  python vqls_cost.py --demo            (the five built-in test problems)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=None,
                    help="tridiag | randherm | random | ising | file (.npy/.npz/.txt/.csv)")
    ap.add_argument("--N", type=int, default=None, help="matrix size (any integer; padded to 2^n)")
    ap.add_argument("--b", default="ones", help="ones | random | e0 | 1,2,3,... | file")
    ap.add_argument("--ask", action="store_true", help="ask for matrix, size and b interactively")
    ap.add_argument("--plot", action="store_true", help="make the 3-panel plot")
    ap.add_argument("--scan-N", default=None,
                    help="sizes for plot panels 2-3, e.g. 4,8,16,32,64 (default: 4 ... 128)")
    ap.add_argument("--out", default="vqls_costs.png", help="plot file name")
    ap.add_argument("--circuits", dest="circuits", action="store_true", default=None,
                    help="force Hadamard-test circuits on")
    ap.add_argument("--no-circuits", dest="circuits", action="store_false",
                    help="exact costs only (fast, any size)")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--entangler", default="cx", choices=["cx", "cz"], help="ansatz entangling gate")
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--pause", action="store_true")
    ap.add_argument("--optimize", default=None, metavar="METHOD",
                    help="run the optimizer: COBYLA | Nelder-Mead | Powell | BFGS | L-BFGS-B | SLSQP")
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--train-on", default="local", choices=["local", "global"])
    ap.add_argument("--init", default="small", choices=["small", "zeros", "random"],
                    help="initial theta guess")
    ap.add_argument("--seed", type=int, default=11, help="seed for the initial theta")
    ap.add_argument("--print-every", type=int, default=1, help="print every k-th theta")
    ap.add_argument("--opt-circuits", action="store_true",
                    help="optimise with the Hadamard-test circuit cost (slow; add --shots for noise)")
    ap.add_argument("--theta-csv", default="vqls_theta_history.csv")
    ap.add_argument("--opt-plot", action="store_true",
                    help="only the optimisation plot (cost + error vs iteration), no slow size scan")
    ap.add_argument("--demo", action="store_true", help="run the five built-in test problems")
    ap.add_argument("--format", default="all", choices=["all", "dense", "nonherm", "sparse", "pauli", "ising"],
                    help="with --demo: which built-in problem")
    ap.add_argument("--n", type=int, default=3, help="with --demo: qubits for the Ising example")
    ap.add_argument("--scan", action="store_true", help="old text-only Ising size scan")
    args = ap.parse_args()
    PAUSE = args.pause

    if args.scan:
        size_scan()
    elif args.demo:
        if args.format in ("all", "dense"):
            A, b = example_dense()
            run_problem("dense numpy (random hermitian 4x4)", A, b, layers=args.layers, shots=args.shots)
        if args.format in ("all", "nonherm"):
            A, b = example_nonhermitian()
            run_problem("dense numpy (NON-hermitian 4x4)", A, b, layers=args.layers, shots=args.shots)
        if args.format in ("all", "sparse"):
            A, b = example_sparse()
            run_problem("scipy.sparse CSR (tridiagonal 6x6, padded)", A, b, layers=args.layers, shots=args.shots)
        if args.format in ("all", "pauli"):
            A, b = example_pauli()
            run_problem("Pauli list", A, b, layers=args.layers, shots=args.shots)
        if args.format in ("all", "ising"):
            A, b, U = example_ising(args.n)
            run_problem(f"Ising family (n={args.n})", A, b, U, layers=args.layers, shots=args.shots)
    else:
        # ---- your own problem ----
        if args.ask:
            kind, A, bspec, b = ask_inputs()
        else:
            kind = args.matrix or "tridiag"
            N = args.N if args.N else 6
            A = make_matrix(kind, N)
            bspec = args.b
            b = make_vector(bspec, A.shape[0])
        N = A.shape[0]
        U = None
        if kind == "ising" and bspec == "ones":
            U = QuantumCircuit(int(round(np.log2(N))), name="U"); U.h(range(U.num_qubits))
        n = max(1, int(np.ceil(np.log2(N))))
        circuits = args.circuits if args.circuits is not None else (n <= 4)
        if args.circuits is None and not circuits:
            print(f"(n = {n} qubits: Hadamard-test circuits off by default above n = 4; "
                  "use --circuits to force them)")
        label = f"{kind}, N = {N}, b = {bspec}"
        prob = run_problem(label, A, b, U, layers=args.layers, shots=args.shots, circuits=circuits,
                           entangler=args.entangler)
        print("\n  your b (first 8 entries, normalised, padded):", np.round(prob.b[:8].real, 4))
        x_true = np.linalg.solve(prob.A, prob.b)
        print("  exact x  (first 8 entries, normalised):     ",
              np.round((x_true / np.linalg.norm(x_true))[:8].real, 4))

        if args.plot:
            if args.scan_N:
                Ns = [int(v) for v in args.scan_N.split(",")]
            else:
                Ns = [4, 8, 16, 32, 64, 128]
            scan_kind = kind if kind in MATRIX_KINDS else "tridiag"
            scan_b = bspec if bspec in ("ones", "random") or (bspec.startswith("e") and bspec[1:].isdigit()) else "ones"
            if scan_b != bspec:
                print(f"\n  note: explicit b values only fit one size; the size scan uses b = ones")
            if scan_kind != kind:
                print(f"  note: matrix from file has one size; the size scan uses kind = tridiag")
            make_plot(prob, label, scan_kind, scan_b, Ns, args.layers, out=args.out, circuits=circuits)

        if args.optimize:
            if args.opt_circuits and prob.terms is None:
                prob._decompose()
            res = optimize(prob, method=args.optimize, maxiter=args.maxiter, train_on=args.train_on,
                           init=args.init, seed=args.seed, use_circuits=args.opt_circuits,
                           shots=args.shots, print_every=args.print_every, csv=args.theta_csv)
            if args.plot or args.opt_plot:
                make_opt_plot(prob, res, label, args.optimize, args.train_on)
