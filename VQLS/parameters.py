"""
layers_table.py -- parameter-count lower bound for VQLS ansatz layers
                    using an RY + linear CX ladder ansatz

PURPOSE
-------
This script estimates a PARAMETER-COUNT LOWER BOUND on the number of
ansatz repetitions ("layers") for a real-amplitude variational state.

For an N x N linear system, choose

    n = ceil(log2(N))

qubits, so that the state space has dimension 2^n.

If N is not a power of two, the linear system may be embedded into a
2^n-dimensional system, for example

        A' = [ A  0 ]
             [ 0  I ]

        b' = [ b ]
             [ 0 ]

so that

        x' = [ x ]
             [ 0 ]

solves A' x' = b' whenever A x = b.

A normalized REAL state on n qubits is a vector in R^(2^n) constrained
to unit norm. Therefore it has

        2^n - 1

independent continuous degrees of freedom.

For the Qiskit RealAmplitudes-style circuit considered here:

    - each rotation layer applies one RY rotation to each of n qubits;
    - there are `layers` repeated RY + CX blocks;
    - a final RY rotation layer is included;
    - CX entanglement is linear/reverse-linear.

Therefore the number of trainable RY angles is

        number of parameters = n * (layers + 1)

and the number of CX gates is

        number of CX gates = layers * (n - 1).

A NECESSARY parameter-counting condition is therefore

        n * (layers + 1) >= 2^n - 1,

which gives

        layers >= ceil((2^n - 1) / n) - 1.

Hence this script defines

       L_count = ceil((2^n - 1) / n) - 1.

IMPORTANT CAVEAT
----------------
L_count is ONLY a parameter-count lower bound.

It does NOT prove that a RealAmplitudes circuit with L_count repetitions
can prepare every normalized real n-qubit state.

Having at least as many trainable parameters as the target manifold has
degrees of freedom is a necessary counting condition, but is not in
general sufficient for expressibility/surjectivity. Circuit structure,
parameter redundancies, entanglement topology, and the rank/dimension of
the reachable state manifold also matter.

Accordingly, the output should NOT be interpreted as:

    "this number of layers guarantees that ANY real solution can
     be represented."

A more accurate interpretation is:

    "below this value there are fewer nominal trainable parameters than
     the number of continuous degrees of freedom of a generic normalized
     real state."

REFERENCES
----------
[1] A. W. Harrow, A. Hassidim, and S. Lloyd,
    "Quantum Algorithm for Linear Systems of Equations,"
    Physical Review Letters 103, 150502 (2009).
    DOI: 10.1103/PhysRevLett.103.150502

    Relevance:
    Quantum linear-system algorithms encode vectors into amplitudes of
    quantum states. An n-qubit state has 2^n computational-basis
    amplitudes, motivating n = ceil(log2(N)) for an N-dimensional vector.

[2] C. Bravo-Prieto, R. LaRose, M. Cerezo, Y. Subasi,
    L. Cincio, and P. J. Coles,
    "Variational Quantum Linear Solver,"
    Quantum 7, 1188 (2023).
    Originally arXiv:1909.05820 (2019).
    DOI: 10.22331/q-2023-11-22-1188

    Relevance:
    Primary reference for the Variational Quantum Linear Solver (VQLS),
    including its variational formulation and local/global cost
    functions.

[3] M. Mottonen, J. J. Vartiainen, V. Bergholm, and M. M. Salomaa,
    "Transformation of quantum states using uniformly controlled
    rotations,"
    Quantum Information and Computation 5, 467-473 (2005).
    arXiv:quant-ph/0407010.

    Relevance:
    Provides a state-preparation construction based on uniformly
    controlled rotations.

    For REAL-amplitude state preparation, a recursive RY construction
    contains

        1 + 2 + 4 + ... + 2^(n-1) = 2^n - 1

    rotation angles.

    NOTE:
    The statement "2^n - 1 degrees of freedom" itself follows directly
    from the dimension of the normalized real-state manifold. It should
    not be presented as though Mottonen et al. simply state that every
    real state requires exactly 2^n - 1 ordinary RY gates in the same
    circuit architecture used here.

[4] IBM Quantum / Qiskit documentation,
    "RealAmplitudes".

    Relevance:
    RealAmplitudes consists of alternating RY rotation layers and
    entanglement layers, followed by a final rotation layer unless that
    layer is explicitly disabled.

    Under the assumptions used in THIS script:

        parameters = n * (layers + 1)

    and with linear/reverse-linear CX entanglement:

        CX gates = layers * (n - 1).

    These formulas are implementation/configuration dependent and should
    not be treated as universal formulas for every hardware-efficient
    ansatz.

[5] A. Kandala et al.,
    "Hardware-efficient variational quantum eigensolver for small
    molecules and quantum magnets,"
    Nature 549, 242-246 (2017).
    DOI: 10.1038/nature23879

    Relevance:
    Standard reference for hardware-efficient variational ansatzes.

    NOTE:
    Use the Qiskit RealAmplitudes documentation, rather than Kandala
    et al., as the source for the exact parameter and CX counts used by
    this particular script.

[6] M. Larocca et al.,
    "Theory of overparametrization in quantum neural networks,"
    Nature Computational Science 3, 542-551 (2023).

    Relevance:
    Discusses parameterization, effective model dimension,
    overparameterization, and why simply counting nominal circuit
    parameters does not by itself guarantee reachability or
    expressibility.

[7] M. Cerezo et al.,
    "Cost function dependent barren plateaus in shallow parametrized
    quantum circuits,"
    Nature Communications 12, 1791 (2021).
    DOI: 10.1038/s41467-021-21728-w

    Relevance:
    Shows that barren-plateau behaviour can depend strongly on whether
    local or global cost functions are used.

[8] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush,
    and H. Neven,
    "Barren plateaus in quantum neural network training landscapes,"
    Nature Communications 9, 4812 (2018).
    DOI: 10.1038/s41467-018-07090-4

    Relevance:
    Foundational barren-plateau reference.


SUMMARY OF FORMULAS USED BY THIS SCRIPT
---------------------------------------

    n               = ceil(log2(N))

    padded dimension
                    = 2^n

    real-state DOF  = 2^n - 1

    parameters      = n * (layers + 1)

    CX gates        = layers * (n - 1)

    L_count         = ceil((2^n - 1) / n) - 1

where the final three formulas assume an RY + linear-CX RealAmplitudes
structure with a final RY rotation layer.

Usage
-----
    python layers_table.py
        Print the table for n = 1 ... 10.

    python layers_table.py 6
        Show the counting bound for an N = 6 linear system.

    python layers_table.py 6 100 1000
        Show results for several matrix dimensions.
"""

FORMULAS = """\
Parameter-count lower bound for VQLS ansatz
(RY + linear CX ladder)
------------------------------------------------------------

  qubits n
      = ceil(log2(N))

      Reference:
      Harrow, Hassidim & Lloyd,
      "Quantum Algorithm for Linear Systems of Equations,"
      Phys. Rev. Lett. 103, 150502 (2009).
      DOI: 10.1103/PhysRevLett.103.150502

      Also:
      Bravo-Prieto et al.,
      "Variational Quantum Linear Solver,"
      Quantum 7, 1188 (2023);
      arXiv:1909.05820.

      Note:
      An n-qubit register has Hilbert-space dimension 2^n.
      Therefore 2^n >= N is required for an N-dimensional vector.


  padded dimension
      = 2^n

      Reference / justification:
      This follows directly from the dimension of an n-qubit register.

      If N is not a power of two, one possible embedding is

          A' = [ A  0 ]
               [ 0  I ]

          b' = [ b ]
               [ 0 ]

      so that x' = [x; 0] solves A'x' = b' whenever Ax = b.

      This padding construction is a direct algebraic construction and
      does not require a separate canonical literature reference.


  real-state DOF
      = 2^n - 1

      Reference / justification:
      A normalized real n-qubit state is a vector in R^(2^n)
      satisfying one normalization constraint,

          sum_i x_i^2 = 1.

      Therefore the normalized real-state manifold has

          2^n - 1

      continuous degrees of freedom.

      Related state-preparation reference:
      Mottonen, Vartiainen, Bergholm & Salomaa,
      "Transformation of quantum states using uniformly controlled
      rotations,"
      Quantum Information and Computation 5, 467-473 (2005);
      arXiv:quant-ph/0407010.

      For recursive real-amplitude RY state preparation,

          1 + 2 + 4 + ... + 2^(n-1) = 2^n - 1

      rotation angles occur.

      Important:
      The 2^n - 1 value is fundamentally a dimension-counting result;
      it should not be interpreted as saying that every ansatz with
      2^n - 1 nominal parameters can necessarily reach every real state.


  parameters
      = n * (layers + 1)

      Reference:
      IBM Quantum / Qiskit RealAmplitudes documentation.

      Assumptions used here:
      - one RY rotation per qubit in each rotation layer,
      - `layers` repeated rotation + entanglement blocks,
      - one final RY rotation layer is included.

      Therefore there are

          layers + 1

      rotation layers, each containing n trainable RY angles, giving

          n * (layers + 1)

      trainable parameters.

      General hardware-efficient ansatz reference:
      Kandala et al.,
      "Hardware-efficient variational quantum eigensolver for small
      molecules and quantum magnets,"
      Nature 549, 242-246 (2017).
      DOI: 10.1038/nature23879


  counting lower bound
      = ceil((2^n - 1) / n) - 1

      Derivation:
      Require the number of nominal trainable parameters to be at least
      the number of continuous degrees of freedom:

          n * (layers + 1) >= 2^n - 1.

      Rearranging gives

          layers >= (2^n - 1)/n - 1.

      Since layers must be an integer,

          layers >= ceil((2^n - 1)/n) - 1.

      Reference / caveat:
      This is a DERIVED parameter-count lower bound, not a guarantee of
      full state expressibility.

      See:
      Larocca et al.,
      "Theory of overparametrization in quantum neural networks,"
      Nature Computational Science 3, 542-551 (2023).

      Circuit topology, parameter redundancy, and the dimension of the
      reachable state manifold can impose additional restrictions.


  CNOTs / CX gates
      = layers * (n - 1)

      Reference:
      IBM Quantum / Qiskit RealAmplitudes documentation.

      Assumption:
      linear or reverse-linear CX entanglement.

      A linear entangling layer on n qubits contains

          n - 1

      CX gates.

      With `layers` entangling layers, the total is therefore

          layers * (n - 1).

      Important:
      This formula changes if a different entanglement topology is used,
      for example full entanglement.


Additional VQLS / optimization references
-----------------------------------------

  VQLS:
      Bravo-Prieto et al.,
      "Variational Quantum Linear Solver,"
      Quantum 7, 1188 (2023);
      arXiv:1909.05820.

  Local/global cost functions and barren plateaus:
      Cerezo et al.,
      "Cost function dependent barren plateaus in shallow parametrized
      quantum circuits,"
      Nature Communications 12, 1791 (2021).
      DOI: 10.1038/s41467-021-21728-w

  Foundational barren-plateau reference:
      McClean et al.,
      "Barren plateaus in quantum neural network training landscapes,"
      Nature Communications 9, 4812 (2018).
      DOI: 10.1038/s41467-018-07090-4
"""

import math
import sys


def qubits(N):
    """
    Return the number of qubits required for an N-dimensional vector.

    An n-qubit register has dimension 2^n, so we require

        2^n >= N,

    giving

        n = ceil(log2(N)).

    We return at least one qubit.

    References
    ----------
    Harrow, Hassidim & Lloyd, Phys. Rev. Lett. 103, 150502 (2009).
    Bravo-Prieto et al., Quantum 7, 1188 (2023).
    """
    return max(1, math.ceil(math.log2(N)))


def real_state_dof(n):
    """
    Continuous degrees of freedom of a normalized REAL n-qubit state.

    A real n-qubit state has 2^n real amplitudes:

        x in R^(2^n).

    Normalization imposes one continuous constraint,

        sum_i x_i^2 = 1,

    leaving

        2^n - 1

    continuous degrees of freedom.

    Note
    ----
    This is a manifold-dimension/counting statement. It does not imply
    that an arbitrary ansatz with this many nominal parameters can
    necessarily reach every such state.
    """
    return 2**n - 1


def num_parameters(n, layers):
    """
    Number of trainable RY angles for the assumed ansatz.

    Assumptions
    -----------
    - n RY rotations per rotation layer;
    - `layers` repeated RY + CX blocks;
    - one final RY rotation layer.

    Therefore

        parameters = n * (layers + 1).

    Reference
    ---------
    IBM Quantum / Qiskit RealAmplitudes documentation.
    """
    return n * (layers + 1)


def num_cnot(n, layers):
    """
    Number of CX gates for linear/reverse-linear entanglement.

    A linear CX entangling layer connecting n qubits contains n - 1
    CX gates.

    With `layers` entangling layers:

        CNOTs = layers * (n - 1).

    This formula is NOT valid for every possible entanglement topology.

    Reference
    ---------
    IBM Quantum / Qiskit RealAmplitudes documentation.
    """
    return layers * (n - 1)


def min_layers(n):
    """
    Parameter-count lower bound on the number of ansatz repetitions.

    We impose the necessary counting condition

        n * (layers + 1) >= 2^n - 1.

    Solving for `layers` gives

        layers >= ceil((2^n - 1) / n) - 1.

    IMPORTANT
    ---------
    This is a NECESSARY PARAMETER-COUNT CONDITION ONLY.

    It does not prove that the corresponding RealAmplitudes circuit is
    capable of representing every normalized real n-qubit state.

    Circuit structure, parameter redundancy, entanglement topology, and
    reachable-manifold dimension can impose additional restrictions.

    See
    ---
    Larocca et al.,
    "Theory of overparametrization in quantum neural networks,"
    Nature Computational Science 3, 542-551 (2023).
    """
    return math.ceil(real_state_dof(n) / n) - 1


def row(n):
    """Create one row of the summary table."""
    L = min_layers(n)

    # Matrix dimensions that require exactly n qubits.
    N_lo = 2 if n == 1 else 2 ** (n - 1) + 1

    return {
        "N range": f"{N_lo}-{2**n}" if N_lo < 2**n else f"{2**n}",
        "qubits n": n,
        "DOF 2^n-1": real_state_dof(n),
        "count-bound layers": L,
        "parameters": num_parameters(n, L),
        "CNOTs": num_cnot(n, L),
    }


def print_table(rows):
    """Print rows as a simple ASCII table."""
    cols = list(rows[0].keys())

    width = {
        c: max(len(c), *(len(str(r[c])) for r in rows))
        for c in cols
    }

    line = "+-" + "-+-".join("-" * width[c] for c in cols) + "-+"

    print(line)
    print("| " + " | ".join(c.rjust(width[c]) for c in cols) + " |")
    print(line)

    for r in rows:
        print(
            "| "
            + " | ".join(str(r[c]).rjust(width[c]) for c in cols)
            + " |"
        )

    print(line)


if __name__ == "__main__":
    print(FORMULAS)

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            N = int(arg)

            if N < 1:
                raise ValueError("Matrix dimension N must be >= 1.")

            n = qubits(N)
            L = min_layers(n)

            print(
                f"N = {N:>6}  ->  "
                f"padded to {2**n:>6},  "
                f"n = {n:>2} qubits,  "
                f"count-bound --layers {L}  "
                f"({num_parameters(n, L)} parameters, "
                f"{num_cnot(n, L)} CNOTs)"
            )

    else:
        print_table([row(n) for n in range(1, 11)])
