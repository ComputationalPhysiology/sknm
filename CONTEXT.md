# SKNM

A Python implementation of the Simplified Kirchhoff Network Model of Jæger & Tveito
(*Sci Rep* 13:16434, 2023), together with KNM and SKNM(uₑ=0) so that the paper's model
comparisons can be reproduced. The domain is cell-based cardiac and islet electrophysiology:
excitable cells coupled through gap junctions, resolved per cell rather than homogenized into a
continuum.

## Language

### The network

**Cell network**:
The collection of cells and the connections between them, together with their geometry and
conductances. An arbitrary graph, not necessarily a sheet.
_Avoid_: tissue (implies a homogenized continuum, which is what these models are not), mesh,
grid, sheet (a sheet is one shape of network)

**Cell**:
One excitable cell, the smallest unit that carries a membrane potential. Has a size and a
membrane area; never resolved internally.
_Avoid_: node, element, compartment

**Connection**:
An adjacency between two cells through which current can pass. Carries its own geometry and gap
junction conductance, which may be zero.
_Avoid_: edge, junction, coupling, link

**Gap junction**:
The protein channels through which two connected cells exchange current. Its conductance `Gg` is
a property of a connection.

### The models

**KNM**:
The Kirchhoff Network Model: both the membrane potential and the extracellular potential are
unknowns, solved together.

**SKNM**:
The Simplified Kirchhoff Network Model: the extracellular potential is eliminated, leaving the
membrane potential as the only unknown.

**SKNM(uₑ=0)**:
SKNM with the extracellular potential taken as zero rather than eliminated. A third model used
in the paper for comparison, not an approximation of the other two.

**Variant**:
Which of the three models a simulation solves.

**Membrane model**:
The system of ordinary differential equations governing one cell's membrane potential and its
internal state. Interchangeable; the network is indifferent to which one is used.
_Avoid_: cell model, ionic model, ODE model

### Quantities

**Membrane potential** (`v`):
The potential difference across a cell's membrane. The quantity that propagates as a wave.

**Extracellular potential** (`uₑ`):
The potential in the space outside the cells. An unknown in KNM, eliminated in SKNM, zero in
SKNM(uₑ=0).

**λ** (`lam`):
The single number by which SKNM summarizes the extracellular space: the least-squares ratio
between the extracellular and intracellular conductances over all connections of a network. A
property of a network's conductances, not a free parameter.

**Conductance misfit** (`F(λ)`):
How far a network is from letting a single λ relate every connection's extracellular and
intracellular conductance — the weighted sum of squares that λ is chosen to minimize. Zero
means SKNM's assumption holds exactly; it grows as gap junctions are spread or cells are
elongated, which is where SKNM and KNM start to disagree.

**Extracellular volume fraction** (`δe`):
How much of the network's volume lies outside the cells.

**Anisotropy factor** (`α`):
A cell's length-to-width ratio. In the paper it takes five measured values, each with its own
cell dimensions.

**Gap junction variation** (`γ`):
How widely gap junction conductances are spread around their nominal value across a network.
γ = 0 is uniform coupling.

**Conduction velocity**:
The speed at which the wave of membrane potential travels across a network, measured between two
cells from the times their membrane potentials cross a threshold.

**Activation time**:
The time at which one cell's membrane potential first reaches a threshold. Undefined for a cell
the wave never reached.
_Avoid_: depolarization time, arrival time, crossing time

**Maximal upstroke velocity**:
The greatest rate at which one cell's membrane potential rises during its action potential,
`max dv/dt`. A property of a single cell, unlike a conduction velocity, which needs two.
_Avoid_: dvdtmax, rise rate, slope
