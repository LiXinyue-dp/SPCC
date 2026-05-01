# SPCC

This project implements algorithms for counting cliques in graphs under Local Differential Privacy (LDP), as described in the SPCC paper. The implementation includes two main methods:

- **EPCC** (Edge-based Private Clique Counting) — a baseline method
- **SPCC** (Star-based Private Clique Counting) — the proposed optimized framework

## Project Structure

```
.
├── cpp/                      # C++ implementation
│   ├── SPCC.cpp              # Main implementation file
│   ├── Makefile              # Build configuration
│   ├── MemoryOperation.h     # Memory management utilities
│   ├── mt19937ar.h           # Mersenne Twister random number generator
│   └── include/              # Third-party libraries
│       ├── gcem.hpp          # Compile-time mathematical functions
│       └── stats.hpp         # Statistical distributions library
├── python/                   # Python preprocessing scripts
│   └── IMDB.py              # IMDB dataset preprocessing
├── data/                     # Dataset directory
│   ├── dblp/                 # DBLP collaboration network
│   ├── IMDB/                 # IMDB actor collaboration network
│   ├── mit/                  # MIT network data
│   └── Orkut/                # Orkut social network
└── LICENSE
```

## Features

- **Privacy-Preserving Clique Counting**: Implements Local Differential Privacy for q-clique counting (including 3-clique/triangle and 4-clique)
- **Two-Phase Framework (SPCC)**:
  - **Phase 1 — Degree Estimation & Parameter Setup**: Each user perturbs their true degree with Laplace noise; the server computes optimal parameters (k, L) and assigns degree-homogeneous buckets
  - **Phase 2 — Local Perturbation & Data Collection**: Users construct local k-star representations, apply PD and AGP mechanisms, then the server performs targeted structural estimation
- **Two Core Mechanisms (SPCC)**:
  - **Padding-and-Dropping (PD)**: Standardizes report lengths to a theoretically optimal value L to conceal true degrees and minimize estimation error
  - **Adaptive Grouping Perturbation (AGP)**: Partitions users into fine-grained degree-homogeneous buckets based on privatized degrees, applying localized GRR perturbation to amplify structural signals
- **Optimal Parameter Selection**: Automatically computes the optimal k-star size (Section 4.6) and padding length L (Section 4.2) to minimize estimation error
- **Graph Projection**: Caps each node's neighborhood to the maximum noisy degree to bound the maximum report size
- **Targeted Structural Estimation**: Instead of synthesizing a full noisy graph, the server performs algebraic rescaling on perturbed k-star reports to reconstruct clique counts efficiently

## Requirements

### For C++ Code
- C++ compiler with C++11 support (g++ recommended)
- Make utility
- Operating System: Linux/Unix (for compilation), or Windows with appropriate toolchain

### For Python Preprocessing
- Python 3.x
- NumPy
- SciPy

## Installation

### Linux/Unix/MacOS

1. Clone or download this repository
2. Navigate to the cpp directory:
```bash
cd cpp
```

3. Compile the program:
```bash
make
```

This will create an executable named `SPCC`.

### Windows

**Option 1: Using WSL (Recommended)**
```bash
wsl
cd ../cpp
make
```

**Option 2: Using MinGW or Cygwin**
- Install MinGW-w64 or Cygwin
- Use the same commands as Linux

**Option 3: Using Visual Studio**
- Create a new C++ console project
- Add SPCC.cpp and header files
- Configure include paths for the `include/` directory
- Build the project

## Usage

### Basic Command

```bash
./SPCC [DatasetPath] [q] [Method] [Eps] ([EdgeFile])
```

### Parameters

- `[DatasetPath]`: Path to the dataset directory (e.g., `../data/dblp`)
- `[q]`: Target clique size
  - `3` for triangle (3-clique) counting
  - `4` for 4-clique counting
- `[Method]`: Algorithm selection
  - `1` for EPCC (Edge-based Private Clique Counting) — baseline
  - `2` for SPCC (Star-based Private Clique Counting) — optimized framework
- `[Eps]`: Total privacy budget ε (= ε₁ + ε₂, where ε₁ is for degree perturbation and ε₂ for k-star perturbation)
- `[EdgeFile]`: (Optional) Edge file name, default is `edges.csv`

### Examples

**Count triangles on DBLP dataset using SPCC with ε=1.0:**
```bash
./SPCC ../data/dblp 3 2 1.0
```

**Count 4-cliques on IMDB dataset using EPCC with ε=2.0:**
```bash
./SPCC ../data/IMDB 4 1 2.0
```

**Using a custom edge file:**
```bash
./SPCC ../data/dblp 3 2 1.0 edges_subgraph_10000.csv
```

## Algorithm Details

### EPCC (Section 3) — Baseline
EPCC is a baseline method grounded in standard unbiased estimation theory. It consists of three stages:
1. **Local Perturbation**: Each user independently perturbs their adjacency vector using Randomized Response (RR), where each bit is retained with probability p = e^ε/(e^ε+1)
2. **Algebraic Rescaling**: The server applies element-wise rescaling to obtain unbiased edge estimators, avoiding noisy graph synthesis
3. **q-Clique Counting**: The server enumerates all q-node subsets and aggregates products of edge estimators

**Limitations**: Independent edge perturbation causes severe variance explosion; the global search space of C(n,q) makes it computationally intractable for large graphs.

### SPCC (Section 4) — Proposed Framework
SPCC shifts the perturbation unit from isolated edges to cohesive k-stars, which naturally serve as structural building blocks for cliques.

**Phase 1 — Degree Estimation & Parameter Setup (Section 4.2):**
1. Each user perturbs their true degree with Laplace noise: d̃_u = d_u + Lap(2/ε₁)
2. The server aggregates the noisy degree distribution, identifies maximum noisy degree d̂_max
3. The server computes optimal k-star size k and padding length L
4. Users are partitioned into degree-homogeneous buckets B based on noisy degrees

**Phase 2 — Local Perturbation & Data Collection (Sections 4.3–4.5):**
1. **Graph Projection**: Each user uniformly samples to retain at most d̂_max neighbors
2. **k-star Construction**: Users construct the local set of all k-stars from their projected neighborhood
3. **Padding-and-Dropping (PD)**: Standardizes all k-star sets to a universal length L
4. **Adaptive Grouping Perturbation (AGP)**: Each neighbor in a k-star is perturbed over its localized bucket domain B+(v) using GRR, with privacy budget ε' = ε₂/(k·L)
5. **Targeted Structural Estimation**: The server reconstructs clique counts via algebraic rescaling on the perturbed k-star reports, with estimated structural redundancy R̂_uv = C(d̃_u - 1, k - 1)

## Dataset Format

The program expects edge data in CSV format with the following structure:

```csv
node1,node2
0,1
0,2
1,2
...
```

- First line can be a header (will be skipped)
- Each subsequent line represents an undirected edge
- Node IDs should be integers

## Output

The program will:
1. Estimate the degree distribution (Phase 1)
2. Compute the optimal padding parameter (L) and k-star size (k) for the dataset
3. Run 10 independent experiments
4. Report results for each experiment
5. Output final statistics including:
   - Original clique count (ground truth)
   - Average relative error across experiments
   - Privacy parameters used (ε₁, ε₂, k, L)

### Sample Output

```
=== Optimal Padding Length Results ===
Optimal L: 12
Total nodes: 317080
Average k-stars per node: 45.2

=== Running 10 experiments ===
Dataset: ../data/dblp
Target clique: 3-clique
Method: SPCC

--- Experiment 1 ---
Calling SPCC_3Clique...
Actual number of nodes: 317080
Original triangle count: 2224385
...
Relative error (SPCC, 3-clique): 0.0234
...

=== Final Results ===
Number of experiments: 10
Target clique: 3-clique
Original triangle count: 2224385
Method: SPCC
Optimal padding (Lopt): 12
Average relative error: 0.0234
```

## Python Preprocessing

The `python/IMDB.py` script can be used to preprocess IMDB data:

```bash
cd python
python3 IMDB.py
```

This script reads the IMDB matrix format data, constructs actor collaboration networks, and outputs edge lists and degree information.

## Citation

If you use this code in your research, please cite the SPCC paper.
