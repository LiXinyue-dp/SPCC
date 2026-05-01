# SPCC: Secure Privacy-Preserving Clique Counting

This project implements algorithms for counting cliques in graphs under local differential privacy constraints, as described in the SPCC paper. The implementation includes two main methods:

- **EPCC** (Enhanced Privacy-Preserving Clique Counting): Enhanced baseline with optimal padding (Lopt) and degree-based grouping
- **SPCC** (Secure Privacy-Preserving Clique Counting): Builds on EPCC with cross-checking and candidate validation

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

- **Privacy-Preserving Clique Counting**: Implements local differential privacy for 3-clique (triangle) and 4-clique counting
- **Multiple Algorithms**:
  - **EPCC** (Method 1): Enhanced baseline with optimal padding (Lopt) and degree-based grouping
  - **SPCC** (Method 2): Full secure method with cross-checking and candidate validation
- **Optimal Padding (Lopt)**: Automatically computes the optimal padding parameter to minimize estimation error (Section 4.2)
- **Degree-Based Grouping**: Uses Laplace noise + quantile-based splitting for privacy-preserving degree grouping (Section 4.3)
- **Cross-Checking**: Verifies reported edges between nodes for improved accuracy (Section 5.2)
- **Candidate Validation**: For 4-cliques, validates all mutual edges among candidate nodes (Section 5.2)

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
  - `3` for triangle counting
  - `4` for 4-clique counting
- `[Method]`: Algorithm selection
  - `1` for EPCC (Enhanced Privacy-Preserving Clique Counting)
  - `2` for SPCC (Secure Privacy-Preserving Clique Counting)
- `[Eps]`: Privacy parameter epsilon
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

### EPCC (Section 5.1)
EPCC extends the basic RR baseline with:
1. **Optimal Padding (Lopt)**: Computed via loss function minimization to balance padding and dropping errors
2. **Degree-Based Grouping**: Nodes are grouped by noisy degree (with Laplace noise) for targeted RR perturbation
3. Each node creates (k-1)-tuples from its neighbors, pads/drops to Lopt elements, then applies RR perturbation

### SPCC (Section 5.2)
SPCC builds on EPCC with additional verification:
1. **Cross-Checking**: For each reported edge (i,j) in G_i's report, verifies that i is also in G_j's report
2. **Candidate Validation**: For 4-cliques, after finding potential 3-clique candidates, verifies all mutual cross-edges
3. These steps filter out false edges introduced by RR perturbation

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
1. Compute the optimal padding parameter (Lopt) for the dataset
2. Run 10 independent experiments
3. Report results for each experiment
4. Output final statistics including:
   - Original clique count
   - Average relative error across experiments
   - Privacy parameters used

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
