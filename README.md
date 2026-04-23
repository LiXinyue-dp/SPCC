# Local Differentially Private Clique Counting

This project implements algorithms for counting cliques in graphs under local differential privacy constraints. The implementation includes both baseline randomized response methods and optimized K-Star methods.

## Project Structure

```
.
├── cpp/                      # C++ implementation
│   ├── LFCK.cpp             # Main implementation file
│   ├── Makefile             # Build configuration
│   ├── MemoryOperation.h    # Memory management utilities
│   ├── mt19937ar.h          # Mersenne Twister random number generator
│   └── include/             # Third-party libraries
│       ├── gcem.hpp         # Compile-time mathematical functions
│       └── stats.hpp        # Statistical distributions library
├── python/                  # Python preprocessing scripts
│   └── IMDB.py             # IMDB dataset preprocessing
├── data/                    # Dataset directory
│   ├── dblp/               # DBLP collaboration network
│   ├── IMDB/               # IMDB actor collaboration network
│   ├── mit/                # MIT network data
│   └── Orkut/              # Orkut social network
└── LICENSE
```

## Features

- **Privacy-Preserving Clique Counting**: Implements local differential privacy for 3-clique (triangle) and 4-clique counting
- **Multiple Algorithms**: 
  - Baseline Randomized Response (RR) methods
  - K-Star optimized methods for improved accuracy
- **Automatic Optimization**: Computes optimal padding parameters for the dataset
- **Experimental Framework**: Runs multiple experiments and reports average relative errors

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

This will create an executable named `LFCK`.

### Windows

You have several options:

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
- Add LFCK.cpp and header files
- Configure include paths for the `include/` directory
- Build the project

## Usage

### Basic Command

```bash
./LFCK [DatasetPath] [q] [Method] [Eps] ([EdgeFile])
```

### Parameters

- `[DatasetPath]`: Path to the dataset directory (e.g., `../data/dblp`)
- `[q]`: Target clique size
  - `3` for triangle counting
  - `4` for 4-clique counting
- `[Method]`: Algorithm selection
  - `1` for Baseline Randomized Response (RR)
  - `2` for K-Star method (optimized)
- `[Eps]`: Privacy parameter epsilon (will be automatically adjusted by q)
- `[EdgeFile]`: (Optional) Edge file name, default is `edges.csv`

### Examples

**Count triangles on DBLP dataset using K-Star method with ε=1.0:**
```bash
./LFCK ../data/dblp 3 2 1.0
```

**Count 4-cliques on IMDB dataset using Baseline RR with ε=2.0:**
```bash
./LFCK ../data/IMDB 4 1 2.0
```

**Using a custom edge file:**
```bash
./LFCK ../data/dblp 3 2 1.0 edges_subgraph_10000.csv
```

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

### Example Datasets Included

- **DBLP**: Co-authorship network from DBLP computer science bibliography
- **IMDB**: Actor collaboration network from IMDB
- **MIT**: MIT network data
- **Orkut**: Social network from Orkut

## Output

The program will:
1. Compute the optimal padding parameter for the dataset
2. Run 10 independent experiments
3. Report results for each experiment
4. Output final statistics including:
   - Original clique count
   - Average relative error across experiments
   - Privacy parameters used

### Sample Output

```
Computed optimal padding: 12.5
Detected NodeNum: 317080

=== Running 10 experiments ===
Dataset: ../data/dblp
Target clique: 3-clique
Method: K-Star

--- Experiment 1 ---
...
Experiment 1 completed.

...

=== Final Results ===
Number of experiments: 10
Target clique: 3-clique
Original triangle count: 2224385
Method: K-Star
Optimal padding: 12.5
Average relative error: 0.0234
```


## Python Preprocessing

The `python/IMDB.py` script can be used to preprocess IMDB data:

```bash
cd python
python3 IMDB.py
```

This script:
- Reads the IMDB matrix format data
- Constructs actor collaboration networks
- Outputs edge lists and degree information

## Troubleshooting

### Compilation Errors

- Ensure you have g++ with C++11 support: `g++ --version`
- Check that the include path is correct in the Makefile
- Verify all header files are present in `cpp/include/`

### Runtime Errors

- **"cannot open [filename]"**: Check that the dataset path and edge file exist
- **Memory issues**: Large graphs may require significant RAM; consider using subgraph samples
- **Segmentation fault**: Ensure the edge file format is correct (CSV with integer node IDs)



## Citation

If you use this code in your research, please cite LF-CK.


