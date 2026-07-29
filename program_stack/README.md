# program_stack

A growing, didactic collection of **classic computer-science algorithms and
data structures**, each shipping with:

- **README** — usage & public API for every language
- **PRINCIPLE** — math / invariants / complexity
- **EXTENSION** — how to extend safely
- **Original reference implementations** in **Java**, **C++**, **Rust**, and **Go**

No language-specific framework magic in the core logic — each demo is a small,
self-checking program you can compile and run with the stock toolchain.

---

## Repository layout

```
program_stack/
├── README.md
├── docs/
│   ├── CONVENTIONS.md
│   ├── ROADMAP.md
│   └── templates/
├── scripts/
│   ├── run.sh            # run one algo in one language
│   └── test_all.sh       # compile+run every demo
└── algorithms/
    ├── sorting/
    ├── searching/
    ├── data_structures/
    ├── graphs/
    ├── dynamic_programming/
    └── strings/
```

Every algorithm folder follows the same convention
(see [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md)):

```
algorithms/<category>/<name>/
├── README.md
├── PRINCIPLE.md
├── EXTENSION.md
├── java/   *.java
├── cpp/    *.cpp
├── rust/   main.rs
└── go/     main.go
```

---

## Quick start

Requirements: JDK 17+, g++ (C++17), rustc, go 1.20+.

```bash
cd program_stack

# Run one algorithm in one language
./scripts/run.sh sorting/bubble_sort java
./scripts/run.sh sorting/bubble_sort cpp
./scripts/run.sh sorting/bubble_sort rust
./scripts/run.sh sorting/bubble_sort go

# Compile & run every demo (smoke test)
./scripts/test_all.sh
```

---

## Currently implemented (20+)

| # | Category | Algorithm | Path |
| - | -------- | --------- | ---- |
| 1 | Sorting | Bubble Sort | `algorithms/sorting/bubble_sort` |
| 2 | Sorting | Insertion Sort | `algorithms/sorting/insertion_sort` |
| 3 | Sorting | Selection Sort | `algorithms/sorting/selection_sort` |
| 4 | Sorting | Merge Sort | `algorithms/sorting/merge_sort` |
| 5 | Sorting | Quick Sort | `algorithms/sorting/quick_sort` |
| 6 | Sorting | Heap Sort | `algorithms/sorting/heap_sort` |
| 7 | Searching | Linear Search | `algorithms/searching/linear_search` |
| 8 | Searching | Binary Search | `algorithms/searching/binary_search` |
| 9 | Data structures | Singly Linked List | `algorithms/data_structures/linked_list` |
| 10 | Data structures | Stack | `algorithms/data_structures/stack` |
| 11 | Data structures | Queue | `algorithms/data_structures/queue` |
| 12 | Data structures | Binary Search Tree | `algorithms/data_structures/binary_search_tree` |
| 13 | Data structures | Hash Table (chaining) | `algorithms/data_structures/hash_table` |
| 14 | Data structures | Union–Find (DSU) | `algorithms/data_structures/union_find` |
| 15 | Graphs | BFS | `algorithms/graphs/bfs` |
| 16 | Graphs | DFS | `algorithms/graphs/dfs` |
| 17 | Graphs | Dijkstra | `algorithms/graphs/dijkstra` |
| 18 | Graphs | Topological Sort | `algorithms/graphs/topological_sort` |
| 19 | Dynamic programming | 0/1 Knapsack | `algorithms/dynamic_programming/knapsack_01` |
| 20 | Dynamic programming | LCS | `algorithms/dynamic_programming/lcs` |
| 21 | Strings | KMP | `algorithms/strings/kmp` |
| 22 | Strings | Trie | `algorithms/strings/trie` |

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the planned expansion list.

---

## Contributing a new algorithm

1. Read [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).
2. `mkdir -p algorithms/<category>/<name>/{java,cpp,rust,go}`
3. Copy templates from `docs/templates/` and fill them in.
4. Implement the same algorithm in all four languages with a self-checking `main`.
5. `./scripts/run.sh <category>/<name> {java,cpp,rust,go}`
6. Update `docs/ROADMAP.md` and the table above.
