# Directory & Coding Conventions

Every algorithm in `program_stack` follows the **same layout and same
multi-language contract**. A reader who understood one module can predict
exactly where to find things in any other.

---

## 1. Directory layout per algorithm

```
algorithms/<category>/<algorithm_name>/
├── README.md              # quick-start, API per language, how to run
├── PRINCIPLE.md           # idea, correctness argument, complexity
├── EXTENSION.md           # variants, pitfalls, secondary-dev guide
├── java/                  # *.java  (entry: public class with main)
├── cpp/                   # *.cpp   (entry: int main)
├── rust/                  # main.rs (entry: fn main)
└── go/                    # *.go    (package main; func main)
```

`<category>` is one of:

- `sorting`
- `searching`
- `data_structures`
- `graphs`
- `dynamic_programming`
- `strings`
- `math` *(extend as needed)*
- `misc` *(extend as needed)*

`<algorithm_name>` is **lower_snake_case** and matches the canonical name
(e.g. `binary_search`, not `binSearch`).

---

## 2. The three documentation files

### `README.md` — usage-facing
- One-sentence description.
- How to run each language via `../../scripts/run.sh …` (or relative path).
- Public API sketch (function / type names) — keep names **aligned across
  languages** where idiomatic.
- Tiny usage example (pseudocode or one language).

### `PRINCIPLE.md` — theory-facing
- Problem statement.
- Algorithm idea / invariants.
- Correctness sketch.
- Time & space complexity.
- Language-agnostic notes (e.g. in-place vs. out-of-place).

### `EXTENSION.md` — developer-facing
- What is safe to change (API surface).
- Common variants (stable sort, 3-way partition, iterative DFS, …).
- Pitfalls (integer overflow, off-by-one, mutability).
- Suggested follow-up algorithms.

Use [`docs/templates/`](templates/) as starting points.

---

## 3. Implementation rules (all four languages)

### Behaviour
- Each language directory must contain a **runnable program** with `main`
  that exercises the algorithm on a few fixed cases and **exits non-zero on
  failure** (assertions / explicit checks).
- Print a short success line, e.g. `bubble_sort: ok`, so `test_all.sh` can
  grep for it.
- Prefer clarity over micro-optimisations; comments explain non-obvious
  steps.
- Do **not** pull in heavy third-party libraries.

### Naming alignment (guideline)

| Concept | Java | C++ | Rust | Go |
| ------- | ---- | --- | ---- | -- |
| Sort fn | `sort(arr)` | `sort(a)` | `sort(&mut a)` | `Sort(a)` |
| Search  | `search(arr, x)` | `search(a, x)` | `search(&a, x)` | `Search(a, x)` |

Exported / documented identifiers should be easy to map across languages.

### Language specifics

- **Java**: one public top-level class per file matching the file name;
  use a package-less demo for simplicity (`javac` from that directory).
- **C++**: C++17; `#include <bits>` is discouraged — use standard headers;
  compile with `-std=c++17 -O0 -Wall`.
- **Rust**: single `main.rs` compilable with `rustc main.rs -o out`
  (no Cargo required for demos); use `assert!` / `assert_eq!`.
- **Go**: `package main`; prefer `go run main.go` from the `go/` directory.
  A tiny `go.mod` (`module …`) is optional but recommended so tooling behaves
  well inside a parent git repo.

---

## 4. Adding a new algorithm — checklist

1. Create the folder skeleton (docs + four language dirs).
2. Fill README / PRINCIPLE / EXTENSION.
3. Implement + self-check in Java, C++, Rust, Go.
4. `./scripts/run.sh <category>/<name> {java,cpp,rust,go}`
5. Update `docs/ROADMAP.md` and the main README table.
