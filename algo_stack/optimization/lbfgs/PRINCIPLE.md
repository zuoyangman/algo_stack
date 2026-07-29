# L-BFGS — Principle

## 1. Problem statement

Minimise a smooth scalar objective `f: ℝⁿ → ℝ` given access to `∇f`. Newton's
method uses `p = −(∇²f)⁻¹ ∇f`, which is expensive and requires the Hessian.
Quasi-Newton methods build a cheap approximation `H_k ≈ (∇²f)⁻¹` from gradient
differences alone.

## 2. BFGS / L-BFGS update

Given step `s_k = x_{k+1} − x_k` and gradient change `y_k = ∇f_{k+1} − ∇f_k`,
the BFGS inverse-Hessian update satisfies the secant equation `H_{k+1} y_k = s_k`.

**L-BFGS** never stores `H` explicitly. It keeps the last `m` pairs `(s_i, y_i)`
and applies the **two-loop recursion** to compute `H_k ∇f` in `O(m n)` time.

## 3. Strong Wolfe line search

Along direction `p`, choose step length `α` such that:

```
f(x + αp) ≤ f(x) + c₁ α ∇f·p          (Armijo / sufficient decrease)
|∇f(x + αp)·p| ≤ c₂ |∇f·p|             (curvature)
```

Typical constants: `c₁ = 10⁻⁴`, `c₂ = 0.9` (quasi-Newton).

## 4. Algorithm

```
x ← x0; evaluate f, g
for k = 1 .. max_iter:
    if ||g||_∞ < tol: stop
    p ← − two_loop_recursion(g, history)
    α ← wolfe_line_search(f, ∇f, x, p)
    s ← α p;  x ← x + s
    y ← ∇f(x) − g;  g ← ∇f(x)
    if y·s > 0: push (s, y) into history (drop oldest if |history| > m)
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Per iteration | `O(m n)` + line-search evaluations |
| Memory | `O(m n)` for history pairs |

## 6. Notes

- The initial Hessian scale uses `γ = (yᵀs)/(yᵀy)`.
- Pairs with `yᵀs ≤ 0` are skipped (curvature condition fails).
- This implementation is didactic; production codes add more safeguards
  (damped BFGS, More–Thuente line search, boxed constraints, …).
