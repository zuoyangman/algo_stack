use std::collections::VecDeque;

/// Kahn's algorithm. Returns None if the graph has a cycle.
fn topological_sort(adj: &[Vec<usize>]) -> Option<Vec<usize>> {
    let n = adj.len();
    let mut indeg = vec![0usize; n];
    for u in 0..n {
        for &v in &adj[u] {
            indeg[v] += 1;
        }
    }
    let mut q = VecDeque::new();
    for i in 0..n {
        if indeg[i] == 0 {
            q.push_back(i);
        }
    }
    let mut order = Vec::with_capacity(n);
    while let Some(u) = q.pop_front() {
        order.push(u);
        for &v in &adj[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                q.push_back(v);
            }
        }
    }
    if order.len() == n {
        Some(order)
    } else {
        None
    }
}

fn main() {
    let adj: Vec<Vec<usize>> = vec![vec![1, 2], vec![3], vec![3], vec![]];
    assert_eq!(topological_sort(&adj), Some(vec![0, 1, 2, 3]));
    let cyclic: Vec<Vec<usize>> = vec![vec![1], vec![0]];
    assert_eq!(topological_sort(&cyclic), None);
    println!("topological_sort: ok");
}
