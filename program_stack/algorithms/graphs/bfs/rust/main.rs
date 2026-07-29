use std::collections::VecDeque;
use std::process;

/// BFS visit order from start on an adjacency-list graph.
fn bfs(adj: &[Vec<usize>], start: usize) -> Vec<usize> {
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut q = VecDeque::new();
    visited[start] = true;
    q.push_back(start);
    while let Some(u) = q.pop_front() {
        order.push(u);
        for &v in &adj[u] {
            if !visited[v] {
                visited[v] = true;
                q.push_back(v);
            }
        }
    }
    order
}

fn main() {
    // 0 → 1,2 ; 1 → 3 ; 2 → 3 ; 3 → 4
    let adj: Vec<Vec<usize>> = vec![vec![1, 2], vec![3], vec![3], vec![4], vec![]];
    let order = bfs(&adj, 0);
    assert_eq!(order, vec![0, 1, 2, 3, 4]);
    let order = bfs(&adj, 3);
    assert_eq!(order, vec![3, 4]);
    println!("bfs: ok");
    let _ = process::exit(0);
}
