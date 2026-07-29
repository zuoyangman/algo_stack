/// Recursive DFS visit order (preorder) from start.
fn dfs(adj: &[Vec<usize>], start: usize) -> Vec<usize> {
    let mut visited = vec![false; adj.len()];
    let mut order = Vec::new();
    dfs_rec(adj, start, &mut visited, &mut order);
    order
}

fn dfs_rec(adj: &[Vec<usize>], u: usize, visited: &mut [bool], order: &mut Vec<usize>) {
    visited[u] = true;
    order.push(u);
    for &v in &adj[u] {
        if !visited[v] {
            dfs_rec(adj, v, visited, order);
        }
    }
}

fn main() {
    // 0 → 1,2 ; 1 → 3 ; 2 → 3 ; 3 → 4  → order 0,1,3,4,2
    let adj: Vec<Vec<usize>> = vec![vec![1, 2], vec![3], vec![3], vec![4], vec![]];
    assert_eq!(dfs(&adj, 0), vec![0, 1, 3, 4, 2]);
    assert_eq!(dfs(&adj, 2), vec![2, 3, 4]);
    println!("dfs: ok");
}
