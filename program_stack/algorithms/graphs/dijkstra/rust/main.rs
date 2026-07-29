const INF: i32 = 1_000_000_000;

/// O(V^2) Dijkstra. edges: (u, v, w). Unreachable = INF.
fn dijkstra(n: usize, edges: &[[i32; 3]], source: usize) -> Vec<i32> {
    let mut w = vec![vec![-1_i32; n]; n];
    for e in edges {
        w[e[0] as usize][e[1] as usize] = e[2];
    }
    let mut dist = vec![INF; n];
    let mut done = vec![false; n];
    dist[source] = 0;

    for _ in 0..n {
        let mut u: Option<usize> = None;
        for i in 0..n {
            if !done[i] && u.map_or(true, |j| dist[i] < dist[j]) {
                u = Some(i);
            }
        }
        let u = match u {
            Some(u) if dist[u] != INF => u,
            _ => break,
        };
        done[u] = true;
        for v in 0..n {
            if w[u][v] >= 0 && dist[u] + w[u][v] < dist[v] {
                dist[v] = dist[u] + w[u][v];
            }
        }
    }
    dist
}

fn main() {
    let edges = [[0, 1, 1], [0, 2, 4], [0, 3, 2], [1, 2, 1], [3, 2, 1]];
    assert_eq!(dijkstra(4, &edges, 0), vec![0, 1, 2, 2]);
    let dist = dijkstra(3, &[[0, 1, 5]], 0);
    assert_eq!(dist[0], 0);
    assert_eq!(dist[1], 5);
    assert_eq!(dist[2], INF);
    println!("dijkstra: ok");
}
