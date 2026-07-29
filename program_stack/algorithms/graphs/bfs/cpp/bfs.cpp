#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <vector>

// BFS visit order from start on an adjacency-list graph.
std::vector<int> bfs(const std::vector<std::vector<int>>& adj, int start) {
    int n = static_cast<int>(adj.size());
    std::vector<char> visited(n, 0);
    std::vector<int> order;
    std::queue<int> q;
    visited[start] = 1;
    q.push(start);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        order.push_back(u);
        for (int v : adj[u]) {
            if (!visited[v]) {
                visited[v] = 1;
                q.push(v);
            }
        }
    }
    return order;
}

int main() {
    // 0 → 1,2 ; 1 → 3 ; 2 → 3 ; 3 → 4
    std::vector<std::vector<int>> adj = {{1, 2}, {3}, {3}, {4}, {}};
    auto order = bfs(adj, 0);
    std::vector<int> expect = {0, 1, 2, 3, 4};
    if (order != expect) {
        std::cerr << "unexpected order\n";
        return 1;
    }
    order = bfs(adj, 3);
    expect = {3, 4};
    if (order != expect) {
        std::cerr << "unexpected order from 3\n";
        return 1;
    }
    std::cout << "bfs: ok\n";
    return 0;
}
