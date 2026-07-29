#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

static void dfsRec(const std::vector<std::vector<int>>& adj, int u,
                   std::vector<char>& visited, std::vector<int>& order) {
    visited[u] = 1;
    order.push_back(u);
    for (int v : adj[u]) {
        if (!visited[v]) dfsRec(adj, v, visited, order);
    }
}

// Recursive DFS visit order (preorder) from start.
std::vector<int> dfs(const std::vector<std::vector<int>>& adj, int start) {
    std::vector<char> visited(adj.size(), 0);
    std::vector<int> order;
    dfsRec(adj, start, visited, order);
    return order;
}

int main() {
    // 0 → 1,2 ; 1 → 3 ; 2 → 3 ; 3 → 4  → order 0,1,3,4,2
    std::vector<std::vector<int>> adj = {{1, 2}, {3}, {3}, {4}, {}};
    auto order = dfs(adj, 0);
    std::vector<int> expect = {0, 1, 3, 4, 2};
    if (order != expect) {
        std::cerr << "unexpected order\n";
        return 1;
    }
    order = dfs(adj, 2);
    expect = {2, 3, 4};
    if (order != expect) {
        std::cerr << "unexpected order from 2\n";
        return 1;
    }
    std::cout << "dfs: ok\n";
    return 0;
}
