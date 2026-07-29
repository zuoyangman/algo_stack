#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <queue>
#include <vector>

// Kahn's algorithm. nullopt if the graph has a cycle.
std::optional<std::vector<int>> topologicalSort(const std::vector<std::vector<int>>& adj) {
    int n = static_cast<int>(adj.size());
    std::vector<int> indeg(n, 0);
    for (int u = 0; u < n; ++u)
        for (int v : adj[u]) ++indeg[v];
    std::queue<int> q;
    for (int i = 0; i < n; ++i)
        if (indeg[i] == 0) q.push(i);
    std::vector<int> order;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        order.push_back(u);
        for (int v : adj[u]) {
            if (--indeg[v] == 0) q.push(v);
        }
    }
    if (static_cast<int>(order.size()) != n) return std::nullopt;
    return order;
}

int main() {
    std::vector<std::vector<int>> adj = {{1, 2}, {3}, {3}, {}};
    auto order = topologicalSort(adj);
    std::vector<int> expect = {0, 1, 2, 3};
    if (!order || *order != expect) {
        std::cerr << "unexpected order\n";
        return 1;
    }
    adj = {{1}, {0}};
    order = topologicalSort(adj);
    if (order) {
        std::cerr << "expected cycle detection failure\n";
        return 1;
    }
    std::cout << "topological_sort: ok\n";
    return 0;
}
