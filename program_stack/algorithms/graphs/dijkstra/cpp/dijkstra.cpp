#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

constexpr int INF = 1'000'000'000;

// O(V^2) Dijkstra. edges: {u, v, w}. Unreachable = INF.
std::vector<int> dijkstra(int n, const std::vector<std::vector<int>>& edges, int source) {
    std::vector<std::vector<int>> w(n, std::vector<int>(n, -1));
    for (const auto& e : edges) w[e[0]][e[1]] = e[2];

    std::vector<int> dist(n, INF);
    std::vector<char> done(n, 0);
    dist[source] = 0;

    for (int iter = 0; iter < n; ++iter) {
        int u = -1;
        for (int i = 0; i < n; ++i) {
            if (!done[i] && (u < 0 || dist[i] < dist[u])) u = i;
        }
        if (u < 0 || dist[u] == INF) break;
        done[u] = 1;
        for (int v = 0; v < n; ++v) {
            if (w[u][v] >= 0 && dist[u] + w[u][v] < dist[v]) {
                dist[v] = dist[u] + w[u][v];
            }
        }
    }
    return dist;
}

int main() {
    std::vector<std::vector<int>> edges = {
        {0, 1, 1}, {0, 2, 4}, {0, 3, 2}, {1, 2, 1}, {3, 2, 1}};
    auto dist = dijkstra(4, edges, 0);
    std::vector<int> expect = {0, 1, 2, 2};
    if (dist != expect) {
        std::cerr << "unexpected dist\n";
        return 1;
    }
    dist = dijkstra(3, {{0, 1, 5}}, 0);
    if (dist[0] != 0 || dist[1] != 5 || dist[2] != INF) {
        std::cerr << "unexpected unreachable case\n";
        return 1;
    }
    std::cout << "dijkstra: ok\n";
    return 0;
}
