import java.util.Arrays;

public class Dijkstra {
    public static final int INF = 1_000_000_000;

    /**
     * O(V^2) Dijkstra. edges[i] = {u, v, w}. Returns dist[]; unreachable = INF.
     */
    public static int[] dijkstra(int n, int[][] edges, int source) {
        int[][] w = new int[n][n];
        for (int i = 0; i < n; i++) Arrays.fill(w[i], -1);
        for (int[] e : edges) w[e[0]][e[1]] = e[2];

        int[] dist = new int[n];
        Arrays.fill(dist, INF);
        boolean[] done = new boolean[n];
        dist[source] = 0;

        for (int iter = 0; iter < n; iter++) {
            int u = -1;
            for (int i = 0; i < n; i++) {
                if (!done[i] && (u < 0 || dist[i] < dist[u])) u = i;
            }
            if (u < 0 || dist[u] == INF) break;
            done[u] = true;
            for (int v = 0; v < n; v++) {
                if (w[u][v] >= 0 && dist[u] + w[u][v] < dist[v]) {
                    dist[v] = dist[u] + w[u][v];
                }
            }
        }
        return dist;
    }

    public static void main(String[] args) {
        // 0-1:1, 0-2:4, 0-3:2, 1-2:1, 3-2:1  → dist [0,1,2,2]
        int[][] edges = {{0,1,1},{0,2,4},{0,3,2},{1,2,1},{3,2,1}};
        int[] dist = dijkstra(4, edges, 0);
        if (!Arrays.equals(dist, new int[]{0, 1, 2, 2})) {
            System.err.println("unexpected dist: " + Arrays.toString(dist));
            System.exit(1);
        }
        // unreachable node 4 with no edges
        dist = dijkstra(3, new int[][]{{0,1,5}}, 0);
        if (dist[0] != 0 || dist[1] != 5 || dist[2] != INF) {
            System.err.println("unexpected unreachable case: " + Arrays.toString(dist));
            System.exit(1);
        }
        System.out.println("dijkstra: ok");
    }
}
