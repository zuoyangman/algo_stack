import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class TopologicalSort {
    /**
     * Kahn's algorithm. Returns a valid order, or null if the graph has a cycle.
     */
    public static List<Integer> topologicalSort(List<List<Integer>> adj) {
        int n = adj.size();
        int[] indeg = new int[n];
        for (int u = 0; u < n; u++) {
            for (int v : adj.get(u)) indeg[v]++;
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < n; i++) if (indeg[i] == 0) q.add(i);
        List<Integer> order = new ArrayList<>();
        while (!q.isEmpty()) {
            int u = q.poll();
            order.add(u);
            for (int v : adj.get(u)) {
                if (--indeg[v] == 0) q.add(v);
            }
        }
        return order.size() == n ? order : null;
    }

    private static List<List<Integer>> graph(int[][] edges, int n) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        for (int[] e : edges) adj.get(e[0]).add(e[1]);
        return adj;
    }

    public static void main(String[] args) {
        // 0→1, 0→2, 1→3, 2→3  → [0,1,2,3]
        List<Integer> order = topologicalSort(graph(new int[][]{{0,1},{0,2},{1,3},{2,3}}, 4));
        if (order == null || !order.equals(Arrays.asList(0, 1, 2, 3))) {
            System.err.println("unexpected order: " + order);
            System.exit(1);
        }
        // cycle 0→1→0
        order = topologicalSort(graph(new int[][]{{0,1},{1,0}}, 2));
        if (order != null) {
            System.err.println("expected cycle detection failure");
            System.exit(1);
        }
        System.out.println("topological_sort: ok");
    }
}
