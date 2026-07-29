import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Dfs {
    /** Recursive DFS visit order (preorder) from start. */
    public static List<Integer> dfs(List<List<Integer>> adj, int start) {
        boolean[] visited = new boolean[adj.size()];
        List<Integer> order = new ArrayList<>();
        dfsRec(adj, start, visited, order);
        return order;
    }

    private static void dfsRec(List<List<Integer>> adj, int u, boolean[] visited,
                               List<Integer> order) {
        visited[u] = true;
        order.add(u);
        for (int v : adj.get(u)) {
            if (!visited[v]) dfsRec(adj, v, visited, order);
        }
    }

    private static List<List<Integer>> graph(int[][] edges, int n) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        for (int[] e : edges) adj.get(e[0]).add(e[1]);
        return adj;
    }

    public static void main(String[] args) {
        // 0 → 1,2 ; 1 → 3 ; 2 → 3 ; 3 → 4  → order 0,1,3,4,2
        List<List<Integer>> adj = graph(new int[][]{{0,1},{0,2},{1,3},{2,3},{3,4}}, 5);
        List<Integer> order = dfs(adj, 0);
        if (!order.equals(Arrays.asList(0, 1, 3, 4, 2))) {
            System.err.println("unexpected order: " + order);
            System.exit(1);
        }
        order = dfs(adj, 2);
        if (!order.equals(Arrays.asList(2, 3, 4))) {
            System.err.println("unexpected order from 2: " + order);
            System.exit(1);
        }
        System.out.println("dfs: ok");
    }
}
