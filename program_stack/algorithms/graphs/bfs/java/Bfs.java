import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class Bfs {
    /** BFS visit order from start on an adjacency-list graph. */
    public static List<Integer> bfs(List<List<Integer>> adj, int start) {
        int n = adj.size();
        boolean[] visited = new boolean[n];
        List<Integer> order = new ArrayList<>();
        Queue<Integer> q = new LinkedList<>();
        visited[start] = true;
        q.add(start);
        while (!q.isEmpty()) {
            int u = q.poll();
            order.add(u);
            for (int v : adj.get(u)) {
                if (!visited[v]) {
                    visited[v] = true;
                    q.add(v);
                }
            }
        }
        return order;
    }

    private static List<List<Integer>> graph(int[][] edges, int n) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        for (int[] e : edges) adj.get(e[0]).add(e[1]);
        return adj;
    }

    public static void main(String[] args) {
        // 0 → 1,2 ; 1 → 3 ; 2 → 3 ; 3 → 4
        List<List<Integer>> adj = graph(new int[][]{{0,1},{0,2},{1,3},{2,3},{3,4}}, 5);
        List<Integer> order = bfs(adj, 0);
        if (!order.equals(Arrays.asList(0, 1, 2, 3, 4))) {
            System.err.println("unexpected order: " + order);
            System.exit(1);
        }
        // start at 3 → only 3,4
        order = bfs(adj, 3);
        if (!order.equals(Arrays.asList(3, 4))) {
            System.err.println("unexpected order from 3: " + order);
            System.exit(1);
        }
        System.out.println("bfs: ok");
    }
}
