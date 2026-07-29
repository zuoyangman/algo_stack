public class UnionFindDemo {
    static final class UnionFind {
        private final int[] parent;
        private final int[] rank;

        UnionFind(int n) {
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }

        int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }

        boolean union(int a, int b) {
            int ra = find(a);
            int rb = find(b);
            if (ra == rb) {
                return false;
            }
            if (rank[ra] < rank[rb]) {
                parent[ra] = rb;
            } else if (rank[ra] > rank[rb]) {
                parent[rb] = ra;
            } else {
                parent[rb] = ra;
                rank[ra]++;
            }
            return true;
        }

        boolean connected(int a, int b) {
            return find(a) == find(b);
        }
    }

    static void expect(boolean cond, String msg) {
        if (!cond) {
            System.err.println("FAIL: " + msg);
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        UnionFind uf = new UnionFind(6);
        expect(!uf.connected(0, 1), "initially separate");
        expect(uf.union(0, 1), "union 0-1");
        expect(uf.union(1, 2), "union 1-2");
        expect(uf.connected(0, 2), "0 connected 2");
        expect(!uf.connected(0, 3), "0 not 3");
        expect(uf.union(3, 4), "union 3-4");
        expect(uf.union(2, 4), "merge components");
        expect(uf.connected(0, 3), "0 connected 3");
        expect(!uf.union(0, 2), "already united");
        expect(uf.find(5) == 5, "singleton 5");
        System.out.println("union_find: ok");
    }
}
