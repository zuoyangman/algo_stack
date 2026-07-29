import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BinarySearchTreeDemo {
    static final class Node {
        int key;
        Node left;
        Node right;

        Node(int key) {
            this.key = key;
        }
    }

    static final class BST {
        private Node root;

        void insert(int x) {
            root = insertRec(root, x);
        }

        private Node insertRec(Node n, int x) {
            if (n == null) {
                return new Node(x);
            }
            if (x < n.key) {
                n.left = insertRec(n.left, x);
            } else if (x > n.key) {
                n.right = insertRec(n.right, x);
            }
            return n;
        }

        boolean contains(int x) {
            Node cur = root;
            while (cur != null) {
                if (x == cur.key) {
                    return true;
                }
                cur = x < cur.key ? cur.left : cur.right;
            }
            return false;
        }

        int[] inorder() {
            List<Integer> out = new ArrayList<>();
            inorderRec(root, out);
            int[] arr = new int[out.size()];
            for (int i = 0; i < out.size(); i++) {
                arr[i] = out.get(i);
            }
            return arr;
        }

        private void inorderRec(Node n, List<Integer> out) {
            if (n == null) {
                return;
            }
            inorderRec(n.left, out);
            out.add(n.key);
            inorderRec(n.right, out);
        }
    }

    static void expect(boolean cond, String msg) {
        if (!cond) {
            System.err.println("FAIL: " + msg);
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        BST t = new BST();
        int[] vals = {5, 3, 7, 1, 4, 6, 8};
        for (int v : vals) {
            t.insert(v);
        }
        t.insert(5); // duplicate ignored
        expect(t.contains(4), "contains 4");
        expect(!t.contains(2), "missing 2");
        expect(Arrays.equals(t.inorder(), new int[] {1, 3, 4, 5, 6, 7, 8}), "inorder");
        System.out.println("binary_search_tree: ok");
    }
}
