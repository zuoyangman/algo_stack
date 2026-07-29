import java.util.HashMap;
import java.util.Map;

public class Trie {
    private static class Node {
        Map<Character, Node> next = new HashMap<>();
        boolean end;
    }

    private final Node root = new Node();

    public void insert(String word) {
        Node cur = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            cur = cur.next.computeIfAbsent(c, k -> new Node());
        }
        cur.end = true;
    }

    public boolean search(String word) {
        Node cur = walk(word);
        return cur != null && cur.end;
    }

    public boolean startsWith(String prefix) {
        return walk(prefix) != null;
    }

    private Node walk(String s) {
        Node cur = root;
        for (int i = 0; i < s.length(); i++) {
            cur = cur.next.get(s.charAt(i));
            if (cur == null) return null;
        }
        return cur;
    }

    public static void main(String[] args) {
        Trie t = new Trie();
        t.insert("apple");
        if (!t.search("apple") || t.search("app") || !t.startsWith("app")) {
            System.err.println("apple/app checks failed");
            System.exit(1);
        }
        t.insert("app");
        if (!t.search("app") || !t.startsWith("ap") || t.search("appl")) {
            System.err.println("after insert app failed");
            System.exit(1);
        }
        if (t.search("") || t.startsWith("b")) {
            System.err.println("edge checks failed");
            System.exit(1);
        }
        System.out.println("trie: ok");
    }
}
