import java.util.Arrays;

public class LinkedListDemo {
    static final class Node {
        int value;
        Node next;

        Node(int value) {
            this.value = value;
        }
    }

    static final class LinkedList {
        private Node head;
        private Node tail;
        private int size;

        void pushFront(int x) {
            Node n = new Node(x);
            n.next = head;
            head = n;
            if (tail == null) {
                tail = n;
            }
            size++;
        }

        void pushBack(int x) {
            Node n = new Node(x);
            if (tail == null) {
                head = tail = n;
            } else {
                tail.next = n;
                tail = n;
            }
            size++;
        }

        int popFront() {
            if (head == null) {
                throw new IllegalStateException("empty");
            }
            int v = head.value;
            head = head.next;
            if (head == null) {
                tail = null;
            }
            size--;
            return v;
        }

        boolean find(int x) {
            for (Node cur = head; cur != null; cur = cur.next) {
                if (cur.value == x) {
                    return true;
                }
            }
            return false;
        }

        int[] toArray() {
            int[] out = new int[size];
            int i = 0;
            for (Node cur = head; cur != null; cur = cur.next) {
                out[i++] = cur.value;
            }
            return out;
        }

        int size() {
            return size;
        }
    }

    static void expect(boolean cond, String msg) {
        if (!cond) {
            System.err.println("FAIL: " + msg);
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        LinkedList list = new LinkedList();
        list.pushBack(2);
        list.pushFront(1);
        list.pushBack(3);
        expect(Arrays.equals(list.toArray(), new int[] {1, 2, 3}), "values after pushes");
        expect(list.find(2), "find 2");
        expect(!list.find(9), "missing 9");
        expect(list.popFront() == 1, "pop_front");
        expect(Arrays.equals(list.toArray(), new int[] {2, 3}), "after pop");
        expect(list.size() == 2, "size");
        System.out.println("linked_list: ok");
    }
}
