public class QueueDemo {
    static final class Node {
        int value;
        Node next;

        Node(int value) {
            this.value = value;
        }
    }

    static final class Queue {
        private Node head;
        private Node tail;

        void enqueue(int x) {
            Node n = new Node(x);
            if (tail == null) {
                head = tail = n;
            } else {
                tail.next = n;
                tail = n;
            }
        }

        int dequeue() {
            if (head == null) {
                throw new IllegalStateException("empty");
            }
            int v = head.value;
            head = head.next;
            if (head == null) {
                tail = null;
            }
            return v;
        }

        int front() {
            if (head == null) {
                throw new IllegalStateException("empty");
            }
            return head.value;
        }

        boolean isEmpty() {
            return head == null;
        }
    }

    static void expect(boolean cond, String msg) {
        if (!cond) {
            System.err.println("FAIL: " + msg);
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        Queue q = new Queue();
        expect(q.isEmpty(), "empty initially");
        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);
        expect(q.front() == 1, "front");
        expect(q.dequeue() == 1, "dequeue 1");
        expect(q.dequeue() == 2, "dequeue 2");
        q.enqueue(4);
        expect(q.front() == 3, "front 3");
        expect(q.dequeue() == 3, "dequeue 3");
        expect(q.dequeue() == 4, "dequeue 4");
        expect(q.isEmpty(), "empty finally");
        System.out.println("queue: ok");
    }
}
