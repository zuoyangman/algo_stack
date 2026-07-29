import java.util.ArrayList;

public class StackDemo {
    static final class Stack {
        private final ArrayList<Character> data = new ArrayList<>();

        void push(char x) {
            data.add(x);
        }

        char pop() {
            if (isEmpty()) {
                throw new IllegalStateException("empty");
            }
            return data.remove(data.size() - 1);
        }

        char peek() {
            if (isEmpty()) {
                throw new IllegalStateException("empty");
            }
            return data.get(data.size() - 1);
        }

        boolean isEmpty() {
            return data.isEmpty();
        }
    }

    static boolean isBalanced(String s) {
        Stack st = new Stack();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c == '{') {
                st.push(c);
            } else if (c == ')' || c == ']' || c == '}') {
                if (st.isEmpty()) {
                    return false;
                }
                char open = st.pop();
                if ((c == ')' && open != '(') || (c == ']' && open != '[') || (c == '}' && open != '{')) {
                    return false;
                }
            }
        }
        return st.isEmpty();
    }

    static String reverse(String s) {
        Stack st = new Stack();
        for (int i = 0; i < s.length(); i++) {
            st.push(s.charAt(i));
        }
        StringBuilder sb = new StringBuilder();
        while (!st.isEmpty()) {
            sb.append(st.pop());
        }
        return sb.toString();
    }

    static void expect(boolean cond, String msg) {
        if (!cond) {
            System.err.println("FAIL: " + msg);
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        Stack st = new Stack();
        expect(st.isEmpty(), "empty initially");
        st.push('a');
        st.push('b');
        expect(st.peek() == 'b', "peek");
        expect(st.pop() == 'b', "pop");
        expect(st.pop() == 'a', "pop a");
        expect(st.isEmpty(), "empty after pops");
        expect(isBalanced("({[]})"), "balanced");
        expect(!isBalanced("([)]"), "unbalanced");
        expect(reverse("abc").equals("cba"), "reverse");
        System.out.println("stack: ok");
    }
}
