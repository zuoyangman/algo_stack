#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class Stack {
    std::vector<char> data_;

public:
    void push(char x) { data_.push_back(x); }

    char pop() {
        if (is_empty()) {
            throw std::runtime_error("empty");
        }
        char v = data_.back();
        data_.pop_back();
        return v;
    }

    char peek() const {
        if (is_empty()) {
            throw std::runtime_error("empty");
        }
        return data_.back();
    }

    bool is_empty() const { return data_.empty(); }
};

static bool is_balanced(const std::string& s) {
    Stack st;
    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') {
            st.push(c);
        } else if (c == ')' || c == ']' || c == '}') {
            if (st.is_empty()) {
                return false;
            }
            char open = st.pop();
            if ((c == ')' && open != '(') || (c == ']' && open != '[') ||
                (c == '}' && open != '{')) {
                return false;
            }
        }
    }
    return st.is_empty();
}

static std::string reverse(const std::string& s) {
    Stack st;
    for (char c : s) {
        st.push(c);
    }
    std::string out;
    while (!st.is_empty()) {
        out.push_back(st.pop());
    }
    return out;
}

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

int main() {
    Stack st;
    expect(st.is_empty(), "empty initially");
    st.push('a');
    st.push('b');
    expect(st.peek() == 'b', "peek");
    expect(st.pop() == 'b', "pop");
    expect(st.pop() == 'a', "pop a");
    expect(st.is_empty(), "empty after pops");
    expect(is_balanced("({[]})"), "balanced");
    expect(!is_balanced("([)]"), "unbalanced");
    expect(reverse("abc") == "cba", "reverse");
    std::cout << "stack: ok\n";
    return 0;
}
