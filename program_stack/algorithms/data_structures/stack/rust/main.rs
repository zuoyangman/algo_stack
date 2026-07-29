struct Stack {
    data: Vec<char>,
}

impl Stack {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn push(&mut self, x: char) {
        self.data.push(x);
    }

    fn pop(&mut self) -> Option<char> {
        self.data.pop()
    }

    fn peek(&self) -> Option<char> {
        self.data.last().copied()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

fn is_balanced(s: &str) -> bool {
    let mut st = Stack::new();
    for c in s.chars() {
        match c {
            '(' | '[' | '{' => st.push(c),
            ')' | ']' | '}' => {
                let Some(open) = st.pop() else {
                    return false;
                };
                let ok = matches!(
                    (open, c),
                    ('(', ')') | ('[', ']') | ('{', '}')
                );
                if !ok {
                    return false;
                }
            }
            _ => {}
        }
    }
    st.is_empty()
}

fn reverse(s: &str) -> String {
    let mut st = Stack::new();
    for c in s.chars() {
        st.push(c);
    }
    let mut out = String::new();
    while let Some(c) = st.pop() {
        out.push(c);
    }
    out
}

fn main() {
    let mut st = Stack::new();
    assert!(st.is_empty());
    st.push('a');
    st.push('b');
    assert_eq!(st.peek(), Some('b'));
    assert_eq!(st.pop(), Some('b'));
    assert_eq!(st.pop(), Some('a'));
    assert!(st.is_empty());
    assert!(is_balanced("({[]})"));
    assert!(!is_balanced("([)]"));
    assert_eq!(reverse("abc"), "cba");
    println!("stack: ok");
}
