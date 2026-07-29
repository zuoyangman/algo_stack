use std::collections::HashMap;

struct Node {
    children: HashMap<char, Box<Node>>,
    end: bool,
}

impl Node {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            end: false,
        }
    }
}

struct Trie {
    root: Node,
}

impl Trie {
    fn new() -> Self {
        Self { root: Node::new() }
    }

    fn insert(&mut self, word: &str) {
        let mut cur = &mut self.root;
        for c in word.chars() {
            cur = cur
                .children
                .entry(c)
                .or_insert_with(|| Box::new(Node::new()));
        }
        cur.end = true;
    }

    fn search(&self, word: &str) -> bool {
        match self.walk(word) {
            Some(n) => n.end,
            None => false,
        }
    }

    fn starts_with(&self, prefix: &str) -> bool {
        self.walk(prefix).is_some()
    }

    fn walk(&self, s: &str) -> Option<&Node> {
        let mut cur = &self.root;
        for c in s.chars() {
            cur = cur.children.get(&c)?;
        }
        Some(cur)
    }
}

fn main() {
    let mut t = Trie::new();
    t.insert("apple");
    assert!(t.search("apple"));
    assert!(!t.search("app"));
    assert!(t.starts_with("app"));
    t.insert("app");
    assert!(t.search("app"));
    assert!(t.starts_with("ap"));
    assert!(!t.search("appl"));
    assert!(!t.search(""));
    assert!(!t.starts_with("b"));
    println!("trie: ok");
}
