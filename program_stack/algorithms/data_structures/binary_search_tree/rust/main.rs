struct Node {
    key: i32,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

struct BST {
    root: Option<Box<Node>>,
}

impl BST {
    fn new() -> Self {
        Self { root: None }
    }

    fn insert(&mut self, x: i32) {
        Self::insert_rec(&mut self.root, x);
    }

    fn insert_rec(slot: &mut Option<Box<Node>>, x: i32) {
        match slot {
            None => {
                *slot = Some(Box::new(Node {
                    key: x,
                    left: None,
                    right: None,
                }));
            }
            Some(node) => {
                if x < node.key {
                    Self::insert_rec(&mut node.left, x);
                } else if x > node.key {
                    Self::insert_rec(&mut node.right, x);
                }
            }
        }
    }

    fn contains(&self, x: i32) -> bool {
        let mut cur = self.root.as_ref();
        while let Some(node) = cur {
            if x == node.key {
                return true;
            }
            cur = if x < node.key {
                node.left.as_ref()
            } else {
                node.right.as_ref()
            };
        }
        false
    }

    fn inorder(&self) -> Vec<i32> {
        let mut out = Vec::new();
        Self::inorder_rec(self.root.as_ref(), &mut out);
        out
    }

    fn inorder_rec(n: Option<&Box<Node>>, out: &mut Vec<i32>) {
        if let Some(node) = n {
            Self::inorder_rec(node.left.as_ref(), out);
            out.push(node.key);
            Self::inorder_rec(node.right.as_ref(), out);
        }
    }
}

fn main() {
    let mut t = BST::new();
    for v in [5, 3, 7, 1, 4, 6, 8] {
        t.insert(v);
    }
    t.insert(5);
    assert!(t.contains(4));
    assert!(!t.contains(2));
    assert_eq!(t.inorder(), vec![1, 3, 4, 5, 6, 7, 8]);
    println!("binary_search_tree: ok");
}
