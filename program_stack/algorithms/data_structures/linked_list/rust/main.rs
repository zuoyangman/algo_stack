struct Node {
    value: i32,
    next: Option<Box<Node>>,
}

struct LinkedList {
    head: Option<Box<Node>>,
    len: usize,
}

impl LinkedList {
    fn new() -> Self {
        Self { head: None, len: 0 }
    }

    fn push_front(&mut self, x: i32) {
        let n = Box::new(Node {
            value: x,
            next: self.head.take(),
        });
        self.head = Some(n);
        self.len += 1;
    }

    fn push_back(&mut self, x: i32) {
        let n = Box::new(Node {
            value: x,
            next: None,
        });
        match self.head.as_mut() {
            None => self.head = Some(n),
            Some(mut cur) => {
                while cur.next.is_some() {
                    cur = cur.next.as_mut().unwrap();
                }
                cur.next = Some(n);
            }
        }
        self.len += 1;
    }

    fn pop_front(&mut self) -> Option<i32> {
        self.head.take().map(|node| {
            self.head = node.next;
            self.len -= 1;
            node.value
        })
    }

    fn find(&self, x: i32) -> bool {
        let mut cur = self.head.as_ref();
        while let Some(node) = cur {
            if node.value == x {
                return true;
            }
            cur = node.next.as_ref();
        }
        false
    }

    fn to_array(&self) -> Vec<i32> {
        let mut out = Vec::with_capacity(self.len);
        let mut cur = self.head.as_ref();
        while let Some(node) = cur {
            out.push(node.value);
            cur = node.next.as_ref();
        }
        out
    }
}

fn main() {
    let mut list = LinkedList::new();
    list.push_back(2);
    list.push_front(1);
    list.push_back(3);
    assert_eq!(list.to_array(), vec![1, 2, 3]);
    assert!(list.find(2));
    assert!(!list.find(9));
    assert_eq!(list.pop_front(), Some(1));
    assert_eq!(list.to_array(), vec![2, 3]);
    assert_eq!(list.len, 2);
    println!("linked_list: ok");
}
