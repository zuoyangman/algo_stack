struct Node {
    value: i32,
    next: Option<Box<Node>>,
}

struct Queue {
    head: Option<Box<Node>>,
    // Tail as raw ownership is awkward in safe Rust; walk or use indices.
    // For clarity we keep only head and append by walking (fine for a demo).
}

impl Queue {
    fn new() -> Self {
        Self { head: None }
    }

    fn enqueue(&mut self, x: i32) {
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
    }

    fn dequeue(&mut self) -> Option<i32> {
        self.head.take().map(|node| {
            self.head = node.next;
            node.value
        })
    }

    fn front(&self) -> Option<i32> {
        self.head.as_ref().map(|n| n.value)
    }

    fn is_empty(&self) -> bool {
        self.head.is_none()
    }
}

fn main() {
    let mut q = Queue::new();
    assert!(q.is_empty());
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);
    assert_eq!(q.front(), Some(1));
    assert_eq!(q.dequeue(), Some(1));
    assert_eq!(q.dequeue(), Some(2));
    q.enqueue(4);
    assert_eq!(q.front(), Some(3));
    assert_eq!(q.dequeue(), Some(3));
    assert_eq!(q.dequeue(), Some(4));
    assert!(q.is_empty());
    println!("queue: ok");
}
