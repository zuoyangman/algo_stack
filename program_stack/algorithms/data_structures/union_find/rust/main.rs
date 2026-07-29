struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.find(p);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
        true
    }

    fn connected(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }
}

fn main() {
    let mut uf = UnionFind::new(6);
    assert!(!uf.connected(0, 1));
    assert!(uf.union(0, 1));
    assert!(uf.union(1, 2));
    assert!(uf.connected(0, 2));
    assert!(!uf.connected(0, 3));
    assert!(uf.union(3, 4));
    assert!(uf.union(2, 4));
    assert!(uf.connected(0, 3));
    assert!(!uf.union(0, 2));
    assert_eq!(uf.find(5), 5);
    println!("union_find: ok");
}
