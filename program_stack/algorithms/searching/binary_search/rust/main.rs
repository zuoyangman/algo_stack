fn search(a: &[i32], x: i32) -> isize {
    if a.is_empty() {
        return -1;
    }
    let mut lo: isize = 0;
    let mut hi: isize = (a.len() - 1) as isize;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let v = a[mid as usize];
        if v == x {
            return mid;
        } else if v < x {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    -1
}

fn expect_exact(a: &[i32], x: i32, want: isize) {
    assert_eq!(search(a, x), want);
}

fn expect_hit(a: &[i32], x: i32) {
    let got = search(a, x);
    assert!(got >= 0);
    assert_eq!(a[got as usize], x);
}

fn main() {
    expect_exact(&[], 1, -1);
    expect_exact(&[42], 42, 0);
    expect_exact(&[42], 7, -1);
    expect_hit(&[1, 2, 3, 4, 5], 1);
    expect_hit(&[1, 2, 3, 4, 5], 5);
    expect_hit(&[1, 2, 3, 4, 5], 3);
    expect_exact(&[1, 2, 3, 4, 5], 6, -1);
    expect_hit(&[1, 1, 2, 2, 3], 2);
    println!("binary_search: ok");
}
