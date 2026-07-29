fn search(a: &[i32], x: i32) -> isize {
    for (i, &v) in a.iter().enumerate() {
        if v == x {
            return i as isize;
        }
    }
    -1
}

fn expect(a: &[i32], x: i32, want: isize) {
    assert_eq!(search(a, x), want);
}

fn main() {
    expect(&[], 1, -1);
    expect(&[42], 42, 0);
    expect(&[42], 7, -1);
    expect(&[3, 1, 4, 1, 5], 1, 1);
    expect(&[3, 1, 4, 1, 5], 5, 4);
    expect(&[3, 1, 4, 1, 5], 9, -1);
    println!("linear_search: ok");
}
