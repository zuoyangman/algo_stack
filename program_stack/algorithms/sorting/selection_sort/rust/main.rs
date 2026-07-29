fn sort(a: &mut [i32]) {
    if a.len() < 2 {
        return;
    }
    for i in 0..a.len() - 1 {
        let mut min = i;
        for j in (i + 1)..a.len() {
            if a[j] < a[min] {
                min = j;
            }
        }
        if min != i {
            a.swap(i, min);
        }
    }
}

fn expect(input: &[i32], want: &[i32]) {
    let mut a = input.to_vec();
    sort(&mut a);
    assert_eq!(a, want);
}

fn main() {
    expect(&[], &[]);
    expect(&[42], &[42]);
    expect(&[1, 2, 3], &[1, 2, 3]);
    expect(&[3, 2, 1], &[1, 2, 3]);
    expect(&[5, 1, 4, 2, 8], &[1, 2, 4, 5, 8]);
    expect(&[3, 1, 2, 1, 3], &[1, 1, 2, 3, 3]);
    println!("selection_sort: ok");
}
