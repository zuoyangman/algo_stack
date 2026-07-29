fn sort(a: &mut [i32]) {
    if a.len() < 2 {
        return;
    }
    let mut end = a.len() - 1;
    while end > 0 {
        let mut swapped = false;
        for i in 0..end {
            if a[i] > a[i + 1] {
                a.swap(i, i + 1);
                swapped = true;
            }
        }
        if !swapped {
            break;
        }
        end -= 1;
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
    println!("bubble_sort: ok");
}
