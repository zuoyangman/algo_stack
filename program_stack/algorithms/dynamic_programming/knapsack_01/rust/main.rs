/// Classic 0/1 knapsack: maximum value with capacity W.
fn knapsack01(weights: &[i32], values: &[i32], capacity: usize) -> i32 {
    let mut dp = vec![0_i32; capacity + 1];
    for i in 0..weights.len() {
        let w = weights[i] as usize;
        let v = values[i];
        for j in (w..=capacity).rev() {
            dp[j] = dp[j].max(dp[j - w] + v);
        }
    }
    dp[capacity]
}

fn main() {
    assert_eq!(knapsack01(&[2, 3, 4, 5], &[3, 4, 5, 6], 5), 7);
    assert_eq!(knapsack01(&[1, 2, 3], &[6, 10, 12], 5), 22);
    assert_eq!(knapsack01(&[], &[], 10), 0);
    println!("knapsack_01: ok");
}
