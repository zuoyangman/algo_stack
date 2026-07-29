public class Knapsack01 {
    /** Classic 0/1 knapsack: maximum value with capacity W. */
    public static int knapsack01(int[] weights, int[] values, int capacity) {
        int[] dp = new int[capacity + 1];
        for (int i = 0; i < weights.length; i++) {
            int w = weights[i], v = values[i];
            for (int j = capacity; j >= w; j--) {
                dp[j] = Math.max(dp[j], dp[j - w] + v);
            }
        }
        return dp[capacity];
    }

    public static void main(String[] args) {
        // w=[2,3,4,5], v=[3,4,5,6], W=5 → 7 (items 0+1)
        int ans = knapsack01(new int[]{2, 3, 4, 5}, new int[]{3, 4, 5, 6}, 5);
        if (ans != 7) {
            System.err.println("expected 7, got " + ans);
            System.exit(1);
        }
        ans = knapsack01(new int[]{1, 2, 3}, new int[]{6, 10, 12}, 5);
        if (ans != 22) {
            System.err.println("expected 22, got " + ans);
            System.exit(1);
        }
        if (knapsack01(new int[]{}, new int[]{}, 10) != 0) {
            System.err.println("empty items should be 0");
            System.exit(1);
        }
        System.out.println("knapsack_01: ok");
    }
}
