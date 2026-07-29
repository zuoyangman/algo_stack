public class Lcs {
    /** Length of a longest common subsequence of a and b. */
    public static int lcsLength(String a, String b) {
        int n = a.length(), m = b.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (a.charAt(i - 1) == b.charAt(j - 1)) dp[i][j] = dp[i - 1][j - 1] + 1;
                else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return dp[n][m];
    }

    /** One concrete LCS string (any valid). */
    public static String lcsString(String a, String b) {
        int n = a.length(), m = b.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (a.charAt(i - 1) == b.charAt(j - 1)) dp[i][j] = dp[i - 1][j - 1] + 1;
                else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        StringBuilder sb = new StringBuilder();
        int i = n, j = m;
        while (i > 0 && j > 0) {
            if (a.charAt(i - 1) == b.charAt(j - 1)) {
                sb.append(a.charAt(i - 1));
                i--;
                j--;
            } else if (dp[i - 1][j] >= dp[i][j - 1]) i--;
            else j--;
        }
        return sb.reverse().toString();
    }

    public static void main(String[] args) {
        String a = "ABCBDAB", b = "BDCABA";
        if (lcsLength(a, b) != 4) {
            System.err.println("expected length 4");
            System.exit(1);
        }
        String s = lcsString(a, b);
        if (s.length() != 4 || lcsLength(s, a) != 4 || lcsLength(s, b) != 4) {
            System.err.println("bad LCS string: " + s);
            System.exit(1);
        }
        if (lcsLength("", "xyz") != 0 || lcsLength("abc", "abc") != 3) {
            System.err.println("edge cases failed");
            System.exit(1);
        }
        System.out.println("lcs: ok");
    }
}
