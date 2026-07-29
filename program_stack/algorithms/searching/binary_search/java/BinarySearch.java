public class BinarySearch {
    /**
     * On a sorted (non-decreasing) array, return an index of x, or -1 if absent.
     */
    public static int search(int[] a, int x) {
        if (a == null || a.length == 0) {
            return -1;
        }
        int lo = 0;
        int hi = a.length - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (a[mid] == x) {
                return mid;
            } else if (a[mid] < x) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return -1;
    }

    private static void expect(int[] a, int x, int want) {
        int got = search(a, x);
        if (want < 0) {
            if (got != -1) {
                System.err.println("expected miss, got " + got);
                System.exit(1);
            }
            return;
        }
        if (got < 0 || got >= a.length || a[got] != x) {
            System.err.println("search got " + got + " want index of " + x);
            System.exit(1);
        }
    }

    private static void expectExact(int[] a, int x, int want) {
        int got = search(a, x);
        if (got != want) {
            System.err.println("search got " + got + " want " + want);
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        expectExact(new int[] {}, 1, -1);
        expectExact(new int[] {42}, 42, 0);
        expectExact(new int[] {42}, 7, -1);
        expect(new int[] {1, 2, 3, 4, 5}, 1, 0);
        expect(new int[] {1, 2, 3, 4, 5}, 5, 4);
        expect(new int[] {1, 2, 3, 4, 5}, 3, 2);
        expectExact(new int[] {1, 2, 3, 4, 5}, 6, -1);
        expect(new int[] {1, 1, 2, 2, 3}, 2, 2); // any index with value 2 is ok
        int[] dups = new int[] {1, 1, 2, 2, 3};
        int idx = search(dups, 2);
        if (idx < 0 || dups[idx] != 2) {
            System.exit(1);
        }
        System.out.println("binary_search: ok");
    }
}
