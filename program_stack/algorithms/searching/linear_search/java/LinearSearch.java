public class LinearSearch {
    /** Return index of first x in a, or -1 if absent. */
    public static int search(int[] a, int x) {
        if (a == null) {
            return -1;
        }
        for (int i = 0; i < a.length; i++) {
            if (a[i] == x) {
                return i;
            }
        }
        return -1;
    }

    private static void expect(int[] a, int x, int want) {
        int got = search(a, x);
        if (got != want) {
            System.err.println("search got " + got + " want " + want);
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        expect(new int[] {}, 1, -1);
        expect(new int[] {42}, 42, 0);
        expect(new int[] {42}, 7, -1);
        expect(new int[] {3, 1, 4, 1, 5}, 1, 1);
        expect(new int[] {3, 1, 4, 1, 5}, 5, 4);
        expect(new int[] {3, 1, 4, 1, 5}, 9, -1);
        System.out.println("linear_search: ok");
    }
}
