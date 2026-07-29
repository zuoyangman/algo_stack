import java.util.Arrays;

public class BubbleSort {
    /** In-place bubble sort (non-decreasing). Stable. */
    public static void sort(int[] a) {
        if (a == null || a.length < 2) {
            return;
        }
        int n = a.length;
        for (int end = n - 1; end > 0; end--) {
            boolean swapped = false;
            for (int i = 0; i < end; i++) {
                if (a[i] > a[i + 1]) {
                    int tmp = a[i];
                    a[i] = a[i + 1];
                    a[i + 1] = tmp;
                    swapped = true;
                }
            }
            if (!swapped) {
                break;
            }
        }
    }

    private static void expect(int[] input, int[] want) {
        int[] a = Arrays.copyOf(input, input.length);
        sort(a);
        if (!Arrays.equals(a, want)) {
            System.err.println("got " + Arrays.toString(a) + " want " + Arrays.toString(want));
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        expect(new int[] {}, new int[] {});
        expect(new int[] {42}, new int[] {42});
        expect(new int[] {1, 2, 3}, new int[] {1, 2, 3});
        expect(new int[] {3, 2, 1}, new int[] {1, 2, 3});
        expect(new int[] {5, 1, 4, 2, 8}, new int[] {1, 2, 4, 5, 8});
        expect(new int[] {3, 1, 2, 1, 3}, new int[] {1, 1, 2, 3, 3});
        System.out.println("bubble_sort: ok");
    }
}
