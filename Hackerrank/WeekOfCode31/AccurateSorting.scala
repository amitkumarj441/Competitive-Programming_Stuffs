// Author : Amit Kumar Jaiswal

import java.util.Arrays;
import java.util.Scanner;

public class AccurateSorting {
    public static void main(String[] args) { 
        Scanner in = new Scanner(System.in);

        int Q = in.nextInt();
        while (Q-- > 0) {
            int N = in.nextInt();
            int[] nums = new int[N];
            for (int i = 0; i < N; i++) {
                nums[i] = in.nextInt();
            }

            System.out.println(solve(nums) ? "Yes" : "No");
        }
    }

    static boolean solve(int[] nums) {
        boolean result = true;

        for (int i = 0, j = i + 1; j < nums.length; i++, j++) {
            if (nums[i] - 1 == nums[j]) {
                nums[i] = nums[i] ^ nums[j] ^ (nums[j] = nums[i]); 
            }
        }

        for (int i = 1; i < nums.length; i++) {
            if (nums[i] <= nums[i - 1])
                return false;
        }

        return result;
    }
}
