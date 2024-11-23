package org.example.services;
import java.util.*;

class ClassicalDP {

    public int knapSack(int maxWeight, List<Integer> vals, List<Integer> weights) {
        if (maxWeight < 0) {
            return -1;
        }

        if (vals.size() != weights.size()) {
            return -1;
        }

        if (vals.isEmpty()) {
            return -1;
        }

        if (vals.stream().anyMatch(v -> v <= 0) || weights.stream().anyMatch(w -> w <= 0)) {
            return -1;
        }

        int n = weights.size();
        int[][] OPT = new int[n + 1][maxWeight + 1];

        for (int w = 0; w <= maxWeight; w++) {
            OPT[0][w] = 0;
        }

        for (int i = 0; i <= n; i++) {
            OPT[i][0] = 0;
        }

        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= maxWeight; w++) {
                if (w - weights.get(i - 1) < 0) {
                    OPT[i][w] = OPT[i - 1][w];
                } else {
                    OPT[i][w] = Math.max(OPT[i - 1][w], OPT[i - 1][w - weights.get(i - 1)] + vals.get(i - 1));
                }
            }
        }

        return OPT[n][maxWeight];
    }


    public int longestIncreasingSubsequence(List<Integer> nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }

        List<Integer> OPT = new ArrayList<>(Collections.nCopies(n, 1));

        for (int i = 0; i < n; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (nums.get(j) < nums.get(i) && OPT.get(j) + 1 > OPT.get(i)) {
                    OPT.set(i, OPT.get(j) + 1);
                }
            }
        }

        return Collections.max(OPT);
    }


    public int longestPalindrome(String arg) {
        int n = arg.length();
        if (n == 0) {
            return 0;
        }

        int[][] OPT = new int[n][n];
        for (int i = 0; i < n; i++) {
            OPT[i][i] = 1;
        }

        for (int length = 1; length < n; length++) {
            for (int i = 0; i < n - length; i++) {
                int j = i + length;
                if (arg.charAt(i) == arg.charAt(j)) {
                    OPT[i][j] = OPT[i + 1][j - 1] + 2;
                } else {
                    OPT[i][j] = Math.max(OPT[i + 1][j], OPT[i][j - 1]);
                }
            }
        }

        return OPT[0][n - 1];
    }


    public int houseRobber(List<Integer> input) {
        if (input.isEmpty()) {
            return 0;
        } else if (input.size() == 1) {
            return input.get(0);
        } else if (input.size() == 2) {
            return Math.max(input.get(0), input.get(1));
        }

        int n = input.size();
        List<Integer> OPT = new ArrayList<>(Collections.nCopies(n, -1));

        OPT.set(0, input.get(0));
        OPT.set(1, Math.max(input.get(0), input.get(1)));

        for (int i = 2; i < n; i++) {
            OPT.set(i, Math.max(OPT.get(i - 2) + input.get(i), OPT.get(i - 1)));
        }

        return OPT.get(n - 1);
    }


    public int binarySearchFinish(List<Interval> intervals, int k) {
        int l = 0, r = intervals.size() - 1;

        if (intervals.get(0).end > k) {
            return -1;
        }

        while (r - l > 1) {
            int m = l + (r - l) / 2;
            if (intervals.get(m).end <= k && intervals.get(m + 1).end > k) {
                return m;
            } else if (intervals.get(m).end > k) {
                r = m - 1;
            } else {
                l = m + 1;
            }
        }

        if (intervals.get(r).end <= k) {
            return r;
        } else if (intervals.get(l).end > k) {
            return -1;
        } else {
            return l;
        }
    }


    public int getSchedule(List<Integer> startTime, List<Integer> endTime, List<Integer> weights) {
        if (startTime.size() != endTime.size() || startTime.size() != weights.size()) {
            return -1;
        }

        if (startTime.isEmpty()) {
            return 0;
        }

        if (weights.stream().anyMatch(x -> x <= 0)) {
            return -1;
        }

        for (int i = 0; i < startTime.size(); i++) {
            if (startTime.get(i) >= endTime.get(i)) {
                return -1;
            }
        }

        List<Interval> intervals = new ArrayList<>();
        for (int i = 0; i < startTime.size(); i++) {
            intervals.add(new Interval(startTime.get(i), endTime.get(i), weights.get(i)));
        }

        intervals.sort(Comparator.comparingInt((Interval x) -> x.end).thenComparingInt(x -> x.start));

        int n = intervals.size();
        List<Integer> OPT = new ArrayList<>(Collections.nCopies(n, 0));
        List<Integer> lastEnding = new ArrayList<>(Collections.nCopies(n, -1));
        
        OPT.set(0, intervals.get(0).weight);

        for (int j = 1; j < n; j++) {
            lastEnding.set(j, binarySearchFinish(intervals, intervals.get(j).start));
        }

        for (int i = 1; i < n; i++) {
            int include = intervals.get(i).weight;
            if (lastEnding.get(i) != -1) {
                include += OPT.get(lastEnding.get(i));
            }

            int exclude = OPT.get(i - 1);
            OPT.set(i, Math.max(include, exclude));
        }

        return OPT.get(n - 1);
    }


    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        int n = s.length();
        if (n == 0) {
            return true;
        }

        List<Boolean> dp = new ArrayList<>(Collections.nCopies(n + 1, false));
        dp.set(0, true);

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (dp.get(j) && wordSet.contains(s.substring(j, i))) {
                    dp.set(i, true);
                    break;
                }
            }
        }

        return dp.get(n);
    }


    public int coinChange(List<Integer> coins, int amount) {
        if (amount <= 0){
            return -1;
        }else if (coins.isEmpty()){
            return -1;
        }else if (coins.stream().anyMatch(coin -> coin <= 0)){
            return -1;
        }

        int INT_MAX = Integer.MAX_VALUE;
        List<Integer> OPT = new ArrayList<>(Collections.nCopies(amount + 1, INT_MAX));
        OPT.set(0, 0);

        for (int sum = 1; sum <= amount; sum++) {
            for (int coin : coins) {
                if (sum - coin >= 0 && OPT.get(sum - coin) != INT_MAX) {
                    OPT.set(sum, Math.min(OPT.get(sum), OPT.get(sum - coin) + 1));
                }
            }
        }

        return OPT.get(amount) == INT_MAX ? -1 : OPT.get(amount);
    }


    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }

        return dp[m][n];
    }


    // Function to calculate the nth Catalan number recursively
    public int catalan_recursive(int n) {
        if (n < 0){
            return -1;
        }else if (n == 0 || n == 1){
            return 1;
        }

        // Table to store results of subproblems
        int[] catalan = new int[n + 1];

        // Initialize first two values in table
        catalan[0] = 1;
        catalan[1] = 1;

        // Fill entries in catalan[] using the recursive formula
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                catalan[i] += catalan[j] * catalan[i - j - 1];
            }
        }

        // Return the nth Catalan number
        return catalan[n];
    }


    // Function to calculate the nth Catalan number using closed-form
    public int catalan_closed_form(int n) {
        if (n < 0) {
            return -1;
        }

        int res = 1;
        // Iterate till N to calculate Catalan number
        for (int i = 1; i <= n; i++) {
            res = (res * (4 * i - 2)) / (i + 1);
        }
        return res;
    }

    // Function to compute the factorial of a number
    public int factorial(int n) {
        if (n < 0) {
            return -1;
        }
        // 0! and 1! are both 1
        if (n == 0 || n == 1) {
            return 1;
        }

        // Compute factorial using a loop
        int res = 1;
        for (int i = 2; i <= n; i++) {
            res *= i;
        }

        return res;
    }

    // Function to calculate the Stirling number of the second kind S(n, r)
    public int stirling_number(int r, int n) {
        if (n < 0 || r < 0) {
            return -1;
        }

        if (r < n) {
            return 0;
        }

        if (r == 0) {
            return 1;
        }

        if (n == 0) {
            return 0;
        }

        if (n == r) {
            return 1;
        }

        // Create a 2D list to store the Stirling numbers
        int[][] dp = new int[n + 1][r + 1];

        // Base cases
        for (int i = 1; i <= n; i++) {
            dp[i][i] = 1;
        }
        for (int i = 1; i <= r; i++) {
            dp[1][i] = 1;
        }

        // Fill in the rest of the dp table
        for (int j = 2; j <= r; j++){
            for (int i = 2; i <= n; i++){
                dp[i][j] = dp[i - 1][j - 1] + (i) * dp[i][j - 1];
            }
        }

        // Return the Stirling number
        return dp[n][r];
    }



    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();

        int[][] dp = new int[m + 1][n + 1];

        // Base case: transforming empty string to another string
        for (int i = 1; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = j;
        }

        // Fill the DP table
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
            }
        }

        return dp[m][n];
    }


    // Function to find the minimum number of operations to multiply matrices
    public int matrixMultiplication(int[] arr) {
        int N = arr.length;
        if (N < 2) {
            return -1;
        }
        if (Arrays.stream(arr).anyMatch(dim -> dim <= 0)){
            return -1;
        }

        int[][] dp = new int[N][N];

        // Loop through chain lengths from 2 to N
        for (int length = 2; length < N; length++) {
            for (int i = 0; i < N - length; i++) {
                int j = i + length;
                dp[i][j] = Integer.MAX_VALUE;
                for (int k = i + 1; k < j; k++) {
                    dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j]);
                }
            }
        }

        return dp[0][N - 1];
    }


    // Function to find the maximum product subarray
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        int maxProd = nums[0], minProd = nums[0], result = nums[0];

        for (int i = 1; i < nums.length; i++) {
            if (nums[i] < 0) {
                int temp = maxProd;
                maxProd = minProd;
                minProd = temp;
            }

            maxProd = Math.max(nums[i], maxProd * nums[i]);
            minProd = Math.min(nums[i], minProd * nums[i]);

            result = Math.max(result, maxProd);
        }

        return result;
    }


    // Function to calculate Fibonacci numbers using dynamic programming
    public int fibonacci(int n) {
        if (n < 0) {
            return -1;
        }else if (n == 0){
            return 0;
        }

        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;

        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }

        return dp[n];
    }

    // Function to compute the binomial coefficient C(n, k)
    public int binomialCoefficient(int n, int k) {
        if (k < 0 || k > n) {
            return -1;
        }

        int[][] dp = new int[n + 1][k + 1];

        for (int i = 0; i <= n; i++) {
            dp[i][0] = 1;  // Base case

            for (int j = 1; j <= Math.min(i, k); j++) {
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
            }
        }

        return dp[n][k];
    }
    

    // Function to calculate derangement (count of permutations where no element is in its original position)
    public int derangementCount(int n) {
        if (n < 0) {
            return -1;
        }
        if (n == 0){
            return 1;
        }

        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 0;  // Base case

        for (int i = 2; i <= n; i++) {
            dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2]);
        }

        return dp[n];
    }
}