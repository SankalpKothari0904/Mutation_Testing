package org.example.services;
import java.util.*;

class ClassicalDP {

    public int knapSack(int maxWeight, List<Integer> vals, List<Integer> weights) {
        // Valid input conditions:
        // - maxWeight should be a non-negative integer.
        // - vals and weights should have the same size.
        // - vals and weights should not be empty or contain non-positive values.
        if (maxWeight < 0) {
            return -1; // Invalid input: negative maxWeight.
        }
        if (vals.size() != weights.size()) {
            return -1; // Invalid input: mismatched vals and weights list sizes.
        }
        if (vals.isEmpty()) {
            return -1; // Invalid input: empty lists.
        }
        if (vals.stream().anyMatch(v -> v <= 0) || weights.stream().anyMatch(w -> w <= 0)) {
            return -1; // Invalid input: contains non-positive values in vals or weights.
        }
    
        int n = weights.size();
        int[][] OPT = new int[n + 1][maxWeight + 1]; // Initialize DP table.
    
        // Base case: no items to include (OPT[0][w] = 0).
        for (int w = 0; w <= maxWeight; w++) {
            OPT[0][w] = 0;
        }
        // Base case: no weight to carry (OPT[i][0] = 0).
        for (int i = 0; i <= n; i++) {
            OPT[i][0] = 0;
        }
    
        // Fill DP table with optimal values.
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= maxWeight; w++) {
                // If the current item exceeds weight, don't include it.
                if (w - weights.get(i - 1) < 0) {
                    OPT[i][w] = OPT[i - 1][w];
                } else {
                    // Take maximum of excluding or including current item.
                    OPT[i][w] = Math.max(OPT[i - 1][w], OPT[i - 1][w - weights.get(i - 1)] + vals.get(i - 1));
                }
            }
        }
    
        // Return the optimal value for maxWeight.
        return OPT[n][maxWeight];
    }    

    public int longestIncreasingSubsequence(List<Integer> nums) {
        // Valid input conditions:
        // - nums should be a non-empty list of integers.
        if (nums == null || nums.isEmpty()) {
            return 0; // Invalid input: empty list.
        }
    
        int n = nums.size();
        List<Integer> OPT = new ArrayList<>(Collections.nCopies(n, 1)); // Initialize DP table.
    
        // Fill DP table with the lengths of increasing subsequences.
        for (int i = 0; i < n; i++) {
            for (int j = i - 1; j >= 0; j--) {
                // Update the subsequence length if the current number is larger than a previous one.
                if (nums.get(j) < nums.get(i) && OPT.get(j) + 1 > OPT.get(i)) {
                    OPT.set(i, OPT.get(j) + 1);
                }
            }
        }
    
        // Return the maximum length of the increasing subsequences.
        return Collections.max(OPT);
    }    

    public int longestPalindrome(String arg) {
        // Function to find the length of the longest palindromic subsequence in a string.
        // Valid input conditions:
        // - arg should be a non-null string.
        if (arg == null) {
            return -1; // Invalid input: null string.
        }
    
        int n = arg.length();
        if (n == 0) {
            return 0; // Empty string has no palindrome.
        }
    
        int[][] OPT = new int[n][n];
        for (int i = 0; i < n; i++) {
            OPT[i][i] = 1; // Base case: single character is a palindrome.
        }
    
        // Build the DP table for substrings of increasing lengths.
        for (int length = 1; length < n; length++) {
            for (int i = 0; i < n - length; i++) {
                int j = i + length;
                if (arg.charAt(i) == arg.charAt(j)) {
                    OPT[i][j] = OPT[i + 1][j - 1] + 2; // Expand palindrome.
                } else {
                    OPT[i][j] = Math.max(OPT[i + 1][j], OPT[i][j - 1]); // Max of excluding one character.
                }
            }
        }
    
        // Return the longest palindrome length.
        return OPT[0][n - 1];
    }    

    public int houseRobber(List<Integer> input) {
        // Function to compute the maximum amount of money a robber can steal without robbing two adjacent houses.
        // Valid input conditions:
        // - input should be a non-null list of integers, representing house values.
        if (input == null) {
            return -1; // Invalid input: null list.
        }
        if (input.isEmpty()) {
            return 0; // No houses to rob.
        } else if (input.size() == 1) {
            return input.get(0); // Only one house to rob.
        } else if (input.size() == 2) {
            return Math.max(input.get(0), input.get(1)); // Choose the richer of the two houses.
        }
    
        int n = input.size();
        List<Integer> OPT = new ArrayList<>(Collections.nCopies(n, -1)); 
        OPT.set(0, input.get(0)); // Base case: max money after robbing the first house.
        OPT.set(1, Math.max(input.get(0), input.get(1))); // Base case: max money after robbing the first or second house.
    
        // Build the DP table for optimal solutions.
        for (int i = 2; i < n; i++) {
            OPT.set(i, Math.max(OPT.get(i - 2) + input.get(i), OPT.get(i - 1))); // Max of robbing or skipping the current house.
        }
    
        // Return the maximum money that can be stolen from all houses.
        return OPT.get(n - 1);
    }    

    public int binarySearchFinish(List<Interval> intervals, int k) {
        // Function to find the last interval whose 'end' is less than or equal to k.
        // Valid input conditions:
        // - intervals should be a non-null list of Interval objects.
        // - k should be a valid integer.
        if (intervals == null || intervals.isEmpty()) {
            return -1; // Invalid input: empty or null list.
        }
    
        int l = 0, r = intervals.size() - 1;
    
        // If the first interval's end is greater than k, return -1.
        if (intervals.get(0).end > k) {
            return -1;
        }
    
        // Binary search for the last interval that ends <= k.
        while (r - l > 1) {
            int m = l + (r - l) / 2;
            if (intervals.get(m).end <= k && intervals.get(m + 1).end > k) {
                return m; // Found the last interval.
            } else if (intervals.get(m).end > k) {
                r = m - 1; // Search in the left half.
            } else {
                l = m + 1; // Search in the right half.
            }
        }
    
        // Final check to ensure the last valid interval is returned.
        if (intervals.get(r).end <= k) {
            return r;
        } else if (intervals.get(l).end > k) {
            return -1; // No valid interval found.
        } else {
            return l;
        }
    }    

    public int getSchedule(List<Integer> startTime, List<Integer> endTime, List<Integer> weights) {
        // Function to find the maximum weight schedule with non-overlapping intervals.
        // Valid input conditions:
        // - startTime, endTime, and weights must have the same size.
        // - All weights should be positive integers.
        // - Each startTime should be less than endTime for each interval.
        
        if (startTime.size() != endTime.size() || startTime.size() != weights.size()) {
            return -1; // Invalid input: inconsistent list sizes.
        }
        if (startTime.isEmpty()) {
            return 0; // No intervals provided.
        }
        if (weights.stream().anyMatch(x -> x <= 0)) {
            return -1; // Invalid input: non-positive weight.
        }
        for (int i = 0; i < startTime.size(); i++) {
            if (startTime.get(i) >= endTime.get(i)) {
                return -1; // Invalid input: start time is greater than or equal to end time.
            }
        }
    
        // Create a list of intervals with their start, end, and weight.
        List<Interval> intervals = new ArrayList<>();
        for (int i = 0; i < startTime.size(); i++) {
            intervals.add(new Interval(startTime.get(i), endTime.get(i), weights.get(i)));
        }
    
        // Sort intervals based on end time, then start time.
        intervals.sort(Comparator.comparingInt((Interval x) -> x.end).thenComparingInt(x -> x.start));
    
        int n = intervals.size();
        List<Integer> OPT = new ArrayList<>(Collections.nCopies(n, 0)); // Initialize OPT list.
        List<Integer> lastEnding = new ArrayList<>(Collections.nCopies(n, -1)); // List for storing the last non-overlapping interval.
    
        OPT.set(0, intervals.get(0).weight); // Base case: the first interval's weight.
    
        // For each interval, find the last non-overlapping interval.
        for (int j = 1; j < n; j++) {
            lastEnding.set(j, binarySearchFinish(intervals, intervals.get(j).start));
        }
    
        // Fill the OPT list by considering whether to include or exclude the current interval.
        for (int i = 1; i < n; i++) {
            int include = intervals.get(i).weight;
            if (lastEnding.get(i) != -1) {
                include += OPT.get(lastEnding.get(i)); // Include the previous non-overlapping interval's weight.
            }
    
            int exclude = OPT.get(i - 1); // Exclude the current interval.
            OPT.set(i, Math.max(include, exclude)); // Choose the better option.
        }
    
        return OPT.get(n - 1); // Return the maximum weight schedule.
    }    

    public boolean wordBreak(String s, List<String> wordDict) {
        // Function to determine if a string can be segmented into words from the dictionary.
        // Valid input conditions:
        // - `s` is a non-null string.
        // - `wordDict` is a non-null list of words.
    
        Set<String> wordSet = new HashSet<>(wordDict); // Convert wordDict to a Set for faster lookups.
        int n = s.length();
        if (n == 0) {
            return true; // Empty string can always be segmented.
        }
    
        List<Boolean> dp = new ArrayList<>(Collections.nCopies(n + 1, false)); // DP list to store results.
        dp.set(0, true); // Base case: empty string is always segmentable.
    
        // Iterate through each substring of `s` to check if it can be segmented.
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                // If substring s[j..i) is a word and s[0..j) can be segmented, mark dp[i] as true.
                if (dp.get(j) && wordSet.contains(s.substring(j, i))) {
                    dp.set(i, true);
                    break; // No need to check further, as we found a valid segmentation.
                }
            }
        }
    
        return dp.get(n); // Return if the entire string can be segmented.
    }    

    public int coinChange(List<Integer> coins, int amount) {
        // Function to find the minimum number of coins needed to make up a given amount.
        // Valid input conditions:
        // - `coins` is a non-null list with values greater than zero.
        // - `amount` is a non-negative integer.
    
        if (amount <= 0){
            return -1; // Invalid amount (negative or zero).
        } else if (coins.isEmpty()){
            return -1; // No coins available.
        } else if (coins.stream().anyMatch(coin -> coin <= 0)){
            return -1; // Invalid coin values.
        }
    
        int INT_MAX = Integer.MAX_VALUE; // Maximum value to represent an unreachable amount.
        List<Integer> OPT = new ArrayList<>(Collections.nCopies(amount + 1, INT_MAX)); // DP list to store the minimum coins for each amount.
        OPT.set(0, 0); // Base case: no coins are needed to make amount 0.
    
        // Calculate the minimum coins for each possible amount.
        for (int sum = 1; sum <= amount; sum++) {
            for (int coin : coins) {
                if (sum - coin >= 0 && OPT.get(sum - coin) != INT_MAX) {
                    OPT.set(sum, Math.min(OPT.get(sum), OPT.get(sum - coin) + 1)); // Update the minimum coins needed.
                }
            }
        }
    
        return OPT.get(amount) == INT_MAX ? -1 : OPT.get(amount); // Return the result or -1 if not possible.
    }
    
    public int longestCommonSubsequence(String text1, String text2) {
        // Function to find the length of the longest common subsequence (LCS) between two strings.
        // Valid input conditions:
        // - `text1` and `text2` are non-null strings.
    
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1]; // DP table to store the length of LCS at each point.
    
        // Build the DP table for LCS.
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1; // If characters match, extend the LCS.
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]); // Otherwise, take the maximum of previous results.
                }
            }
        }
    
        return dp[m][n]; // Return the length of LCS.
    }
    

    // Function to calculate the nth Catalan number using dynamic programming.
    public int catalan_recursive(int n) {
        // Valid input conditions:
        // - `n` should be a non-negative integer.

        if (n < 0) {
            return -1; // Return -1 for invalid input.
        } else if (n == 0 || n == 1) {
            return 1; // Base cases: Catalan(0) = 1, Catalan(1) = 1.
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

    // Function to calculate the nth Catalan number using closed-form formula.
    public int catalan_closed_form(int n) {
        // Valid input conditions:
        // - `n` should be a non-negative integer.

        if (n < 0) {
            return -1; // Return -1 for invalid input.
        }

        int res = 1;
        // Iterate till N to calculate Catalan number using closed-form formula
        for (int i = 1; i <= n; i++) {
            res = (res * (4 * i - 2)) / (i + 1); // Apply closed-form formula
        }
        return res; // Return the nth Catalan number
    }

    // Function to compute the factorial of a number.
    public int factorial(int n) {
        // Valid input conditions:
        // - `n` should be a non-negative integer.

        if (n < 0) {
            return -1; // Return -1 for invalid input.
        }
        // 0! and 1! are both 1
        if (n == 0 || n == 1) {
            return 1; // Base case for factorial
        }

        // Compute factorial using a loop
        int res = 1;
        for (int i = 2; i <= n; i++) {
            res *= i; // Multiply by current number in loop
        }

        return res; // Return the computed factorial
    }


    // Function to calculate the Stirling number of the second kind S(n, r)
    public int stirling_number(int r, int n) {
        // Valid input conditions:
        // - `n` and `r` should be non-negative integers.
        // Invalid input conditions:
        // - Return -1 if `n` or `r` is negative.
        // - Return 0 if `r < n`, as there are no ways to partition `n` elements into `r` non-empty sets.

        if (n < 0 || r < 0) {
            return -1; // Return -1 for invalid input.
        }
        if (r < n) {
            return 0; // No way to partition if r < n.
        }
        if (r == 0) {
            return 1; // Base case: S(n, 0) = 1.
        }
        if (n == 0) {
            return 0; // No way to partition 0 elements into a non-zero number of sets.
        }
        if (n == r) {
            return 1; // Base case: S(n, n) = 1.
        }

        // Create a 2D array to store the Stirling numbers.
        int[][] dp = new int[n + 1][r + 1];

        // Base cases initialization
        for (int i = 1; i <= n; i++) {
            dp[i][i] = 1; // S(i, i) = 1.
        }
        for (int i = 1; i <= r; i++) {
            dp[1][i] = 1; // S(1, i) = 1 for all i >= 1.
        }

        // Fill the DP table using the recurrence relation.
        for (int j = 2; j <= r; j++) {
            for (int i = 2; i <= n; i++) {
                dp[i][j] = dp[i - 1][j - 1] + (i) * dp[i][j - 1]; // Recurrence formula.
            }
        }

        // Return the Stirling number S(n, r).
        return dp[n][r];
    }

    // Function to calculate the minimum distance between two words (edit distance)
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();

        // Create a DP table to store results of subproblems
        int[][] dp = new int[m + 1][n + 1];

        // Base case: transforming empty string to another string
        // If one word is empty, the distance is the length of the other word
        for (int i = 1; i <= m; i++) {
            dp[i][0] = i; // Edit distance for word1 to empty string is i (deletion of i characters)
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = j; // Edit distance for empty string to word2 is j (insertion of j characters)
        }

        // Fill the DP table based on recurrence relation
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1]; // No change needed if characters match
                } else {
                    // Otherwise, take the minimum of:
                    // - Deletion (dp[i-1][j]),
                    // - Insertion (dp[i][j-1]),
                    // - Substitution (dp[i-1][j-1])
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
            }
        }

        // Return the minimum edit distance for transforming word1 to word2
        return dp[m][n];
    }

    // Function to find the minimum number of operations to multiply matrices
    public int matrixMultiplication(int[] arr) {
        int N = arr.length;

        // Handle edge cases for invalid input
        if (N < 2) {
            return -1; // Invalid input if fewer than two matrices
        }
        if (Arrays.stream(arr).anyMatch(dim -> dim <= 0)) {
            return -1; // Invalid input if any dimension is zero or negative
        }

        // Create a DP table to store minimum multiplication costs
        int[][] dp = new int[N][N];

        // Loop through chain lengths from 2 to N
        for (int length = 2; length < N; length++) {
            for (int i = 0; i < N - length; i++) {
                int j = i + length;
                dp[i][j] = Integer.MAX_VALUE; // Initialize with a large value
                // Try all possible positions of the split (k) to minimize cost
                for (int k = i + 1; k < j; k++) {
                    dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j]);
                }
            }
        }

        // Return the minimum number of operations required for multiplying the matrices
        return dp[0][N - 1];
    }

    // Function to find the maximum product subarray
    public int maxProduct(int[] nums) {
        // Check for null or empty input
        if (nums == null || nums.length == 0) {
            return 0;
        }

        // Initialize variables to store the maximum and minimum product so far,
        // as well as the result
        int maxProd = nums[0], minProd = nums[0], result = nums[0];

        // Iterate through the array, updating the max and min product for each element
        for (int i = 1; i < nums.length; i++) {
            // If the current number is negative, swap max and min
            if (nums[i] < 0) {
                int temp = maxProd;
                maxProd = minProd;
                minProd = temp;
            }

            // Update max and min product by considering the current number
            maxProd = Math.max(nums[i], maxProd * nums[i]);
            minProd = Math.min(nums[i], minProd * nums[i]);

            // Update result to store the maximum product found so far
            result = Math.max(result, maxProd);
        }

        // Return the maximum product found
        return result;
    }

    // Function to calculate Fibonacci numbers using dynamic programming
    public int fibonacci(int n) {
        // Return -1 for invalid input
        if (n < 0) {
            return -1;
        } else if (n == 0) {
            return 0; // Base case: Fibonacci(0) = 0
        }

        // Create an array to store Fibonacci numbers up to n
        int[] dp = new int[n + 1];
        dp[0] = 0; // Base case: Fibonacci(0) = 0
        dp[1] = 1; // Base case: Fibonacci(1) = 1

        // Calculate Fibonacci numbers from 2 to n
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }

        // Return the nth Fibonacci number
        return dp[n];
    }

    // Function to compute the binomial coefficient C(n, k)
    public int binomialCoefficient(int n, int k) {
        // Return -1 for invalid input
        if (k < 0 || k > n) {
            return -1;
        }

        // Create a 2D array to store binomial coefficients
        int[][] dp = new int[n + 1][k + 1];

        // Initialize base case where C(i, 0) = 1 for all i
        for (int i = 0; i <= n; i++) {
            dp[i][0] = 1;  // Base case: C(i, 0) = 1

            // Calculate the binomial coefficients for other values
            for (int j = 1; j <= Math.min(i, k); j++) {
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
            }
        }

        // Return the binomial coefficient C(n, k)
        return dp[n][k];
    }

    // Function to calculate derangement (count of permutations where no element is in its original position)
    public int derangementCount(int n) {
        // Return -1 for invalid input
        if (n < 0) {
            return -1;
        }

        // Base case for n == 0
        if (n == 0) {
            return 1;
        }

        // Initialize a dp array to store derangement values
        int[] dp = new int[n + 1];
        dp[0] = 1;  // Base case: D(0) = 1
        dp[1] = 0;  // Base case: D(1) = 0 (cannot derange a single element)

        // Fill the dp array using the recurrence relation
        for (int i = 2; i <= n; i++) {
            dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2]);
        }

        // Return the derangement count for n elements
        return dp[n];
    }

}