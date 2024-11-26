package org.example.services;


import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import static org.junit.Assert.*;

public class ClassicalDPTest {

    private final ClassicalDP dp = new ClassicalDP();

    // KnapSack Tests

    @Test
    public void testKnapsack_case1() {
        List<Integer> vals = Arrays.asList(60, 100, 120);
        List<Integer> weights = Arrays.asList(10, 20, 30);
        assertEquals(220, dp.knapSack(50, vals, weights));
    }

    @Test
    public void testKnapsack_case2() {
        List<Integer> vals = Arrays.asList(10, 20);
        List<Integer> weights = Arrays.asList(5, 5);
        assertEquals(20, dp.knapSack(5, vals, weights));
    }

    @Test
    public void testKnapsack_case3(){
        List<Integer> vals = Arrays.asList(1, 2);
        List<Integer> weights = Arrays.asList(1, 3);
        assertEquals(-1, dp.knapSack(-2, vals, weights));
    }

    @Test
    public void testKnapsack_case4(){
        List<Integer> vals = Arrays.asList(1, 2);
        List<Integer> weights = Arrays.asList(1, 3);
        assertEquals(0, dp.knapSack(0, vals, weights));
    }

    @Test
    public void testKnapsack_case5(){
        List<Integer> vals = Arrays.asList(1, 2, 4);
        List<Integer> weights = Arrays.asList(1, 3);
        assertEquals(-1, dp.knapSack(5, vals, weights));
    }

    @Test
    public void testKnapsack_case6(){
        List<Integer> vals = List.of();
        List<Integer> weights = List.of();
        assertEquals(-1, dp.knapSack(5, vals, weights));
    }

    @Test
    public void testKnapsack_case7(){
        List<Integer> vals = List.of(1,3,-1);
        List<Integer> weights = List.of(2,4,5);
        assertEquals(-1, dp.knapSack(5, vals, weights));
    }

    @Test
    public void testKnapsack_case8(){
        List<Integer> vals = List.of(1,3,1);
        List<Integer> weights = List.of(2,4,-5);
        assertEquals(-1, dp.knapSack(5, vals, weights));
    }

    @Test
    public void testKnapsack_case9(){
        List<Integer> vals = List.of(1,3,0);
        List<Integer> weights = List.of(2,4,-5);
        assertEquals(-1, dp.knapSack(5, vals, weights));
    }

    @Test
    public void testKnapsack_case10(){
        List<Integer> vals = List.of(1,3,1);
        List<Integer> weights = List.of(2,4,0);
        assertEquals(-1, dp.knapSack(5, vals, weights));
    }

    // Longest Increasing Subsequence Tests
    @Test
    public void testLongestIncreasingSubsequence_case1() {
        List<Integer> nums = Arrays.asList(10, 9, 2, 5, 3, 7, 101, 18);
        assertEquals(4, dp.longestIncreasingSubsequence(nums));
    }

    @Test
    public void testLongestIncreasingSubsequence_case2() {
        List<Integer> nums = Arrays.asList(0, 1, 0, 3, 2, 3);
        assertEquals(4, dp.longestIncreasingSubsequence(nums));
    }

    @Test
    public void testLongestIncreasingSubsequence_case3() {
        List<Integer> nums = Arrays.asList(0, 1, 2, 2);
        assertEquals(3, dp.longestIncreasingSubsequence(nums));
    }

    @Test
    public void testLongestIncreasingSubsequence_case4() {
        List<Integer> nums = Arrays.asList(0, 1, 2, 0, 2);
        assertEquals(3, dp.longestIncreasingSubsequence(nums));
    }

    // Longest Palindrome Tests
    @Test
    public void testLongestPalindrome_case1() {
        assertEquals(4, dp.longestPalindrome("bbbab"));
    }

    @Test
    public void testLongestPalindrome_case2() {
        assertEquals(1, dp.longestPalindrome("abcde"));
    }

    // House Robber Tests
    @Test
    public void testHouseRobber_case1() {
        List<Integer> input = Arrays.asList(2, 7, 9, 3, 1);
        assertEquals(12, dp.houseRobber(input));
    }

    @Test
    public void testHouseRobber_case2() {
        List<Integer> input = Arrays.asList(1, 2, 3, 1);
        assertEquals(4, dp.houseRobber(input));
    }

    @Test
    public void testHouseRobber_case3() {
        List<Integer> input = List.of(1);
        assertEquals(1, dp.houseRobber(input));
    }

    @Test
    public void testHouseRobber_case4() {
        List<Integer> input = List.of(1, 2);
        assertEquals(2, dp.houseRobber(input));
    }

    //Binary Search Intervals testcases
    @Test
    public void searchIntervals_case1() {
        // Case 1: Valid intervals, k is within the range
        List<Interval> intervals = Arrays.asList(
                new Interval(1, 3, 10),
                new Interval(2, 5, 20),
                new Interval(4, 8, 30)
        );
        int k = 5;
        int result = dp.binarySearchFinish(intervals, k);
        assertEquals(1, result);
    }

    @Test
    public void searchIntervals_case2() {
        // Case 2: No intervals with end time <= k
        List<Interval> intervals = Arrays.asList(
                new Interval(1, 3, 10),
                new Interval(4, 6, 20),
                new Interval(7, 9, 30)
        );
        int k = 2;
        int result = dp.binarySearchFinish(intervals, k);
        assertEquals(-1, result);
    }

    @Test
    public void searchIntervals_case3() {
        // Case 3: k matches the end time of the last interval
        List<Interval> intervals = Arrays.asList(
                new Interval(1, 3, 10),
                new Interval(2, 5, 20),
                new Interval(4, 8, 30)
        );
        int k = 8;
        int result = dp.binarySearchFinish(intervals, k);
        assertEquals(2, result);
    }

    @Test
    public void searchIntervals_case4() {
        // Case 4: k is greater than all end times
        List<Interval> intervals = Arrays.asList(
                new Interval(1, 3, 10),
                new Interval(2, 5, 20),
                new Interval(4, 8, 30)
        );
        int k = 10;
        int result = dp.binarySearchFinish(intervals, k);
        assertEquals(2, result);
    }

    @Test
    public void searchIntervals_case5() {
        // Case 5: k is less than all end times
        List<Interval> intervals = Arrays.asList(
                new Interval(1, 3, 10),
                new Interval(4, 6, 20),
                new Interval(7, 9, 30)
        );
        int k = 0;
        int result = dp.binarySearchFinish(intervals, k);
        assertEquals(-1, result);
    }

    @Test
    public void searchIntervals_case6() {
        // Case 6: Single interval, k is within range
        List<Interval> intervals = Arrays.asList(
                new Interval(1, 3, 10)
        );
        int k = 3;
        int result = dp.binarySearchFinish(intervals, k);
        assertEquals(0, result);
    }

    @Test
    public void searchIntervals_case7() {
        // Case 7: Single interval, k is out of range
        List<Interval> intervals = Arrays.asList(
                new Interval(1, 3, 10)
        );
        int k = 0;
        int result = dp.binarySearchFinish(intervals, k);
        assertEquals(-1, result);
    }

    //Weighted Scheduling testcases
    @Test
    public void testGetSchedule_case1() {
        // Test with valid input
        List<Integer> startTime = Arrays.asList(1, 2, 3, 3);
        List<Integer> endTime = Arrays.asList(3, 4, 5, 6);
        List<Integer> weights = Arrays.asList(50, 10, 40, 70);

        int result = dp.getSchedule(startTime, endTime, weights);
        assertEquals(120, result);
    }

    @Test
    public void testGetSchedule_case2() {
        // Test with empty input
        List<Integer> startTime = List.of();
        List<Integer> endTime = List.of();
        List<Integer> weights = List.of();

        int result = dp.getSchedule(startTime, endTime, weights);
        assertEquals(0, result);
    }

    @Test
    public void testGetSchedule_case3() {
        // Test with mismatched input sizes
        List<Integer> startTime = Arrays.asList(1, 2);
        List<Integer> endTime = Arrays.asList(3, 4, 5);
        List<Integer> weights = Arrays.asList(50, 10);

        int result = dp.getSchedule(startTime, endTime, weights);
        assertEquals(-1, result);
    }

    @Test
    public void testGetSchedule_case4() {
        // Test with non-positive weights
        List<Integer> startTime = Arrays.asList(1, 2, 3);
        List<Integer> endTime = Arrays.asList(3, 4, 5);
        List<Integer> weights = Arrays.asList(50, -10, 40);

        int result = dp.getSchedule(startTime, endTime, weights);
        assertEquals(-1, result);
    }

    @Test
    public void testGetSchedule_case5() {
        // Test with invalid intervals (start time >= end time)
        List<Integer> startTime = Arrays.asList(3, 4, 5);
        List<Integer> endTime = Arrays.asList(3, 4, 6);
        List<Integer> weights = Arrays.asList(50, 10, 40);

        int result = dp.getSchedule(startTime, endTime, weights);
        assertEquals(-1, result);
    }

    @Test
    public void testGetSchedule_case6() {
        // Test with non-overlapping intervals
        List<Integer> startTime = Arrays.asList(1, 4, 6);
        List<Integer> endTime = Arrays.asList(3, 5, 8);
        List<Integer> weights = Arrays.asList(50, 60, 70);

        int result = dp.getSchedule(startTime, endTime, weights);
        assertEquals(180, result);
    }

    @Test
    public void testGetSchedule_case7() {
        // Test with overlapping intervals
        List<Integer> startTime = Arrays.asList(1, 2, 3, 3);
        List<Integer> endTime = Arrays.asList(4, 5, 6, 7);
        List<Integer> weights = Arrays.asList(20, 30, 40, 50);

        int result = dp.getSchedule(startTime, endTime, weights);
        assertEquals(50, result);
    }

    // Word Break Tests
    @Test
    public void testWordBreak_case1() {
        List<String> wordDict = Arrays.asList("leet", "code");
        assertTrue(dp.wordBreak("leetcode", wordDict));
    }

    @Test
    public void testWordBreak_case2() {
        List<String> wordDict = Arrays.asList("apple", "pen");
        assertTrue(dp.wordBreak("applepenapple", wordDict));
    }

    @Test
    public void testWordBreak_case3() {
        List<String> wordDict = Arrays.asList("apple", "pen");
        assertTrue(dp.wordBreak("", wordDict));
    }

    @Test
    public void testWordBreak_case4() {
        List<String> wordDict = Arrays.asList("apple", "pen");
        assertFalse(dp.wordBreak("applepenapplee", wordDict));
    }

    // Coin Change Tests
    @Test
    public void testCoinChange_case1() {
        List<Integer> coins = Arrays.asList(1, 2, 5);
        assertEquals(3, dp.coinChange(coins, 11));
    }

    @Test
    public void testCoinChange_case2() {
        List<Integer> coins = List.of(2);
        assertEquals(-1, dp.coinChange(coins, 3));
    }

    @Test
    public void testCoinChange_case3() {
        List<Integer> coins = Arrays.asList(1, 2, 5);
        assertEquals(-1, dp.coinChange(coins, -2));
    }

    @Test
    public void testCoinChange_case4() {
        List<Integer> coins = List.of();
        assertEquals(-1, dp.coinChange(coins, 3));
    }

    @Test
    public void testCoinChange_case5() {
        List<Integer> coins = List.of(5,3,-1);
        assertEquals(-1, dp.coinChange(coins, 8));
    }

    @Test
    public void testCoinChange_case6() {
        List<Integer> coins = List.of(5,3,0);
        assertEquals(-1, dp.coinChange(coins, 8));
    }

    // Longest Common Subsequence Tests
    @Test
    public void testLongestCommonSubsequence_case1() {
        assertEquals(3, dp.longestCommonSubsequence("abcde", "ace"));
    }

    @Test
    public void testLongestCommonSubsequence_case2() {
        assertEquals(0, dp.longestCommonSubsequence("abc", "def"));
    }

    //Recursive Catalan Tests
    @Test
    public void catalan_recursive_case1() {
        // Case 1: Base case n = 0
        int n = 0;
        int result = dp.catalan_recursive(n);
        assertEquals(1, result);
    }

    @Test
    public void catalan_recursive_case2() {
        // Case 2: Valid n = 3
        int n = 3;
        int result = dp.catalan_recursive(n);
        assertEquals(5, result);
    }

    @Test
    public void catalan_recursive_case3() {
        // Case 3: Invalid n = -1
        int n = -1;
        int result = dp.catalan_recursive(n);
        assertEquals(-1, result);
    }

    //Catalan Closed Form

    @Test
    public void catalan_closed_form_case1() {
        // Case 1: Base case n = 0
        int n = 0;
        int result = dp.catalan_closed_form(n);
        assertEquals(1, result);
    }

    @Test
    public void catalan_closed_form_case2() {
        // Case 2: Valid n = 3
        int n = 3;
        int result = dp.catalan_closed_form(n);
        assertEquals(5, result);
    }

    @Test
    public void catalan_closed_form_case3() {
        // Case 3: Invalid n = -1
        int n = -1;
        int result = dp.catalan_closed_form(n);
        assertEquals(-1, result);
    }

    //Factorial Testcases

    @Test
    public void factorial_case1() {
        // Case 1: n = 0 (base case)
        int n = 0;
        int result = dp.factorial(n);
        assertEquals(1, result);
    }

    @Test
    public void factorial_case2() {
        // Case 2: n = 5 (valid input)
        int n = 5;
        int result = dp.factorial(n);
        assertEquals(120, result);
    }

    @Test
    public void factorial_case3() {
        // Case 3: n = -1 (invalid input)
        int n = -1;
        int result = dp.factorial(n);
        assertEquals(-1, result);
    }

    //Stirling Number Testcases

    @Test
    public void stirling_number_case1() {
        // Case 1: n = 0, r = 0 (base case)
        int n = 0, r = 0;
        int result = dp.stirling_number(r, n);
        assertEquals(1, result);
    }

    @Test
    public void stirling_number_case2() {
        // Case 2: n = 5, r = 5 (n == r)
        int n = 5, r = 5;
        int result = dp.stirling_number(r, n);
        assertEquals(1, result);
    }

    @Test
    public void stirling_number_case3() {
        // Case 3: r = 3, n = 2 (valid input)
        int r = 3, n = 2;
        int result = dp.stirling_number(r, n);
        assertEquals(3, result); // Stirling number S(3, 2) = 3
    }

    @Test
    public void stirling_number_case4() {
        // Case 4: r = 2, n = 3 (n > r)
        int r = 2, n = 3;
        int result = dp.stirling_number(r, n);
        assertEquals(0, result);
    }

    @Test
    public void stirling_number_case5() {
        // Case 5: n = -1, r = 3 (invalid input)
        int n = -1, r = 3;
        int result = dp.stirling_number(r, n);
        assertEquals(-1, result);
    }

    @Test
    public void stirling_number_case6() {
        // Case 5: n = 3, r = 9 (valid input)
        int n = 3, r = 9;
        int result = dp.stirling_number(r, n);
        assertEquals(3025, result);
    }

    //Min Distance Test

    @Test
    public void minDistance_case1() {
        // Case 1: Both strings are empty
        String word1 = "";
        String word2 = "";
        int result = dp.minDistance(word1, word2);
        assertEquals(0, result);
    }

    @Test
    public void minDistance_case2() {
        // Case 2: One string is empty
        String word1 = "abc";
        String word2 = "";
        int result = dp.minDistance(word1, word2);
        assertEquals(3, result); // Deleting all characters in "abc"
    }

    @Test
    public void minDistance_case3() {
        // Case 3: Both strings are identical
        String word1 = "abc";
        String word2 = "abc";
        int result = dp.minDistance(word1, word2);
        assertEquals(0, result); // No edits required
    }

    @Test
    public void minDistance_case4() {
        // Case 4: One string is a substring of the other
        String word1 = "abc";
        String word2 = "abcd";
        int result = dp.minDistance(word1, word2);
        assertEquals(1, result); // Add 'd' to "abc"
    }

    @Test
    public void minDistance_case5() {
        // Case 5: General case with edits required
        String word1 = "horse";
        String word2 = "ros";
        int result = dp.minDistance(word1, word2);
        assertEquals(3, result); // Replace 'h' -> 'r', delete 'o', delete 'e'
    }
    
    //Matrix Multiplication Test

    @Test
    public void matrixMultiplication_case1() {
        // Case 1: Array has fewer than 2 elements
        int[] arr = {10};
        int result = dp.matrixMultiplication(arr);
        assertEquals(-1, result);
    }

    @Test
    public void matrixMultiplication_case2() {
        // Case 2: Array contains a non-positive dimension
        int[] arr = {10, -20, 30};
        int result = dp.matrixMultiplication(arr);
        assertEquals(-1, result);
    }

    @Test
    public void matrixMultiplication_case3() {
        // Case 3: Minimal valid input (2 matrices)
        int[] arr = {10, 20, 30};
        int result = dp.matrixMultiplication(arr);
        assertEquals(6000, result); // 10x20 * 20x30 = 6000
    }

    @Test
    public void matrixMultiplication_case4() {
        // Case 4: Valid input with multiple matrices
        int[] arr = {10, 20, 30, 40, 30};
        int result = dp.matrixMultiplication(arr);
        assertEquals(30000, result); // Optimal cost calculation
    }

    @Test
    public void matrixMultiplication_case5() {
        // Case 5: Array with all equal dimensions
        int[] arr = {10, 10, 10, 10};
        int result = dp.matrixMultiplication(arr);
        assertEquals(2000, result); // Optimal multiplication cost
    }

    @Test
    public void matrixMultiplication_case6() {
        // Case 5: Array with 2 elements
        int[] arr = {10, 20};
        int result = dp.matrixMultiplication(arr);
        assertEquals(0, result); // Optimal multiplication cost
    }

    @Test
    public void matrixMultiplication_case7() {
        // Case 5: Array with 2 elements
        int[] arr = {10, 20, 0};
        int result = dp.matrixMultiplication(arr);
        assertEquals(-1, result); // Optimal multiplication cost
    }
    
    //Max Product Testcases
    @Test
    public void maxProduct_case1() {
        // Case 1: Empty array
        int[] nums = {};
        int result = dp.maxProduct(nums);
        assertEquals(0, result);
    }

    @Test
    public void maxProduct_case2() {
        // Case 2: Single positive number
        int[] nums = {5};
        int result = dp.maxProduct(nums);
        assertEquals(5, result);
    }

    @Test
    public void maxProduct_case3() {
        // Case 3: Single negative number
        int[] nums = {-3};
        int result = dp.maxProduct(nums);
        assertEquals(-3, result);
    }

    @Test
    public void maxProduct_case4() {
        // Case 4: Multiple positive numbers
        int[] nums = {1, 2, 3, 4};
        int result = dp.maxProduct(nums);
        assertEquals(24, result); // 1*2*3*4 = 24
    }

    @Test
    public void maxProduct_case5() {
        // Case 5: Mixed positive and negative numbers
        int[] nums = {2, 3, -2, 4};
        int result = dp.maxProduct(nums);
        assertEquals(6, result); // 2*3 = 6
    }

    @Test
    public void maxProduct_case6() {
        // Case 6: Contains zero
        int[] nums = {-2, 0, -1};
        int result = dp.maxProduct(nums);
        assertEquals(0, result); // Single 0 is the max product
    }

    @Test
    public void maxProduct_case7() {
        // Case 7: Mixed positive, negative, and zero
        int[] nums = {-2, 3, -4};
        int result = dp.maxProduct(nums);
        assertEquals(24, result); // (-2)*(-4)*3 = 24
    }

    @Test
    public void maxProduct_case8() {
        // Case 8: All negative numbers
        int[] nums = {-1, -3, -10, -2};
        int result = dp.maxProduct(nums);
        assertEquals(60, result);
    }

    @Test
    public void maxProduct_case9() {
        // Case 6: Contains zero
        int[] nums = {2, 0, 1};
        int result = dp.maxProduct(nums);
        assertEquals(2, result); // Single 0 is the max product
    }

    // Fibonacci Tests
    @Test
    public void testFibonacci_case1() {
        assertEquals(0, dp.fibonacci(0));
    }

    @Test
    public void testFibonacci_case2() {
        assertEquals(21, dp.fibonacci(8));
    }

    @Test
    public void testFibonacci_case3() {
        assertEquals(-1, dp.fibonacci(-1));
    }

    @Test
    public void testFibonacci_case4() {
        assertEquals(1, dp.fibonacci(1));
    }

    //Binomial Coefficients Test

    @Test
    public void binomialCoefficient_case1() {
        // Case 1: Invalid input, k < 0
        int result = dp.binomialCoefficient(5, -1);
        assertEquals(-1, result);
    }

    @Test
    public void binomialCoefficient_case2() {
        // Case 2: Invalid input, k > n
        int result = dp.binomialCoefficient(4, 5);
        assertEquals(-1, result);
    }

    @Test
    public void binomialCoefficient_case3() {
        // Case 3: k = 0 (base case)
        int result = dp.binomialCoefficient(5, 0);
        assertEquals(1, result);
    }

    @Test
    public void binomialCoefficient_case4() {
        // Case 4: n = k (base case)
        int result = dp.binomialCoefficient(4, 4);
        assertEquals(1, result);
    }

    @Test
    public void binomialCoefficient_case5() {
        // Case 5: General case, small n and k
        int result = dp.binomialCoefficient(5, 2);
        assertEquals(10, result); // C(5, 2) = 10
    }

    @Test
    public void binomialCoefficient_case6() {
        // Case 6: General case, larger n and k
        int result = dp.binomialCoefficient(10, 3);
        assertEquals(120, result); // C(10, 3) = 120
    }

    //Derangement Tests

    @Test
    public void derangementCount_case1() {
        // Case 1: Invalid input, n < 0
        int result = dp.derangementCount(-1);
        assertEquals(-1, result);
    }

    @Test
    public void derangementCount_case2() {
        // Case 2: Base case, n = 0
        int result = dp.derangementCount(0);
        assertEquals(1, result); // D(0) = 1
    }

    @Test
    public void derangementCount_case3() {
        // Case 3: Base case, n = 1
        int result = dp.derangementCount(1);
        assertEquals(0, result); // D(1) = 0
    }

    @Test
    public void derangementCount_case4() {
        // Case 4: Small n, n = 2
        int result = dp.derangementCount(2);
        assertEquals(1, result); // D(2) = 1
    }

    @Test
    public void derangementCount_case5() {
        // Case 5: Small n, n = 3
        int result = dp.derangementCount(3);
        assertEquals(2, result); // D(3) = 2
    }

    @Test
    public void derangementCount_case6() {
        // Case 6: General case, n = 4
        int result = dp.derangementCount(4);
        assertEquals(9, result); // D(4) = 9
    }

    @Test
    public void derangementCount_case7() {
        // Case 7: Larger n, n = 5
        int result = dp.derangementCount(5);
        assertEquals(44, result); // D(5) = 44
    }
}
