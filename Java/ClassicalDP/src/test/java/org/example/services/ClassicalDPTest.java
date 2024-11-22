package org.example.services;


import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import static org.junit.Assert.*;

public class ClassicalDPTest {

    private final ClassicalDP dp = new ClassicalDP();

    // KnapSack Tests
    @Test
    public void testKnapSack_case1() {
        List<Integer> vals = Arrays.asList(60, 100, 120);
        List<Integer> weights = Arrays.asList(10, 20, 30);
        assertEquals(220, dp.knapSack(50, vals, weights));
    }

    @Test
    public void testKnapSack_case2() {
        List<Integer> vals = Arrays.asList(10, 20);
        List<Integer> weights = Arrays.asList(5, 5);
        assertEquals(20, dp.knapSack(5, vals, weights));
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

    // Coin Change Tests
    @Test
    public void testCoinChange_case1() {
        List<Integer> coins = Arrays.asList(1, 2, 5);
        assertEquals(3, dp.coinChange(coins, 11));
    }

    @Test
    public void testCoinChange_case2() {
        List<Integer> coins = Arrays.asList(2);
        assertEquals(-1, dp.coinChange(coins, 3));
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

    // Fibonacci Tests
    @Test
    public void testFibonacci_case1() {
        assertEquals(0, dp.fibonacci(0));
    }

    @Test
    public void testFibonacci_case2() {
        assertEquals(21, dp.fibonacci(8));
    }
}
