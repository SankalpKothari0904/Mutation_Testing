package org.example.services;


import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import static org.junit.Assert.*;

// import org.example.services.ClassicalDP;

public class ClassicalDPTest {

    private final ClassicalDP dp = new ClassicalDP();

    // KnapSack Tests
    @Test
    void testKnapSack_case1() {
        List<Integer> vals = Arrays.asList(60, 100, 120);
        List<Integer> weights = Arrays.asList(10, 20, 30);
        assertEquals(220, dp.knapSack(50, vals, weights));
    }

    @Test
    void testKnapSack_case2() {
        List<Integer> vals = Arrays.asList(10, 20);
        List<Integer> weights = Arrays.asList(5, 5);
        assertEquals(20, dp.knapSack(5, vals, weights));
    }

    // Longest Increasing Subsequence Tests
    @Test
    void testLongestIncreasingSubsequence_case1() {
        List<Integer> nums = Arrays.asList(10, 9, 2, 5, 3, 7, 101, 18);
        assertEquals(4, dp.longestIncreasingSubsequence(nums));
    }

    @Test
    void testLongestIncreasingSubsequence_case2() {
        List<Integer> nums = Arrays.asList(3, 10, 2, 1, 20);
        assertEquals(3, dp.longestIncreasingSubsequence(nums));
    }

    // Longest Palindrome Tests
    @Test
    void testLongestPalindrome_case1() {
        assertEquals(4, dp.longestPalindrome("bbbab"));
    }

    @Test
    void testLongestPalindrome_case2() {
        assertEquals(1, dp.longestPalindrome("abcde"));
    }

    // House Robber Tests
    @Test
    void testHouseRobber_case1() {
        List<Integer> input = Arrays.asList(2, 7, 9, 3, 1);
        assertEquals(12, dp.houseRobber(input));
    }

    @Test
    void testHouseRobber_case2() {
        List<Integer> input = Arrays.asList(1, 2, 3, 1);
        assertEquals(4, dp.houseRobber(input));
    }

    // Word Break Tests
    @Test
    void testWordBreak_case1() {
        List<String> wordDict = Arrays.asList("leet", "code");
        assertTrue(dp.wordBreak("leetcode", wordDict));
    }

    @Test
    void testWordBreak_case2() {
        List<String> wordDict = Arrays.asList("apple", "pen");
        assertTrue(dp.wordBreak("applepenapple", wordDict));
    }

    // Coin Change Tests
    @Test
    void testCoinChange_case1() {
        List<Integer> coins = Arrays.asList(1, 2, 5);
        assertEquals(3, dp.coinChange(coins, 11));
    }

    @Test
    void testCoinChange_case2() {
        List<Integer> coins = Arrays.asList(2);
        assertEquals(-1, dp.coinChange(coins, 3));
    }

    // Longest Common Subsequence Tests
    @Test
    void testLongestCommonSubsequence_case1() {
        assertEquals(3, dp.longestCommonSubsequence("abcde", "ace"));
    }

    @Test
    void testLongestCommonSubsequence_case2() {
        assertEquals(0, dp.longestCommonSubsequence("abc", "def"));
    }

    // Fibonacci Tests
    @Test
    void testFibonacci_case1() {
        assertEquals(0, dp.fibonacci(0));
    }

    @Test
    void testFibonacci_case2() {
        assertEquals(21, dp.fibonacci(8));
    }
}
