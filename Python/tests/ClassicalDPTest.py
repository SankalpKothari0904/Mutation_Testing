import sys
import os
import unittest
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))
from ClassicalDP import ClassicalDP, Interval

class ClassicalDPTest(unittest.TestCase):
    def setUp(self):
        self.dp = ClassicalDP()

    # Knapsack tests
    def test_knapsack_case1(self):
        vals = [60, 100, 120]
        weights = [10, 20, 30]
        self.assertEqual(self.dp.knapSack(50, vals, weights), 220)

    def test_knapsack_case2(self):
        vals = [10, 20]
        weights = [5, 5]
        self.assertEqual(self.dp.knapSack(5, vals, weights), 20)

    def test_knapsack_case3(self):
        vals = [1, 2]
        weights = [1, 3]
        self.assertEqual(self.dp.knapSack(-2, vals, weights), -1)

    def test_knapsack_case4(self):
        vals = [1, 2]
        weights = [1, 3]
        self.assertEqual(self.dp.knapSack(0, vals, weights), 0)

    def test_knapsack_case5(self):
        vals = [1, 2, 4]
        weights = [1, 3]
        self.assertEqual(self.dp.knapSack(5, vals, weights), -1)

    def test_knapsack_case6(self):
        vals = []
        weights = []
        self.assertEqual(self.dp.knapSack(5, vals, weights), -1)

    def test_knapsack_case7(self):
        vals = [1, 3, -1]
        weights = [2, 4, 5]
        self.assertEqual(self.dp.knapSack(5, vals, weights), -1)

    def test_knapsack_case8(self):
        vals = [1, 3, 1]
        weights = [2, 4, -5]
        self.assertEqual(self.dp.knapSack(5, vals, weights), -1)

    def test_knapsack_case9(self):
        vals = [1, 3, 0]
        weights = [2, 4, -5]
        self.assertEqual(self.dp.knapSack(5, vals, weights), -1)

    def test_knapsack_case10(self):
        vals = [1, 3, 1]
        weights = [2, 4, 0]
        self.assertEqual(self.dp.knapSack(5, vals, weights), -1)

    # Longest Increasing Subsequence tests
    def test_longest_increasing_subsequence_case1(self):
        nums = [10, 9, 2, 5, 3, 7, 101, 18]
        self.assertEqual(self.dp.longestIncreasingSubsequence(nums), 4)

    def test_longest_increasing_subsequence_case2(self):
        nums = [0, 1, 0, 3, 2, 3]
        self.assertEqual(self.dp.longestIncreasingSubsequence(nums), 4)

    def test_longest_increasing_subsequence_case3(self):
        nums = [0, 1, 2, 2]
        self.assertEqual(self.dp.longestIncreasingSubsequence(nums), 3)

    def test_longest_increasing_subsequence_case4(self):
        nums = [0, 1, 2, 0, 2]
        self.assertEqual(self.dp.longestIncreasingSubsequence(nums), 3)

    # Longest Palindrome Tests
    def test_longest_palindrome_case1(self):
        self.assertEqual(self.dp.longestPalindrome("bbbab"), 4)

    def test_longest_palindrome_case2(self):
        self.assertEqual(self.dp.longestPalindrome("abcde"), 1)

    # House Robber Tests
    def test_house_robber_case1(self):
        input_data = [2, 7, 9, 3, 1]
        self.assertEqual(self.dp.house_robber(input_data), 12)

    def test_house_robber_case2(self):
        input_data = [1, 2, 3, 1]
        self.assertEqual(self.dp.house_robber(input_data), 4)

    def test_house_robber_case3(self):
        input_data = [1]
        self.assertEqual(self.dp.house_robber(input_data), 1)

    def test_house_robber_case4(self):
        input_data = [1, 2]
        self.assertEqual(self.dp.house_robber(input_data), 2)

    # Binary Search Intervals Test Cases
    def test_search_intervals_case1(self):
        # Case 1: Valid intervals, k is within the range
        intervals = [
            Interval(1, 3, 10),
            Interval(2, 5, 20),
            Interval(4, 8, 30),
        ]
        k = 5
        result = self.dp.binarySearchFinish(intervals, k)
        self.assertEqual(result, 1)

    def test_search_intervals_case2(self):
        # Case 2: No intervals with end time <= k
        intervals = [
            Interval(1, 3, 10),
            Interval(4, 6, 20),
            Interval(7, 9, 30),
        ]
        k = 2
        result = self.dp.binarySearchFinish(intervals, k)
        self.assertEqual(result, -1)

    def test_search_intervals_case3(self):
        # Case 3: k matches the end time of the last interval
        intervals = [
            Interval(1, 3, 10),
            Interval(2, 5, 20),
            Interval(4, 8, 30),
        ]
        k = 8
        result = self.dp.binarySearchFinish(intervals, k)
        self.assertEqual(result, 2)

    def test_search_intervals_case4(self):
        # Case 4: k is greater than all end times
        intervals = [
            Interval(1, 3, 10),
            Interval(2, 5, 20),
            Interval(4, 8, 30),
        ]
        k = 10
        result = self.dp.binarySearchFinish(intervals, k)
        self.assertEqual(result, 2)

    def test_search_intervals_case5(self):
        # Case 5: k is less than all end times
        intervals = [
            Interval(1, 3, 10),
            Interval(4, 6, 20),
            Interval(7, 9, 30),
        ]
        k = 0
        result = self.dp.binarySearchFinish(intervals, k)
        self.assertEqual(result, -1)

    def test_search_intervals_case6(self):
        # Case 6: Single interval, k is within range
        intervals = [
            Interval(1, 3, 10),
        ]
        k = 3
        result = self.dp.binarySearchFinish(intervals, k)
        self.assertEqual(result, 0)

    def test_search_intervals_case7(self):
        # Case 7: Single interval, k is out of range
        intervals = [
            Interval(1, 3, 10),
        ]
        k = 0
        result = self.dp.binarySearchFinish(intervals, k)
        self.assertEqual(result, -1)

    # Weighted Scheduling Test Cases
    def test_get_schedule_case1(self):
        # Test with valid input
        start_time = [1, 2, 3, 3]
        end_time = [3, 4, 5, 6]
        weights = [50, 10, 40, 70]

        result = self.dp.get_schedule(start_time, end_time, weights)
        self.assertEqual(result, 120)

    def test_get_schedule_case2(self):
        # Test with empty input
        start_time = []
        end_time = []
        weights = []

        result = self.dp.get_schedule(start_time, end_time, weights)
        self.assertEqual(result, 0)

    def test_get_schedule_case3(self):
        # Test with mismatched input sizes
        start_time = [1, 2]
        end_time = [3, 4, 5]
        weights = [50, 10]

        result = self.dp.get_schedule(start_time, end_time, weights)
        self.assertEqual(result, -1)

    def test_get_schedule_case4(self):
        # Test with non-positive weights
        start_time = [1, 2, 3]
        end_time = [3, 4, 5]
        weights = [50, -10, 40]

        result = self.dp.get_schedule(start_time, end_time, weights)
        self.assertEqual(result, -1)

    def test_get_schedule_case5(self):
        # Test with invalid intervals (start time >= end time)
        start_time = [3, 4, 5]
        end_time = [3, 4, 6]
        weights = [50, 10, 40]

        result = self.dp.get_schedule(start_time, end_time, weights)
        self.assertEqual(result, -1)

    def test_get_schedule_case6(self):
        # Test with non-overlapping intervals
        start_time = [1, 4, 6]
        end_time = [3, 5, 8]
        weights = [50, 60, 70]

        result = self.dp.get_schedule(start_time, end_time, weights)
        self.assertEqual(result, 180)

    def test_get_schedule_case7(self):
        # Test with overlapping intervals
        start_time = [1, 2, 3, 3]
        end_time = [4, 5, 6, 7]
        weights = [20, 30, 40, 50]

        result = self.dp.get_schedule(start_time, end_time, weights)
        self.assertEqual(result, 50)

    # Word Break Test Cases
    def test_word_break_case1(self):
        word_dict = ["leet", "code"]
        result = self.dp.wordBreak("leetcode", word_dict)
        self.assertTrue(result)

    def test_word_break_case2(self):
        word_dict = ["apple", "pen"]
        result = self.dp.wordBreak("applepenapple", word_dict)
        self.assertTrue(result)

    def test_word_break_case3(self):
        word_dict = ["apple", "pen"]
        result = self.dp.wordBreak("", word_dict)
        self.assertTrue(result)

    def test_word_break_case4(self):
        word_dict = ["apple", "pen"]
        result = self.dp.wordBreak("applepenapplee", word_dict)
        self.assertFalse(result)

    # Coin Change Test Cases
    def test_coin_change_case1(self):
        coins = [1, 2, 5]
        result = self.dp.coinChange(coins, 11)
        self.assertEqual(result, 3)

    def test_coin_change_case2(self):
        coins = [2]
        result = self.dp.coinChange(coins, 3)
        self.assertEqual(result, -1)

    def test_coin_change_case3(self):
        coins = [1, 2, 5]
        result = self.dp.coinChange(coins, -2)
        self.assertEqual(result, -1)

    def test_coin_change_case4(self):
        coins = []
        result = self.dp.coinChange(coins, 3)
        self.assertEqual(result, -1)

    def test_coin_change_case5(self):
        coins = [5, 3, -1]
        result = self.dp.coinChange(coins, 8)
        self.assertEqual(result, -1)

    def test_coin_change_case6(self):
        coins = [5, 3, 0]
        result = self.dp.coinChange(coins, 8)
        self.assertEqual(result, -1)

    # Longest Common Subsequence Test Cases
    def test_longest_common_subsequence_case1(self):
        result = self.dp.longestCommonSubsequence("abcde", "ace")
        self.assertEqual(result, 3)

    def test_longest_common_subsequence_case2(self):
        result = self.dp.longestCommonSubsequence("abc", "def")
        self.assertEqual(result, 0)

    # Recursive Catalan Number Test Cases
    def test_catalan_recursive_case1(self):
        # Base case n = 0
        result = self.dp.catalan_recursive(0)
        self.assertEqual(result, 1)

    def test_catalan_recursive_case2(self):
        # Valid n = 3
        result = self.dp.catalan_recursive(3)
        self.assertEqual(result, 5)

    def test_catalan_recursive_case3(self):
        # Invalid n = -1
        result = self.dp.catalan_recursive(-1)
        self.assertEqual(result, -1)

    # Catalan Closed Form Test Cases
    def test_catalan_closed_form_case1(self):
        # Case 1: Base case n = 0
        n = 0
        result = self.dp.catalan_closed_form(n)
        self.assertEqual(result, 1)

    def test_catalan_closed_form_case2(self):
        # Case 2: Valid n = 3
        n = 3
        result = self.dp.catalan_closed_form(n)
        self.assertEqual(result, 5)

    def test_catalan_closed_form_case3(self):
        # Case 3: Invalid n = -1
        n = -1
        result = self.dp.catalan_closed_form(n)
        self.assertEqual(result, -1)

    # Factorial Test Cases
    def test_factorial_case1(self):
        # Case 1: n = 0 (base case)
        n = 0
        result = self.dp.factorial(n)
        self.assertEqual(result, 1)

    def test_factorial_case2(self):
        # Case 2: n = 5 (valid input)
        n = 5
        result = self.dp.factorial(n)
        self.assertEqual(result, 120)

    def test_factorial_case3(self):
        # Case 3: n = -1 (invalid input)
        n = -1
        result = self.dp.factorial(n)
        self.assertEqual(result, -1)

    # Stirling Number Test Cases
    def test_stirling_number_case1(self):
        # Case 1: n = 0, r = 0 (base case)
        n, r = 0, 0
        result = self.dp.stirling_number(r, n)
        self.assertEqual(result, 1)

    def test_stirling_number_case2(self):
        # Case 2: n = 5, r = 5 (n == r)
        n, r = 5, 5
        result = self.dp.stirling_number(r, n)
        self.assertEqual(result, 1)

    def test_stirling_number_case3(self):
        # Case 3: n = 2, r = 3 (valid input)
        n, r = 2, 3
        result = self.dp.stirling_number(r, n)
        self.assertEqual(result, 3)  # S(3, 2) = 3

    def test_stirling_number_case4(self):
        # Case 4: n = 3, r = 2 (n > r)
        n, r = 3, 2
        result = self.dp.stirling_number(r, n)
        self.assertEqual(result, 0)

    def test_stirling_number_case5(self):
        # Case 5: n = -1, r = 3 (invalid input)
        n, r = -1, 3
        result = self.dp.stirling_number(r, n)
        self.assertEqual(result, -1)

    def test_stirling_number_case6(self):
        # Case 6: n = 3, r = 9 (valid input)
        n, r = 3, 9
        result = self.dp.stirling_number(r, n)
        self.assertEqual(result, 3025)

    # Min distance test cases
    def test_min_distance_case1(self):
        # Case 1: Both strings are empty
        word1 = ""
        word2 = ""
        result = self.dp.min_distance(word1, word2)
        self.assertEqual(result, 0)

    def test_min_distance_case2(self):
        # Case 2: One string is empty
        word1 = "abc"
        word2 = ""
        result = self.dp.min_distance(word1, word2)
        self.assertEqual(result, 3)  # Deleting all characters in "abc"

    def test_min_distance_case3(self):
        # Case 3: Both strings are identical
        word1 = "abc"
        word2 = "abc"
        result = self.dp.min_distance(word1, word2)
        self.assertEqual(result, 0)  # No edits required

    def test_min_distance_case4(self):
        # Case 4: One string is a substring of the other
        word1 = "abc"
        word2 = "abcd"
        result = self.dp.min_distance(word1, word2)
        self.assertEqual(result, 1)  # Add 'd' to "abc"

    def test_min_distance_case5(self):
        # Case 5: General case with edits required
        word1 = "horse"
        word2 = "ros"
        result = self.dp.min_distance(word1, word2)
        self.assertEqual(result, 3)  # Replace 'h' -> 'r', delete 'o', delete 'e'

    # Matrix Chain Multiplication Tests
    def test_matrix_multiplication_case1(self):
        # Case 1: Array has fewer than 2 elements
        arr = [10]
        result = self.dp.matrix_multiplication(arr)
        self.assertEqual(result, -1)

    def test_matrix_multiplication_case2(self):
        # Case 2: Array contains a non-positive dimension
        arr = [10, -20, 30]
        result = self.dp.matrix_multiplication(arr)
        self.assertEqual(result, -1)

    def test_matrix_multiplication_case3(self):
        # Case 3: Minimal valid input (2 matrices)
        arr = [10, 20, 30]
        result = self.dp.matrix_multiplication(arr)
        self.assertEqual(result, 6000)  # 10x20 * 20x30 = 6000

    def test_matrix_multiplication_case4(self):
        # Case 4: Valid input with multiple matrices
        arr = [10, 20, 30, 40, 30]
        result = self.dp.matrix_multiplication(arr)
        self.assertEqual(result, 30000)  # Optimal cost calculation

    def test_matrix_multiplication_case5(self):
        # Case 5: Array with all equal dimensions
        arr = [10, 10, 10, 10]
        result = self.dp.matrix_multiplication(arr)
        self.assertEqual(result, 2000)  # Optimal multiplication cost

    def test_matrix_multiplication_case6(self):
        # Case 6: Array with 2 elements
        arr = [10, 20]
        result = self.dp.matrix_multiplication(arr)
        self.assertEqual(result, 0)  # No multiplication required

    # Maximum Product Subarray Tests
    def test_max_product_case1(self):
        # Case 1: Empty array
        nums = []
        result = self.dp.max_product(nums)
        self.assertEqual(result, 0)

    def test_max_product_case2(self):
        # Case 2: Single positive number
        nums = [5]
        result = self.dp.max_product(nums)
        self.assertEqual(result, 5)

    def test_max_product_case3(self):
        # Case 3: Single negative number
        nums = [-3]
        result = self.dp.max_product(nums)
        self.assertEqual(result, -3)

    def test_max_product_case4(self):
        # Case 4: Multiple positive numbers
        nums = [1, 2, 3, 4]
        result = self.dp.max_product(nums)
        self.assertEqual(result, 24)  # 1*2*3*4 = 24

    def test_max_product_case5(self):
        # Case 5: Mixed positive and negative numbers
        nums = [2, 3, -2, 4]
        result = self.dp.max_product(nums)
        self.assertEqual(result, 6)  # 2*3 = 6

    def test_max_product_case6(self):
        # Case 6: Contains zero
        nums = [-2, 0, -1]
        result = self.dp.max_product(nums)
        self.assertEqual(result, 0)  # Max product is 0

    def test_max_product_case7(self):
        # Case 7: Mixed positive, negative, and zero
        nums = [-2, 3, -4]
        result = self.dp.max_product(nums)
        self.assertEqual(result, 24)  # (-2)*(-4)*3 = 24

    def test_max_product_case8(self):
        # Case 8: All negative numbers
        nums = [-1, -3, -10, -2]
        result = self.dp.max_product(nums)
        self.assertEqual(result, 60)

    def test_max_product_case9(self):
        # Case 9: Contains zero
        nums = [2, 0, 1]
        result = self.dp.max_product(nums)
        self.assertEqual(result, 2)

    # Fibonacci Test Cases
    def test_fibonacci_case1(self):
        # Case 1: n = 0 (base case)
        self.assertEqual(self.dp.fibonacci(0), 0)

    def test_fibonacci_case2(self):
        # Case 2: Valid n = 8
        self.assertEqual(self.dp.fibonacci(8), 21)

    def test_fibonacci_case3(self):
        # Case 3: Invalid n = -1
        self.assertEqual(self.dp.fibonacci(-1), -1)

    def test_fibonacci_case4(self):
        # Case 4: n = 1 (base case)
        self.assertEqual(self.dp.fibonacci(1), 1)

    # Binomial Coefficient Tests
    def test_binomial_coefficient_case1(self):
        # Case 1: Invalid input, k < 0
        result = self.dp.binomial_coefficient(5, -1)
        self.assertEqual(result, -1)

    def test_binomial_coefficient_case2(self):
        # Case 2: Invalid input, k > n
        result = self.dp.binomial_coefficient(4, 5)
        self.assertEqual(result, -1)

    def test_binomial_coefficient_case3(self):
        # Case 3: k = 0 (base case)
        result = self.dp.binomial_coefficient(5, 0)
        self.assertEqual(result, 1)

    def test_binomial_coefficient_case4(self):
        # Case 4: n = k (base case)
        result = self.dp.binomial_coefficient(4, 4)
        self.assertEqual(result, 1)

    def test_binomial_coefficient_case5(self):
        # Case 5: General case, small n and k
        result = self.dp.binomial_coefficient(5, 2)
        self.assertEqual(result, 10)  # C(5, 2) = 10

    def test_binomial_coefficient_case6(self):
        # Case 6: General case, larger n and k
        result = self.dp.binomial_coefficient(10, 3)
        self.assertEqual(result, 120)  # C(10, 3) = 120

    # Derangement Tests
    def test_derangement_count_case1(self):
        # Case 1: Invalid input, n < 0
        result = self.dp.derangement_count(-1)
        self.assertEqual(result, -1)

    def test_derangement_count_case2(self):
        # Case 2: Base case, n = 0
        result = self.dp.derangement_count(0)
        self.assertEqual(result, 1)  # D(0) = 1

    def test_derangement_count_case3(self):
        # Case 3: Base case, n = 1
        result = self.dp.derangement_count(1)
        self.assertEqual(result, 0)  # D(1) = 0

    def test_derangement_count_case4(self):
        # Case 4: Small n, n = 2
        result = self.dp.derangement_count(2)
        self.assertEqual(result, 1)  # D(2) = 1

    def test_derangement_count_case5(self):
        # Case 5: Small n, n = 3
        result = self.dp.derangement_count(3)
        self.assertEqual(result, 2)  # D(3) = 2

    def test_derangement_count_case6(self):
        # Case 6: General case, n = 4
        result = self.dp.derangement_count(4)
        self.assertEqual(result, 9)  # D(4) = 9

    def test_derangement_count_case7(self):
        # Case 7: Larger n, n = 5
        result = self.dp.derangement_count(5)
        self.assertEqual(result, 44)  # D(5) = 44


if __name__ == "__main__":
    unittest.main()
