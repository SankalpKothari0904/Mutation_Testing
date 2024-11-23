from typing import List

class Interval:
    def __init__(self, start, end, weight):
        self.start = start
        self.end = end
        self.weight = weight

class ClassicalDP():
    
    def knapSack(self, max_weight: int, vals: List[int], weights: List[int]) -> int:
        # Function to solve the 0/1 Knapsack problem using dynamic programming.
        # Valid input conditions:
        # - `max_weight` should be a non-negative integer.
        # - `vals` and `weights` must be non-empty lists of equal length.
        # - All elements in `vals` and `weights` must be positive integers.
        
        if max_weight < 0:
            return -1  # Invalid input: max_weight cannot be negative.
        if len(vals) != len(weights):
            return -1  # Invalid input: `vals` and `weights` must have the same length.
        if len(vals) < 1 or len(weights) < 1:
            return -1  # Invalid input: `vals` and `weights` cannot be empty.
        if any(v <= 0 for v in vals) or any(w <= 0 for w in weights):
            return -1  # Invalid input: All values and weights must be positive.

        n = len(weights)  # Number of items.
        # Initialize a 2D list for dynamic programming:
        # OPT[i][w] represents the maximum value achievable using the first `i` items with a weight limit `w`.
        OPT = [[0] * (max_weight + 1) for _ in range(n + 1)]

        # Base cases: No items or weight capacity of zero results in zero value.
        for w in range(max_weight + 1):
            OPT[0][w] = 0  # No items, max value is 0.
        for i in range(n + 1):
            OPT[i][0] = 0  # Zero weight capacity, max value is 0.

        # Fill the DP table.
        for i in range(1, n + 1):  # Iterate over items.
            for w in range(1, max_weight + 1):  # Iterate over possible weights.
                if w - weights[i - 1] < 0:  # Current item cannot fit in the weight limit.
                    OPT[i][w] = OPT[i - 1][w]  # Exclude the current item.
                else:
                    # Include or exclude the current item, whichever gives the higher value.
                    if OPT[i - 1][w] > OPT[i - 1][w - weights[i - 1]] + vals[i - 1]:
                        OPT[i][w] = OPT[i - 1][w]
                    else:
                        OPT[i][w] = OPT[i - 1][w - weights[i - 1]] + vals[i - 1]

        # Return the maximum value achievable with the given weight capacity and items.
        return OPT[n][max_weight]
    
    def longestIncreasingSubsequence(self, nums: List[int]) -> int:
        # Function to find the length of the longest increasing subsequence (LIS) in a list of integers.
        # Valid input conditions:
        # - `nums` is a list of integers, which can be empty or non-empty.

        n = len(nums)  # Number of elements in the list.

        if (n == 0): 
            return 0  # If the list is empty, the LIS length is 0.

        # OPT[i] stores the length of the LIS ending at index `i`.
        # Initialize all elements to 1, as the minimum LIS for any single element is 1.
        OPT = [1] * n

        # Compute the LIS for each element.
        for i in range(n):  # Iterate over each element as the endpoint of a subsequence.
            for j in range(i - 1, -1, -1):  # Check all previous elements.
                # If nums[j] is less than nums[i], and including nums[i] in the subsequence
                # ending at nums[j] increases its length, update OPT[i].
                if nums[j] < nums[i] and OPT[j] + 1 > OPT[i]:
                    OPT[i] = OPT[j] + 1

        # Find the maximum LIS length from the OPT array.
        max_length = max(OPT)
        return max_length


    def longestPalindrome(self, arg: str) -> int:
        # Function to find the length of the longest palindromic subsequence in a given string.
        # Valid input conditions:
        # - `arg` is a string that can be empty or non-empty.

        n = len(arg)  # Length of the input string.

        if (n == 0):
            return 0  # If the string is empty, the length of the longest palindromic subsequence is 0.

        # OPT[i][j] stores the length of the longest palindromic subsequence 
        # in the substring from index `i` to index `j` (inclusive).
        OPT = [[0] * n for _ in range(n)]

        # Base case: Single-character substrings are palindromes of length 1.
        for i in range(n):
            OPT[i][i] = 1

        # Dynamic programming to compute the length of the longest palindromic subsequence.
        for length in range(1, n):  # `length` is the difference between `j` and `i`.
            for i in range(n - length):  # `i` is the starting index of the substring.
                j = i + length  # `j` is the ending index of the substring.
                if arg[i] == arg[j]:  # If characters at `i` and `j` are equal:
                    OPT[i][j] = OPT[i + 1][j - 1] + 2  # Include both characters.
                else:  # Otherwise, take the maximum length by excluding one character.
                    if OPT[i + 1][j] > OPT[i][j - 1]:
                        OPT[i][j] = OPT[i + 1][j]
                    else:
                        OPT[i][j] = OPT[i][j - 1]

        # The result is stored in OPT[0][n-1], which represents the longest palindromic subsequence
        # for the entire string (from index 0 to n-1).
        return OPT[0][n-1]
    
    def house_robber(self, input: List[int]) -> int:
        # Function to solve the "House Robber" problem using dynamic programming.
        # Valid input conditions:
        # - `input` is a list of non-negative integers representing the amount of money in each house.

        if (len(input) == 0):
            return 0  # If no houses are present, the maximum money that can be robbed is 0.
        elif len(input) == 1:
            return input[0]  # If there is only one house, rob it.
        elif len(input) == 2:
            return max(input[0], input[1])  # If there are two houses, rob the one with more money.

        n = len(input)  # Number of houses.
        # OPT[i] stores the maximum sum of money that can be robbed up to the i-th house, 
        # without alerting the police (no two adjacent houses can be robbed).
        OPT = [-1] * n

        # Initial conditions for the first two houses.
        OPT[0] = input[0]  # Rob the first house.
        OPT[1] = max(input[0], input[1])  # Rob the house with more money between the first two.

        # Fill the OPT array using the recurrence relation:
        # OPT[i] = max(OPT[i-1], OPT[i-2] + input[i])
        for i in range(2, n):
            if OPT[i - 2] + input[i] > OPT[i - 1]:  # Rob the current house.
                OPT[i] = OPT[i - 2] + input[i]
            else:  # Skip the current house.
                OPT[i] = OPT[i - 1]

        # Return the maximum money that can be robbed without alerting the police.
        return OPT[-1]

    
    def binarySearchFinish(self, intervals, k):
        # Function to find the largest index of an interval whose `end` is less than or equal to `k`.
        # Valid input conditions:
        # - `intervals` is a sorted list of objects with an `end` attribute.
        # - `k` is a comparable value (e.g., integer or float).
        # - The `end` attribute of intervals is sorted in ascending order.

        l, r = 0, len(intervals) - 1  # Initialize binary search bounds.

        # Edge case: If the smallest interval's `end` is greater than `k`, return -1 (no valid interval).
        if intervals[0].end > k:
            return -1

        # Binary search loop.
        while r - l > 1:  # Continue until `r` and `l` converge or are adjacent.
            m = l + (r - l) // 2  # Calculate the middle index.
            # Check if `m` is the desired index.
            if intervals[m].end <= k < intervals[m + 1].end:
                return m  # Found the largest valid interval.
            elif intervals[m].end > k:
                r = m - 1  # Narrow the search to the left half.
            else:
                l = m + 1  # Narrow the search to the right half.

        # Final checks after binary search loop.
        if intervals[r].end <= k:
            return r  # Return `r` if its `end` is within the valid range.
        elif intervals[l].end > k:
            return -1  # No valid interval if even `l` is too large.
        else:
            return l  # Return `l` if it is the largest valid interval.


    def get_schedule(self, startTime: List[int], endTime: List[int], weights: List[int]) -> int:
        # Function to find the maximum weight of a non-overlapping interval schedule using dynamic programming.
        # Valid input conditions:
        # - `startTime`, `endTime`, and `weights` are lists of the same length.
        # - All elements in `weights` must be positive.
        # - Each interval's start time must be strictly less than its end time.
        # - Lists may be empty, in which case the result is 0.

        if len(startTime) != len(endTime) or len(startTime) != len(weights):
            return -1  # Invalid input: All input lists must have the same length.
        if len(startTime) == 0:
            return 0  # No intervals to schedule, maximum weight is 0.
        if any(x <= 0 for x in weights):  # Ensure all weights are positive.
            return -1
        if any(startTime[i] >= endTime[i] for i in range(len(startTime))):  # Ensure valid intervals.
            return -1

        # Create a list of Interval objects representing the intervals with their start, end, and weight.
        intervals = [Interval(startTime[i], endTime[i], weights[i]) for i in range(len(startTime))]

        # Sort intervals by their finish time, and by start time as a secondary key for tie-breaking.
        intervals.sort(key=lambda x: (x.end, x.start))

        n = len(intervals)  # Number of intervals.
        OPT = [0] * n  # OPT[i] stores the maximum weight achievable considering intervals up to i.
        lastEnding = [-1] * n  # lastEnding[j] stores the largest index of an interval that ends before intervals[j] starts.

        # Base case: The maximum weight considering only the first interval is its weight.
        OPT[0] = intervals[0].weight

        # Build lastEnding using binary search to find the last non-overlapping interval for each interval.
        for j in range(1, n):
            lastEnding[j] = self.binarySearchFinish(intervals, intervals[j].start)

        # Fill the OPT array using dynamic programming.
        for i in range(1, n):
            # Include the current interval.
            include = intervals[i].weight
            if lastEnding[i] != -1:  # Add the weight of the last non-overlapping interval if it exists.
                include += OPT[lastEnding[i]]

            # Exclude the current interval.
            exclude = OPT[i - 1]

            # Take the maximum of including or excluding the current interval.
            if include > exclude:
                OPT[i] = include
            else:
                OPT[i] = exclude

        # The result is the maximum weight achievable using all intervals.
        return OPT[n-1]
    
    
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # Function to determine if a string can be segmented into a space-separated sequence of dictionary words.
        # Valid input conditions:
        # - `s` is a string to be checked.
        # - `wordDict` is a list of words that can be used to segment `s`.
        # - `wordDict` may contain an empty list, in which case the function returns False.
        
        word_set = set(wordDict)  # Convert the list of words into a set for O(1) lookups.
        n = len(s)  # Length of the string `s`.

        if (n == 0):
            return True  # If the string is empty, it's considered valid.

        dp = [False] * (n + 1)  # dp[i] is True if the substring s[0:i] can be segmented.
        dp[0] = True  # Base case: empty substring is always valid.

        # Iterate over the string `s`.
        for i in range(1, n + 1):
            for j in range(i):
                # If the substring s[j:i] is in the word dictionary and the prefix s[0:j] is valid.
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True  # Mark dp[i] as True, meaning s[0:i] can be segmented.
                    break

        # The result is whether the entire string `s` can be segmented.
        return dp[n]

    
    def coinChange(self, coins: List[int], amount: int) -> int:
        # Function to find the minimum number of coins required to make a given amount using dynamic programming.
        # Valid input conditions:
        # - `coins` is a list of positive integers representing the coin denominations.
        # - `amount` is a non-negative integer representing the total amount to be formed.
        # - `coins` must not contain non-positive values, and `amount` must be positive.

        if (amount <= 0):
            return -1  # If the amount is less than or equal to 0, it's not valid to form any sum.
        if (len(coins) == 0):
            return -1  # If no coins are available, the sum cannot be formed.
        if (any(coin <= 0 for coin in coins)):  
            return -1  # Ensure that all coins are positive.

        INT_MAX = float('inf')  # A placeholder for an impossible large value (infinity).
        OPT = [INT_MAX] * (amount + 1)  # OPT[i] stores the minimum number of coins to make sum i.
        OPT[0] = 0  # Base case: 0 coins are needed to make amount 0.

        # For each sum from 1 to the target amount, find the minimum number of coins needed.
        for sum in range(1, amount + 1):
            for coin in coins:
                # If it's possible to make the current sum with the current coin.
                if sum - coin >= 0 and OPT[sum - coin] != INT_MAX:
                    # Update OPT[sum] with the minimum number of coins required.
                    OPT[sum] = min(OPT[sum], OPT[sum - coin] + 1)

        # If no valid combination was found for the target amount, return -1.
        if (OPT[amount] == INT_MAX):
            return -1
        
        return OPT[amount]  # Return the minimum number of coins to form the given amount.
    
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # Function to find the length of the longest common subsequence (LCS) between two strings.
        # Valid input conditions:
        # - `text1` and `text2` are non-empty strings to be compared.
        # - The function assumes the inputs are valid strings and does not check for non-string inputs.

        m, n = len(text1), len(text2)  # m and n are the lengths of text1 and text2.
        dp = [[0] * (n + 1) for _ in range(m + 1)]  # dp[i][j] stores the length of LCS of text1[0:i] and text2[0:j].

        # Build the dp table by comparing characters from both strings.
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    # If characters match, the LCS length increases by 1.
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    # If characters don't match, take the maximum LCS length from previous options.
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])

        # The result is the LCS length of the entire strings text1 and text2.
        return dp[m][n]

    
    def catalan_recursive(self, n: int) -> int:
        # Function to calculate the nth Catalan number using dynamic programming.
        # Valid input conditions:
        # - `n` is a non-negative integer. If `n` is negative, the function returns -1.

        if (n == 0 or n == 1):
            return 1  # The first two Catalan numbers are both 1.
        elif (n < 0):
            return -1  # Invalid input for negative n.

        # Table to store results of subproblems. catalan[i] will store the ith Catalan number.
        catalan = [0]*(n+1)

        # Initialize the first two values of the Catalan sequence.
        catalan[0] = 1
        catalan[1] = 1

        # Fill entries in catalan[] using the recursive formula:
        # C(i) = sum(C(j) * C(i-j-1)) for all j from 0 to i-1.
        for i in range(2, n + 1):
            for j in range(i):
                catalan[i] += catalan[j] * catalan[i-j-1]

        # Return the nth Catalan number.
        return catalan[n]

    
    def catalan_closed_form(self, n: int) -> int:
        # Function to calculate the nth Catalan number using the closed-form formula.
        # Valid input conditions:
        # - `n` is a non-negative integer. If `n` is negative, the function returns -1.

        if (n < 0):
            return -1  # Invalid input for negative n.

        res = 1  # Initialize result to 1, which is the 0th Catalan number.

        # Iterate through 1 to n to calculate the nth Catalan number using the closed-form formula:
        # C(n) = (4n - 2) * C(n-1) / (n + 1).
        for i in range(1, n+1):
            res = (res * (4 * i - 2)) // (i + 1)  # Update result using the formula.

        # Return the nth Catalan number.
        return res
    
    
    def factorial(self, n: int) -> int:
        # Function to calculate the factorial of a non-negative integer n.
        # Valid input conditions:
        # - `n` must be a non-negative integer. If `n` is negative, the function returns -1.
        
        if (n < 0):
            return -1  # Invalid input for negative n.
        
        # Base case: 0! and 1! are both 1.
        if n == 0 or n == 1:
            return 1
        
        # Compute the factorial using a loop for n >= 2.
        res = 1  # Initialize result to 1.
        for i in range(2, n + 1):  # Multiply all integers from 2 to n.
            res *= i
        
        # Return the calculated factorial.
        return res

    
    def stirling_number(self, r: int, n: int) -> int:
        # Function to compute the Stirling number of the second kind, S(n, r), 
        # which represents the number of ways to partition a set of n elements into r non-empty subsets.
        # Valid input conditions:
        # - `r` and `n` must be non-negative integers.
        # - If `n < r`, return 0 as it's impossible to partition n elements into more than n subsets.
        # - If `r == 0`, return 1 (base case for Stirling numbers).
        # - If `n == 0`, return 0 (there are no ways to partition zero elements into non-zero subsets).
        # - If `n == r`, return 1 (only one way to partition n elements into exactly n subsets).

        if (n < 0 or r < 0):
            return -1  # Invalid input for negative r or n.
        if (r < n):
            return 0  # It's impossible to partition n elements into more than n subsets.
        if (r == 0):
            return 1  # Base case: S(0, 0) = 1.
        if (n == 0):
            return 0  # Base case: S(0, r) = 0 for r > 0.
        if (n == r):
            return 1  # Base case: S(n, n) = 1, only one way to partition n elements into n subsets.

        # Create a 2D list to store the Stirling numbers.
        dp = [[0] * (r + 1) for _ in range(n + 1)]
    
        # Base cases: S(i, i) = 1 and S(1, j) = (j-1)!
        for i in range(n + 1):
            dp[i][i] = 1  # S(i, i) = 1, there is one way to partition i elements into i subsets.
        for i in range(1, r + 1):
            dp[1][i] = 1  # S(1, j) = 1, only one way to partition j element into 1 subsets.
    
        # Fill in the rest of the table using the recurrence relation:
        # S(n, r) = S(n-1, r-1) + (r-1) * S(n, r-1)

        for j in range(2, r + 1):
            for i in range(2, n + 1):
                dp[i][j] = i * dp[i][j - 1] + dp[i - 1][j - 1]
    
        # Return the computed Stirling number S(n, r).
        return dp[n][r]
    
    def minDistance(self, word1: str, word2: str) -> int:
        # Function to compute the minimum number of operations required to convert word1 to word2.
        # Operations allowed: insert, delete, and replace a character.
        # Valid input conditions:
        # - Both `word1` and `word2` should be strings.
        
        m, n = len(word1), len(word2)
        
        # dp[i][j] represents the minimum number of operations to convert word1[0..i) to word2[0..j).
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base case: transforming an empty string to another string.
        # If word1 is empty, all characters of word2 need to be inserted.
        for i in range(1, m + 1):
            dp[i][0] = i  # i deletions are required to turn word1[0..i] to an empty string.
        
        # If word2 is empty, all characters of word1 need to be deleted.
        for j in range(1, n + 1):
            dp[0][j] = j  # j insertions are required to turn an empty string to word2[0..j].

        # Fill the DP table using the recurrence relation:
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # If characters match, no operation is needed.
                else:
                    # Take the minimum of:
                    # - deleting a character from word1 (dp[i-1][j]),
                    # - inserting a character into word1 (dp[i][j-1]),
                    # - replacing a character in word1 (dp[i-1][j-1]),
                    # and add 1 for the operation.
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

        # Return the minimum number of operations to convert word1 to word2.
        return dp[m][n]

    def matrixMultiplication(self, arr: List[int]) -> int:

        # Function to compute the minimum number of scalar multiplications needed
        # to multiply a chain of matrices represented by the dimensions in `arr`.
        # Valid input conditions:
        # - `arr` should be a list of integers where each value represents the dimension of a matrix.
        # - All dimensions should be positive integers.

        if (any(dim <= 0 for dim in arr)):
            return -1  # Return -1 for invalid dimensions.

        N = len(arr)
        if (N < 2):
            return -1
        # Create a 2D array `opt` to store the minimum multiplication costs for matrix chain multiplication.

        opt = [[0] * N for _ in range(N)]

        # Loop through chain lengths from 2 to N, i.e., consider subchains of increasing size.
        for length in range(2, N):
            for i in range(N - length):
                j = i + length
                m = float('inf')  # Initialize to infinity to find the minimum cost.
                
                # Try all possible places to split the matrix chain.
                for k in range(i + 1, j):
                    m = min(m, opt[i][k] + opt[k][j] + arr[i] * arr[k] * arr[j])
                
                # Store the minimum multiplication cost for the chain from matrix i to matrix j.
                opt[i][j] = m

        # Return the minimum cost of multiplying the entire chain of matrices.
        return opt[0][N - 1]

    def maxProduct(self, nums: List[int]) -> int:
        # Function to find the maximum product of a contiguous subarray.
        # Valid input conditions:
        # - `nums` should be a list of integers with at least one element.

        n = len(nums)

        if (n == 0):
            return 0  # Return 0 for invalid input.

        # Initialize variables:
        # - `max_prod` and `min_prod` track the maximum and minimum products at the current index.
        # - `result` stores the maximum product encountered so far.

        max_prod, min_prod, result = nums[0], nums[0], nums[0]

        # Traverse the array starting from the second element.
        for i in range(1, n):
            # If the current number is negative, swap `max_prod` and `min_prod` to account for sign change.
            if nums[i] < 0:
                max_prod, min_prod = min_prod, max_prod

            # Update `max_prod` to the maximum of the current element or the product including it.
            max_prod = max(nums[i], max_prod * nums[i])
            
            # Update `min_prod` to the minimum of the current element or the product including it.
            min_prod = min(nums[i], min_prod * nums[i])

            # Update `result` to the maximum product found so far.
            result = max(result, max_prod)

        # Return the maximum product of any contiguous subarray.
        return result

    
    def fibonacci(self, n: int) -> int:
        # Function to compute the nth Fibonacci number using dynamic programming.
        # Valid input conditions:
        # - `n` should be a non-negative integer.

        if (n < 0):
            return -1
        elif (n == 0):
            return 0  # Base cases: Fibonacci(0) = 0, Fibonacci(1) = 1.

        # Initialize a DP array to store Fibonacci numbers up to n.
        dp = [0] * (n + 1)
        dp[1] = 1  # Base case: Fibonacci(1) = 1.

        # Fill the DP array using the Fibonacci recurrence relation.
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]  # Fibonacci(i) = Fibonacci(i-1) + Fibonacci(i-2).

        # Return the nth Fibonacci number.
        return dp[n]

    
    def binomial_coefficient(self, n: int, k: int) -> int:
        # Function to compute the binomial coefficient C(n, k) using dynamic programming.
        # Valid input conditions:
        # - `n` and `k` should be non-negative integers.
        # - `k` should not be greater than `n`.

        if (n < 0 or k < 0):
            return -1  # Return -1 for invalid input.
        if (k > n):
            return -1  # Return -1 if k > n.

        # Initialize a DP table with dimensions (n+1) x (k+1).
        dp = [[0] * (k + 1) for _ in range(n + 1)]

        # Fill the DP table for binomial coefficient calculation.
        for i in range(n + 1):
            dp[i][0] = 1  # Base case: C(i, 0) = 1.

            # Calculate C(i, j) for 1 <= j <= min(i, k).
            for j in range(1, min(i, k) + 1):
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]  # Pascal's triangle relation.

        # Return the binomial coefficient C(n, k).
        return dp[n][k]

    
    def derangement_count(self, n: int) -> int:
        # Valid input condition:
        # - n should be a non-negative integer.
        if (n < 0):
            return -1
        
        elif (n == 0):
            return 1

        # Initialize the DP table
        dp = [0] * (n + 1)
        dp[0], dp[1] = 1, 0  # Base cases: D(0) = 1, D(1) = 0

        # Fill the DP table using recurrence relation
        for i in range(2, n + 1):
            dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2])

        return dp[n]