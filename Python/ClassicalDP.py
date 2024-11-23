from typing import List

class Interval:
    def __init__(self, start, end, weight):
        self.start = start
        self.end = end
        self.weight = weight

class ClassicalDP():
    
    def knapSack(self, max_weight: int, vals: List[int], weights: List[int]) -> int:

        if max_weight < 0:
            return -1
        if len(vals) != len(weights):
            return -1
        if len(vals) < 1 or len(weights) < 1:
            return -1
        if any(v <= 0 for v in vals) or any(w <= 0 for w in weights):
            return -1

        n = len(weights)
        OPT = [[0] * (max_weight + 1) for _ in range(n + 1)]

        # Initialize OPT for the base cases
        for w in range(max_weight + 1):
            OPT[0][w] = 0
        for i in range(n + 1):
            OPT[i][0] = 0
    
        for i in range(1, n + 1):
            for w in range(1, max_weight + 1):
                if w - weights[i - 1] < 0:
                    OPT[i][w] = OPT[i - 1][w]
                else:
                    if OPT[i - 1][w] > OPT[i - 1][w - weights[i - 1]] + vals[i - 1]:
                        OPT[i][w] = OPT[i - 1][w]
                    else:
                        OPT[i][w] = OPT[i - 1][w - weights[i - 1]] + vals[i - 1]
        
        return OPT[n][max_weight]
    
    def longestIncreasingSubsequence(self, nums: List[int]) -> int:
        n = len(nums)

        if (n == 0): 
            return 0
        # OPT[i] stores the length of the LIS ending at index i.
        OPT = [1] * n

        for i in range(n):
            for j in range(i - 1, -1, -1):
                if nums[j] < nums[i] and OPT[j] + 1 > OPT[i]:
                    OPT[i] = OPT[j] + 1

        # Find the maximum length and its index
        max_length = max(OPT)
        return max_length

    def longestPalindrome(self, arg: str) -> int:
        n = len(arg)

        if (n == 0):
            return 0

        # OPT[i][j] stores the length of the longest palindromic subsequence between i and j (inclusive)
        OPT = [[0] * n for _ in range(n)]

        # Base case: single character substrings are palindromes of length 1
        for i in range(n):
            OPT[i][i] = 1

        # Dynamic programming to calculate OPT
        for length in range(1, n):  # Length of the substring
            for i in range(n - length):
                j = i + length
                if arg[i] == arg[j]:
                    OPT[i][j] = OPT[i + 1][j - 1] + 2
                else:
                    if OPT[i + 1][j] > OPT[i][j - 1]:
                        OPT[i][j] = OPT[i + 1][j]
                    else:
                        OPT[i][j] = OPT[i][j - 1]
        
        return OPT[0][n-1]
    
    def house_robber(self, input: List[int]) -> int:
        if (len(input) == 0):
            return 0
        elif len(input) == 1:
            return input[0]
        elif len(input) == 2:
            return max(input[0], input[1])

        n = len(input)
        OPT = [-1] * n  # OPT[i] stores the max sum till index i

        # Initial conditions
        OPT[0] = input[0]
        OPT[1] = max(input[0], input[1])

        # Fill OPT and sequence using dynamic programming
        for i in range(2, n):
            if OPT[i - 2] + input[i] > OPT[i - 1]:
                OPT[i] = OPT[i - 2] + input[i]
            else:
                OPT[i] = OPT[i - 1]
        
        return OPT[-1]
    
    def binarySearchFinish(self, intervals, k):
        l, r = 0, len(intervals) - 1

        if intervals[0].end > k:
            return -1

        while r - l > 1:
            m = l + (r - l) // 2
            if intervals[m].end <= k < intervals[m + 1].end:
                return m
            elif intervals[m].end > k:
                r = m - 1
            else:
                l = m + 1

        if intervals[r].end <= k:
            return r
        elif intervals[l].end > k:
            return -1
        else:
            return l


    def get_schedule(self, startTime: List[int], endTime: List[int], weights: List[int]) -> int:

        if len(startTime) != len(endTime) or len(startTime) != len(weights):
            return -1
        if len(startTime) == 0:
            return 0
        if any(x <= 0 for x in weights):  # Weights must be positive
            return -1
        if any(startTime[i] >= endTime[i] for i in range(len(startTime))):  # Invalid interval
            return -1
        
        intervals = [Interval(startTime[i], endTime[i], weights[i]) for i in range(len(startTime))]

        # Sort intervals by finish time
        intervals.sort(key=lambda x: (x.end, x.start))

        n = len(intervals)
        OPT = [0] * n
        lastEnding = [-1] * n
        # Base case
        OPT[0] = intervals[0].weight

        # Build lastEnding using binary search
        for j in range(1, n):
            lastEnding[j] = self.binarySearchFinish(intervals, intervals[j].start)

        # Fill OPT using dynamic programming
        for i in range(1, n):
            include = intervals[i].weight
            if lastEnding[i] != -1:
                include += OPT[lastEnding[i]]

            exclude = OPT[i - 1]
            if include > exclude:
                OPT[i] = include
            else:
                OPT[i] = exclude

        return OPT[n-1]
    
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        word_set = set(wordDict)
        n = len(s)

        if (n == 0):
            return True
        
        dp = [False] * (n + 1)
        dp[0] = True  # Base case: empty string is valid

        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break

        return dp[n]
    
    def coinChange(self, coins: List[int], amount: int) -> int:
        # OPT[i] -> minimum number of coins to make sum i

        if (amount <= 0):
            return -1
        if (len(coins) == 0):
            return -1
        if (any(coin <= 0 for coin in coins)):
            return -1

        INT_MAX = float('inf')
        OPT = [INT_MAX] * (amount + 1)
        OPT[0] = 0  # Base case: no coins needed for amount 0

        for sum in range(1, amount + 1):
            for coin in coins:
                if sum - coin >= 0 and OPT[sum - coin] != INT_MAX:
                    OPT[sum] = min(OPT[sum], OPT[sum - coin] + 1)

        if (OPT[amount] == INT_MAX):
            return -1
        
        return OPT[amount]
    
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])

        return dp[m][n]
    
    def catalan_recursive(self, n: int) -> int:
        if (n < 0):
            return -1
        elif (n == 0 or n == 1):
            return 1

        # Table to store results of subproblems
        catalan = [0]*(n+1)

        # Initialize first two values in table
        catalan[0] = 1
        catalan[1] = 1

        # Fill entries in catalan[]
        # using recursive formula
        for i in range(2, n + 1):
            for j in range(i):
                catalan[i] += catalan[j] * catalan[i-j-1]

        # Return last entry
        return catalan[n]
    
    def catalan_closed_form(self, n: int)-> int:
        if (n < 0):
            return -1
        
        res = 1
        # Iterate till N
        for i in range(1, n+1):
            # Calculate the ith Catalan number
            res = (res * (4 * i - 2)) // (i + 1)
        return res
    
    def factorial(self, n: int) -> int:
        if (n < 0):
            return -1
        # 0! and 1! = 1
        if n == 0 or n == 1:
            return 1
    
        # Compute the factorial using a loop
        res = 1
        for i in range(2, n + 1):
            res *= i
    
        return res
    
    def stirling_number(self, r: int, n: int) -> int:

        if (n < 0 or r < 0):
            return -1
        if (r < n):
            return 0
        if (r == 0):
            return 1
        if (n == 0):
            return 0
        if (n == r):
            return 1

        # Create a 2D list to store the Stirling numbers
        dp = [[0] * (r + 1) for _ in range(n + 1)]
    
        # Fill in the base cases
        for i in range(n + 1):
            dp[i][i] = 1
        for i in range(1, r + 1):
            dp[1][i] = 1

        for j in range(2, r + 1):
            for i in range(2, n + 1):
                dp[i][j] = i * dp[i][j - 1] + dp[i - 1][j - 1]
    
        # Return the computed value
        return dp[n][r]
    
    
    def min_distance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        
        # dp[i][j] will be the minimum number of operations required to convert word1[0..i) to word2[0..j)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base case: transforming empty string to another string
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j

        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

        return dp[m][n]
    
    def matrix_multiplication(self, arr: List[int]) -> int:

        N = len(arr)
        if (N < 2):
            return -1
        
        if (any(dim <= 0 for dim in arr)):
            return -1
        
        # Create a 2D array to store the minimum multiplication costs
        opt = [[0] * N for _ in range(N)]

        # Loop through chain lengths from 2 to N
        for length in range(2, N):
            for i in range(N - length):
                j = i + length
                m = float('inf')
                # Try all possible places to split the chain
                for k in range(i + 1, j):
                    m = min(m, opt[i][k] + opt[k][j] + arr[i] * arr[k] * arr[j])
                opt[i][j] = m

        return opt[0][N - 1]
    
    def max_product(self, nums: List[int]) -> int:
        n = len(nums)

        if (n == 0):
            return 0
        
        # Initialize the variables for maximum and minimum products at the current index
        max_prod, min_prod, result = nums[0], nums[0], nums[0]
        
        for i in range(1, n):
            # If the current number is negative, swap max_prod and min_prod
            if nums[i] < 0:
                max_prod, min_prod = min_prod, max_prod

            # Update max_prod and min_prod by including the current element
            max_prod = max(nums[i], max_prod * nums[i])
            min_prod = min(nums[i], min_prod * nums[i])
            
            # Update the result with the maximum product found so far
            result = max(result, max_prod)

        return result
    
    def fibonacci(self, n: int) -> int:

        if (n < 0):
            return -1
        elif (n == 0):
            return 0
        
        dp = [0] * (n + 1)
        dp[1] = 1

        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[n]
    
    def binomial_coefficient(self, n: int, k: int) -> int:

        if (n < 0 or k < 0):
            return -1
        if (k > n):
            return -1

        dp = [[0] * (k + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = 1  # Base case

            for j in range(1, min(i, k) + 1):
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]

        return dp[n][k]
    
    def derangement_count(self, n: int) -> int:

        if (n < 0):
            return -1
        
        elif (n == 0):
            return 1

        dp = [0] * (n + 1)
        dp[0], dp[1] = 1, 0  # Base cases

        for i in range(2, n + 1):
            dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2])

        return dp[n]