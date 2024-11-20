import java.util.*;

class Interval {
    int start, end, weight;

    public Interval(int start, int end, int weight) {
        this.start = start;
        this.end = end;
        this.weight = weight;
    }
}

public class ClassicalDP {
    
    public int knapSack(int maxWeight, int[] vals, int[] weights) {
        int n = weights.length;
        int[][] OPT = new int[n + 1][maxWeight + 1];

        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= maxWeight; w++) {
                if (w < weights[i - 1]) {
                    OPT[i][w] = OPT[i - 1][w];
                } else {
                    OPT[i][w] = Math.max(OPT[i - 1][w], OPT[i - 1][w - weights[i - 1]] + vals[i - 1]);
                }
            }
        }
        return OPT[n][maxWeight];
    }

    public int longestIncreasingSubsequence(int[] nums) {
        if (nums.length == 0) return 0;
        int[] OPT = new int[nums.length];
        Arrays.fill(OPT, 1);

        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    OPT[i] = Math.max(OPT[i], OPT[j] + 1);
                }
            }
        }
        return Arrays.stream(OPT).max().getAsInt();
    }

    public int longestPalindrome(String s) {
        int n = s.length();
        int[][] OPT = new int[n][n];

        for (int i = 0; i < n; i++) OPT[i][i] = 1;

        for (int length = 2; length <= n; length++) {
            for (int i = 0; i <= n - length; i++) {
                int j = i + length - 1;
                if (s.charAt(i) == s.charAt(j)) {
                    OPT[i][j] = OPT[i + 1][j - 1] + 2;
                } else {
                    OPT[i][j] = Math.max(OPT[i + 1][j], OPT[i][j - 1]);
                }
            }
        }
        return OPT[0][n - 1];
    }

    public int houseRobber(int[] nums) {
        if (nums.length == 1) return nums[0];
        if (nums.length == 2) return Math.max(nums[0], nums[1]);

        int[] OPT = new int[nums.length];
        OPT[0] = nums[0];
        OPT[1] = Math.max(nums[0], nums[1]);

        for (int i = 2; i < nums.length; i++) {
            OPT[i] = Math.max(OPT[i - 1], OPT[i - 2] + nums[i]);
        }
        return OPT[nums.length - 1];
    }

    public int getSchedule(int[] startTime, int[] endTime, int[] weights) {
        List<Interval> intervals = new ArrayList<>();
        for (int i = 0; i < startTime.length; i++) {
            intervals.add(new Interval(startTime[i], endTime[i], weights[i]));
        }

        intervals.sort(Comparator.comparingInt(a -> a.end));
        int n = intervals.size();
        int[] OPT = new int[n];
        OPT[0] = intervals.get(0).weight;

        for (int i = 1; i < n; i++) {
            int include = intervals.get(i).weight;
            int l = binarySearchFinish(intervals, intervals.get(i).start);
            if (l != -1) include += OPT[l];
            OPT[i] = Math.max(include, OPT[i - 1]);
        }

        return OPT[n - 1];
    }

    private int binarySearchFinish(List<Interval> intervals, int k) {
        int low = 0, high = intervals.size() - 1;
        while (low <= high) {
            int mid = (low + high) / 2;
            if (intervals.get(mid).end <= k) {
                if (mid + 1 == intervals.size() || intervals.get(mid + 1).end > k) {
                    return mid;
                }
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return -1;
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }

    public int coinChange(int[] coins, int amount) {
        int[] OPT = new int[amount + 1];
        Arrays.fill(OPT, amount + 1);
        OPT[0] = 0;

        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (i >= coin) {
                    OPT[i] = Math.min(OPT[i], OPT[i - coin] + 1);
                }
            }
        }
        return OPT[amount] > amount ? -1 : OPT[amount];
    }

    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }

    public int matrixMultiplication(int[] arr) {
        int N = arr.length;
        int[][] dp = new int[N][N];

        for (int len = 2; len < N; len++) {
            for (int i = 0; i < N - len; i++) {
                int j = i + len;
                dp[i][j] = Integer.MAX_VALUE;
                for (int k = i + 1; k < j; k++) {
                    dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j]);
                }
            }
        }
        return dp[0][N - 1];
    }

    public int maxProduct(int[] nums) {
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
}
