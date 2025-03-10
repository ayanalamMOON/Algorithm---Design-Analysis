/*
 * LeetCode Medium Difficulty Problem:
 * 
 * Longest Palindromic Subsequence
 * 
 * Given a string s, find the longest palindromic subsequence's length in s.
 * 
 * A subsequence is a sequence that can be derived from another sequence by
 * deleting some or no elements without changing the order of the remaining elements.
 * 
 * Example 1:
 * Input: s = "bbbab"
 * Output: 4
 * Explanation: One possible longest palindromic subsequence is "bbbb".
 * 
 * Example 2:
 * Input: s = "cbbd"
 * Output: 2
 * Explanation: One possible longest palindromic subsequence is "bb".
 * 
 * Example 3:
 * Input: s = "abcdef"
 * Output: 1
 * Explanation: The longest palindromic subsequence is any single character.
 * 
 * Constraints:
 * - 1 <= s.length <= 1000
 * - s consists only of lowercase English letters.
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
    public:
        int longestPalindromeSubseq(const string& s){
            int n = s.size();
            vector<vector<int>> dp(n, vector<int>(n, 0));
            for(int i=n-1; i>=0; i--){
                dp[i][i] =1;
                for(int j=i+1; j<n; j++){
                    if(s[i] == s[j]){
                        dp[i][j] = dp[i+1][j-1] +2;
                    } else {
                        dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
                    }
                }
            }
            return dp[0][n-1];
        }
};
int main() {
    Solution sol;
    cout << sol.longestPalindromeSubseq("bbbab") << endl;
    cout << sol.longestPalindromeSubseq("cbbd") << endl;
    cout << sol.longestPalindromeSubseq("abcdef") << endl;
    return 0;
}
