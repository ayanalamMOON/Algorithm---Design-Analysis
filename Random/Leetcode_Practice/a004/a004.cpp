/*
 * LeetCode Medium Difficulty Problem:
 * 
 * Substring with Concatenation of All Words
 * 
 * You are given a string s and an array of strings words of the same length.
 * Return all starting indices of substring(s) in s that is a concatenation
 * of each word in words exactly once, in any order, and without any intervening characters.
 * 
 * You can return the answer in any order.
 * 
 * Example 1:
 * Input: s = "barfoothefoobarman", words = ["foo","bar"]
 * Output: [0,9]
 * Explanation: Substrings starting at index 0 and 9 are "barfoo" and "foobar".
 * These are concatenations of words "foo" and "bar" in any order.
 * 
 * Example 2:
 * Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
 * Output: []
 * Explanation: Since words has duplicate "word", no solution exists.
 * 
 * Example 3:
 * Input: s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
 * Output: [6,9,12]
 * 
 * Constraints:
 * - 1 <= s.length <= 10^4
 * - 1 <= words.length <= 5000
 * - 1 <= words[i].length <= 30
 * - s and words[i] consist of lowercase English letters.
 */


#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> findSubstring(const string &s, vector<string> &words) {
        vector<int> result;
        if (words.empty() || s.empty()) {
            return result;
        }
        unordered_map<string, int> freq;
        for (auto &w : words) {
            freq[w]++;
        }
        int wordCount = words.size();
        int wordLength = words[0].size();
        int totalWordsLength = wordCount * wordLength;
        if (totalWordsLength > (int)s.size()) {
            return result;
        }
        for (int offset = 0; offset < wordLength; offset++) {
            unordered_map<string, int> windowFreq;
            int start = offset, matchedWords = 0;
            for (int end = offset; end + wordLength <= (int)s.size(); end += wordLength) {
                string current = s.substr(end, wordLength);
                if (freq.count(current)) {
                    windowFreq[current]++;
                    if (windowFreq[current] == freq[current]) {
                        matchedWords++;
                    }
                    while (windowFreq[current] > freq[current]) {
                        string leftWord = s.substr(start, wordLength);
                        windowFreq[leftWord]--;
                        if (windowFreq[leftWord] == freq[leftWord] - 1) {
                            matchedWords--;
                        }
                        start += wordLength;
                    }
                    if (matchedWords == (int)freq.size()) {
                        if (end - start + wordLength == totalWordsLength) {
                            result.push_back(start);
                        }
                        string leftWord = s.substr(start, wordLength);
                        windowFreq[leftWord]--;
                        if (windowFreq[leftWord] == freq[leftWord] - 1) {
                            matchedWords--;
                        }
                        start += wordLength;
                    }
                } else {
                    windowFreq.clear();
                    matchedWords = 0;
                    start = end + wordLength;
                }
            }
        }
        return result;
    }
};

int main() {
    Solution sol;
    string s1 = "barfoothefoobarman";
    vector<string> words1 = {"foo","bar"};
    auto ans1 = sol.findSubstring(s1, words1);
    for (int x : ans1) cout << x << " ";
    cout << endl;
    string s2 = "wordgoodgoodgoodbestword";
    vector<string> words2 = {"word","good","best","word"};
    auto ans2 = sol.findSubstring(s2, words2);
    for (int x : ans2) cout << x << " ";
    cout << endl;
    string s3 = "barfoofoobarthefoobarman";
    vector<string> words3 = {"bar","foo","the"};
    auto ans3 = sol.findSubstring(s3, words3);
    for (int x : ans3) cout << x << " ";
    cout << endl;
    return 0;
}
