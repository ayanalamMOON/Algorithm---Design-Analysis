/*
You are given a string word and a non-negative integer k.

Return the total number of substrings of word that contain every vowel ('a', 'e', 'i', 'o', and 'u') at least once and exactly k consonants.

 

Example 1:

Input: word = "aeioqq", k = 1

Output: 0

Explanation:

There is no substring with every vowel.

Example 2:

Input: word = "aeiou", k = 0

Output: 1

Explanation:

The only substring with every vowel and zero consonants is word[0..4], which is "aeiou".

Example 3:

Input: word = "ieaouqqieaouqq", k = 1

Output: 3

Explanation:

The substrings with every vowel and one consonant are:

word[0..5], which is "ieaouq".
word[6..11], which is "qieaou".
word[7..12], which is "ieaouq".
 

Constraints:

5 <= word.length <= 2 * 105
word consists only of lowercase English letters.
0 <= k <= word.length - 5
*/

class Solution {
private:
    const unordered_set<char> vowels = {'a', 'e', 'i', 'o', 'u'};
    
    bool isVowel(char c) {
        return vowels.count(c) > 0;
    }
    
public:
    long countOfSubstrings(string word, int k) {
        long result = 0;
        int n = word.length();
        
        for (int i = 0; i < n; i++) {
            unordered_set<char> currentVowels;
            int consonants = 0;
            
            for (int j = i; j < n; j++) {
                if (isVowel(word[j])) {
                    currentVowels.insert(word[j]);
                } else {
                    consonants++;
                }
                
                if (consonants > k) break;
                
                if (currentVowels.size() == 5 && consonants == k) {
                    result++;
                }
            }
        }
        
        return result;
    }
};
