/**
 * DEBUGGING PROBLEM: "Find the Missing Number"
 * 
 * DESCRIPTION:
 * You're given an array of integers containing n distinct numbers taken from the range 0 to n.
 * There's one number missing from the sequence, and your task is to find it.
 * 
 * The function findMissingNumber(nums) should return the missing number.
 * 
 * EXAMPLES:
 * findMissingNumber([3, 0, 1]) => 2
 * findMissingNumber([9, 6, 4, 2, 3, 5, 7, 0, 1]) => 8
 * findMissingNumber([0]) => 1
 * 
 * The code below has several bugs. Find and fix them!
 */
/*
#include <vector>

int findMissingNumber(std::vector<int>& nums) {
    int n = nums.size();
    
    // Calculate expected sum of first n+1 numbers (0 to n)
    int expectedSum = n * (n + 1) / 2;
    
    // Calculate actual sum
    int actualSum = 0;
    for (int i = 1; i <= n; i++) {
        actualSum += nums[i];
    }
    
    // The difference is our missing number
    return expectedSum - actualSum;
}
*/

/**
 * BUGS TO FIND:
 * 1. There's an off-by-one error in the loop
 * 2. There's a logical error in how we're calculating the expected sum
 * 3. There might be an issue with how we're accessing array elements
 * 
 * Try to fix the function so all test cases pass!
 */

#include<bits/stdc++.h>
using namespace std;

int findMissingNumber(const vector<int>& nums){
    int n = nums.size();
    int expectedSum = n * (n+1)/2;
    long long actualSum = 0;
    for(int i = 0; i < n; i++){
        actualSum += nums[i];
    }
    return expectedSum - actualSum;
}

int main(){
    vector<vector<int>> testCases = {
        {3, 0, 1},
        {9, 6, 4, 2, 3, 5, 7, 0, 1},
        {0}

    };

    for (const auto& testCase : testCases) {
        cout << "Missing number for {";
        for (int i = 0; i < testCase.size(); i++) {
            cout << testCase[i] << (i < testCase.size() - 1 ? ", " : "");
        }
        cout << "} is " << findMissingNumber(testCase) << endl;
    }
    return 0;  
}