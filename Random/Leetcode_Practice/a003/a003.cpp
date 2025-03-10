/*
You are given a 0-indexed integer array nums and an integer k. You have a starting score of 0.

In one operation:

choose an index i such that 0 <= i < nums.length,
increase your score by nums[i], and
replace nums[i] with ceil(nums[i] / 3).
Return the maximum possible score you can attain after applying exactly k operations.

The ceiling function ceil(val) is the least integer greater than or equal to val.

 

Example 1:

Input: nums = [10,10,10,10,10], k = 5
Output: 50
Explanation: Apply the operation to each array element exactly once. The final score is 10 + 10 + 10 + 10 + 10 = 50.
Example 2:

Input: nums = [1,10,3,3,3], k = 3
Output: 17
Explanation: You can do the following operations:
Operation 1: Select i = 1, so nums becomes [1,4,3,3,3]. Your score increases by 10.
Operation 2: Select i = 1, so nums becomes [1,2,3,3,3]. Your score increases by 4.
Operation 3: Select i = 2, so nums becomes [1,2,1,3,3]. Your score increases by 3.
The final score is 10 + 4 + 3 = 17.
 

Constraints:

1 <= nums.length, k <= 105
1 <= nums[i] <= 109

*/
#include <iostream>
#include <vector>
#include <queue>
#include <cassert>
using namespace std;

class Solution {
public:
    long long maxScore(vector<int>& nums, int k) {
        priority_queue<long long> maxHeap;
        for(int num : nums){
            maxHeap.push(num);
        }

        long long score = 0;
        for(int i=0; i<k; i++) {
            long long topVal = maxHeap.top();
            maxHeap.pop();
            score += topVal;
            
            long long newVal = (topVal+2)/3;
            maxHeap.push(newVal);
        }
    
        return score;
    }
};

void runTests() {
    Solution solution;
    
    // Test case 1: Example 1 from the problem statement
    vector<int> nums1 = {10, 10, 10, 10, 10};
    int k1 = 5;
    assert(solution.maxScore(nums1, k1) == 50);
    
    // Test case 2: Example 2 from the problem statement
    vector<int> nums2 = {1, 10, 3, 3, 3};
    int k2 = 3;
    assert(solution.maxScore(nums2, k2) == 17);
    
    // Test case 3: Single element with multiple operations
    vector<int> nums3 = {9};
    int k3 = 3;
    // 9 + 3 + 1 = 13
    assert(solution.maxScore(nums3, k3) == 13);
    
    // Test case 4: Large numbers
    vector<int> nums4 = {1000000000, 1000000000};
    int k4 = 4;
    // 1000000000 + 1000000000 + 333333334 + 333333334 = 2666666668
    assert(solution.maxScore(nums4, k4) == 2666666668LL);
    
    // Test case 5: Many small operations on large number
    vector<int> nums5 = {1000000000};
    
    cout << "All tests passed!" << endl;
}

int main() {
    runTests();
    return 0;
}

