#include <iostream>
#include <vector>
#include <cassert>
#include <queue>
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
    int k5 = 10;
    // 10^9 + 333333334 + 111111112 + 37037038 + 12345680 + 4115227 + 1371743 + 457248 + 152416 + 50806 = 1499999604
    assert(solution.maxScore(nums5, k5) == 1499999604LL);
    
    cout << "All tests passed!" << endl;
}

int main() {
    runTests();
    return 0;
}