/*
 * CodeWars Level: 5 kyu (Intermediate-Advanced)
 * 
 * Title: Maximum Path Sum in a Triangle
 * 
 * Description:
 * Given a triangle array, find the maximum path sum from top to bottom.
 * For each step, you may move to an adjacent number on the row below.
 * 
 * Example:
 * Input:
 *    3
 *   7 4
 *  2 4 6
 * 8 5 9 3
 * 
 * Output: 23
 * Explanation: The maximum path sum is 3 + 7 + 4 + 9 = 23
 * 
 * Challenge:
 * 1. Implement a solution with O(n²) time complexity, where n is the height of the triangle
 * 2. Try to solve it using O(n) extra space (instead of modifying the input array)
 * 3. Can you optimize to use only O(1) extra space?
 * 
 * Note:
 * - The triangle will have at least 1 row and at most 200 rows
 * - Each row i contains i+1 elements
 * - Elements in the triangle will be between -10000 and 10000
 * 
 * Input Format: 
 * A vector of vectors, where triangle[i][j] represents the value at row i, column j
 * 
 * Sample Input:
 * {{3}, {7, 4}, {2, 4, 6}, {8, 5, 9, 3}}
 */

#include<bits/stdc++.h>
using namespace std;

class Solution {
    public:
    int maximumPathSum(vector<vector<int>>& triangle){
        for(int i= triangle.size()-2; i>= 0; i--){
            for(int j=0; j<(int)triangle[i].size(); j++){
                triangle[i][j]+= max(triangle[i+1][j], triangle[i+1][j+1]);
            }
        }
        return triangle[0][0];
    }
};

int main(){
    vector<vector<int>> triangle = {
        {3},
        {7, 4},
        {2, 4, 6},
        {8, 5, 9, 3}
    };
    Solution sol;
    int result = sol.maximumPathSum(triangle);
    cout << "Maximum Path Sum: " << result << endl;
    return 0;   
}
