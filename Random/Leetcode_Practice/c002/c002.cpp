/*
Problem: Debug the Merge Sorted Arrays Function

The following function is supposed to merge two sorted arrays into a single sorted array.
However, there are several bugs that prevent it from working correctly.

Find and fix all bugs in the mergeSortedArrays function so that it correctly
merges two sorted arrays and returns the result.

Example:
Input: arr1 = [1, 3, 5], arr2 = [2, 4, 6]
Expected Output: [1, 2, 3, 4, 5, 6]

Current incorrect output: [1, 2, 3, 4, 0, 0]
*/

#include<bits/stdc++.h>
using namespace std;

vector<int> mergeSortedArrays(const vector<int>& arr1, const vector<int>& arr2){
    vector<int> result(arr1.size() + arr2.size());
    int i=0, j=0, k=0;
    while(i< (int)arr1.size() || j<(int)arr2.size()){
        if(i>= (int)arr1.size()){
            result[k++] = arr2[j++];
        } else if(j>= (int)arr2.size()){
            result[k++] = arr1[i++];
        } else if(arr1[i] < arr2[j]){
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
        }
    }
    
    return result;
}
void testMergeSortedArrays() {
    std::vector<int> arr1 = {1, 3, 5};
    std::vector<int> arr2 = {2, 4, 6};
    
    std::vector<int> result = mergeSortedArrays(arr1, arr2);
    
    std::cout << "Merged array: ";
    for (int num : result) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

int main() {
    testMergeSortedArrays();
    return 0;
}