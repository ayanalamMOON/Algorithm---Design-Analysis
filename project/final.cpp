/*
Floyd-Warshall algorithm psudocode:

function alogo(graph):
    lest dist be 2D array of size V x V,
    initialize dist[][] with givemgraph's edge weights
    for each vertex v in graph
    dist[v][v] = 0
    
   for k from 0 to V-1
        for i from 1 to
            for j from 1 to V
                if dist[i][k] != inf and dist[k][j] != inf and dist[i][k] + dist[k][j] < dist[i][j]
                    dist[i][j] = dist[i][k] + dist[k][j] 
    return dist


*/

/*
function mergeSort(array)
    if length of array <= 1
        return array

    mid = length of array / 2
    leftHalf = mergeSort(array[0...mid-1])
    rightHalf = mergeSort(array[mid...end])

    return merge(leftHalf, rightHalf)

function merge(left, right)
    result = empty array
    while left is not empty and right is not empty
        if left[0] <= right[0]
            append left[0] to result
            remove left[0] from left
        else
            append right[0] to result
            remove right[0] from right

    while left is not empty
        append left[0] to result
        remove left[0] from left

    while right is not empty
        append right[0] to result
        remove right[0] from right

    return result
*/

//write a program to insert an ele,emt in a 1 D array

#include <iostream>
using namespace std;

void insertElement(int arr[], int& n, int element, int position) {
    if (position < 0 || position > n) {
        cout << "Invalid position!" << endl;
        return;
    }

    // Shift elements to the right
    for (int i = n; i > position; --i) {
        arr[i] = arr[i - 1];
    }

    // Insert the element
    arr[position] = element;
    ++n;
}

int main() {
    int arr[100] = {1, 2, 3, 4, 5}; // Initial array with some elements
    int n = 5; // Current size of the array
    int element = 10; // Element to be inserted
    int position = 2; // Position at which to insert the element

    cout << "Original array: ";
    for (int i = 0; i < n; ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;

    insertElement(arr, n, element, position);

    cout << "Array after insertion: ";
    for (int i = 0; i < n; ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;

    return 0;
}

/*
algorithm to find search an item in a 2D array

function search2DArray(arr, rows, cols, target)
    for i from 0 to rows-1
        for j from 0 to cols-1
            if arr[i][j] == target
                return (i, j)
    return "Item not found"
*/