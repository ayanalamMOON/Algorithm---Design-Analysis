#include <iostream>
#include <iomanip> // For std::setw
using namespace std;

// Function to calculate factorial
int factorial(int n) {
    if (n <= 1)
        return 1;
    else
        return n * factorial(n - 1);
}

// Function to calculate binomial coefficient
int binomialCoefficient(int n, int k) {
    return factorial(n) / (factorial(k) * factorial(n - k));
}

// Function to print Pascal's Pyramid
void printPascalsPyramid(int rows) {
    int max_width = 4 * rows; // Estimate the maximum width needed for the largest number

    for (int i = 0; i < rows; i++) {
        // Print leading spaces to center-align the pyramid
        cout << string((max_width - 4 * i) / 2, ' ');

        for (int j = 0; j <= i; j++) {
            cout << std::setw(4) << binomialCoefficient(i, j) << " ";
        }
        cout << endl;
    }
}

int main() {
    int rows;
        cout << "Enter the number of rows for Pascal's Pyramid: ";
        cin >> rows;

    printPascalsPyramid(rows);

    return 0;
}