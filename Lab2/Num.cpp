#include <iostream>
#include <cmath>
using namespace std;

int main() {
    // Declare variables to store the user input, a copy of the input, and the sum of the cubes of the digits
    int number, originalNumber, remainder, result = 0;
    
    // Prompt the user to enter a number
    cout << "Enter a number: ";
    
    // Read the input number
    cin >> number;
    
    // Store the original number
    originalNumber = number;
    
    // Use a loop to extract each digit, cube it, and add it to the sum
    while (originalNumber != 0) {
        remainder = originalNumber % 10;
        result += pow(remainder, 3);
        originalNumber /= 10;
    }
    
    // Compare the sum to the original number to determine if it is an Armstrong number
    if (result == number) {
        cout << number << " is an Armstrong number." << endl;
    } else {
        cout << number << " is not an Armstrong number." << endl;
    }
    
    return 0;
}