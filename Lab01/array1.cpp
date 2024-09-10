#include <iostream>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()

int main() {
    int myArray[6]; // Creating an array of 6 integers

    // Initialize random seed
    std::srand(std::time(0));

    // Filling the array with random elements
    for (int i = 0; i < 6; ++i) {
        myArray[i] = std::rand() % 100; // Random number between 0 and 99
    }

    // Printing array elements
    for (int i = 0; i < 6; ++i) {
        std::cout << "Element at index " << i << ": " << myArray[i] << std::endl;
    }

    return 0;
}