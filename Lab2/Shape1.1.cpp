#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

// Function to generate a random alphabet
char getRandomAlphabet() {
    return 'A' + (rand() % 26);
}

int main() {
    int height;

    cout << "Enter the height of the equilateral triangle: ";
    cin >> height;

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // Print the hollow equilateral triangle with random alphabets
    for (int i = 0; i < height; ++i) {
        // Print leading spaces
        for (int j = i; j < height - 1; ++j) {
            cout << " ";
        }
        // Print the alphabets
        for (int j = 0; j <= i; ++j) {
            if (j == 0 || j == i || i == height - 1) {
                cout << getRandomAlphabet() << " ";
            } else {
                cout << "  ";
            }
        }
        cout << endl;
    }

    return 0;
}