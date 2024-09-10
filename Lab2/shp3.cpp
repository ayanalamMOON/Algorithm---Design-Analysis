#include <iostream>
#include <string>

using namespace std;

int main() {
    int height;
    string text;

    cout << "Enter the height of the hexagon (must be an even number): ";
    cin >> height;
    cin.ignore(); // To ignore the newline character left in the buffer

    // Validate height input
    if (height <= 0 || height % 2 != 0) {
        cout << "Invalid height. Please enter a positive even number." << endl;
        return 1;
    }

    cout << "Enter the text to write inside the hexagon: ";
    getline(cin, text);

    int middleRow = height / 2;
    int textStartIndex = (2 * middleRow - 1 - text.length()) / 2;

    // Print the top part of the hexagon
    for (int i = 1; i <= middleRow; ++i) {
        // Print leading spaces
        for (int j = i; j < height; ++j) {
            cout << " ";
        }
        // Print stars and spaces inside the hexagon
        for (int j = 1; j <= (2 * i + height - 2); ++j) {
            if (i == middleRow && j >= textStartIndex + 1 && j < textStartIndex + 1 + text.length()) {
                cout << text[j - textStartIndex - 1];
            } else if (j == 1 || j == (2 * i + height - 2)) {
                cout << "*";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }

    // Print the middle part of the hexagon
    for (int i = middleRow + 1; i <= height; ++i) {
        // Print leading spaces
        for (int j = 1; j <= height - middleRow; ++j) {
            cout << " ";
        }
        // Print stars and spaces inside the hexagon
        for (int j = 1; j <= (2 * middleRow + height - 2); ++j) {
            if (j == 1 || j == (2 * middleRow + height - 2)) {
                cout << "*";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }

    // Print the bottom part of the hexagon
    for (int i = middleRow - 1; i >= 1; --i) {
        // Print leading spaces
        for (int j = height; j > i; --j) {
            cout << " ";
        }
        // Print stars and spaces inside the hexagon
        for (int j = 1; j <= (2 * i + height - 2); ++j) {
            if (j == 1 || j == (2 * i + height - 2)) {
                cout << "*";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }

    return 0;
}