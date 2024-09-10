#include <iostream>
#include <string>

using namespace std;

int main() {
    int height;
    string text;

    cout << "Enter the height of the pentagon: ";
    cin >> height;
    cin.ignore(); // To ignore the newline character left in the buffer

    cout << "Enter the text to write inside the pentagon: ";
    getline(cin, text);

    int middleRow = height / 2 + 1;
    int textStartIndex = (2 * middleRow - 1 - text.length()) / 2;

    // Print the top part of the pentagon
    for (int i = 1; i <= middleRow; ++i) {
        // Print leading spaces
        for (int j = i; j < height; ++j) {
            cout << " ";
        }
        // Print stars and spaces inside the pentagon
        for (int j = 1; j <= (2 * i - 1); ++j) {
            if (i == middleRow && j >= textStartIndex + 1 && j < textStartIndex + 1 + text.length()) {
                cout << text[j - textStartIndex - 1];
            } else if (j == 1 || j == (2 * i - 1)) {
                cout << "*";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }

    // Print the middle part of the pentagon
    for (int i = middleRow + 1; i <= height; ++i) {
        // Print leading spaces
        for (int j = 1; j <= height - middleRow; ++j) {
            cout << " ";
        }
        // Print stars and spaces inside the pentagon
        for (int j = 1; j <= (2 * middleRow - 1); ++j) {
            if (j == 1 || j == (2 * middleRow - 1)) {
                cout << "*";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }

    // Print the bottom part of the pentagon
    for (int i = height - 1; i >= 1; --i) {
        // Print leading spaces
        for (int j = height; j > i; --j) {
            cout << " ";
        }
        // Print stars and spaces inside the pentagon
        for (int j = 1; j <= (2 * i - 1); ++j) {
            if (j == 1 || j == (2 * i - 1)) {
                cout << "*";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }

    return 0;
}