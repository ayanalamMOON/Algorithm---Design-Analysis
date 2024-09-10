#include <iostream>
#include <string>

using namespace std;

int main() {
    int height = 14; // You can change the height of the triangle here
    string text;
    
    cout << "Enter the text to write inside the triangle: ";
    getline(cin, text);

    int middleRow = height / 2 + 1;
    int textStartIndex = (2 * middleRow - 1 - text.length()) / 2;

    for (int i = 1; i <= height; ++i) {
        // Print leading spaces
        for (int j = i; j < height; ++j) {
            cout << " ";
        }
        // Print stars and spaces inside the triangle
        for (int j = 1; j <= (2 * i - 1); ++j) {
            if (i == middleRow && j >= textStartIndex + 1 && j < textStartIndex + 1 + text.length()) {
                cout << text[j - textStartIndex - 1];
            } else if (j == 1 || j == (2 * i - 1) || i == height) {
                cout << "*";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }

    return 0;
}