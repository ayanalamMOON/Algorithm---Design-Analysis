#include <iostream>
using namespace std;

int main() {
    int rows;
    cout << "Enter the number of rows: ";
    cin >> rows;

    int spaces = rows - 1;
    int stars = 1;

    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= spaces; j++) {
            cout << " ";
        }
        for (int k = 1; k <= stars; k++) {
            cout << "*";
        }
        cout << endl;
        spaces--;
        stars += 2;
    }

    return 0;
}