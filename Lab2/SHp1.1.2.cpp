#include <iostream>
#include <cmath>

using namespace std;

// Function to check if a point is on the boundary of a circle
bool isOnCircleBoundary(int x, int y, int centerX, int centerY, int radius) {
    int dx = x - centerX;
    int dy = y - centerY;
    int distanceSquared = dx * dx + dy * dy;
    int radiusSquared = radius * radius;
    return (distanceSquared >= radiusSquared - radius && distanceSquared <= radiusSquared + radius);
}

int main() {
    int height;

    cout << "Enter the height of the equilateral triangle: ";
    cin >> height;

    int centerX = height / 2;
    int centerY = height / 2;
    int radius = height / 3;

    // Print the hollow equilateral triangle with a hollow circle inside
    for (int i = 0; i < height; ++i) {
        // Print leading spaces
        for (int j = i; j < height - 1; ++j) {
            cout << " ";
        }
        // Print the triangle and circle
        for (int j = 0; j <= 2 * i; ++j) {
            if (j == 0 || j == 2 * i || i == height - 1) {
                cout << "*";
            } else if (isOnCircleBoundary(i, j - i + centerX, centerY, radius)) {
                cout << "O";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }

    return 0;
}