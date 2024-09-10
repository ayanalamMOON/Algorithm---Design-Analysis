#include<iostream>

int main() {
    int height = 9; // You can change the height of the triangle here

    for(int i = 1; i <= height; ++i) {
        for(int j = 1; j <= i; ++j) {
            if (j == 1 || j == i || i == height) {
                std::cout << "*";
            } else {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }

    return 0;
}