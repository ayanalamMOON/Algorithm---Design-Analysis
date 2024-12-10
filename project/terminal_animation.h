#ifndef TERMINAL_ANIMATION_H
#define TERMINAL_ANIMATION_H

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>

class TerminalAnimation {
public:
    // Display array with optional highlight
    static void displayArray(const std::vector<int>& arr, 
                              int highlightIndex1 = -1, 
                              int highlightIndex2 = -1) {
        for (size_t i = 0; i < arr.size(); ++i) {
            if (i == highlightIndex1 || i == highlightIndex2) {
                std::cout << "\033[1;31m"; // Red color for highlighted
            }
            std::cout << std::setw(5) << arr[i] << " ";
            if (i == highlightIndex1 || i == highlightIndex2) {
                std::cout << "\033[0m"; // Reset color
            }
        }
        std::cout << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Simple swap animation
    static void animateSwap(std::vector<int>& arr, int i, int j) {
        std::cout << "Swapping " << arr[i] << " and " << arr[j] << std::endl;
        displayArray(arr, i, j);
        std::swap(arr[i], arr[j]);
        displayArray(arr);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Visualization for searching
    static void visualizeSearch(const std::vector<int>& arr, int target, int index) {
        std::cout << "\nSearching for " << target << ":\n";
        for (size_t i = 0; i < arr.size(); ++i) {
            if (i == index) {
                std::cout << "\033[1;32m"; // Green color for found
            }
            std::cout << std::setw(5) << arr[i] << " ";
            if (i == index) {
                std::cout << "\033[0m"; // Reset color
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::cout << std::endl;
    }
};

#endif // TERMINAL_ANIMATION_H