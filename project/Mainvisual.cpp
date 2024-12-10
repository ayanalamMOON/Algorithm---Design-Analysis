#include <iostream>
#include <vector>
#include <random>
#include "terminal_animation.h"
#include "sorting_visualization.h"

class InteractiveSortingDemo {
private:
    std::vector<int> data;

    void generateRandomData(int size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(1, 1000);

        data.clear();
        for (int i = 0; i < size; ++i) {
            data.push_back(distrib(gen));
        }
    }

public:
    void runDemo() {
        int choice, size;

        while (true) {
            std::cout << "\n--- Interactive Sorting Visualization ---\n";
            std::cout << "1. Generate Random Data\n";
            std::cout << "2. Visualize Bubble Sort\n";
            std::cout << "3. Visualize Quick Sort\n";
            std::cout << "4. Search Visualization\n";
            std::cout << "5. Exit\n";
            std::cout << "Enter your choice: ";
            std::cin >> choice;

            switch (choice) {
                case 1: {
                    std::cout << "Enter number of elements: ";
                    std::cin >> size;
                    generateRandomData(size);
                    std::cout << "Data Generated. Current Array:\n";
                    TerminalAnimation::displayArray(data);
                    break;
                }
                case 2: {
                    if (data.empty()) {
                        std::cout << "Generate data first!\n";
                        break;
                    }
                    auto dataCopy = data;
                    SortingVisualizer::bubbleSortVisualized(dataCopy);
                    break;
                }
                case 3: {
                    if (data.empty()) {
                        std::cout << "Generate data first!\n";
                        break;
                    }
                    auto dataCopy = data;
                    SortingVisualizer::quickSortVisualized(dataCopy, 0, dataCopy.size() - 1);
                    break;
                }
                case 4: {
                    if (data.empty()) {
                        std::cout << "Generate data first!\n";
                        break;
                    }
                    std::sort(data.begin(), data.end());
                    int target;
                    std::cout << "Enter search target: ";
                    std::cin >> target;

                    // Linear search visualization
                    for (size_t i = 0; i < data.size(); ++i) {
                        if (data[i] == target) {
                            TerminalAnimation::visualizeSearch(data, target, i);
                            std::cout << "Target found at index: " << i << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 5:
                    return;
                default:
                    std::cout << "Invalid choice!\n";
            }
        }
    }
};

int main() {
    InteractiveSortingDemo demo;
    demo.runDemo();
    return 0;
}