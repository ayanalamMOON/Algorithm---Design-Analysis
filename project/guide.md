# Algorithm Performance Testing Framework User Guide

## Overview

This guide provides step-by-step instructions and examples for using the Algorithm Performance Testing Framework.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Sorting and Searching](#sorting-and-searching)
3. [Graph Algorithms](#graph-algorithms)
4. [Dynamic Programming](#dynamic-programming)
5. [String Matching](#string-matching)
6. [Performance Visualization Guide](#performance-visualization-guide)
7. [Custom Benchmarking from Menu](#custom-benchmarking-from-menu)

## Getting Started

When you run the program, you'll see this menu:
```
1. Test Sorting and Searching Algorithms
2. Test Graph Algorithms
3. Test Dynamic Programming and String Matching
4. Run Custom Benchmark
5. Exit
Enter your choice:
```

## Sorting and Searching

### Example 1: Testing Sorting Algorithms

Input:
```
Enter your choice: 1
Enter the number of elements to generate: 10
```

Output:
```
Generated integers: 423 871 159 742 315 968 247 563 891 104

--- Sorting Algorithm Performance ---
Algorithm            Time (ms)    Time Complexity    Space Complexity
---------------------------------------------------------------
Bubble Sort          0.0842      O(n²)              O(1)
Selection Sort       0.0654      O(n²)              O(1)
Quick Sort           0.0123      O(n log n)         O(log n)
Merge Sort           0.0156      O(n log n)         O(n)
Heap Sort            0.0189      O(n log n)         O(1)
...

Sorted integers: 104 159 247 315 423 563 742 871 891 968
```

### Example 2: Searching

After sorting, you can search for a value:

Input:
```
Enter a value to search: 315
```

Output:
```
--- Searching Algorithm Performance ---
Algorithm            Time (ms)    Time Complexity    Space Complexity    Result
-------------------------------------------------------------------------
Linear Search        0.0012      O(n)              O(1)                Found
Binary Search        0.0008      O(log n)          O(1)                Found
```

## Graph Algorithms

### Example 1: Creating and Analyzing a Graph

Input:
```
Enter your choice: 2
Enter number of vertices (1-1000): 4
Enter number of edges (0-12): 5
Enter edge information (source destination weight):
0 1 10
0 2 15
1 2 5
2 3 20
1 3 25
Enter source vertex: 0
```

Output:
```
--- Graph Algorithm Performance ---
Algorithm            Time (ms)    Time Complexity    Space Complexity
----------------------------------------------------------------
Dijkstra             0.0234      O(V log V + E)     O(V)
Shortest distances from vertex 0:
To vertex 1: 10
To vertex 2: 15
To vertex 3: 35
```

### Example 2: Maximum Flow

Input:
```
Enter sink vertex for Ford-Fulkerson and Edmonds-Karp: 3
```

Output:
```
Ford-Fulkerson Max Flow: 30
Edmonds-Karp Max Flow: 30
```

## Dynamic Programming

### Example 1: 0/1 Knapsack Problem

Input:
```
Enter your choice: 3
Enter number of items for Knapsack: 3
Enter values and weights:
Item 1 - Value: 60
Item 1 - Weight: 10
Item 2 - Value: 100
Item 2 - Weight: 20
Item 3 - Value: 120
Item 3 - Weight: 30
Enter knapsack capacity: 50
```

Output:
```
Maximum value: 220
Selected items: 2 1
```

### Example 2: Matrix Chain Multiplication

Input:
```
Enter number of matrices: 4
Enter matrix dimensions:
p0: 30
p1: 35
p2: 15
p3: 5
p4: 10
```

Output:
```
Minimum operations: 15125
Multiplication order: 0 1 2 3
```

## String Matching

### Example: Pattern Matching

Input:
```
Enter text: AABAACAADAABAAABAA
Enter pattern: AABA
```

Output:
```
KMP matches found at positions: 0 9 13
Rabin-Karp matches found at positions: 0 9 13
```

## Error Handling Examples

### Invalid Input Example

Input:
```
Enter number of elements to generate: -5
```

Output:
```
Error: Invalid size: must be between 1 and 1000000
```

### Invalid Graph Input

Input:
```
Enter number of vertices (1-1000): 0
```

Output:
```
Error: Invalid number of vertices
```

## Performance Visualization

The program displays performance metrics using ASCII bar charts:
```
Algorithm            Performance
--------------------------------
Quick Sort       |##### 0.0123ms
Merge Sort       |###### 0.0156ms
Heap Sort        |####### 0.0189ms
Bubble Sort      |################################ 0.0842ms
```

## Performance Visualization Guide

### Basic Usage

The PerformanceVisualizer provides visual representation of algorithm performance using ASCII bar charts. Here's how to use it:

```cpp
// Create a vector of algorithm results
vector<AlgorithmResult> results;

// Add results to the vector
AlgorithmResult result1 = {
    .execution_time = 0.0123,
    .algorithm_name = "Quick Sort",
    .complexity = "O(n log n)",
    .success = true
};
results.push_back(result1);

// Display the bar chart
PerformanceVisualizer::DisplayBarChart(results);
```

### Example Output

```
Algorithm Performance Visualization
----------------------------------
Quick Sort       |##### 0.0123ms
Merge Sort       |###### 0.0156ms
Heap Sort        |####### 0.0189ms
Bubble Sort      |################################ 0.0842ms
```

### Real-World Examples

1. **Comparing Sorting Algorithms**
```
Enter the number of elements to generate: 1000

--- Sorting Algorithm Performance ---
Algorithm            Performance
--------------------------------
Quick Sort       |##### 0.0123ms
Merge Sort       |###### 0.0156ms
Heap Sort        |####### 0.0189ms
Bubble Sort      |################################ 0.0842ms
Selection Sort   |############################ 0.0754ms
```

2. **Graph Algorithm Comparison**
```
Enter number of vertices: 100
Enter number of edges: 300

--- Graph Algorithm Performance ---
Algorithm            Performance
--------------------------------
DFS              |### 0.0034ms
BFS              |#### 0.0045ms
Dijkstra         |########## 0.0156ms
Floyd-Warshall   |################################ 0.0923ms
```

### Features

1. **Automatic Scaling**
   - Bar lengths are automatically scaled relative to the longest execution time
   - Maximum bar width is 50 characters
   - Times are displayed with 4 decimal places

2. **Visual Elements**
   - Algorithm names are left-aligned in 20-character field
   - Bars are drawn using '#' characters
   - Execution times are shown after the bars

3. **Comparison Features**
   - Easily compare multiple algorithms
   - Visual representation of relative performance
   - Clear indication of fastest/slowest algorithms

### Tips for Using the Visualizer

1. **Grouping Related Algorithms**
```cpp
// Group sorting algorithms
vector<AlgorithmResult> sortingResults;
// Add sorting algorithm results
PerformanceVisualizer::DisplayBarChart(sortingResults);

// Group searching algorithms
vector<AlgorithmResult> searchingResults;
// Add searching algorithm results
PerformanceVisualizer::DisplayBarChart(searchingResults);
```

2. **Custom Benchmarking**
```cpp
// Benchmark with custom parameters
AlgorithmResult result;
result.algorithm_name = "Custom Algorithm";
result.execution_time = measurePerformance(customAlgorithm);
result.complexity = "O(n)";
results.push_back(result);
```

3. **Error Handling**
```cpp
try {
    PerformanceVisualizer::DisplayBarChart(results);
} catch (const exception& e) {
    cout << "Visualization error: " << e.what() << "\n";
}
```

### Integration with Logging

The visualizer works seamlessly with the logging system:

```cpp
// Display and log performance
for (const auto& result : results) {
    // Display in bar chart
    PerformanceVisualizer::DisplayBarChart({result});
    
    // Log the performance
    Logger::LogPerformance(
        result.algorithm_name,
        result.execution_time,
        result.complexity
    );
}
```

### Best Practices

1. **Data Size Considerations**
   - Use appropriate data sizes for meaningful comparisons
   - Consider running multiple iterations for more accurate results
   - Group similar algorithms together

2. **Visualization Tips**
   - Sort results by execution time for better readability
   - Use consistent data sizes when comparing algorithms
   - Include complexity information for context

3. **Output Formatting**
   - Consider terminal width when displaying results
   - Use appropriate precision for time measurements
   - Include units (ms) in the display

## Custom Benchmarking

You can benchmark your own algorithms using the public BenchmarkAlgorithm function:

```cpp
// Example 1: Benchmark a simple sorting algorithm
vector<int> data = {5, 3, 8, 1, 9};
auto result = AlgorithmPerformanceTester::BenchmarkAlgorithm(
    "My Sort",
    [&]() {
        sort(data.begin(), data.end());
    }
);

// Example 2: Benchmark with custom iterations
auto result2 = AlgorithmPerformanceTester::BenchmarkAlgorithm(
    "Complex Algorithm",
    []() {
        // Your algorithm here
    },
    10  // Run 10 iterations
);

// Example 3: Accessing benchmark results
cout << "Execution time: " << result.execution_time << "ms\n"
     << "Success: " << (result.success ? "Yes" : "No") << "\n";
```

### Parameters:
- `name`: String identifier for the algorithm
- `algorithm`: Lambda function containing the code to benchmark
- `iterations`: (Optional) Number of times to run the benchmark

### Return Value:
Returns an AlgorithmResult structure containing:
- `execution_time`: Average execution time in milliseconds
- `algorithm_name`: Name of the algorithm
- `success`: Whether the benchmark completed successfully
- `error_message`: Error description if the benchmark failed

### Best Practices for Custom Benchmarking:
1. Use meaningful algorithm names
2. Consider warm-up runs before actual benchmarking
3. Use appropriate data sizes for your tests
4. Ensure consistent test conditions

## Custom Benchmarking from Menu

The program now includes a dedicated benchmark option in the main menu:

```
=== Algorithm Performance Testing Framework ===
1. Test Sorting and Searching Algorithms
2. Test Graph Algorithms
3. Test Dynamic Programming and String Matching
4. Run Custom Benchmark
5. Exit
```

### Using the Custom Benchmark

When selecting option 4, you can choose from:
1. Vector operations
2. String operations
3. Custom data operation

Example usage:

```
Enter choice: 1
Enter vector size: 1000000

Benchmark Results:
Algorithm: Vector Sort
Time: 45.3216ms
Status: Success
```

For string operations:
```
Enter choice: 2
Enter text to process: Hello World!

Benchmark Results:
Algorithm: String Reverse
Time: 0.0023ms
Status: Success
```

For custom operations:
```
Enter choice: 3
Enter number of iterations: 10

Benchmark Results:
Algorithm: Custom Operation
Time: 12.4567ms
Status: Success
Iterations: 10
```

## Enhanced Benchmarking Features

### Detailed Performance Analysis

The enhanced benchmark function provides detailed statistics:

```
Detailed Statistics for Quick Sort:
Min Time: 0.0121ms
Max Time: 0.0156ms
Avg Time: 0.0134ms
Std Dev: 0.0012ms
Memory Used: 1024KB
Distribution:
0.012ms: ***
0.013ms: *****
0.014ms: ****
0.015ms: **
```

### Memory Usage Analysis

Track memory consumption of algorithms:

```
Enter choice: 5
Enter data structure size: 1000000

Memory Usage Analysis:
Initial Memory: 1024KB
Peak Memory: 5120KB
Memory Delta: 4096KB
```

### Algorithm Comparison

Compare multiple algorithms with detailed statistics:

```
Enter choice: 4
Enter vector size for comparison: 100000

Comparing Sorting Algorithms:
STL Sort:    |##### 0.0123ms (±0.0005ms)
Quick Sort:  |###### 0.0156ms (±0.0008ms)
Merge Sort:  |####### 0.0189ms (±0.0007ms)

Memory Usage:
STL Sort:    1024KB
Quick Sort:  1536KB
Merge Sort:  2048KB
```

### Best Practices for Enhanced Benchmarking

1. **Statistical Analysis**
   - Use multiple iterations for reliable results
   - Consider standard deviation for stability analysis
   - Look for performance outliers

2. **Memory Analysis**
   - Monitor peak memory usage
   - Track memory leaks
   - Consider memory fragmentation

3. **Comparative Analysis**
   - Use consistent data sets
   - Consider different input sizes
   - Account for system state

## Logging

The program creates two log files:
- `algorithm_performance.log`: Records execution times and complexities
- `error.log`: Records any errors that occur during execution

Example log entry:
```
Thu Jan 20 10:30:45 2024
Quick Sort: 0.0123ms, O(n log n)
```

## Best Practices

1. Start with small data sets to verify correctness
2. Use larger data sets for performance testing
3. Consider algorithm trade-offs based on:
   - Input size
   - Data distribution
   - Memory constraints
   - Time constraints

## Tips

1. For sorting/searching:
   - Use Quick Sort for general-purpose sorting
   - Use Binary Search when data is sorted
   - Consider Count Sort for integer data with small range

2. For graph algorithms:
   - Use Dijkstra for non-negative weighted graphs
   - Use Bellman-Ford when negative weights are present
   - Use A* when heuristic information is available

3. For dynamic programming:
   - Verify input constraints carefully
   - Consider space-optimized versions for large inputs