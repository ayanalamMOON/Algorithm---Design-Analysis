import pandas as pd
import matplotlib.pyplot as plt
import sys

def visualize_performance(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Create a bar chart of execution times
    plt.figure(figsize=(10, 6))
    plt.bar(data['Algorithm'], data['Time (ms)'], color='skyblue')
    plt.xlabel('Algorithm')
    plt.ylabel('Execution Time (ms)')
    plt.title('Algorithm Performance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Optionally, save the figure
    # plt.savefig('performance_chart.png')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualizer.py <results.csv>")
    else:
        csv_file = sys.argv[1]
        visualize_performance(csv_file)
