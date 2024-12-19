import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import time
import subprocess
from typing import Dict, List
import sys

class AlgorithmVisualizer:
    def __init__(self):
        self.data = None
        self.results_file = "algorithm_results.json"
        sns.set_style("whitegrid")
        plt.style.use('seaborn')

    def run_cpp_program(self) -> bool:
        """Run the C++ program and wait for results"""
        try:
            result = subprocess.run(["./test"], capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Error running C++ program: {e}")
            return False

    def load_data(self) -> bool:
        """Load performance data from JSON file"""
        try:
            with open(self.results_file, 'r') as f:
                self.data = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def create_performance_plot(self):
        """Create bar plot of algorithm performance"""
        if not self.data:
            return

        algorithms = [algo["name"] for algo in self.data["algorithms"]]
        times = [algo["time"] for algo in self.data["algorithms"]]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(algorithms, times)
        plt.title("Algorithm Performance Comparison")
        plt.xlabel("Algorithm")
        plt.ylabel("Execution Time (ms)")
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}ms',
                    ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('performance_comparison.png')
        plt.close()

    def create_complexity_heatmap(self):
        """Create heatmap showing relative performance"""
        if not self.data:
            return

        algorithms = [algo["name"] for algo in self.data["algorithms"]]
        times = [algo["time"] for algo in self.data["algorithms"]]
        
        # Create correlation matrix
        n = len(algorithms)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = times[i] / times[j] if times[j] != 0 else 0

        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.2f', 
                    xticklabels=algorithms, yticklabels=algorithms)
        plt.title("Relative Performance Comparison")
        plt.tight_layout()
        plt.savefig('performance_heatmap.png')
        plt.close()

    def create_time_distribution(self):
        """Create distribution plot of execution times"""
        if not self.data:
            return

        times = [algo["time"] for algo in self.data["algorithms"]]
        algorithms = [algo["name"] for algo in self.data["algorithms"]]

        plt.figure(figsize=(12, 6))
        sns.violinplot(data=times)
        plt.title("Execution Time Distribution")
        plt.ylabel("Time (ms)")
        plt.xticks([0], ["Algorithms"])
        plt.tight_layout()
        plt.savefig('time_distribution.png')
        plt.close()

    def create_success_rate_plot(self):
        """Create plot showing success rate of algorithms"""
        if not self.data:
            return

        algorithms = [algo["name"] for algo in self.data["algorithms"]]
        success = [1 if algo["success"] else 0 for algo in self.data["algorithms"]]
        
        plt.figure(figsize=(12, 6))
        plt.bar(algorithms, success, color=['green' if s else 'red' for s in success])
        plt.title("Algorithm Success Rate")
        plt.xlabel("Algorithm")
        plt.ylabel("Success (1) / Failure (0)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('success_rate.png')
        plt.close()

    def generate_report(self):
        """Generate HTML report with all visualizations"""
        if not self.data:
            return

        html_content = """
        <html>
        <head>
            <title>Algorithm Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: auto; }
                .visualization { margin: 20px 0; }
                img { max-width: 100%; height: auto; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Algorithm Performance Analysis Report</h1>
                
                <h2>Performance Comparison</h2>
                <div class="visualization">
                    <img src="performance_comparison.png" alt="Performance Comparison">
                </div>

                <h2>Performance Heatmap</h2>
                <div class="visualization">
                    <img src="performance_heatmap.png" alt="Performance Heatmap">
                </div>

                <h2>Time Distribution</h2>
                <div class="visualization">
                    <img src="time_distribution.png" alt="Time Distribution">
                </div>

                <h2>Success Rate</h2>
                <div class="visualization">
                    <img src="success_rate.png" alt="Success Rate">
                </div>

                <h2>Detailed Results</h2>
                <table>
                    <tr>
                        <th>Algorithm</th>
                        <th>Time (ms)</th>
                        <th>Success</th>
                        <th>Complexity</th>
                    </tr>
        """

        for algo in self.data["algorithms"]:
            html_content += f"""
                    <tr>
                        <td>{algo['name']}</td>
                        <td>{algo['time']:.2f}</td>
                        <td>{'Yes' if algo['success'] else 'No'}</td>
                        <td>{algo.get('complexity', 'N/A')}</td>
                    </tr>
            """

        html_content += """
                </table>
            </div>
        </body>
        </html>
        """

        with open('performance_report.html', 'w') as f:
            f.write(html_content)

    def run_visualization(self):
        """Run the complete visualization pipeline"""
        print("Starting visualization process...")
        
        if not self.run_cpp_program():
            print("Failed to run C++ program")
            return

        if not self.load_data():
            print("Failed to load data")
            return

        print("Generating visualizations...")
        self.create_performance_plot()
        self.create_complexity_heatmap()
        self.create_time_distribution()
        self.create_success_rate_plot()
        self.generate_report()
        print("Visualization complete! Check performance_report.html for results.")

if __name__ == "__main__":
    visualizer = AlgorithmVisualizer()
    visualizer.run_visualization()
