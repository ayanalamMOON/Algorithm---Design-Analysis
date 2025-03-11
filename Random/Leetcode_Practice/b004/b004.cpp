/**
 * @title: "Persistent Bugger"
 * @difficulty: Intermediate
 * 
 * @description:
 * Write a function that takes a positive integer and returns its multiplicative persistence,
 * which is the number of times you must multiply the digits together until you reach a single digit.
 * 
 * @examples:
 * - persistence(39) => 3
 *   Because: 3*9 = 27, 2*7 = 14, 1*4 = 4 and 4 has only one digit.
 * - persistence(999) => 4
 *   Because: 9*9*9 = 729, 7*2*9 = 126, 1*2*6 = 12, and finally 1*2 = 2.
 * - persistence(4) => 0
 *   Because: 4 is already a one-digit number.
 * 
 * @constraints:
 * - 0 < n < 10^18
 * 
 * @task:
 * Implement the function int persistence(long long n)
 */

#include <bits/stdc++.h>
using namespace std;

int persistence(long long n){
    int steps = 0;
    while(n>=10){
        long long prod=1;
        while(n>0){
            prod*=n%10;
            n/=10;
        }
        n=prod; steps++;
    }
    return steps;
}

int main(){
    cout << persistence(39) << endl; // Output: 3
    cout << persistence(999) << endl; // Output: 4
    cout << persistence(4) << endl; // Output: 0
    return 0;
}
