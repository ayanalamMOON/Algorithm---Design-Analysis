/*
    *Codewars style kata: "Digital Root Sum"
    
    *Difficulty: Medium

    *Description:
    *Digital root is the recursive sum of all the digits in a number.
    *Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.
    *For example, if the input is 16, then the output should be 7 because 1 + 6 = 7.
    *If the input is 942, then the output should be 6 because 9 + 4 + 2 = 15 and 1 + 5 = 6.

    Examples:
        digitalRootSum(16) => 7
        digitalRootSum(942) => 6
        digitalRootSum(132189) => 6
        digitalRootSum(493193) => 2

    Constraints:
    0 <= num <= 10^9

    Write a function that calculates the digital root sum of a number.
*/

#include<bits/stdc++.h>
using namespace std;
int digitalRootSUm(int num){
    while(num>9){
        int s=0;
        while(num>0){
            s+=num % 10;
            num/=10;
        }
        num=s;
    }
    return num;
}

int main(){

    cout << digitalRootSUm(16) << endl;
    cout << digitalRootSUm(942) << endl;
    cout << digitalRootSUm(132189) << endl;
    cout << digitalRootSUm(493193) << endl;
    return 0;
}
