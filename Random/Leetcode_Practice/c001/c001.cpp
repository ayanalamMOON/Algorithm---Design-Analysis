/* 
 * Problem: Debug the following code which is supposed to count the number of occurrences of each character in a string.
 * 
 * The code has several bugs that need to be fixed:
 * 1. The program should print the character and its count only if the character appears at least once.
 * 2. Currently, the program doesn't count characters correctly.
 * 3. The program doesn't handle spaces properly.
 * 
 * Your task is to identify and fix the bugs in the code below.
 */

/*

#include <iostream>
#include <string>
using namespace std;

void countCharacters(string str) {
    int count[256] = {0}; // Array to store count of each character
    
    // Count occurrences of each character
    for (int i = 0; i <= str.length(); i++) {
        count[str[i]]++;
    }
    
    // Print the count of each character
    for (int i = 0; i < 256; i++) {
        // If character is present in the string
        if (count[i] > 0 && i != ' ')
            cout << "Character " << (char)i << " occurs " << count[i] << " times." << endl;
    }
}

int main() {
    string str;
    cout << "Enter a string: ";
    cin >> str;
    
    countCharacters(str);
    
    return 0;
}

*/

/* 
 * Debug this code:
 * 1. Fix the loop that counts characters
 * 2. Fix the character printing logic
 * 3. Make sure spaces are handled correctly
 * 4. Ensure the program reads the entire line of input, not just the first word
 */

#include<bits/stdc++.h>
using namespace std;

void countCharacter(const string &str){
    int count[256]={0};
    for(int i=0; i<str.length(); i++){
        count[(unsigned char)str[i]]++;
    }
    for(int i=0; i<256; i++){
        if(count[i]>0){
            cout << "Character " << (char)i << " occurs " << count[i] << " times." << endl;
        }
    }
}

int main(){
    cout<<"Enter a string:";
    string str;
    getline(cin, str); // Use getline to read the entire line
    countCharacter(str);
    return 0;
}