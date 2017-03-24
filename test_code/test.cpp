#include "stdint.h"
#include <iostream>
#include <bitset>

#define bitRead(value, bit) (((value) >> (bit)) & 0x01)
#define bitSet(value, bit) ((value) |= (1UL << (bit)))
#define bitClear(value, bit) ((value) &= ~(1UL << (bit)))
#define bitWrite(value, bit, bitvalue) (bitvalue ? bitSet(value, bit) : bitClear(value, bit))


typedef uint8_t byte;

using namespace std;

int main(){

	byte val_in[4] = {0, 0, 0, 0};
	unsigned long int val = 70000;
	cout << bitset<32>(val) << endl;
	for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            bitWrite(val, i + j * 8, bitRead(val_in[j], i));
    		cout << i << " " << j << " " << bitset<32>(val) << endl;  
        }
    }

	return 0;
}