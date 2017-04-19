#include <stdint.h>

int t = 0;

int get_t();

int main(){
    t++;
}

int get_t(){
    return t;
}