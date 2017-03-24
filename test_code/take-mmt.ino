#include "Waveforms.h"

#define oneHzSample 1000000/maxSamplesNum  // sample for the 1Hz signal expressed in microseconds 

int i = 0;
int sample;
volatile int wave0 = 0, wave1 = 0;

void setup() {
  // put your setup code here, to run once:
  SerialUSB.begin(115200);
  analogWriteResolution(12);  // set the analog output resolution to 12 bit (4096 levels)
  analogReadResolution(12);   // set the analog input resolution to 12 bit (4096 levels)
  pinMode(DAC1, OUTPUT);
  pinMode(A0, INPUT);
}
