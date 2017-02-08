void loop() {
  // Read the the potentiometer and map the value  between the maximum and the minimum sample available
  // 1 Hz is the minimum freq for the complete wave
  // 170 Hz is the maximum freq for the complete wave. Measured considering the loop and the analogRead() time
  
  sample = map(1, 0, 4095, 0, oneHzSample);
  sample = constrain(sample, 0, oneHzSample);
  analogWrite(DAC1, waveformsTable[wave1][i]);  // write the selected waveform on DAC1
  SerialUSB.println(analogRead(A0));
  
  i++;
  if(i == maxSamplesNum)  // Reset the counter to repeat the wave
    i = 0;

  delayMicroseconds(sample);  // Hold the sample value for the sample time
}
