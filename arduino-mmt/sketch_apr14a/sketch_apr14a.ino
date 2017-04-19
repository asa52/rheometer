void setup() {
  // put your setup code here, to run once:
   SerialUSB.begin(115200);
    analogWriteResolution(12);  // set the analog output resolution to 12 bit (4096 levels)
    analogReadResolution(12);   // set the analog input resolution to 12 bit (4096 levels)
    pinMode(DAC0, OUTPUT);
    pinMode(A0, INPUT);
    pinMode(22, OUTPUT);
    pinMode(2, OUTPUT);   // port B pin 25
    analogWrite(2, 255);  // sets up some other registers I haven't worked out yet
    
}

void loop() {
  // put your main code here, to run repeatedly:
  //for (int i = 0; i < 21; i++){
  //  SerialUSB.println(i*200 + 25);
   // analogWrite(DAC0, i*200 + 25); //change to lower level code TODO
   // delay(10000);
    //int pos = analogRead(DAC0);
  //}
    //SerialUSB.println(4000 + i);
    //analogWrite(DAC0, 4000 + i); //change to lower level code TODO
    //delay(5000);
    int pos = analogRead(A0);
    SerialUSB.println(pos);
    delay(1000);
}
