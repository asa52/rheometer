void setup() {
  // put your setup code here, to run once:
  SerialUSB.begin(115200);
  analogWriteResolution(12);  // set the analog output resolution to 12 bit (4096 levels)
  analogReadResolution(12);   // set the analog input resolution to 12 bit (4096 levels)
  pinMode(DAC1, OUTPUT);
  pinMode(A0, INPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(DAC1, HIGH);
  int val = analogRead(A0);    // read the input pin
  SerialUSB.println(val);      // debug value
  delay(1000);
  digitalWrite(DAC1, LOW);
  val = analogRead(A0);    // read the input pin
  SerialUSB.println(val);         // debug value  
  delay(1000);
}
