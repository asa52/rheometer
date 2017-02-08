/*File to declare: Initial setup functions to enable communication between the Arduino and the Processing GUI file.*/

void setup() {
    SerialUSB.begin(115200);
    analogWriteResolution(12);  // set the analog output resolution to 12 bit (4096 levels)
    analogReadResolution(12);   // set the analog input resolution to 12 bit (4096 levels)
    pinMode(DAC1, OUTPUT);
    pinMode(22, OUTPUT);
    pinMode(2, OUTPUT);   // port B pin 25
    analogWrite(2, 255);  // sets up some other registers I haven't worked out yet
    REG_PIOB_PDR = 1 << 25; // disable PIO, enable peripheral
    REG_PIOB_ABSR = 1 << 25; // select peripheral B
    REG_TC0_WPMR = 0x54494D00; // enable write to registers
    REG_TC0_CMR0 = 0b00000000000010011100010000000000; // set channel mode register (see datasheet)
    REG_TC0_RC0 = val; // counter period
    // needs to be HALF of (1/freq) * (84 MHz/sample_num) for some reason
    // 2 is smallest prescale factor, that's why
    // clock speed is 84 MHz
    REG_TC0_RA0 = 30000000; // PWM value
    REG_TC0_CCR0 = 0b101;  // start counter
    REG_TC0_IER0 = 0b00010000; // enable interrupt on counter=rc
    REG_TC0_IDR0 = 0b11101111; // disable other interrupts

    NVIC_EnableIRQ(TC0_IRQn); // enable TC0 interrupts

    REG_PMC_WPMR = 0x504D43;
    bitSet(REG_PMC_PCER1, 6);
    REG_DACC_CR = 1;
    REG_DACC_WPMR = 0x444143;
    REG_DACC_ACR = 0b00000000000000000000000100001010;
    REG_DACC_IER = 1;
    REG_DACC_MR = 0b00001100000000010000100000000000;
    bitSet(REG_DACC_CHER, 1);
}

void loop() {

}
