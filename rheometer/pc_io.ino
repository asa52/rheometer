/*For functions that handle PC input and output.
*/

void handle_pc_input() {
    if (SerialUSB.available() > 0) {
        mode = SerialUSB.read(); //first byte encodes input type
        //REG_TC0_RC0 *= 2;
        if (mode == 0 || mode == 1 || mode == 3 || mode == 5) {
            SerialUSB.readBytes(val_in, 4);
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 4; j++) {
                    bitWrite(val, i + j * 8, bitRead(val_in[j], i));
                }
            }
            /*
            if(val == 1024){
              REG_TC0_RC0 *= 2;
            }
            */

            if (mode == 0) {
                REG_TC0_RC0 = val;
                // cannot predict response at new frequency
                // hence set to zero
                if (0 == run_option) {
                    amp = 0;
                }

                amp_step = 0;
                old_amp_step = 0;
                strain_closing_in = 0;

                // interrupt handler intervals have changed size
                // restart data aqcuisition processes
                freq_check = 0;
                p = 0;
                A0_period_count = 0;
                t_diff = 0;
                e_num = -1;
                delta_num = 0;
                pest_num = -1;
                feed_num = -1;

            } else if (mode == 1) { //bitRead(mode,0) == 1)
                if (0 == run_option) {
                    set_strain = val;
                    amp_step = 0;
                    old_amp_step = 0;
                    strain_closing_in = 0;
                } else if (1 == run_option) {
                    amp = val;
                } else if (2 == run_option) {
                    DC_func = val;
                }
            } else if (mode == 3) {
                if (0 == korb) {
                    simu_k = val;
                } else if (1 == korb) {
                    simu_b = val;
                }
            } else if (mode == 5) {
                equilibrium_A0 = val;
            }

        } else if ((mode == 0b00000010 || mode == 0b00000100) && (1 == run_option || 2 == run_option)) {
            if (mode == 0b00000010) {
                if (1 == run_option && 0 < amp) {
                    amp -= 1;
                }

                if (2 == run_option && DC_func > 0) {
                    DC_func -= 1;
                }
                //amp_step /= 2;

            } else if (mode == 0b00000100) {
                if (1 == run_option && 2048 > amp) {
                    amp += 1;
                }
                if (2 == run_option && DC_func < 4095) {
                    DC_func += 1;
                }
                
                /* 
                amp_step *= 2;
              if(amp_step == 0){
                amp_step = 1;
              }
              */

            }

        } else if (mode == 0b00001000) {
            //attempt to keep strain amplitude constant
            run_option = 0;
            amp_step = 0;
            old_amp_step = 0;
            strain_closing_in = 0;

        } else if (mode == 0b00010000) {
            //keep stress amplitude constant
            run_option = 1;
        } else if (mode == 0b00100000) {
            //deactivate spring and damping simulation
            NR = 0;
        } else if (mode == 0b01000000) {
            //activate spring and damping simulation
            NR = 1;
        } else if (mode == 0b00000111) {
            //send a DC current to oscillator
            run_option = 2;
            amp = 0;
        } else if (mode == 0b00000110) {
            //toggle spring/damping value user input
            korb = 1 - korb;
        } else if (mode == 0b00001100) {
            //only send simulation via USB, but do not apply
            NR = 2;
        } else if (mode == 0b00011000) {
            //toggle driving torgue/measured velocity being sent via USB
            Torv = 1 - Torv;
        } else if (mode == 0b00110000) {
            //toggle floating centre/permanent equilibrium displacement estimate
            centre_mode = 1 - centre_mode;
            if (-1 == equilibrium_A0) {
                equilibrium_A0 = centre;
            }
        }
    }
}

// ~~~~~~~~~~ beginning of space for functions that send data via SerialUSB ~~~~~~~~~~

void send_3_byte_value(int value, byte third_byte) {
    // give 3rd byte in binary 0b7th6th5th4th3rd2nd1st0th

    b[0] = value;

    for (int i = 8; i <= 14; i++) { //had problems with 14th bit being set for no reason
        bitWrite(b[1], i - 8, bitRead(value, i));
    }

    bitSet(b[1], 7);
    b[2] = third_byte;
    SerialUSB.write(b, 3);
    bitClear(b[1], 7);
    bitClear(b[1], 6);
    b[2] = 0;
}

void send_out_of_bounds_values() {
    int excess;
    if (pos <= 1558) {
        excess = 0;
    } else {
        excess = 12000;
    }

    b[0] = excess;

    for (int i = 8; i <= 13; i++) {
        bitWrite(b[1], i - 8, bitRead(excess, i));
    }

    bitSet(b[1], 6);
    SerialUSB.write(b, 2);
    bitClear(b[1], 6);
    b[2] = 0;
    b[0] = amp;

    for (int i = 8; i <= 13; i++) {
        bitWrite(b[1], i - 8, bitRead(amp, i));
    }

    bitSet(b[1], 7);
    SerialUSB.write(b, 3);
    bitClear(b[1], 7);
}

void send_func() {
    b[0] = func;

    for (int i = 8; i <= 13; i++) {
        bitWrite(b[1], i - 8, bitRead(func, i));
    }

    bitClear(b[1], 4);
    bitClear(b[1], 5);
    b[2] = t;
    SerialUSB.write(b, 3);
}

void send_mu() {
    b[0] = mu; //set byte b[0] equal to rightmost / lowest byte of 32 bit integer mu

    for (int i = 8; i <= 13; i++) {  // set byte b[1]'s bits # 0 to 5 equal
        bitWrite(b[1], i - 8, bitRead(mu, i)); // to bits 8 to 13 of 32 bit integer mu
    }

    bitSet(b[1], 6);   // set byte b[1]'s bit # 6 to 1 to signal data of different kind
    SerialUSB.write(b, 2); // bit-wise operators count bits from zero, like the
    bitClear(b[1], 6);     // powers of two they represent
}

void send_pos() {
    b[0] = pos;

    for (int i = 8; i <= 13; i++) {
        bitWrite(b[1], i - 8, bitRead(pos, i));
    }

    //bitSet(b[1],6);
    SerialUSB.write(b, 2);
    //bitClear(b[1],6);
}

void map_centre_back_to_pos(int last_pos, int try_num, int twos) {
    if (10 > try_num && A0mu[last_pos] != centre) {
        if (A0mu[last_pos] < centre) {
            try_num += 1; //start counting from 1 not 0 tries
            twos *= 2; //avoid pow function as uses float
            int new_pos = (last_pos + (1038 / twos)); //need powers of 2 fractions of half maximum
            map_centre_back_to_pos(new_pos, try_num, twos);//recrusive call with advanced values
        } else if (A0mu[last_pos] > centre) {
            try_num += 1; //start counting from 1 not 0 tries
            twos *= 2; //avoid pow function as uses float
            int new_pos = (last_pos - (1038 / twos)); //need powers of 2 fractions of half maximum
            map_centre_back_to_pos(new_pos, try_num, twos);//recrusive call with advanced values
        }

    } else if (A0mu[last_pos] == centre) {
        pos_rec = (pos_0 + last_pos);
    } else { // try_num == 10, so 2^try_num == 1024 and int 2076 / 1024 == 2
        int get_out = 0;
        for (int j = 1; j < 128; j++) { //only widen value search range once entry range has been searched
            for (int i = last_pos; i <= (last_pos + 2); i++) { // last_pos + 2076/2^try_num
                if (A0mu[i] == centre + j) {
                    pos_rec = (pos_0 + i);  //break out i-loop
                    get_out = 1;
                    break;
                } else if (A0mu[i] == centre - j) {
                    pos_rec = (pos_0 + i);
                    get_out = 1;
                    break;
                }
            }
            if (1 == get_out) {
                break; //break out j-loop
            }
        } //in case no match is found
        if (0 == get_out) { // though unlikely, check entries 0, 1 and 2 as well
            for (int j = 1; j < 128; j++) {
                for (int i = 0; i <= 2; i++) {
                    if (A0mu[i] == centre + j) {
                        pos_rec = (pos_0 + i);  //break out i-loop
                        get_out = 1;
                        break;
                    } else if (A0mu[i] == centre - j) {
                        pos_rec = (pos_0 + i);
                        get_out = 1;
                        break;
                    }
                }
                if (1 == get_out) {
                    break; //break out j-loop
                }
            }
        }
    }
}
