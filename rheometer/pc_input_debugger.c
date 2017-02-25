void handle_pc_input() {
    if (SerialUSB.available() > 0) {
        mode = SerialUSB.read(); //first byte encodes input type
        //REG_TC0_RC0 *= 2;
        if (mode == 0 || mode == 1 || mode == 3 || mode == 5) {
            SerialUSB.readBytes(val_in, 4); // byte type stores 8-bit unsigned int from 0 to 255.
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