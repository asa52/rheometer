/*Main timer-counter handler function. is this the main function instead of loop?*/

void TC0_Handler() {
	// why does this function run: it is never explicitly called???? is this the main function?
    long dummy = REG_TC0_SR0; // vital - reading this clears some flag
    // otherwise you get infinite interrupts

    //bitSet(REG_PIOB_SODR, 26);

    pos = analogRead(measure); //reads the voltage at analog pin A0

    if ((pos >= 1558 && pos <= 3633)) {
        // TODO Why this limit on the pos value?
        int num = (pos - pos_0); // Convert to index from which the voltage from A0 can be converted into TODO 'distance values'
        //mu_two_back = mu_one_back;
        //mu_one_back = mu;
        mu = A0mu[num]; //0.1 microns
        //send_mu();
    }

    if (centre_mode == 1) {
        used_zero_A0 = equilibrium_A0; //set zero point to some value you set
    } else {
        used_zero_A0 = centre; // CHECK - ERROR centre does not appear to have been defined before this inequality is done - all ok?
    }

    if (NR != 1 && (run_option == 0 || run_option == 1)) {
        // If we are not in normalised resonance mode and the run option is either one or two
        func = (((waveformsTable[t] - waveformsTable[0]) * amp) / 2048) + 2047; // func is sine wave added onto mid-value
    } else if (NR != 1 && run_option == 2) {
        // 2nd run option is constant output mode
        func = DC_func;
    } else if (NR == 1 && (run_option == 0 || run_option == 1)) {
        func = ((((waveformsTable[t] - waveformsTable[0]) * amp) / 2048)
                + ((//( (mu_two_back - used_zero_A0) + (mu_one_back - used_zero_A0) + (mu - used_zero_A0) )
                    //( (mu_one_back - used_zero_A0) + (mu - used_zero_A0) )
                    (mu - used_zero_A0) * simu_k) / (simu_k_unit))  // elastic response - CHECK
                + ((dmudt * simu_b) / (simu_b_unit)))  // viscous response - CHECK
               + 2047;  // midpoint
    } else if (NR == 1 && run_option == 2) {
        func = DC_func 
               + ((//( (mu_two_back - used_zero_A0) + (mu_one_back - used_zero_A0) + (mu - used_zero_A0) )
                   //( (mu_one_back - used_zero_A0) + (mu - used_zero_A0) )
                   (mu - used_zero_A0) * simu_k) / (simu_k_unit))
               + ((dmudt * simu_b) / (simu_b_unit));
    }

    //analogWrite(DAC1, func); // writes this sine wave to the DAC output pin 1
    if (func > 4095) {
        // if function too large, clip
        func = 4095;
    }
    if (func < 0) {
        // if too small, clip
        func = 0;
    }

    REG_DACC_CDR = func; // analog write to DAC1 

    if (NR == 2 && (run_option == 0 || run_option == 1)) {
        func = ((((waveformsTable[t] - waveformsTable[0]) * amp) / 2048)
                + ((((mu_one_back - used_zero_A0) + (mu - used_zero_A0)) * simu_k) / (simu_k_unit * 2))
                + ((dmudt * simu_b) / (simu_b_unit)))
               + 2047;
    } else if (NR == 2 && run_option == 2) {
        func = DC_func 
               + ((((mu_one_back - used_zero_A0) + (mu - used_zero_A0)) * simu_k) / (simu_k_unit * 2))
               + ((dmudt * simu_b) / (simu_b_unit));
    } //prioritise NR view over velocity view
    else if (Torv == 1 && NR != 1) {
        func = dmudt + 2047;
    }

    //send_func(); //send the DAC1 output value via SerialUSB to the PC
    if ((pos >= 1558 && pos <= 3633)) {
        send_mu(); // send mu via serialUSB ???? why after all of the stuff above? commented out for now in favour of println
        //SerialUSB.println(mu);	// is this too slow? why was this not implemented?
    }

    //cycle_counter = 1 - cycle_counter;
    //digitalWrite(42,cycle_counter);

    handle_const_strain_feedback();
    handle_pc_input();

    t++; // counts number of completed computing cycles
    if (t == sample_num) {
        t = 0; // Reset the counter to repeat the wave
    }

    //bitSet(REG_PIOB_CODR,26); what does this do?
}
