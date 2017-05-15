/*Main timer-counter handler function. The main function instead of loop.*/

void TC0_Handler() {
  // why does this function run: it is never explicitly called???? is this the main function?
  long dummy = REG_TC0_SR0; // vital - reading this clears some flag
  // otherwise you get infinite interrupts
  
  pos = analogRead(measure); //reads the voltage at analog pin A0

  if ((pos >= 1558 && pos <= 3633)) {
    int num = (pos - pos_0);
    mu = A0mu[num]; //0.1 microns
  }

  if (centre_mode == 1) {
    used_zero_A0 = equilibrium_A0; //set zero point to some value you set
  } else {
    used_zero_A0 = centre;
  }

  if (NR != 1 && (run_option == 0 || run_option == 1)) {
    // If we are not in normalised resonance mode and the run option is either one or two
    func = (((waveformsTable[t] - waveformsTable[0]) * amp) / 2048) + 2047; // func is sine wave added onto mid-value
  } else if (NR != 1 && run_option == 2) {
    // 2nd run option is constant output mode
    func = DC_func;
  } else if (NR == 1 && (run_option == 0 || run_option == 1)) {
    func = ((((waveformsTable[t] - waveformsTable[0]) * amp) / 2048)
            + (((mu - used_zero_A0) * simu_k) / (simu_k_unit))  // elastic response - CHECK
            + ((dmudt * simu_b) / (simu_b_unit)))  // viscous response - CHECK
            + 2047;  // midpoint
  } else if (NR == 1 && run_option == 2) {
    func = //DC_func +
      (((mu - used_zero_A0) * simu_k) / (simu_k_unit))
      + ((dmudt * simu_b) / (simu_b_unit));
  }
  if (func > 4095) {
    // if function too large, clip
    func = 4095;
  }
  if (func < 0) {
    // if too small, clip
    func = 0;
  }
  //REG_DACC_CDR = func; // analog write to DAC1
  analogWrite(DAC0, func); //change to lower level code
  SerialUSB.println("Dataset-t-timeElapsed-func");
  SerialUSB.println(t);
  SerialUSB.println(timeElapsed);
  SerialUSB.println(func);
  //SerialUSB.println(simu_k);
  //SerialUSB.println(simu_k_unit);
  //SerialUSB.println(simu_b);
  //SerialUSB.println(simu_b_unit);

  if (NR == 2 && (run_option == 0 || run_option == 1)) {
    func = ((((waveformsTable[t] - waveformsTable[0]) * amp) / 2048)
            + ((((mu_one_back - used_zero_A0) + (mu - used_zero_A0)) * simu_k) / (simu_k_unit * 2))
            + ((dmudt * simu_b) / (simu_b_unit)))
           + 2047;
  } else if (NR == 2 && run_option == 2) {
    func = //DC_func
      + ((((mu_one_back - used_zero_A0) + (mu - used_zero_A0)) * simu_k) / (simu_k_unit * 2))
      + ((dmudt * simu_b) / (simu_b_unit));
  } //prioritise NR view over velocity view
  else if (Torv == 1 && NR != 1) {
    func = dmudt + 2047;
  }

  //if ((pos >= 1558 && pos <= 3633)) {
  //send_mu(); // send mu via serialUSB to pc???? why after all of the stuff above? commented out for now in favour of println
  //}

  //handle_const_strain_feedback();
  //handle_pc_input();

  t++; // counts number of completed computing cycles
  if (t == sample_num) {
    t = 0; // Reset the counter to repeat the wave
  }
}

