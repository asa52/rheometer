// Feedback functions.

void handle_const_strain_feedback() {
// what does this function do?

    if ((pos <= 1558 || pos >= 3633) && amp >= 1) {
        if (0 == run_option) {
            amp -= 1; // If outside range, decrease oscillation amplitude
        }
        //send_out_of_bounds_values(); CHANGE BACK
        if (freq_check == 1) {// WHAT IS THIS
            t_diff++; // also count time of out of bounds events
        }
    } else {//within range
        if (freq_check == 1) {
            if ((pos + 16) >= pos_rec && (pos - 16) <= pos_rec) { //points are sparse, so a strict recurrence condition can miss many cycles
                if (val >= 175000) { //at freq < 2Hz shorter periods are presumably the 2Hz fundamental
                    if (t_diff > 6) {
                        if ((f_count % 2) == 0 && f_count > 1 && (t_diff + rec_times[f_count]) < 15) {
                            f_count -= 2;
                        } else if ((f_count % 2) == 0) { //2nd recurrence of a value implies completed period
                            A0_period_estimates[A0_period_count] = t_diff + rec_times[f_count];
                            pest_check = (120 - A0_period_estimates[A0_period_count]);
                            pest_num = A0_period_count;
                            A0_period_count++;
                        }

                        rec_times[f_count] = t_diff;
                        f_count++;
                    }
                    t_diff = 0;

                } else if (val < 175000) { //at freq > 2Hz reject periods shorter than half a driving cycle
                    if (t_diff > 6) {
                        if ((f_count % 2) == 0 && f_count >= 1 && (t_diff + rec_times[f_count]) < 60) {
                            f_count -= 2;
                        } else if ((f_count % 2) == 0) { //2nd recurrence of a value implies completed period
                            A0_period_estimates[A0_period_count] = t_diff + rec_times[f_count];
                            pest_check = (120 - A0_period_estimates[A0_period_count]);
                            pest_num = A0_period_count;
                            A0_period_count++;
                        }

                        rec_times[f_count] = t_diff;
                        f_count++;
                    }
                    t_diff = 0;
                }

                if (f_count == 32) {
                    int two_rec_mean = 0;
                    for (int i = 0; i < f_count; i++) {
                        two_rec_mean += rec_times[i];}
                    A0_period_estimate_mean = (two_rec_mean / 16);
                    pest_check = (120 - A0_period_estimates[A0_period_count]);
                    f_count = 0;
                    freq_check = 2;
                }
                if (A0_period_count == 16) {
                    A0_period_count = 0;
                }
            } else if (t_diff > 1600) { //handle time overflow for non-recurring pos values
                freq_check = 2;//possibly guessed range doesn't contain value, so take centre instead
                t_diff = 0;
            } else {
                t_diff++;
            }
        }/* else if((pos >= 2200 && pos <= 2900) && 0 == freq_check){
          freq_check = 1;
          pos_rec = pos;
      }*/

        if (centre_estimated == 1) {
            switch (p) {
                case 0 :
                    //send_3_byte_value(amp, 0b00000000);CHANGE BACK
                    break;
                case 3 :
                    //send_3_byte_value(sym_check, 0b00000001);CHANGE BACK
                    break;
                case 5 :
                    //send_3_byte_value(centre, 0b00000010);CHANGE BACK
                    break;
                case 7 :
                    //send_3_byte_value(peak_to_peak, 0b00000100);CHANGE BACK
                    break;
                case 9 :
                    //send_3_byte_value(A0_period_estimates[A0_period_count], 0b00001000);CHANGE BACK
                    break;
                case 11 :
                    if (phase_estimates[delta_num] < 0) {
                        //send_3_byte_value(phase_estimates[delta_num], 0b00010001);CHANGE BACK
                    } else {
                        //send_3_byte_value(phase_estimates[delta_num], 0b00010000);CHANGE BACK
                    }
                    //centre_estimated = 0;
                    break;
                case 125 :
                    //send_3_byte_value(amp, 0b00000000);CHANGE BACK
                    break;
                case 129 :
                    //send_3_byte_value(A0_period_estimates[A0_period_count], 0b00001000);CHANGE BACK
                    break;
                case 131 :
                    if (phase_estimates[delta_num] < 0) {
                        //send_3_byte_value(phase_estimates[delta_num], 0b00010001);CHANGE BACK
                    } else {
                        //send_3_byte_value(phase_estimates[delta_num], 0b00010000);CHANGE BACK
                    }
                    break;
                case 245 :
                    //send_3_byte_value(amp, 0b00000000);CHANGE BACK
                    break;
                case 249 :
                    //send_3_byte_value(A0_period_estimates[A0_period_count], 0b00001000);CHANGE BACK
                    break;
                case 251 :
                    if (phase_estimates[delta_num] < 0) {
                        //send_3_byte_value(phase_estimates[delta_num], 0b00010001);CHANGE BACK
                    } else {
                        //send_3_byte_value(phase_estimates[delta_num], 0b00010000);CHANGE BACK
                    }
                    break;
                case 365 :
                    //send_3_byte_value(amp, 0b00000000);CHANGE BACK
                    break;
                case 369 :
                    //send_3_byte_value(A0_period_estimates[A0_period_count], 0b00001000);CHANGE BACK
                    break;
                case 371 :
                    if (phase_estimates[delta_num] < 0) {
                        //send_3_byte_value(phase_estimates[delta_num], 0b00010001);CHANGE BACK
                    } else {
                        //send_3_byte_value(phase_estimates[delta_num], 0b00010000);CHANGE BACK
                    }
                    centre_estimated = 0;
                    break;
            }

        }

        if (p < range) {
            array[p] = mu;

            if (last_t > t) {
                dt = t + (120 - last_t);
            } else {
                dt = t - last_t;
            }

            if (p > 0 && range > p) { // time differential quotient (new - old) / dt
                dmudt = ((array[p] - array[p - 1]) / dt);
            } else if (p == 0) {
                dmudt = ((array[0] - array[range - 1]) / dt);
            }

            darraydt[p] = dmudt;

            if ((p % sample_num) == 1) { // reset estimates every full cycle, for p == 1, 121, 241 and 361
                if (e_num >= 0) {
                    peaksnt[e_num][0] = peak;
                    peaksnt[e_num][1] = t_peak;
                    peaksnt[e_num][2] = dpeakdt;
                    peaksnt[e_num][3] = t_dpeakdt;
                    troughsnt[e_num][0] = trough;
                    troughsnt[e_num][1] = t_trough;
                    troughsnt[e_num][2] = dtroughdt;
                    troughsnt[e_num][3] = t_dtroughdt;

                    A0_amp = ((peaksnt[feed_num][0] - troughsnt[feed_num][0]) / 2);
                    dA0dt_amp = ((peaksnt[feed_num][2] - troughsnt[feed_num][2]) / 2);
                    // positive phase_estimate means that the response leads the driving

                    if ((delta_num % 2) == 0) { // even numbered elements are estimates from peaks
                        phase_estimates[delta_num] = (30 - t_peak);
                        if (phase_estimates[delta_num] < -60) {
                            phase_estimates[delta_num] = sample_num + phase_estimates[delta_num];
                        } else if (phase_estimates[delta_num] > 60) {
                            phase_estimates[delta_num] = sample_num - phase_estimates[delta_num];
                        }

                        delta_num++;

                    } else { // odd numbered elements are estimates from troughs
                        phase_estimates[delta_num] = (90 - t_trough);
                        if (phase_estimates[delta_num] < -60) {
                            phase_estimates[delta_num] = sample_num + phase_estimates[delta_num];
                        } else if (phase_estimates[delta_num] > 60) {
                            phase_estimates[delta_num] = sample_num - phase_estimates[delta_num];
                        }

                        delta_num++;
                    }

                    if (delta_num == 16) {
                        delta_num = 0;
                    }

                    feed_num = e_num;
                }

                e_num++;

                if (e_num == 8) {
                    e_num = 0;
                }

                if (array[p] > array[p - 1]) { //start making new extremum estimates
                    peak = array[p];
                    trough = array[p - 1];
                    t_peak = t;
                    t_trough = last_t;
                }
                if (array[p] < array[p - 1]) {
                    peak = array[p - 1];
                    trough = array[p];
                    t_peak = last_t;
                    t_trough = t;
                }
                if (darraydt[p] > darraydt[p - 1]) {
                    dpeakdt = darraydt[p];
                    dtroughdt = darraydt[p - 1];
                    t_dpeakdt = t;
                    t_dtroughdt = last_t;
                }
                if (darraydt[p] < darraydt[p - 1]) {
                    dpeakdt = array[p - 1];
                    dtroughdt = darraydt[p];
                    t_dpeakdt = last_t;
                    t_dtroughdt = t;
                }

            } else if (p > 1 && (p % sample_num) != 1) { //else compare values within cycle
                if (peak < array[p]) {
                    peak = array[p];
                    t_peak = t;
                }
                if (trough > array[p]) {
                    trough = array[p];
                    t_trough = t;
                }
                if (dpeakdt < darraydt[p]) {
                    dpeakdt = darraydt[p];
                    t_dpeakdt = t;
                }
                if (dtroughdt > darraydt[p]) {
                    dtroughdt = darraydt[p];
                    t_dtroughdt = t;
                }

            }

            p++; // sends different things on different cycles to keep cycle time the same
        }

        if (p == range) { // once sampled over 4 cycles, extract information
            int mean = 0;
            for (int j = 0; j < p; j++) {
                mean += array[j];
            }
            centre = (mean / p);
            centre_estimated = 1;

            peak_to_peak = (peak - trough);
            upper_amplitude = (peak - centre);
            lower_amplitude = (centre - trough);

            sym_check = (upper_amplitude - lower_amplitude);

            p = 0;

            map_centre_back_to_pos(1038, 0, 1);
            freq_check = 1;

        }

        last_t = t;
    }

    if (run_option == 0) {
        if (waveformsTable[t] == 0x7ff && feed_num >= 0) {
            // assess whether peak and trough values lie symmetrically about the centre value
            // if not, response likely hasn't settled to a steady state, so just wait
            if (((4 * mu_tol) >= sym_check && ((-4) * mu_tol) <= sym_check) || 
                (pest_num <= 0 && pest_check <= 40 && pest_check >= (-40))) {
                //compare using last 1 cycle estimates
                if (strain_closing_in == 0) {
                    adaptive_step_calculation_for_const_strain();
                } else if (strain_closing_in == 1) {
                    settle_amp();
                }

                // increase amplitude, when it is lower than required
                if ((((peaksnt[feed_num][0] - troughsnt[feed_num][0]) / 2) <= set_strain - mu_tol) && (amp + amp_step) <= 2048){
                    amp += amp_step;
                    old_amp_step = amp_step;
                } // decrease amplitude, when it is higher than required
                else if ((((peaksnt[feed_num][0] - troughsnt[feed_num][0]) / 2) >= set_strain + mu_tol) && 
                         (amp - amp_step) >= 0) {
                    amp -= amp_step;
                    old_amp_step = amp_step;
                }
            }
        }
    }
}

void adaptive_step_calculation_for_const_strain() {
    //( (waveformsTable[t] * 512 * amp) / 1048576 )

    if (old_amp_step != 0 && feed_num >= 1) {
        fiddle = (128 / old_amp_step);
        if (old_amp_step > 64 && old_amp_step < 96) {
            fiddle = 2;
        } else if (old_amp_step > 96) {
            fiddle = 1;
        }

        signed int step_estimate;

        step_estimate = ((fiddle * old_amp_step * ((2 * set_strain) - (2 * (peaksnt[feed_num][0] - troughsnt[feed_num][0])) 
          + (peaksnt[feed_num - 1][0] - troughsnt[feed_num - 1][0]))) / (4 * set_strain));

        past_step_estimates[step_count] = step_estimate;

        if (amp <= 8 && step_estimate > 8) {
            amp_step = 8;
        }

        if (step_estimate < 0) {
            if (step_count > 0 && past_step_estimates[step_count - 1] > 0) {
                sign_change_count++;
            }
            step_estimate = (-step_estimate);
        } else if (step_count > 0 && past_step_estimates[step_count - 1] < 0) {
            sign_change_count++;
        }

        if (((amp + step_estimate) <= 2048) && ((amp - step_estimate) >= 0)) {
            amp_step = step_estimate;
        } else if (((amp + (step_estimate / 2)) <= 2048) && ((amp - (step_estimate / 2)) >= 0)) {
            amp_step = (step_estimate / 2);
        } else if (((amp + (step_estimate / 4)) <= 2048) && ((amp - (step_estimate / 4)) >= 0)) {
            amp_step = (step_estimate / 4);
        } else {
            amp_step = 1;
        }

    } else if (old_amp_step == 0 && feed_num <= 1) {
        if (((((peaksnt[feed_num][0] - troughsnt[feed_num][0]) / 2) <= set_strain - mu_tol) && amp < 2048) || 
          ((((peaksnt[feed_num][0] - troughsnt[feed_num][0]) / 2) >= set_strain + mu_tol) && amp > 0)) {
            amp_step = 1;
        }
    }

    if (sign_change_count > 0) {
        past_amp_steps[step_count] = amp_step;
        step_count++;
    }

    if (step_count == 8) {
        if (sign_change_count >= 3) {
            int d_amp_step = 0, d_amp = 0;
            for (int i = 0; i < step_count; i++) {
                d_amp_step += past_amp_steps[i];
                d_amp += past_amps[i];
                if (i >= 1) {
                    max_amp_step = max(past_amp_steps[i], past_amp_steps[i - 1]);
                    min_amp_step = min(past_amp_steps[i], past_amp_steps[i - 1]);
                }
            }

            mean_amp = (d_amp / step_count);
            mean_amp_step = (d_amp_step / step_count);
            int d_step_dev = 0;

            for (int i = 0; i < step_count; i++) {
                if (mean_amp_step >= past_amp_steps[i]) {
                    d_step_dev += (mean_amp_step - past_amp_steps[i]);
                } else if (mean_amp_step < past_amp_steps[i]) {
                    d_step_dev += (past_amp_steps[i] - mean_amp_step);
                }
            }

            amp_step_deviation = (d_step_dev / step_count);
            strain_closing_in = 1;
        }
        sign_change_count = 0;
        step_count = 0;
    }
}

void settle_amp() {
	 // attempt to have it regulate its amplitude?

    if (mean_tried == 0) {
        amp = mean_amp;
        mean_tried = 1;
    }

    if (set_strain < (2 * set_strain - (peaksnt[feed_num][0] - troughsnt[feed_num][0])) || 
      (-1) * set_strain > (2 * set_strain - (peaksnt[feed_num][0] - troughsnt[feed_num][0]))) {
        strain_closing_in = 0;
    }

    amp_step = 1;

}

