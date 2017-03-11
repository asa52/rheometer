// RK4 solution to the ODE for a spinning ring in a magnetic field
// d^2 theta/dt^2 = G / I - b / I * (d theta/dt) - k / I * theta

#include <cmath>
#include <ctime>
#include <iostream>
#include <chrono>
#define n_eqns 2

using namespace std;
 
// Initial conditions
double t = 0.;
double h = 0.1;
double G = 0.;
double I = 1.;
double b = 6.;
double k = 5.;
double y[n_eqns] = {10., 0.};
double tol = 1.e-8;

// Set up arrays
double dydx[n_eqns];
double c1[n_eqns];
double c2[n_eqns];
double c3[n_eqns];
double c4[n_eqns];
double c5[n_eqns];
double c6[n_eqns];
double s = 1;

// Timing code
class Timer
{
public:
    Timer(): beg_(clock_::now()) {}

    void reset() { 
    	beg_ = clock_::now(); 
    }

    double elapsed() const { 
        return chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef chrono::high_resolution_clock clock_;
    typedef chrono::duration<double, ratio<1> > second_;
    chrono::time_point<clock_> beg_;
};


void calc_G(double time){
	// Calculate the total torque (RHS of eqn) at time t.
	G = sin(t);
}

// Evaluate the derivatives: we work in the transformed variables
// y[0] = theta, y[1] = d(theta)/dt
double calc_coeffs(int i, double time_arg, double pos_arg[n_eqns]) {
	if (i == 0){
		return pos_arg[1]; 	
	} else if (i == 1){
		return G / I - (b / I * pos_arg[1]) - (k / I * pos_arg[0]);
	}
}

void rk4(double step_size, double time_arg, double pos_arg[n_eqns]){
	// Calculate one iteration of the Runge Kutta 4th order algorithm.
	// do c1 to c4 in sequence
	double y2[n_eqns];
	double y3[n_eqns];
	double y4[n_eqns];
	double c1[n_eqns];
	double c2[n_eqns];
	double c3[n_eqns];
	double c4[n_eqns];
	calc_G(t); 	//NOTE that calc_coeffs with terms like t + h/2 need a different value of G. 

	for (int i = 0; i < n_eqns; i++){
		c1[i] = step_size * calc_coeffs(i, time_arg, pos_arg);
		y2[i] = pos_arg[i] + c1[i]/2.;			
	}
	for (int i = 0; i < n_eqns; i++){
		c2[i] = step_size * calc_coeffs(i, time_arg + step_size/2., y2);
		y3[i] = pos_arg[i] + c2[i]/2.;	
	}
	for (int i = 0; i < n_eqns; i++){
		c3[i] = step_size * calc_coeffs(i, time_arg + step_size/2., y3);			
		y4[i] = pos_arg[i] + c3[i];
	}
	for (int i = 0; i < n_eqns; i++){
		c4[i] = step_size * calc_coeffs(i, time_arg + step_size, y4);
		pos_arg[i] = pos_arg[i] + c1[i]/6. + c2[i]/3. + c3[i]/3. + c4[i]/6.;
	}
	time_arg = time_arg + step_size;
}

void rkf45(){
	// Calculate one iteration of the RKF45 algorithm.
	// Do c1 to c6 in sequence. 
	double y2[n_eqns];
	double y3[n_eqns];
	double y4[n_eqns];
	double y5[n_eqns];
	double y6[n_eqns];
	double y_rk4[n_eqns];
	double y_rk5[n_eqns];
	double diff[n_eqns]; 	// difference between rk4 and rk5 approximations
	calc_G(t);	//NOTE that calc_coeffs with terms like t + h/2 need a different value of G. 

	for (int i = 0; i < n_eqns; i++){
		c1[i] = h * calc_coeffs(i, t, y);
		y2[i] = y[i] + c1[i]/4.;			
	}
	for (int i = 0; i < n_eqns; i++){
		c2[i] = h * calc_coeffs(i, t + h/4., y2);
		y3[i] = y[i] + 3.*c1[i]/32. + 9.*c2[i]/32.;	
	}
	for (int i = 0; i < n_eqns; i++){
		c3[i] = h * calc_coeffs(i, t + 3.*h/8., y3);			
		y4[i] = y[i] + 1932.*c1[i]/2197. - 7200.*c2[i]/2197. + 7296.*c3[i]/2197.;
	}
	for (int i = 0; i < n_eqns; i++){
		c4[i] = h * calc_coeffs(i, t + 12.*h/13., y4);
		y5[i] = y[i] + 439.*c1[i]/216. - 8.*c2[i] + 3680.*c3[i]/513. - 845.*c4[i]/4104.;	
	}
	for (int i = 0; i < n_eqns; i++){
		c5[i] = h * calc_coeffs(i, t + h, y5);
		y6[i] = y[i] - 8.*c1[i]/27. + 2.*c2[i] - 3544.*c3[i]/2565. + 1859.*c4[i]/4104. - 11.*c5[i]/40.;	
	}
	for (int i = 0; i < n_eqns; i++){
		c6[i] = h * calc_coeffs(i, t + h/2., y6);
		y_rk4[i] = y[i] + 25.*c1[i]/216. + 1408.*c3[i]/2565. + 2197.*c4[i]/4101. - c5[i]/5.;
		y_rk5[i] = y[i] + 16.*c1[i]/135. + 6656.*c3[i]/12825. + 28561.*c4[i]/56430. - 9*c5[i]/50. + 2.*c6[i]/55.;

		// Next step of iteration
		y[i] = y_rk4[i];
		diff[i] = y_rk5[i] - y_rk4[i];
	}
	// calculate the magnitude of diff for the new step size.
	double sq_mag = 0.;
	for (int i = 0; i < n_eqns; i++){
		sq_mag = sq_mag + pow(diff[i], 2);
	}
	double mag = sqrt(sq_mag);
	if (mag > tol){
		h = h/2.;
	} else if (mag < tol){
		h = 2. * h;
	}
	
	t = t + h;
}

int main(){
	Timer tmr;
	for (int num_run; num_run < 100000; num_run++){
		double runtime = tmr.elapsed();
		cout << t << '\t' << y[0] << '\t' << y[1] << '\t' << runtime << endl;
		rkf45();
		tmr.reset();
	}
	return 0;
}