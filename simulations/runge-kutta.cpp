// RK4 solution to the ODE for a spinning ring in a magnetic field
// d^2 theta/dt^2 = G / I - b / I * (d theta/dt) - k / I * theta

#include <cmath>
#include <iostream>
#include <chrono>
#define n_eqns 2

using namespace std;
 
// Initial conditions
double t = 0.;
double h = 0.0001;
double G = 0.;
double I = 1.;
double b = 6.;
double k = 5.;
double y[n_eqns] = {10., 0.};
double tol = 1.e-8;


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


void rk4(){
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
		c1[i] = h * calc_coeffs(i, t, y);
		y2[i] = y[i] + c1[i]/2.;			
	}
	for (int i = 0; i < n_eqns; i++){
		c2[i] = h * calc_coeffs(i, t + h/2., y2);
		y3[i] = y[i] + c2[i]/2.;	
	}
	for (int i = 0; i < n_eqns; i++){
		c3[i] = h * calc_coeffs(i, t + h/2., y3);			
		y4[i] = y[i] + c3[i];
	}
	for (int i = 0; i < n_eqns; i++){
		c4[i] = h * calc_coeffs(i, t + h, y4);
		y[i] = y[i] + c1[i]/6. + c2[i]/3. + c3[i]/3. + c4[i]/6.;
	}
	t = t + h;
}


int main(){
	Timer tmr;
	for (int num_run; num_run < 100000; num_run++){
		double runtime = tmr.elapsed();
		cout << t << '\t' << y[0] << '\t' << y[1] << '\t' << runtime << endl;
		rk4();
		tmr.reset();
	}
	return 0;
}
