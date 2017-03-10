// GSL solution to the ODE for a spinning ring in a magnetic field
// d^2 theta/dt^2 = G / I - b / I * (d theta/dt) - k / I * theta

#include <iostream>
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv.h>

// Evaluate the derivatives: we work in the transformed variables
// y[0] = theta, y[1] = d(theta)/dt
int calc_derivs(double t, const double y[], double dydx[], void *params) {
	// Extract the parameters from the pointer param*. In this case
	// there are 4: G, I, b, k.
	double G = *(double *)(params[0]);
	double I = *(double *)(params[1]);
	double b = *(double *)(params[2]);
	double k = *(double *)(params[3]);

	dydx[0] = y[1];
	dydx[1] = (G / I) - (b / I * y[1]) - (k / I * y[0]);
	return GSL_SUCCESS;
}

int main() {
	// Initial conditions:
	double G = 1.0;
	double I = 1.0;
	double b = 1.0;
	double k = 1.0;
	double parameters[4] = {G, I, b, k};

	const int n_equations = 2;
	double y[n_equations] = {0, 10};
	double t = 0.0;

	// Create a stepping function:
	gsl_odeiv_step *gsl_step = gsl_odeiv_step_alloc(gsl_odeiv_step_rk4, n_equations);

	// Adaptive step control: let's use fixed steps here:
	gsl_odeiv_control *gsl_control = NULL;

	// Create an evolution function:
	gsl_odeiv_evolve *gsl_evolve = gsl_odeiv_evolve_alloc(n_equations);

	// Set up the system needed by GSL: The 4th arg is a pointer
	// to any parameters needed by the evaluator. The 2nd arg
	// points to the jacobian function if needed (it's not needed here).
	gsl_odeiv_system gsl_sys = {calc_derivs, NULL, n_equations, &parameters};
	double t_max = 20.0;
	double h = 1e-3;

	// Main loop: advance solution until t_max reached.
	while (t < t_max) {
		std::cout << t << " " << y[0] << " " << y[1] << "\n";
		int status = gsl_odeiv_evolve_apply(gsl_evolve, gsl_control, gsl_step, &gsl_sys, &t, t_max, &h, y);
		if (status != GSL_SUCCESS) break;
	}
	
	// Tidy up the GSL objects for neatness:
	gsl_odeiv_evolve_free(gsl_evolve);
	gsl_odeiv_step_free(gsl_step);
	return 0;
}