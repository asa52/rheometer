# rheometer
Code to simulate the feedback system of a stress-controlled oscillatory
rheometer under normalised resonance.

Directory structure:
andreas_code: Work by a previous Part III student on writing Arduino code
for the rheometer.
rheometer: Arduino code, arranged into more readable chunks. The Processing
GUI is in a separate folder.
simulations: My work, containing config files and the Python experiments
conducted. Also contains the measure_feedback.so file, which is a direct
translation of Arduino code into C that can be integrated with Python.

