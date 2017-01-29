cmake_minimum_required(VERSION 3.6)
project(Code)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        .idea/.gitignore
        .idea/Code.iml
        .idea/codeStyleSettings.xml
        .idea/misc.xml
        .idea/modules.xml
        .idea/serialmonitor_settings.xml
        .idea/vcs.xml
        .idea/workspace.xml
        arduino-rheometer/arduino-rheometer.ino
        arduino-rheometer/Waveforms.h
        samples/rheometer_project_1_9/ready_to_print.pdf
        samples/rheometer_project_1_9/rheometer_project_1_9.ino
        samples/rheometer_project_1_9/rheometer_project_1_9.ino.arduino_due_x.bin
        samples/rheometer_project_1_9_w_comments/rheometer_project_1_9_w_comments.ino
        samples/rheometer_project_1_9.pde
        simulations/__init__.py
        simulations/frontend.py
        simulations/main.py
        tests/24-01-17/c=1,b=0,F=0,theta_0=pi_2,omega_0=0.png
        tests/24-01-17/c=1,b=0,F=3sinpi_2-t,theta_0=1,omega_0=0.png
        tests/24-01-17/c=1,b=1,F=0,theta_0=pi_2,omega_0=0.png
        tests/24-01-17/c=1,b=2,F=3sinpi_2-t,theta_0=1,omega_0=0,crit_damping.png
        tests/24-01-17/c=25,b=8,F=3sinpi_2-t,theta_0=1,omega_0=0,light_damping.png
        "tests/25-01-17/Simple High-Low Arduino output through DAC1.png"
        "tests/27-01-17/red-capacitor, blue-supply to amp 2.png"
        "tests/27-01-17/red-capacitor, blue-supply to amp 3.png"
        "tests/27-01-17/red-capacitor, blue-supply to amp 4.png"
        "tests/27-01-17/red-capacitor, blue-supply to amp 5 - const supply.png"
        "tests/27-01-17/red-capacitor, blue-supply to amp 6 - const supply.png"
        "tests/27-01-17/red-capacitor, blue-supply to amp.png"
        "tests/27-01-17/Supply power  to amp vs capacitor output 2.txt"
        "tests/27-01-17/Supply power  to amp vs capacitor output 3 - const supply 4V.txt"
        "tests/27-01-17/Supply power  to amp vs capacitor output.txt"
        .gitignore
        README.md)

add_executable(Code ${SOURCE_FILES})