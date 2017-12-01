# PySPH
PySPH is a lightweight Python smoothed-particle hydrodynamics code designed to simulate the evolution of dust in molecular clouds. While the code is still being fine-tuned to ensure a more accurate transition between molecular clouds and H II regions, the basic functionality (pressure-gradient acceleration, self-gravitation, ionization chemistry, and rudimentary radiative transfer for heating and cooling) is all in place.
##Use
The file code\_running.py contains everything necessary to run the code, including free parameters (such as absorption cross-sections) that one can alter to get a more or less accurate picture of H II regions. All core functionality is stored in navier\_stokes\_cleaned.py.
##Authors
* Dhruv Muley, UC Berkeley and Lawrence Berkeley National Lab (dmuley@berkeley.edu)
* Umut Can Oktem, UC Santa Barbara and Lawrence Berkeley National Lab (umbutcan@gmail.com)
All rights reserved.
