#PySPH

PySPH is a lightweight Python smoothed-particle hydrodynamics code designed to simulate the evolution of dust in molecular clouds. While the code is still being fine-tuned to ensure a more accurate transition between molecular clouds and H II regions, the basic functionality (pressure-gradient acceleration, self-gravitation, ionization chemistry, and rudimentary radiative transfer for heating and cooling) is all in place.
## Use

The file code\_running.py contains everything necessary to run the code, including free parameters (such as absorption cross-sections) that one can alter to get a more or less accurate picture of H II regions. All core functionality is stored in navier\_stokes\_cleaned.py. This will suffice for individual runs, but to generalize the results to galactic evolution the following procedure is required:
* Run the config\_bootstrap.py file to create an initial config file that sets a "base galaxy" with (nearly) zero metallicity.
* Run code\_running.py as many times as desired to produce accurate average results for dust production, AGB production, et cetera.
* Run config\_helper.py to resolve the output files from all instances of code\_running.py into a new config file.
* (Important) delete raw output files produced by code\_running.py.
* Repeat the loop as many times as desired to obtain evolution of dust abundances across cosmological time.
## Authors

* Dhruv Muley, UC Berkeley and Lawrence Berkeley National Lab (dmuley@berkeley.edu)
* Umut Can Oktem, UC Santa Barbara and Lawrence Berkeley National Lab (umutoktem@umail.ucsb.edu)

All rights reserved.
