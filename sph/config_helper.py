import numpy as np
import random
from scipy import constants
import matplotlib.pyplot as plt
import itertools
import copy
from time import sleep
import navier_stokes_cleaned as nsc
import os
from scipy.interpolate import interp2d, RectBivariateSpline
import numpy as np
from scipy import interpolate
import itertools
from mpl_toolkits.mplot3d import Axes3D

amu = constants.physical_constants['atomic mass constant'][0]
solar_mass = 1.989e30 #kilograms
solar_luminosity = 3.846e26 #watts
solar_lifespan = 1e10 #years
year = 60. * 60. * 24. * 365.
dt_0 = year * 250000.

#properties for species in each SPH particle, (H2, He, H,H+,He+,e-,Mg2SiO4,SiO2,C,Si,Fe,MgSiO3,FeSiO3)in that order
species_labels = np.array(['H2', 'He', 'H','H+','He+','e-','Mg2SiO4','SiO2','C','Si','Fe','MgSiO3','FeSiO3', 'SiC'])
mu_specie = np.array([2.0159,4.0026,1.0079,1.0074,4.0021,0.0005,140.69,60.08,12.0107,28.0855,55.834,100.39,131.93, 40.096])
destruction_energies = np.array([7.2418e-19, 3.93938891e-18, 2.18e-18, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000])
clean_gas_composition = np.array([ 0.86,  0.14,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

TIMESTEP = 3e7 * year
MOLECULAR_CLOUD_DURATION = 3e7 * year
LOC_FRAC_CONSUMED = 0.005 * (TIMESTEP/MOLECULAR_CLOUD_DURATION)
FRAC_CONSUMED = (1. - LOC_FRAC_CONSUMED)**np.arange(10000) * LOC_FRAC_CONSUMED #probably way more timesteps than necessary
#fraction of galactic mass consumed by each cycle of GMCs, used to compute dilution of amounts calculated
#galactic gas is renewable, but we use a decreasing exponential because the SFR reduces anyway

'''
#### PURPOSE OF THIS CODE ####
 - Read in output files generated by each instance of code_running.py
 - Take the results and produce a new config file
 - Compute dust production from AGBs and use the derivative to compute dust added to new simulation
 - Output .npz files containing all the information needed (dust composition/fraction, gas composition, etc.)
 - Calculate cumulative change, divide by (MOLECULAR_CLOUD_DURATION/TIMESTEP) (longer timestep means more effect
   from each one).
   
#### SOME WORDS OF WARNING ####
 - Make sure that a 'config_0.npz' file already exists (create this manually or the code won't work!
 - Make sure that code forms a closed loop---i. e. code_running writes to small output files, and these
   small output files contain all the information asked for here, which in turn create config files, which
   in turn determine completely (up to randomness) the working of code_running.py. Code should be totally
   hands-off by the time it is run on supercomputer.
'''
 
############ LOAD CONFIG FILE INFORMATION #############
absolute_path_to_nsc = os.path.dirname(os.path.abspath(nsc.__file__))
absolute_path_to_outputs = absolute_path_to_nsc + '/../savefiles/outputs'
absolute_path_to_config  = absolute_path_to_nsc + '/../savefiles/config'

config_files = os.listdir(absolute_path_to_config)
config_files.append('config_-1.npz')
conf_number = np.array([am[7:-4] for am in config_files]).astype('int')
max_conf_file = (conf_number == max(conf_number))

TIMESTEP_NUMBER = max(conf_number)

conf_filename_selected = np.array(config_files)[max_conf_file][0]
latest_file = np.load(absolute_path_to_config + '/' + conf_filename_selected)
#obtains DUST_FRAC, specie_fraction_array, and dust_base_frac from loaded output file
specie_fraction_array = latest_file['specie_fraction_array']
dust_base_frac = latest_file['dust_base_frac']
DUST_FRAC = latest_file['DUST_FRAC'] 
OVERALL_AGE = latest_file['OVERALL_AGE']

overall_AGB_list = latest_file['overall_AGB_list']
overall_AGB_time_until = latest_file['overall_AGB_time_until']
overall_AGB_metallicity = latest_file['overall_AGB_metallicity']
overall_AGB_composition = latest_file['overall_AGB_composition']

#Create array of AGB stars, loading in each file one at a time. This is a bit slow, but is most accurate.
#append to the end of this and compute later amount created
timestep_AGB_list = np.array([])
timestep_AGB_time_until = np.array([])
timestep_AGB_metallicity = np.array([])
timestep_AGB_composition = np.array([clean_gas_composition])

timestep_gas_mass_by_species = copy.deepcopy(destruction_energies) * 0
timestep_star_mass_by_species = copy.deepcopy(destruction_energies) * 0
timestep_dust_mass_by_species = copy.deepcopy(destruction_energies) * 0

for savefile_name in os.listdir(absolute_path_to_outputs):
	array_file = np.load(unicode(absolute_path_to_outputs + '/' + savefile_name));
	AGB_list = array_file['AGB_list']
	AGB_time_until = array_file['AGB_time_until']
	AGB_metallicity = array_file['AGB_metallicity']
	AGB_comp = array_file['AGB_composition']
	#print AGB_comp
	
	timestep_AGB_list = np.append(timestep_AGB_list, AGB_list)
	timestep_AGB_time_until = np.append(timestep_AGB_time_until, AGB_time_until)
	timestep_AGB_metallicity = np.append(timestep_AGB_metallicity, AGB_metallicity)
	timestep_AGB_composition = np.append(timestep_AGB_composition, AGB_comp, axis=0)
		
	timestep_gas_mass_by_species += array_file['gas_mass_by_species']
	timestep_star_mass_by_species += array_file['star_mass_by_species']
	timestep_dust_mass_by_species += array_file['dust_mass_by_species']
	
	array_file.close()

timestep_AGB_composition = timestep_AGB_composition[1:] #removing dummy element
#append these to overall_lists, found in latest file in output folder

def interpolate_amounts(absolute_path_to_nsc):
	#assume equal fractions of C, S, and M stars, or rather, assume each star is 1/3 each
	#AGB star types, in order:(C                                                         )(S                 )(M          )
	#AGB labels, in order:    (forsterite, fayalite, enstatite, ferrosilite, quartz, iron, quartz, iron, SiC, carbon, iron)
	#Corresponding formulae:  (Mg2SiO4,    Mg2SiO4,  MgSiO3,    FeSiO3,      SiO2,   Fe,   SiO2,   Fe,   SiC, C,      Fe  )
	#AGB indices, in order:   (0,          1,        2,         3,           4,      5,    6,      7,    8,   9,      10  )
	#Corresponding indices:   (6,          6,        11,        12,          7,      10,   7,      13,   8,   10,     13  )
	
	#return interpolation function which is then evaluated
	
	AGB_masses = np.array([1,1.1,1.2,1.25,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,3,3.5,4,4.01,4.5,5.0,5.5,6.0,6.5,7])
	metallicity = np.array([0.001,0.002,0.004,0.008,0.015,0.02,0.03,0.04])
	#in solar masses
	absolute_path_to_AGB = absolute_path_to_nsc + '/../agb_interp'
	AGB_files = np.array(os.listdir(absolute_path_to_AGB))
	by_metallicity = []
	for agbdata in AGB_files[AGB_files != '.DS_Store']:
		array_file = np.genfromtxt(unicode(absolute_path_to_AGB + '/' + agbdata));
		#each file represents a different metallicity.
		#stack these up on one another into a rank 3 tensor, then take slices by the desired species
		#axes in order (metallicity, mass, species)
		species = array_file.T[1:]
		by_metallicity.append(species)
			
		#array_file.close()
		print agbdata
		#we want: (species, metallicity, mass)
	
	species_segregated = np.swapaxes(by_metallicity, 0, 1)
	species_segregated[species_segregated <= 0.] = 1e-30
	mapto = np.array([6, 6, 11, 12, 7, 10, 7, 13, 8, 10, 13])
	AGB_divisor = 3 #assume each star is equal parts C, S, and M in the absence of other data, can modify later if needed
	
	#input_meshgrid = np.meshgrid(metallicity, AGB_masses)
	splines = []
	
	for item in range(len(mapto)):
		new_grid = RectBivariateSpline(metallicity,AGB_masses,species_segregated[item], kx=1, ky=1, s=0.9)
		splines.append(new_grid)
		
	return splines, mapto, AGB_divisor
	
def calculate_interpolation(AGB_masses, AGB_metallicities, splines, mapto, AGB_divisor, mu_specie, AGB_composition):
	dust_mass_created = np.array([mu_specie] * len(AGB_masses)) * 0.
	mass_WD = (0.55 + (AGB_masses/solar_mass - 1.) * 0.45/(7. - 1.)) * solar_mass #based on Dominguez et al. 1999
	
	for pos in range(len(AGB_masses)):
		mass_created[pos][mapto] = np.array([float(splines[obj](AGB_metallicities[pos], AGB_masses[pos])) for obj in range(len(splines))])
		print pos
		
	dust_mass_created /= AGB_divisor
	dust_mass_created[mass_created < 0.] = 0.
	
	gas_mass_created = AGB_masses - np.sum(dust_mass_created, axis=1) - mass_WD
	#move 10% of hydrogen to helium---fusion happens!
	
	AGB_gas_comp_number = (copy.deepcopy(AGB_composition)/(mu_specie)).T
	
	ionized_amount = copy.deepcopy(AGB_gas_comp_number[3] * 0.1)
	neutral_H2 = copy.deepcopy(AGB_gas_comp_number[0] * 0.1)
	neutral_H = copy.deepcopy(AGB_gas_comp_number[2] * 0.1)
	
	AGB_gas_comp_number[0] -= neutral_H2
	AGB_gas_comp_number[1] += neutral_H/4. + neutral_H2/2. + ionized_amount/4.
	AGB_gas_comp_number[2] -= neutral_H
	AGB_gas_comp_number[3] -= ionized_amount
	AGB_gas_comp_number[5] -= ionized_amount
	
	gas_mass_composition = (AGB_gas_comp_number.T * mu_specie)/np.sum(AGB_gas_comp_number.T * mu_specie, axis=1)
	
	return mass_created, (gas_mass_composition.T * gas_mass_created).T

overall_AGB_list = np.append(overall_AGB_list, timestep_AGB_list)
overall_AGB_time_until = np.append(overall_AGB_time_until, timestep_AGB_time_until)
overall_AGB_metallicity = np.append(overall_AGB_metallicity, timestep_AGB_metallicity)
overall_AGB_composition = np.append(overall_AGB_composition, timestep_AGB_composition)

#calculate how much dust is produced by AGBs during the NEXT timestep, and remove all AGBs that have passed already
#compute dust production ONLY for those stars whose time has come, don't needlessly store for others
overall_AGB_list = overall_AGB_list[overall_AGB_time_until >= OVERALL_AGE]
overall_AGB_metallicity = overall_AGB_metallicity[overall_AGB_time_until >= OVERALL_AGE]
#overall_AGB_dust_prod = overall_AGB_dust_prod[overall_AGB_time_until >= OVERALL_AGE]
overall_AGB_time_until = overall_AGB_time_until[overall_AGB_time_until >= OVERALL_AGE]

splines, mapto, AGB_divisor = interpolate_amounts(absolute_path_to_nsc)
if len(overall_AGB_list[overall_AGB_time_until <= OVERALL_AGE + TIMESTEP]) > 0:
	overall_AGB_dust_prod, overall_AGB_gas_prod = calculate_interpolation(overall_AGB_list[overall_AGB_time_until <= OVERALL_AGE + TIMESTEP] , overall_AGB_metallicity[overall_AGB_time_until <= OVERALL_AGE + TIMESTEP], splines, mapto, AGB_divisor, mu_specie, overall_AGB_composition[overall_AGB_time_until <= OVERALL_AGE + TIMESTEP])
else:
	overall_AGB_dust_prod = np.vstack([mu_specie * 0] * 2)
	overall_AGB_gas_prod = np.vstack([mu_specie * 0 * 2])

timestep_AGB_dust_mass_by_species = np.sum(overall_AGB_dust_prod, axis=0)
timestep_AGB_gas_mass_by_species = np.sum(overall_AGB_gas_prod, axis=0)

timestep_mass_composition = timestep_dust_mass_by_species + timestep_gas_mass_by_species + timestep_AGB_dust_mass_by_species + timestep_AGB_gas_mass_by_species
timestep_dust_by_mass_composition = timestep_AGB_dust_mass_by_species + timestep_dust_mass_by_species
timestep_gas_by_mass_composition = timestep_AGB_gas_mass_by_species + timestep_gas_mass_by_species
timestep_dust_frac = np.sum(timestep_dust_by_mass_composition)/np.sum(timestep_mass_composition)

##################obtaining final composition of dust#################
overall_dust_mass_composition_initial = dust_base_frac * mu_specie/(np.sum(dust_base_frac * mu_specie))
timestep_dust_mass_composition_initial = timestep_dust_by_mass_composition/np.sum(timestep_dust_by_mass_composition)

overall_dust_mass_composition = DUST_FRAC * overall_dust_mass_composition_initial * (1 - FRAC_CONSUMED[TIMESTEP_NUMBER]) + timestep_dust_frac * timestep_dust_mass_composition_initial * FRAC_CONSUMED[TIMESTEP_NUMBER]
overall_dust_mass_composition /= np.sum(overall_dust_mass_composition)

overall_dust_number_composition = (overall_dust_mass_composition/mu_specie)/np.sum((overall_dust_mass_composition/mu_specie))
new_dustfrac = DUST_FRAC * (1 - LOC_FRAC_CONSUMED) + timestep_dust_frac * FRAC_CONSUMED[TIMESTEP_NUMBER]

###################obtaining final composition of gas#################
overall_gas_mass_composition_initial = specie_fraction_array * mu_specie/np.sum(specie_fraction_array * mu_specie)
timestep_gas_mass_composition_initial = timestep_gas_by_mass_composition/np.sum(timestep_gas_by_mass_composition)

overall_gas_mass_composition = (1 - DUST_FRAC) * overall_gas_mass_composition_initial * (1 - FRAC_CONSUMED[TIMESTEP_NUMBER]) + (1 - timestep_dust_frac) * timestep_gas_mass_composition_initial * FRAC_CONSUMED[TIMESTEP_NUMBER]
overall_gas_mass_composition /= np.sum(overall_gas_mass_composition)

overall_gas_number_composition = (overall_gas_mass_composition/mu_specie)

### TIME TO NEUTRALIZE NUMBER COMPOSITION ###
### IONS SHOULD NOT PERSIST ###

overall_gas_number_composition[2] += overall_gas_number_composition[3]
overall_gas_number_composition[1] += overall_gas_number_composition[4]
overall_gas_number_composition[3:6] *= 0

overall_gas_number_composition /= np.sum(overall_gas_number_composition)

overall_AGB_composition_2 = copy.deepcopy(overall_AGB_composition).T
overall_AGB_composition_2[2] += overall_AGB_composition_2[3]
overall_AGB_composition_2[1] += overall_AGB_composition_2[4]
overall_AGB_composition_2[3:6] *= 0

overall_AGB_composition = (overall_AGB_composition_2/np.sum(overall_AGB_composition_2, axis=0)).T
#saving new config files

#now just add these to output file and increment timestep number by 1
#make sure to start from config_0.npz

#assume changes in dust *during* each timestep are small, only total amounts matter. This is valid
#because only 0.5% of the galaxy's mass is subject to "processing" at any given time.

#increment OVERALL_AGE by TIMESTEP and write everything to the next output file
np.savez(unicode(absolute_path_to_config + '/config_' + str(int(TIMESTEP_NUMBER + 1))), specie_fraction_array = overall_gas_number_composition, dust_base_frac = overall_dust_number_composition, DUST_FRAC = new_dustfrac, OVERALL_AGE = OVERALL_AGE + TIMESTEP, overall_AGB_list = overall_AGB_list, overall_AGB_time_until = overall_AGB_time_until, overall_AGB_metallicity = overall_AGB_metallicity, overall_AGB_composition = overall_AGB_composition)

#LIST OF VARIABLES IN config_## file:
#specie_fraction_array = overall_gas_number_composition
#dust_base_frac = overall_dust_number_composition
#DUST_FRAC = new_dustfrac
#OVERALL_AGE = OVERALL_AGE + TIMESTEP
#overall_AGB_list = overall_AGB_list
#overall_AGB_time_until = overall_AGB_time_until
#overall_AGB_metallicity = overall_AGB_metallicity
#overall_AGB_composition = overall_AGB_composition
##These need to be bootstrapped initially
#This is it! The loop is now closed.
