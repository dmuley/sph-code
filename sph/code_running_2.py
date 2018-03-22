import numpy as np
import random
from scipy import constants
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial, stats
import copy
from time import sleep
import navier_stokes_cleaned as nsc
import os
import time
'''
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(seed=int(time.time() * rank) % 4294967294)'''

#These constants should be exclusively PHYSICAL---they should be independent
#of any properties of a particular SPH simulation
G = constants.G
k = constants.Boltzmann
sb = constants.Stefan_Boltzmann
AU = constants.au
amu = constants.physical_constants['atomic mass constant'][0]
h = constants.h
c = constants.c
wien = constants.Wien
m_h = 1.0008*amu#mass of hydrogen
solar_mass = 1.989e30 #kilograms
solar_luminosity = 3.846e26 #watts
solar_lifespan = 1e10 #years
t_cmb = 2.732
t_solar = 5776
supernova_energy = 1e44 #in Joules
m_0 = 10**1.5 * solar_mass #solar masses, maximum mass in the kroupa IMF
year = 60. * 60. * 24. * 365.
dt_0 = year * 250000.
#properties for species in each SPH particle, (H2, He, H,H+,He+,e-,Mg2SiO4,SiO2,C,Si,Fe,MgSiO3,FeSiO3)in that order
species_labels = np.array(['H2', 'He', 'H','H+','He+', 'He++','e-','Mg2SiO4','SiO2','C','Si','Fe','MgSiO3','FeSiO3', 'SiC'])
mu_specie = np.array([2.0158,4.0026,1.0079,1.0074,4.0021, 4.0016,0.0005,140.69,60.08,12.0107,28.0855,55.834,100.39,131.93, 40.096])
cross_sections = np.array([6.3e-22, 6.3e-22, 6.3e-22, 1e-60, 1e-60, 1.e-60,6.65e-60 * 80, 0., 0., 0., 0., 0., 0., 0., 0.])/80. + 1e-80
destruction_energies = np.array([7.2418e-19, 3.93938891e-18, 2.18e-18, 10000, 10000, 8.71584082e-18, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000])
mineral_densities = np.array([1.e19, 1e19,1e19,1e19,1e19,1e19,1e19, 3320,2260,2266,2329,7870,3250,3250., 3166.])
sputtering_yields = np.array([0,0,0,0,0,0,0,0.137,0.295,0.137,0.295,0.137,0.137,0.137, 0.137])
f_u = np.array([.86,.14,0,0,0,0,0,0,0,0,0,0,0,0,0]) #relative abundance for species in each SPH particle, an array of arrays
gamma = np.array([7./5,5./3,5./3,5./3,5./3,5./3,5./3,15.6354113,4.913,1.0125,2.364,3.02,10.,10.,10.])#the polytropes of species in each SPH, an array of arrays
W6_constant = (3 * np.pi/80)
critical_density = 1000*amu*10**6 #critical density of star formation
crit_mass = 0.0001 * solar_mass #setting a minimum dust mass to help avoid numerical errors!

W6_constant = (3 * np.pi/80)
mrn_constants = np.array([50e-10, 5000e-10]) #minimum and maximum radii for MRN distribution
cross_sections += nsc.sigma_effective(mineral_densities, mrn_constants, mu_specie)
raise_factor = 715 #8 million keeps the array non jagged, so maintain this number!
#### AND NOW THE FUN BEGINS! THIS IS WHERE THE SIMULATION RUNS HAPPEN. ####
#SETTING VALUES OF BASIC SIMULATION PARAMETERS HERE (TO REPLACE DUMMY VALUES AT BEGINNING)
DIAMETER = 1.25e6 * AU

N_PARTICLES = 3000 * raise_factor
N_INT_PER_PARTICLE = N_PARTICLES/raise_factor
V = (DIAMETER)**3
d = (V/N_PARTICLES * N_INT_PER_PARTICLE)**(1./3.) * raise_factor**(1./3.)
nsc.d = d
d_sq = d**2
d_0 = 1e5 * AU
nsc.d_0 = 1e5 * AU

dt = dt_0
nsc.dt = dt
nsc.dt_0 = dt_0
DUST_MASS = 0.05000000001/raise_factor #mass of each dust SPH particle
N_RADIATIVE = 1 #number of timesteps for radiative transfer, deprecated
MAX_AGE = 3e7 * year #don't want to see any AGB stars undergoing supernovae inadvertently
crit_mass = 0.0001 * solar_mass/raise_factor #setting a minimum dust mass to help avoid numerical errors!
critical_density = 1000 * amu * 10**6 #critical density of star formation
N_NEIGH = 40

specie_fraction_array = np.array([.86,.14,0,0,0,0,0,0,0,0,0,0,0,0,0])
dust_base_frac = np.array([.0,.0,0.,0.,0.,0.,0.,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025])
DUST_FRAC = 0.000
OVERALL_AGE = 0. #age of the galaxy
'''LOAD BASIC PROPERTIES FROM SOME SORT OF FILE, TO INITIALIZE THIS IN SOME PARAMETER SPACE. BASE ALL OF THIS
ON THE MPI NUMBER. IF SOMETHING EXISTS AT A PARTICULAR MPI NUMBER SKIP TO LOADING THE BELOW VALUES FROM THAT FILE
AND RUNNING THE SIMULATION; IF NOT, CONTINUE WITH THE INITIALIZATION.'''

############################
#END LOADING FROM FILE HERE#
############################

dust_base = dust_base_frac/np.sum(dust_base_frac)
base_imf = np.array([0.4]) #np.logspace(np.log10(0.10),np.log10(40.), 200)
d_base_imf = np.array([1.]) #np.append(np.diff(base_imf)[0], np.diff(base_imf))
imf = base_imf * d_base_imf #nsc.kroupa_imf(base_imf) * d_base_imf
imf /= np.sum(imf)

#np.random.seed(rank*time.time())
mass_ind = np.random.choice(np.arange(len(base_imf)), N_PARTICLES, p = imf)
mass = base_imf[mass_ind]
diff_mass = 0. #np.random.rand(len(mass)) * d_base_imf[mass_ind]
mass = (mass + diff_mass) * solar_mass / raise_factor

N_DUST = max(int(DUST_FRAC/(1. - DUST_FRAC) * np.sum(mass)/(DUST_MASS * solar_mass)), 1)
particle_type = np.zeros([N_PARTICLES]) #0 for gas, 1 for stars, 2 for dust
N_PARTICLES += N_DUST
particle_type = np.append(particle_type, [2] * N_DUST)
mass = np.append(mass, [DUST_MASS * solar_mass] * N_DUST)

points = (np.random.rand(N_PARTICLES, 3) - 0.5) * DIAMETER
points2 = np.array(list(points))

#print(nbrs)
#print(points)
velocities = np.random.normal(size=(N_PARTICLES, 3)) * 0.
total_accel = np.random.rand(N_PARTICLES, 3) * 0.
sizes = (mass/m_0)**(1./3.) * d

mu_array = np.zeros([N_PARTICLES])#array of all mu
E_internal = np.zeros([N_PARTICLES]) #array of all Energy
#copy of generate_E_array
#fills in E_internal array specified at the beginning
T = 10 * (np.ones([N_PARTICLES]) + np.random.rand(N_PARTICLES)) #20 kelvins max temperature

#fills the f_u array
#import these from previous as needed
fgas = np.array([specie_fraction_array] * N_PARTICLES).T * (particle_type == 0)
fdust = np.array([dust_base] * N_PARTICLES).T * (particle_type == 2)

f_un = (fgas + fdust).T
f_un = f_un.astype('longdouble')
#sizes[particle_type == 2] = d/raise_factor**(2./3.)
mass = mass.astype('longdouble')

#based on http://iopscience.iop.org/article/10.1088/0004-637X/729/2/133/meta
base_sfr = 0.00 #no star formation in new sim!!!
T_FF = (3./(2 * np.pi * G * critical_density))**0.5/year
#base star formation per free fall rate

#f_un = np.array([specie_fraction_array] * N_PARTICLES)
mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(f_un, axis=1)
gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(f_un, axis=1)
cross_array = np.sum(f_un * cross_sections, axis = 1)/np.sum(f_un, axis=1)
E_internal = gamma_array * mass * k * T/(mu_array * m_h)
optical_depth = mass/(m_h * mu_array) * cross_array

critical_density = 1000 * amu * 10**6 #critical density of star formation

neighbor, neighbor_tree, neighbor_dist, num_nontrivial, sizes_new = nsc.neighbors(points, (base_imf[0] * solar_mass/m_0)**(1./3.) * d, N_NEIGH)
E_internal = E_internal/gamma_array * sizes_new**3
sizes = sizes_new #isothermal equation of state
nontrivial_int = neighbor
num_neighbors = N_NEIGH * np.ones(len(points))

velocities[particle_type == 2] *= 0.
velocities += np.random.normal(size=(N_PARTICLES, 3)) * 1000.

#### STEPS FOR RUNNING THE CODE ####
'''SHOULD HAVE THE OPTION OF INITIALIZING FROM SCRATCH (AND GENERATING THE PARAMETERS AS ABOVE) OR FROM LOADING ALL OF THESE
PARAMETERS FROM FILE. IT IS ESSENTIAL FOR CHECKPOINTING THAT THE RESULTS OF THE SIMULATION ARE ALL WRITTEN OUT TO DISK.
CREATE ONE FOLDER PER ELEMENT OF THE PARAMETER SPACE AND WRITE THE OUTPUTS TO FILE, SO LONG AS THIS DOESN'T TAKE MORE THAN A
FEW SECONDS, WE SHOULD BE OKAY TO DO IT EACH TIMESTEP. BETTER YET IF WE CAN DO DYNAMICAL UPDATING OF INDIVIDUAL PARAMETERS
FASTER THAN THE GLOBAL HYDRO UPDATE TIMESTEP.

RUN HYDRO CODE ON ONE CORE, GRAV ACCELERATION ON ANOTHER CORE, AND RADIATIVE TRANSFER ON THE THIRD CORE. HYDRO UPDATE TIMESCALE
SHOULD BE REDUCED TO ~2500 YEARS DURING A SUPERNOVA EXPLOSION BECAUSE SHOCKS WILL PROPAGATE VERY RAPIDLY THROUGH THE MEDIUM. HAVE
SOME METHOD OF DETERMINING AHEAD OF TIME WHAT THE "GLOBAL TIMESTEP" IS GOING TO BE AND USE THAT TO SET THE PACE OF THE SIMULATION.
SET HYDRO TIMESTEP AS TYPICAL CROSSING TIME.'''

hydro_accel, visc_accel, visc_heat, densities, num_densities, fun_densities, dust_densities = nsc.hydro_update(neighbor, points, mass, sizes, f_un, particle_type, T, mu_array, gamma_array, velocities)
grav_accels = nsc.grav_force_calculation_new(mass, points, sizes)
#rad_transfer_variables, = nsc.radiative_transfer(whatever, parameters, we, need)
#if (supernova condition):
#	nsc.do_supernova(whatever, parameters, we, need)

#SUPERNOVA HANDLING CAN BE PRETTY MUCH AS BEFORE
'''        
		max_rel_age = np.max(star_ages/nsc.luminosity_relation(mass/nsc.solar_mass, np.ones(len(mass)), 1)/(year * 1e10))
        print ("Maximum relative stellar age: " + str(max_rel_age))
        print ("Maximum stellar mass: " + str(np.max(mass[particle_type == 1]/solar_mass)) + " solar masses")
        print ("Number of supernovae: " + str(len(supernova_pos)))
        print (" ")
        
        
        
        #print (star_ages/luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1)/(year * 1e10))[supernova_pos]
        if len(supernova_pos) > 0:
        	#print('beginning supernova impulse')
        	for ku in supernova_pos:
        		impulse, indices = nsc.supernova_impulse(points, mass, ku, particle_type)
        		velocities[indices] += impulse
			#print('end supernova impulse')
			#print supernova_pos
			
			#temporarily reducing timestep to allow AV to kick in
			ct = nsc.crossing_time(neighbor, velocities, sizes, particle_type)
			dt = min(dt_0/100., ct)
			nsc.dt = dt
			
			if len(supernova_pos) > 0:
				dust_comps, gas_comps, star_comps, dust_mass, gas_mass, stars_mass, newpoints, newvels, newgastype, newdusttype, new_eint_stars, new_eint_dust, new_eint_gas, supernova_pos2, dustpoints, dustvels = nsc.supernova_explosion(mass,points,velocities,E_internal,supernova_pos, f_un)
				particle_type = np.concatenate((particle_type, newdusttype, newgastype))
				E_internal[supernova_pos] = new_eint_stars
				f_un[supernova_pos] = star_comps
				mass[supernova_pos] = stars_mass
				E_internal = np.concatenate((E_internal, new_eint_dust, new_eint_gas))
				mass = np.concatenate((mass, dust_mass, gas_mass))
				f_un = np.vstack([f_un, dust_comps, gas_comps])
				velocities = np.concatenate((velocities, dustvels, newvels))
				star_ages = np.concatenate((star_ages, np.ones(len(dustpoints))* (-2), np.ones(len(supernova_pos))* (-2)))
				points = np.vstack([points, dustpoints, newpoints])
				oldsizes = copy.deepcopy(sizes)
				sizes = np.zeros(len(points))
				sizes[:len(oldsizes)] = oldsizes
				sizes[(particle_type == 0) & (sizes == 0)] = (mass[(particle_type == 0) & (sizes == 0)]/m_0)**(1./3.) * d
				sizes[(particle_type == 1) & (sizes == 0)] = d/10000.
				sizes[(particle_type == 2) & (sizes == 0)] = d
				Tnew = np.zeros(len(sizes));
				Tnew[:len(T)] += T
				T = Tnew
				T[T < t_cmb] = t_cmb
				supernova_pos = []
			#supernova_pos = np.where(star_ages/nsc.luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1)/(year * 1e10) > 1.)[0]

			neighbor, neighbor_tree, neighbor_dist, num_nontrivial, sizes = nsc.neighbors(points, base_imf[0]/m_0 * d, N_NEIGH)#now a numpy array---so much faster!
			chems_neighbor = np.copy(neighbor)
			nontrivial_int = neighbor #nsc.nontrivial_neighbors(points, mass, particle_type, neighbor)
			num_neighbors = N_NEIGH * np.ones(len(points))
			
        
        
        mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(f_un, axis=1)
        gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(f_un, axis=1)
        cross_array = np.sum(f_un * cross_sections, axis = 1)/np.sum(f_un, axis=1)
        optical_depth = mass/(m_h * mu_array) * cross_array
        #print("Negative compositions after supernova: " + str(len(f_un[np.sum(f_un/np.abs(f_un),axis=1) < 13])))
'''