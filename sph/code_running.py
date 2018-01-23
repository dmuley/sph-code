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
size = comm.Get_size()'''

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
t_max = 2e4
t_solar = 5776
nsc.supernova_energy = 1e44 #in Joules
m_0 = 10**1.5 * solar_mass #solar masses, maximum mass in the kroupa IMF
year = 60. * 60. * 24. * 365.
dt_0 = year * 250000.
#properties for species in each SPH particle, (H2, He, H,H+,He+,e-,Mg2SiO4,SiO2,C,Si,Fe,MgSiO3,FeSiO3, SiC)in that order
species_labels = np.array(['H2', 'He', 'H','H+','He+','e-','Mg2SiO4','SiO2','C','Si','Fe','MgSiO3','FeSiO3', 'SiC'])
mu_specie = np.array([2.0159,4.0026,1.0079,1.0074,4.0021,0.0005,140.69,60.08,12.0107,28.0855,55.834,100.39,131.93, 40.096])
cross_sections = np.array([6.65e-24/5., 6.65e-24/5., 6.65e-23, 5e-60, 5e-60, 0., 0., 0., 0., 0., 0., 0., 0., 0.]) + 1e-80
destruction_energies = np.array([7.2418e-19, 3.93938891e-18, 2.18e-18, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000])
mineral_densities = np.array([1.e19, 1e19,1e19,1e19,1e19,1e19, 3320,2260,2266,2329,7870,3250,3250., 3166.])
sputtering_yields = np.array([0,0,0,0,0,0,0.137,0.295,0.137,0.295,0.137,0.137,0.137, 0.137])
f_u = np.array([[.86,.14,0,0,0,0,0,0,0,0,0,0,0,0]]) #relative abundance for species in each SPH particle, an array of arrays
gamma = np.array([7./5,5./3,5./3,5./3,5./3,5./3,15.6354113,4.913,1.0125,2.364,3.02,10.,10.,10.])#the polytropes of species in each SPH, an array of arrays

W6_constant = (3 * np.pi/80)
mrn_constants = np.array([50e-10, 5000e-10]) #minimum and maximum radii for MRN distribution
cross_sections += nsc.sigma_effective(mineral_densities, mrn_constants, mu_specie)

#### AND NOW THE FUN BEGINS! THIS IS WHERE THE SIMULATION RUNS HAPPEN. ####
#SETTING VALUES OF BASIC SIMULATION PARAMETERS HERE (TO REPLACE DUMMY VALUES AT BEGINNING)
DIAMETER = 2.2e5 * AU
N_PARTICLES = 400
N_INT_PER_PARTICLE = 50
V = (DIAMETER)**3
d = (V/N_PARTICLES * N_INT_PER_PARTICLE)**(1./3.)
nsc.d = d
d_sq = d**2
d_0 = 1e5 * AU
nsc.d_0 = 1e5 * AU
dt = dt_0
nsc.dt = dt
nsc.dt_0 = dt_0
DUST_MASS = 0.05 #mass of each dust SPH particle
N_RADIATIVE = 1 #number of timesteps for radiative transfer, deprecated
MAX_AGE = 3e7 * year/7. #don't want to see any AGB stars undergoing supernovae inadvertently



specie_fraction_array = np.array([.86,.14,0,0,0,0,0,0,0,0,0,0,0,0])
dust_base_frac = np.array([.0,.0,0.,0.,0.,0.,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025])
DUST_FRAC = 0.000
OVERALL_AGE = 0. #age of the galaxy
'''
WHAT NEEDS TO BE DONE HERE: IMPORT COMPOSITION AND DUST FRACTION FROM FILE, IF SUCH A FILE EXISTS
THIS CODE WILL WRITE TO FILE, HELPER CODE WILL CREATE FILE
'''
#import dust_base_frac from file for future calculations, if the file exists
#Returns last-numbered .npy file
absolute_path_to_nsc = os.path.dirname(os.path.abspath(nsc.__file__))
absolute_path_to_config = absolute_path_to_nsc + '/../savefiles/config'

config_files = os.listdir(absolute_path_to_config)
conf_number = np.array([am[7:-4] for am in config_files]).astype('int')
max_conf_file = (conf_number == max(conf_number))

conf_filename_selected = np.array(config_files)[max_conf_file][0]
latest_file = np.load(unicode(absolute_path_to_config + '/' + conf_filename_selected))

#obtains DUST_FRAC, specie_fraction_array, and dust_base_frac from loaded file
specie_fraction_array = latest_file['specie_fraction_array']
dust_base_frac = latest_file['dust_base_frac']
DUST_FRAC = latest_file['DUST_FRAC']
OVERALL_AGE = latest_file['OVERALL_AGE']

############################
#END LOADING FROM FILE HERE#
############################

dust_base = dust_base_frac/np.sum(dust_base_frac)
base_imf = np.logspace(np.log10(0.1),np.log10(40.), 200)
d_base_imf = np.append(base_imf[0], np.diff(base_imf))
imf = nsc.kroupa_imf(base_imf) * d_base_imf
imf /= np.sum(imf)

#np.random.seed(rank*time.time())
mass = np.random.choice(base_imf, N_PARTICLES, p = imf) * solar_mass

N_DUST = max(int(DUST_FRAC/(1. - DUST_FRAC) * np.sum(mass)/(DUST_MASS * solar_mass)), 1)
particle_type = np.zeros([N_PARTICLES]) #0 for gas, 1 for stars, 2 for dust
N_PARTICLES += N_DUST
particle_type = np.append(particle_type, [2] * N_DUST)
mass = np.append(mass, [DUST_MASS * solar_mass] * N_DUST)

points = (np.random.rand(N_PARTICLES, 3) - 0.5) * DIAMETER
points2 = copy.deepcopy(points)
neighbor = nsc.neighbors(points, d)
#print(nbrs)
#print(points)
velocities = np.random.normal(size=(N_PARTICLES, 3)) * 10.
total_accel = np.random.rand(N_PARTICLES, 3) * 0.
sizes = (mass/m_0)**(1./3.) * d

mu_array = np.zeros([N_PARTICLES])#array of all mu
E_internal = np.zeros([N_PARTICLES]) #array of all Energy
#copy of generate_E_array
#fills in E_internal array specified at the beginning
T = 10 * (np.ones([N_PARTICLES]) + np.random.rand(N_PARTICLES)) #20 kelvins


#fills the f_u array
#import these from previous as needed
fgas = np.array([specie_fraction_array] * N_PARTICLES).T * (particle_type == 0)
fdust = np.array([dust_base] * N_PARTICLES).T * (particle_type == 2)

f_un = (fgas + fdust).T
f_un = f_un.astype('longdouble')
sizes[particle_type == 2] = d
mass = mass.astype('longdouble')

#based on http://iopscience.iop.org/article/10.1088/0004-637X/729/2/133/meta
base_sfr = 0.0057 
T_FF = (3./(2 * np.pi * G * np.sum(mass[particle_type == 0])/V))**0.5/year

#f_un = np.array([specie_fraction_array] * N_PARTICLES)
mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(f_un, axis=1)
gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(f_un, axis=1)
cross_array = np.sum(f_un * cross_sections, axis = 1)/np.sum(f_un, axis=1)
E_internal = gamma_array * mass * k * T/(mu_array * m_h)
optical_depth = mass/(m_h * mu_array) * cross_array

#copy of generate_mu_array

critical_density = 1000*amu*10**6 #critical density of star formation

densities = nsc.density(points,mass,particle_type,neighbor)
dust_densities = nsc.dust_density(points,mass,neighbor,particle_type,sizes)
delp = nsc.del_pressure(points,mass,particle_type,neighbor,E_internal,gamma_array)

star_ages = np.ones(len(points)) * -1.
age = 0

time_coord = np.array([])
dust_temps = np.array([])
star_frac = np.array([])
imf_measure = np.array([])
chems_error = np.array([])
sup_error = np.array([])

print("Simulation time: " + str(MAX_AGE/year) + " y")
print("Estimated free fall time: " + str(T_FF) + " y")
plt.ion()
#RUNNING SIMULATION FOR SPECIFIED TIME!
#simulating supernova asap
particle_type[mass == max(mass)] = 1
star_ages[mass == max(mass)] = 3.31e6 * year
#fig, ax = plt.subplots(nrows=1, ncols = 2)
while (age < MAX_AGE):
    #timestep reset here
    ct = nsc.crossing_time(neighbor, velocities, sizes, particle_type)
    if age == 0:
    	dt = dt_0/10
    else:
    	dt = max(dt_0/5., min(dt_0, ct))
    	
    nsc.dt = dt
    #stop points from going ridiculously far
    points[points > 1e11 * AU] = 1e11 * AU
    points[points < -1e11 * AU] = -1e11 * AU
    velocities[points > 1e11 * AU] = 0.1
    velocities[points < -1e11 * AU] = -0.1
    points = np.nan_to_num(points)
    velocities = np.nan_to_num(velocities)
    print('=====================================================================')
    #print("Negative compositions before radiative transfer: " + str(len(f_un[np.sum(f_un/np.abs(f_un),axis=1) < 13])))
    if np.sum(particle_type[particle_type == 1]) > 0:
        supernova_pos = np.where(star_ages/nsc.luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1)/(year * 1e10) > 1.)[0]
        #N_RADIATIVE = int(10 + np.average(np.nan_to_num(T))**(1./3.)/10.)
        #N_RADIATIVE = 1
        area = (4 * np.pi * sizes**2)
        N_PART = mass/(m_h * mu_array)
        W6_integral = 9./area #evaluating integral of W6 kernel
        optd = 1. - np.exp(-optical_depth * W6_integral)
        for nradiative in range(N_RADIATIVE):
			rh = nsc.rad_heating(points, particle_type, mass, sizes, cross_array, f_un, supernova_pos, mu_array, T, dt/N_RADIATIVE)
		
			f_un = rh[1]
		
			mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(f_un, axis=1)
			gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(f_un, axis=1)
			cross_array = np.sum(f_un * cross_sections, axis = 1)/np.sum(f_un, axis=1)
			optical_depth = mass/(m_h * mu_array) * cross_array
		
			#have to fix radiative cooling
			rc = nsc.rad_cooling(points, particle_type, mass, sizes, cross_array, rh[1], neighbor, mu_array, T, dt/N_RADIATIVE)
			#f_un = rc[0]
		
			mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(f_un, axis=1)
			gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(f_un, axis=1)
			cross_array = np.sum(f_un * cross_sections, axis = 1)/np.sum(f_un, axis=1)
			optical_depth = mass/(m_h * mu_array) * cross_array
				
			area = (4 * np.pi * sizes**2)
			N_PART = mass/(m_h * mu_array)
			W6_integral = 9./area #evaluating integral of W6 kernel
			optd = 1. - np.exp(-optical_depth * W6_integral)

			T[particle_type == 2] = (rh[0][particle_type[particle_type != 1] == 2]/(sb * 4 * np.pi * optd[particle_type == 2] * sizes[particle_type == 2]**2 * dt * 4e-6 * 1))**(1./6.) + t_cmb
			E_internal[particle_type == 2] = (N_PART * k * T * gamma_array)[particle_type == 2]
			E_internal[particle_type == 0] += rh[0][particle_type[particle_type != 1] == 0] #- (rc[1] * N_PART)[particle_type == 0]
			T[particle_type == 0] += rh[0][particle_type[particle_type != 1] == 0]/(gamma_array * N_PART * k)[particle_type == 0] #- (rc[1]/gamma_array/k)[particle_type == 0]
		
			velocities[particle_type != 1] += rh[2]
		
			E_internal[particle_type == 0] *= np.exp(-rc[1]/k/gamma_array/T)[particle_type == 0]
			T[particle_type == 0] *= np.exp(-rc[1]/k/gamma_array/T)[particle_type == 0]

			E_internal[E_internal < t_cmb * (gamma_array * mass * k)/(mu_array * m_h)] = (t_cmb * (gamma_array * mass * k)/(mu_array * m_h))[E_internal < t_cmb * (gamma_array * mass * k)/(mu_array * m_h)]
			E_internal[E_internal > t_max * (gamma_array * mass * k)/(mu_array * m_h)] = (t_max * (gamma_array * mass * k)/(mu_array * m_h))[E_internal > t_max * (gamma_array * mass * k)/(mu_array * m_h)]
			T[T < t_cmb] = t_cmb
			T[T >= t_max] = t_max
			'''
			#another plotting function
			plt.rc('font', family='tex')
			massT = np.cumsum(mass[particle_type == 0][np.argsort(T[particle_type == 0])])
			massT /= max(massT)
			
			massd = np.cumsum(mass[particle_type == 0][np.argsort(densities[particle_type == 0])])
			massd /= max(massd)
			
			ax[0].plot(massd, np.log10(np.sort(densities[particle_type == 0])/critical_density), c='black', alpha=0.15)
			ax[1].plot(massT, np.log10(np.sort(T[particle_type == 0])), c = 'maroon', alpha=0.15)
			
			ax[0].set_xlabel('Percentile'); ax[1].set_xlabel('Percentile')
			ax[0].set_ylabel('Log10(density/critical density)')
			ax[1].set_ylabel('Log10(T/K)')
			plt.suptitle('Curves for density and temperature over time (t = ' + str(age/year) + ' years')
			plt.pause(0.01)'''

        #print("Negative compositions after radiative transfer: " + str(len(f_un[np.sum(f_un/np.abs(f_un),axis=1) < 13])))
        #on supernova event--- add new dust particle (particle_type == 2)
        
        max_rel_age = np.max(star_ages/nsc.luminosity_relation(mass/nsc.solar_mass, np.ones(len(mass)), 1)/(year * 1e10))
        print ("Maximum relative stellar age: " + str(max_rel_age))
        print ("Maximum stellar mass: " + str(np.max(mass[particle_type == 1]/solar_mass)) + " solar masses")
        print ("Calculated timestep: " + str(ct/year) + " years")
        print ("Number of supernovae: " + str(len(supernova_pos)))
        #print (star_ages/luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1)/(year * 1e10))[supernova_pos]
        if len(supernova_pos) > 0:
        	#print('beginning supernova impulse')
        	for ku in supernova_pos:
        		impulse, indices = nsc.supernova_impulse(points, mass, ku, particle_type)
        		velocities[indices] += impulse
			#print('end supernova impulse')
			print supernova_pos
			
			#temporarily reducing timestep to allow AV to kick in
			ct = nsc.crossing_time(neighbor, velocities, sizes, particle_type)
			dt = min(dt_0/100., ct)
			nsc.dt = dt
			
			dust_comps, gas_comps, star_comps, dust_mass, gas_mass, stars_mass, newpoints, newvels, newgastype, newdusttype, new_eint_stars, new_eint_dust, new_eint_gas, supernova_pos, dustpoints, dustvels = nsc.supernova_explosion(mass,points,velocities,E_internal,supernova_pos, f_un)
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
			sizes = np.zeros(len(points))
			sizes[particle_type == 0] = (mass[particle_type == 0]/m_0)**(1./3.) * d
			sizes[particle_type == 1] = d/10000.
			sizes[particle_type == 2] = d
			Tnew = np.zeros(len(sizes));
			Tnew[:len(T)] += T
			T = Tnew
			T[T < t_cmb] = t_cmb
			supernova_pos = np.where(star_ages/nsc.luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1)/(year * 1e10) > 1.)[0]

			neighbor = nsc.neighbors(points, d)
        mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(f_un, axis=1)
        gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(f_un, axis=1)
        cross_array = np.sum(f_un * cross_sections, axis = 1)/np.sum(f_un, axis=1)
        optical_depth = mass/(m_h * mu_array) * cross_array
        #print("Negative compositions after supernova: " + str(len(f_un[np.sum(f_un/np.abs(f_un),axis=1) < 13])))
    
    f_un = (f_un.T/np.sum(f_un, axis=1)).T #normalizing composition 
    densities = nsc.density(points,mass,particle_type,neighbor)
    dust_densities = nsc.dust_density(points,mass,neighbor,particle_type,sizes)
    num_densities = nsc.num_dens(mass, points, mu_array, neighbor)
    
    chems = nsc.chemisputtering_2(points, neighbor, mass, f_un, mu_array, sizes, T, particle_type)
    deltam_chems = (np.sum(mass) - np.sum(chems[0]))/solar_mass
    print ("Mass error from chemisputtering: " + str(deltam_chems) + " solar masses")
    f_un = chems[1]; mass = chems[0]
    supd = nsc.supernova_destruction(points, velocities, neighbor, mass, f_un, mu_array, sizes, densities, particle_type)    
    mass_reduction = -np.sum(np.sum((mass/(mu_array) * supd[0].T).T * mu_specie,axis=1))
    mass_increase = np.sum(np.sum((mass/(mu_array) * supd[1].T).T * mu_specie,axis=1))
    print("Negative compositions after chemisputtering: " + str(len(f_un[np.sum(f_un/np.abs(f_un),axis=1) < 13])))
    #increase *= mass_reduction/mass_increase
    delta_n = (np.nan_to_num(-supd[0] + supd[1]).astype('longdouble').T * (mass > 0.001 * solar_mass)).T
    mass_change = np.nan_to_num(np.sum((mass/(mu_array) * delta_n.T).T * mu_specie,axis=1)) * (mass > 0.001 * solar_mass)
    deltam_sup = np.sum(mass_change)/solar_mass
    print ("Mass error from supernova sputtering: " + str(deltam_sup) + " solar masses")
    f_un += delta_n
    #f_un = np.abs(f_un)
    f_un[(np.isnan(f_un))] = 1e-15
    f_un = (f_un.T/np.sum(f_un, axis=1)).T #normalizing composition
    mass += mass_change
    #mass = np.nan_to_num(mass)
    mass[mass < 0.001 * solar_mass] = 0.001 * solar_mass
    mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(f_un, axis=1)
    gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(f_un, axis=1)
    cross_array = np.sum(f_un * cross_sections, axis = 1)/np.sum(f_un, axis=1)
    optical_depth = mass/(m_h * mu_array) * cross_array
    sizes[particle_type == 0] = (mass[particle_type == 0]/m_0)**(1./3.) * d
    sizes[particle_type == 2] = d
    print("Negative compositions after supernova sputtering: " + str(len(f_un[np.sum(f_un/np.abs(f_un),axis=1) < 13])))

    neighbor = nsc.neighbors(points, d)#find neighbors in each timestep
    num_neighbors = np.array([len(adjoining) for adjoining in neighbor])
    age += dt
    gfcalc = nsc.grav_force_calculation(mass, points, sizes);
    grav_accel = gfcalc[0]
    '''bg = nsc.bin_generator(mass, points, [5, 4, 4]); age += dt
    com = bg[0] #center of masses of each bin
    grav_accel = nsc.compute_gravitational_force(points, bg[0], bg[1], np.median(sizes)).T #gravity is always acting, thus no cutoff distance introduced for gravity
    '''
    densities = nsc.density(points,mass,particle_type,neighbor)
    dust_densities = nsc.dust_density(points,mass,neighbor,particle_type,sizes)
    num_densities = nsc.num_dens(mass, points, mu_array, neighbor)
    #viscous force causing dust to accelerate/decelerate along with gas
    viscous_drag = nsc.net_impulse(points,mass,sizes,velocities,particle_type,neighbor,f_un)
    delp = nsc.del_pressure(points,mass,particle_type,neighbor,E_internal,gamma_array)
    #artificial viscosity to ensure proper blast wave
    av = nsc.artificial_viscosity(neighbor, points, particle_type, sizes, mass, densities, velocities, T, gamma_array, mu_array)
    
    pressure_accel = -np.nan_to_num((delp.T/densities * (particle_type == 0).astype('float')).T)
    #drag is a very small factor, only seems to matter at the solar-system scale
    drag_accel_gas = np.nan_to_num(((viscous_drag[0].T) * dust_densities/densities * (particle_type == 0).astype('float')).T)
    drag_accel_dust = np.nan_to_num(viscous_drag[1])
    
    #leapfrog integration
    old_accel = copy.deepcopy(total_accel)
    visc_accel = drag_accel_gas + drag_accel_dust + av[0]
    #want perturbations of artificial viscosity to die out in time
    visc_accel[np.sum(velocities**2, axis=1)**0.5 - np.sum(visc_accel**2, axis=1)**0.5 * dt < 0] = (-velocities/dt)[np.sum(velocities**2, axis=1)**0.5 - np.sum(visc_accel**2, axis=1)**0.5 * dt < 0]
    
    total_accel = grav_accel + pressure_accel + visc_accel
    
    #removed sputtering until numerical errors can be eliminated
                
    points += ((total_accel * (dt)**2)/2.) + velocities * dt
    if np.shape(total_accel) == np.shape(old_accel):
    	dv = (total_accel + old_accel)/2. * dt
    else:
    	dv = total_accel * dt
    velocities += dv
    
    E_internal = np.nan_to_num(E_internal) + np.nan_to_num(av[1] * dt)
    T = np.nan_to_num(E_internal * (mu_array * m_h)/(gamma_array * mass * k))
        
    vel_condition = np.sum(velocities**2, axis=1)
    
    xmin = np.percentile(points.T[0][vel_condition < 80000**2], 10)/AU
    xmax =  np.percentile(points.T[0][vel_condition < 80000**2], 90)/AU
    ymin =  np.percentile(points.T[1][vel_condition < 80000**2], 10)/AU
    ymax =  np.percentile(points.T[1][vel_condition < 80000**2], 90)/AU
    zmin = np.percentile(points.T[2][vel_condition < 80000**2], 10)/AU
    zmax = np.percentile(points.T[2][vel_condition < 80000**2], 90)/AU

    x_dist = zmax - zmin
    y_dist = xmax - xmin
    z_dist = ymax - ymin
    
    #dynamic sizing of SPH particles helps ensure
    #a more accurate simulation once gas is dispersed
    #and moreover allows us to reduce the number of nominal intersections per particle
    #which provides a speed boost at the beginning
    
    #moreover, this gives us a greater dynamic range of possible densities
    #to work with, and helps avoid star formation at some arbitrary density floor
    V_new = np.abs(x_dist * y_dist * z_dist) * AU**3
    d = (V_new/len(points[vel_condition < 80000**2])/(0.9 - 0.1) * N_INT_PER_PARTICLE)**(1./3.)
    d_sq = d**2 
    sizes[particle_type == 0] = (np.abs(mass[particle_type == 0])/m_0)**(1./3.) * d
    sizes[particle_type == 2] = d
    
    dist_sq = np.sum(points**2,axis=1)
    
    min_dist = np.percentile(dist_sq[vel_condition < 80000**2], 0)
    max_dist = np.percentile(dist_sq[vel_condition < 80000**2], 90)
    '''
    #PLOTTING: THIS CAN BE ADDED OR REMOVED AT WILL
    xpts = points.T[1:][0][particle_type == 0]/constants.parsec
    ypts = points.T[1:][1][particle_type == 0]/constants.parsec
    
    xstars = points.T[1:][0][particle_type == 1]/constants.parsec
    ystars = points.T[1:][1][particle_type == 1]/constants.parsec
    sstars = (mass[particle_type == 1]/solar_mass) * 2.
    
    xdust = points.T[1:][0][particle_type == 2]/constants.parsec
    ydust = points.T[1:][1][particle_type == 2]/constants.parsec
    
    #colors = (f_un.T[5]/np.sum(f_un, axis=1))[particle_type == 0]
    colors = np.log10(T[particle_type == 0])
    col_dust = np.log10(T[particle_type == 2])
    
    plt.clf()
    plt.axis('equal')
    
    max_val = max(max(xpts[(dist_sq[particle_type == 0] < max_dist * 11./9.)]), max(ypts[(dist_sq[particle_type == 0] < max_dist * 11./9.)]))
    min_val = min(min(xpts[(dist_sq[particle_type == 0] < max_dist * 11./9.)]), min(ypts[(dist_sq[particle_type == 0] < max_dist * 11./9.)]))
    
    #BEGIN PLOTTING
    
    plt.scatter(np.append(xdust[(dist_sq[particle_type == 2] < max_dist * 11./9.)],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))]), np.append(ydust[(dist_sq[particle_type == 2] < max_dist * 11./9.)],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))]), c=np.append(col_dust[(dist_sq[particle_type == 2] < max_dist * 11./9.)],[0,np.log10(t_max)]), s = np.append(100 * np.ones(len(col_dust[(dist_sq[particle_type == 2] < max_dist * 11./9.)])), [0.01, 0.01]), alpha=0.25)
    plt.scatter(np.append(xpts[(dist_sq[particle_type == 0] < max_dist * 11./9.)],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))]), np.append(ypts[(dist_sq[particle_type == 0] < max_dist * 11./9.)],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))]), c=np.append(colors[(dist_sq[particle_type == 0] < max_dist * 11./9.)],[0,np.log10(t_max)]),s=np.append((sizes[(dist_sq[particle_type == 0] < max_dist * 11./9.)]/d) * 100 * 4, [0.01, 0.01]), edgecolor='none', alpha=0.1)
    #plt.scatter([max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))],c=[1,0],s=0.01,alpha=0.1)
    #plt.scatter(np.append(xdust[(dist_sq[particle_type == 2] < max_dist * 11./9.)],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))]), np.append(ydust[(dist_sq[particle_type == 2] < max_dist * 11./9.)],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))]), c=np.append(col_dust[(dist_sq[particle_type == 2] < max_dist * 11./9.)],[0,7]), s = np.append((m_0/solar_mass) * np.ones(len(col_dust[(dist_sq[particle_type == 2] < max_dist * 11./9.)])), [0.01, 0.01]), alpha=0.25)
    plt.colorbar()
    plt.scatter(xstars[(dist_sq[particle_type == 1] < max_dist * 11./9.)], ystars[(dist_sq[particle_type == 1] < max_dist * 11./9.)], c='black', s=sstars[(dist_sq[particle_type == 1] < max_dist * 11./9.)])
    #plt.scatter(xdust[(dist_sq[particle_type == 2] < max_dist * 11./9.)], ydust[(dist_sq[particle_type == 2] < max_dist * 11./9.)], c=col_dust[(dist_sq[particle_type == 2] < max_dist * 11./9.)], s = (m_0/solar_mass), alpha=0.25)
    plt.xlabel('Position (parsecs)')
    plt.ylabel('Position (parsecs)')
    plt.title('Temperature in H II region (t = ' + str(age/year/1e6) + ' Myr)')
    plt.pause(1)
    
    #END PLOTTING'''
    
    star_ages[(particle_type == 1) & (star_ages > -2)] += dt
    #print star_ages[(particle_type == 1)]/year
    probability = base_sfr * (densities/critical_density)**(1.6) * ((dt/year)/T_FF)
    diceroll = np.random.rand(len(probability))
    particle_type[(particle_type == 0) & (num_neighbors > 1)] = ((diceroll < probability).astype('float'))[(particle_type == 0) & (num_neighbors > 1)]
    #this helps ensure that lone SPH particles don't form stars at late times in the simulation
    #ideally, there is an infinite number of SPH particles, each with infinitesimal density
    #that only has any real physical effects in conjunction with other particles

    star_massfrac = float(np.sum(mass[particle_type == 1]))/np.sum(mass[particle_type != 2])
    star_numfrac = np.sum(float(len(particle_type[particle_type == 1]))/float(len(particle_type[particle_type != 2])))
    
    time_coord = np.append(time_coord, [age] * len(T[particle_type == 2]))
    dust_temps = np.append(dust_temps, (E_internal/optical_depth)[particle_type == 2])
    star_frac = np.append(star_frac, [star_massfrac] * len(T[particle_type == 2]))
    imf_measure = np.append(imf_measure, [star_massfrac/star_numfrac] * len(T[particle_type == 2]))
    chems_error = np.append(chems_error, [deltam_chems] * len(T[particle_type == 2]))
    sup_error = np.append(sup_error, [deltam_sup] * len(T[particle_type == 2]))
    
    print ("Total mass of system: " + str(np.sum(mass)/solar_mass) + " solar masses")
    print ('Age:', age/year)
    #print (d/AU)
    print ('Stellar mass/nondust mass = ', star_massfrac)
    print ('(stellar mass/nondust mass)/(Stellar number/total number)' + str(star_massfrac/star_numfrac))
    print ('=====================================================================')

'''WE WANT SEVERAL OUTPUTS FROM THIS FILE
1. mass fraction of dust
2. Average composition of dust weighted by mass
3. List of AGB progenitors and when they will form AGBs.
4. Dust produced by type of source? (AGB, supernovae, nucleation)
5. Find some way to calculate dilution based on mass fraction in GMCs at any given time.
'''

gas_mass_by_species = np.sum((((f_un * mu_specie)[particle_type == 0].T/np.sum((f_un * mu_specie)[particle_type == 0], axis=1)) * mass[particle_type == 0]).T, axis=0)
star_mass_by_species = np.sum((((f_un * mu_specie)[particle_type == 1].T/np.sum((f_un * mu_specie)[particle_type == 1], axis=1)) * mass[particle_type == 1]).T, axis=0)
dust_mass_by_species = np.sum((((f_un * mu_specie)[particle_type == 2].T/np.sum((f_un * mu_specie)[particle_type == 2], axis=1)) * mass[particle_type == 2]).T, axis=0)
#assume even split between C, S, and M stars because there is no other assumption to make from observation
AGB_condition = ((particle_type == 1) & (mass/solar_mass > 1.) & (mass/solar_mass < 7.))
AGB_list = mass[AGB_condition]
AGB_time_until = nsc.luminosity_relation(mass[AGB_condition]/solar_mass, np.ones(len(mass[AGB_condition])), 1) * 1e10 * year - star_ages[AGB_condition] + OVERALL_AGE #time of formation of AGB
AGB_metallicity = np.sum((f_un * mu_specie)[AGB_condition].T[6:], axis=0)/np.sum((f_un * mu_specie)[AGB_condition], axis=1)

absolute_path_to_outputs = absolute_path_to_nsc + '/../savefiles/outputs'
list_of_outputs = np.append(os.listdir(unicode(absolute_path_to_outputs)), '-1.npz')
output_number = np.max(np.array([bm[:-4] for bm in list_of_outputs]).astype('int'))
np.savez(unicode(absolute_path_to_outputs + '/' + str(output_number + 1)), gas_mass_by_species = gas_mass_by_species, star_mass_by_species = star_mass_by_species, dust_mass_by_species = dust_mass_by_species, AGB_condition = AGB_condition, AGB_list = AGB_list, AGB_time_until = AGB_time_until, AGB_metallicity = AGB_metallicity)
#save all the above to a Numpy binary which will then be read in by config_helper.py to create a new single config file

'''
utime = np.unique(time_coord)
dustt = np.array([np.average(dust_temps[time_coord == med]) for med in np.unique(time_coord)])
dusts = np.array([np.std(dust_temps[time_coord == med]) for med in np.unique(time_coord)])
starf = np.array([np.average(star_frac[time_coord == med]) for med in np.unique(time_coord)])
photodissociation = (f_un.T[3] + f_un.T[2] + f_un.T[4])/(f_un.T[2] + f_un.T[0] + f_un.T[3] + f_un.T[1] + f_un.T[4]

plt.scatter(np.log10(time_coord/year), np.log10(dust_temps), alpha=0.2, c='grey', s = 10, edgecolor='none')
plt.plot(np.log10(utime/year), np.log10(dustt), c='maroon', alpha=0.5)
plt.plot(np.log10(utime/year), np.log10(dustt + 2 * dusts), c='maroon', alpha=0.25)
plt.plot(np.log10(utime/year), np.log10(dustt - 2 * dusts), c='maroon', alpha=0.25)

plt.scatter((time_coord/year), (dust_temps), alpha=0.2, c='grey', s = 10, edgecolor='none')
plt.plot((utime/year), (dustt), c='maroon', alpha=0.5)
plt.plot((utime/year), (dustt + 2 * dusts), c='maroon', alpha=0.25)
plt.plot((utime/year), (dustt - 2 * dusts), c='maroon', alpha=0.25)


VARIOUS FORMS OF PLOTTING THAT ONE CAN USE TO REPRESENT THE FINAL STATE OF THE SIMULATION

3D PLOTTING:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
[ax.scatter(points.T[0][particle_type == 0]/AU, points.T[1][particle_type == 0]/AU, points.T[2][particle_type == 0]/AU, alpha=0.1)]
[ax.scatter(points.T[0][particle_type == 1]/AU, points.T[1][particle_type == 1]/AU, points.T[2][particle_type == 1]/AU, alpha=0.2)]
[ax.set_xlim3d(-DIAMETER * 10/AU, DIAMETER * 10/AU)]
[ax.set_ylim3d(-DIAMETER * 10/AU, DIAMETER * 10/AU)]
[ax.set_zlim3d(-DIAMETER * 10/AU, DIAMETER * 10/AU)]
[plt.show()]

PROJECTION OF ALL 3 PAIRS OF COORDINATES ONTO A 2D COLOR PLOT:
[plt.scatter(points.T[0][particle_type == 0]/AU, points.T[1][particle_type == 0]/AU, c = np.log10(T)[particle_type == 0], s=30, edgecolor='none', alpha=0.1)]
#[plt.scatter(points.T[0][particle_type == 0]/AU, points.T[2][particle_type == 0]/AU, c = np.log10(T)[particle_type == 0], s=30, edgecolor='none', alpha=0.1)]
#[plt.scatter(points.T[1][particle_type == 0]/AU, points.T[2][particle_type == 0]/AU, c = np.log10(T)[particle_type == 0], s=30, edgecolor='none', alpha=0.1)]
[plt.colorbar()]
[plt.scatter(points.T[0][particle_type == 2]/AU, points.T[1][particle_type == 2]/AU, c = 'black', s=30, edgecolor='face', alpha=0.01)]
[plt.scatter(points.T[0][particle_type == 1]/AU, points.T[1][particle_type == 1]/AU, c = 'black', s=(mass[particle_type == 1]/solar_mass), alpha=1)]
[plt.axis('equal'), plt.show()]
plt.xlabel('Position (astronomical units)')
plt.ylabel('Position (astronomical units)')
plt.title('Temperature in H II region')

[plt.scatter(points.T[0][particle_type == 0]/AU, points.T[1][particle_type == 0]/AU, c = np.log10(densities/critical_density)[particle_type == 0], s=30, edgecolor='none', alpha=0.1)]
#[plt.scatter(points.T[0][particle_type == 0]/AU, points.T[2][particle_type == 0]/AU, c = np.log10(densities/critical_density)[particle_type == 0], s=30, edgecolor='none', alpha=0.1)]
#[plt.scatter(points.T[1][particle_type == 0]/AU, points.T[2][particle_type == 0]/AU, c = np.log10(densities/critical_density)[particle_type == 0], s=30, edgecolor='none', alpha=0.1)]
[plt.colorbar()]
[plt.scatter(points.T[0][particle_type == 2]/AU, points.T[1][particle_type == 2]/AU, c = 'black', s=30, edgecolor='face', alpha=0.02)]
[plt.scatter(points.T[0][particle_type == 1]/AU, points.T[1][particle_type == 1]/AU, c = 'black', s=(mass[particle_type == 1]/solar_mass), alpha=1)]
[plt.axis('equal'), plt.show()]
plt.xlabel('Position (astronomical units)')
plt.ylabel('Position (astronomical units)')
plt.title('Density in H II region')

#INTERPOLATED PLOTTING:
arb_points = (np.random.rand(N_PARTICLES * 50, 3) - 0.5) * max_dist**0.5 * 10./9. * 2
narb = nsc.neighbors_arb(points, arb_points)
darb = nsc.density_arb(points, arb_points, mass, particle_type, narb)
ddarb = nsc.dust_density_arb(points, arb_points, mass, particle_type, sizes, narb)
tarb = nsc.temperature_arb(points, arb_points, mass, particle_type, T, narb)
dtarb = nsc.dust_temperature_arb(points, arb_points, mass, particle_type, sizes, T, narb)
parb = nsc.photoionization_arb(points, arb_points, mass, N_PART, photio, particle_type, narb)

fig, ax = plt.subplots(nrows=2, ncols = 2, sharex = True, sharey = True)
[plt.axis('equal')]
[ax[0][0].scatter(arb_points.T[1][darb!= 0]/constants.parsec, arb_points.T[2][darb!= 0]/constants.parsec, c = np.log10(darb[darb!= 0]/critical_density), s=30, alpha=0.01, edgecolor='none')]
ax00 = ax[0][0].scatter(arb_points.T[1][darb!= 0]/constants.parsec, arb_points.T[2][darb!= 0]/constants.parsec, c = np.log10(darb[darb!= 0]/critical_density), s=0, alpha=0.5, edgecolor='none')

[ax[1][0].scatter(arb_points.T[1][tarb != 0]/constants.parsec, arb_points.T[2][tarb != 0]/constants.parsec, c = np.log10(tarb[tarb != 0]), s=30, alpha=0.025, edgecolor='none')]
ax10 = ax[1][0].scatter(arb_points.T[1][tarb != 0]/constants.parsec, arb_points.T[2][tarb != 0]/constants.parsec, c = np.log10(tarb[tarb != 0]), s=0, alpha=0.5, edgecolor='none')

[ax[0][1].scatter(arb_points.T[1][ddarb != 0]/constants.parsec, arb_points.T[2][ddarb != 0]/constants.parsec, c = np.log10(ddarb[ddarb != 0]/critical_density), s=30, alpha=0.01, edgecolor='none')]
#to ensure equal bounds to the others
[ax[0][1].scatter(arb_points.T[1][tarb != 0]/constants.parsec, arb_points.T[2][tarb != 0]/constants.parsec, c = np.log10(tarb[tarb != 0]), s=0, alpha=0.005, edgecolor='none')]
ax01 = ax[0][1].scatter(arb_points.T[1][ddarb != 0]/constants.parsec, arb_points.T[2][ddarb != 0]/constants.parsec, c = np.log10(ddarb[ddarb != 0]/critical_density), s=0, alpha=0.5, edgecolor='none')

[ax[1][1].scatter(arb_points.T[1][dtarb != 0]/constants.parsec, arb_points.T[2][dtarb != 0]/constants.parsec, c = np.log10(dtarb[dtarb != 0]), s=30, alpha=0.01, edgecolor='none')]
ax11 = ax[1][1].scatter(arb_points.T[1][dtarb != 0]/constants.parsec, arb_points.T[2][dtarb != 0]/constants.parsec, c = np.log10(dtarb[dtarb != 0]), s=0, alpha=0.5, edgecolor='none')
[ax[1][1].scatter(arb_points.T[1][tarb != 0]/constants.parsec, arb_points.T[2][tarb != 0]/constants.parsec, c = np.log10(tarb[tarb != 0]), s=0, alpha=0.005, edgecolor='none')]

c00 = plt.colorbar(ax00, ax = ax[0][0])
c10 = plt.colorbar(ax10, ax = ax[1][0])
c01 = plt.colorbar(ax01, ax = ax[0][1])
c11 = plt.colorbar(ax11, ax = ax[1][1])

c00.set_label(r'$\log{(\rho/\rho_{crit})}$')
c10.set_label(r'$\log{(T / K)}$')
c01.set_label(r'$\log{(\rho/\rho_{crit})}$')
c11.set_label(r'$\log{(T / K)}$')

[qrst.scatter(points.T[1][particle_type == 1]/constants.parsec, points.T[2][particle_type == 1]/constants.parsec, c = 'black', s=(mass[particle_type == 1]/solar_mass) * 2, alpha=1) for qrst in ax.flatten()]

[qrst.set_xlabel(r'Position (parsec)') for qrst in ax[1]]
[qrst.set_ylabel(r'Position (parsec)') for qrst in ax.T[0]]
[ax[0][0].set_title(r'Gas density in H II region')]
[ax[1][0].set_title(r'Gas temperature in H II region')]
[ax[0][1].set_title(r'Dust density in H II region')]
[ax[1][1].set_title(r'Dust temperature in H II region')]
plt.suptitle(r'Post-supernova molecular cloud')

fig, ax = plt.subplots(nrows=1, ncols=2, sharex = True, sharey = True)
plt.axis('equal')
ax[0].scatter(arb_points.T[1][tarb != 0]/constants.parsec, arb_points.T[2][tarb != 0]/constants.parsec, c = np.log10(tarb[tarb != 0]), s=40, alpha=0.025, edgecolor='none')
ax0 = ax[0].scatter(arb_points.T[1][tarb != 0]/constants.parsec, arb_points.T[2][tarb != 0]/constants.parsec, c = np.log10(tarb[tarb != 0]), s=0, alpha=0.25, edgecolor='none')
ax[1].scatter(arb_points.T[1][parb > 0]/constants.parsec, arb_points.T[2][parb > 0]/constants.parsec, c = np.log10(parb[parb > 0]), s=40, alpha=0.025, edgecolor='none')
ax1 = ax[1].scatter(arb_points.T[1][parb > 0]/constants.parsec, arb_points.T[2][parb > 0]/constants.parsec, c = np.log10(parb[parb > 0]), s=0, alpha=0.25, edgecolor='none')
[qrst.scatter(points.T[1][particle_type == 1]/constants.parsec, points.T[2][particle_type == 1]/constants.parsec, c = 'black', s=(mass[particle_type == 1]/solar_mass) * 2, alpha=1) for qrst in ax.flatten()]

[qrst.set_xlabel(r'Position (parsec)') for qrst in ax]
ax[0].set_ylabel(r'Position (parsec)')

c0 = plt.colorbar(ax0, ax = ax[0])
c1 = plt.colorbar(ax1, ax = ax[1])
c0.set_label(r'$\log{(T / K)}$')
c1.set_label(r'$\log{f_{diss}}$')
[qrst.set(aspect='equal') for qrst in ax]
ax[0].set_title(r'Temperature in H II region')
ax[1].set_title(r'Photodissociation fraction in H II region')
plt.suptitle(r'H II region surrounding 20.01 $M_\odot$ star')
'''
