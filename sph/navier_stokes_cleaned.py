'''
Created on Aug 21, 2017

@author: umbut, dhruv
'''
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
species_labels = np.array(['H2', 'He', 'H','H+','He+','e-','Mg2SiO4','SiO2','C','Si','Fe','MgSiO3','FeSiO3', 'SiC'])
mu_specie = np.array([2.0159,4.0026,1.0079,1.0074,4.0021,0.0005,140.69,60.08,12.0107,28.0855,55.834,100.39,131.93, 40.096])
cross_sections = np.array([6.65e-24/5., 6.65e-24/5., 6.65e-23, 5e-60, 5e-60, 0., 0., 0., 0., 0., 0., 0., 0., 0.]) + 1e-80
destruction_energies = np.array([7.2418e-19, 3.93938891e-18, 2.18e-18, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000])
mineral_densities = np.array([1.e19, 1e19,1e19,1e19,1e19,1e19, 3320,2260,2266,2329,7870,3250,3250., 3166.])
sputtering_yields = np.array([0,0,0,0,0,0,0.137,0.295,0.137,0.295,0.137,0.137,0.137, 0.137])
f_u = np.array([.86,.14,0,0,0,0,0,0,0,0,0,0,0,0,0]) #relative abundance for species in each SPH particle, an array of arrays
gamma = np.array([7./5,5./3,5./3,5./3,5./3,5./3,15.6354113,4.913,1.0125,2.364,3.02,10.,10.,10.])#the polytropes of species in each SPH, an array of arrays
W6_constant = (3 * np.pi/80)
critical_density = 1000*amu*10**6 #critical density of star formation
crit_mass = 0.0001 * solar_mass #setting a minimum dust mass to help avoid numerical errors!

mrn_constants = np.array([50e-10, 5000e-10]) #minimum and maximum radii for MRN distribution

#DUMMY VALUES
'''DIAMETER = 1e6 * AU
N_PARTICLES = 15000
N_INT_PER_PARTICLE = 100
V = (DIAMETER)**3
d = (V/N_PARTICLES * N_INT_PER_PARTICLE)**(1./3.)
d_0 = 1e5 * AU
d_sq = d**2
specie_fraction_array = np.array([.86,.14,0,0,0,0,0,0,0,0,0,0,0])'''
	
#These are "constant functions" which you can just call once and forget. They encapsulate the results
#of previous literature about dust grain radius distribution and destruction in supernovae.

#There are also general utility functions that need to be called repeatedly, but which nevertheless
#contain either previous cited literature or otherwise established physics such as Planck's law.
def sigma_effective(mineral_densities, mrn_constants,mu_specie):
	#computes effective cross-sectional area per particle
	#assuming an MRN distribution. This gives a measure of
	#the optical activity of dust
	seff = mu_specie * amu/mineral_densities * (3./4.) * -np.diff(mrn_constants**-0.5)/np.diff(mrn_constants**0.5)
	return seff

def grain_mass(mineral_densities, mrn_constants):
	meff = mineral_densities * -np.diff(mrn_constants**0.5)/np.diff(mrn_constants**-2.5) * (4./5.) * 4 * np.pi/3.
	
	return meff

def mixed_CCSN_interpolate(): 
    progen_mass = np.array([0,20,25,30,35,40])
    #dust_mass_al = np.array([10**(-4),10**(-3),2*10**(-3),10**(-2),10**(-2)*8,10**(-1)])
    dust_mass_Fe = np.array([10**(-2),7*10**(-2),7*10**(-2),7*10**(-2),7*10**(-2),7*10**(-2)])
    dust_mass_MgSiO3 = np.array([2*10**(-2),5*10**(-2),6*10**(-2),9*10**(-3),0,0])
    dust_mass_Mg2SiO4 = np.array([6*10**(-2),10**(-1)*1.5,2*10**(-1),8*10**(-1),1,1])
    dust_mass_SiO2 = np.array([4*10**(-2),4*10**(-1),4.1*10**(-1),5*10**(-1),10**(-1)*8,10**(-1)*8])
    
    #f2 = interp1d(progen_mass, dust_mass_al, kind='linear')
    f3 = interp1d(progen_mass, dust_mass_Fe, kind='linear')
    f4 = interp1d(progen_mass, dust_mass_MgSiO3, kind='linear')
    f5 = interp1d(progen_mass, dust_mass_Mg2SiO4, kind='linear')
    f6 = interp1d(progen_mass, dust_mass_SiO2, kind='linear')
    
    def zero_function(v):
        return np.zeros(len(v));
    #(H2, He, H,H+,He+,e-,Mg2SiO4,SiO2,C,Si,Fe,MgSiO3,FeSiO3)
    #list of interpolation functions for each species
    return (zero_function, zero_function, zero_function, zero_function, zero_function, zero_function, f5, f6, zero_function, zero_function, f3, f4, zero_function, zero_function)

int_supernova = mixed_CCSN_interpolate(); #calling this beforehand 

def mixed_PISN_interpolate():
    progen_mass = np.array([140,170,200,225,250,260])
    dust_mass_al = np.array([7*10**(-2),6*10**(-2),3*10**(-2),10**(-2),0,0])
    dust_mass_Fe = np.array([0,4,10,12,15,20])
    dust_mass_MgSiO3 = np.array([2,6,10,12,15,20])
    dust_mass_Mg2SiO4 = np.array([7,4,2,0,0,0])
    dust_mass_SiO2 = np.array([5,20,30,55,60,70])
    
    f2 = interp1d(progen_mass, dust_mass_al, kind='linear')
    f3 = interp1d(progen_mass, dust_mass_Fe, kind='linear')
    f4 = interp1d(progen_mass, dust_mass_MgSiO3, kind='linear')
    f5 = interp1d(progen_mass, dust_mass_Mg2SiO4, kind='linear')
    f6 = interp1d(progen_mass, dust_mass_SiO2, kind='linear')
    

    xnew = np.arange(140,260)
    plt.plot(progen_mass, dust_mass_al, 'o', label = '$Al_2O_3$',color = 'blue')
    plt.plot(xnew, f2(xnew), '--', label = '$Al_2O_3$', color = 'red')
    plt.legend(['data', 'linear'], loc='best')
    
    plt.plot(progen_mass, dust_mass_Fe, 'x', label = '$Fe_3O_4$', color ='red')
    plt.plot(xnew, f3(xnew), '--',label = '$Fe_3O_4$',color = 'red')
    plt.legend(['data', 'linear'], loc='best')
    
    plt.plot(progen_mass, dust_mass_MgSiO3, '^',label = '$MgSiO_3$', color = 'green')
    plt.plot(xnew, f4(xnew), '--',label = '$MgSiO_3$', color = 'green')
    plt.legend(['data', 'linear'], loc='best')
    
    plt.plot(progen_mass, dust_mass_Mg2SiO4, 'x',color = 'brown')
    plt.plot(xnew, f5(xnew), '--',label = '$Mg_2SiO_4$', color = 'brown')
    plt.legend(['data', 'linear'], loc='best')
    
    plt.plot(progen_mass, dust_mass_SiO2, 'o',label = '$SiO_2$',color = 'cyan')
    plt.plot(xnew, f6(xnew), '--', label = '$SiO_2$', color = 'cyan')
    plt.legend(['data', 'linear'], loc='best')
    
    
    pylab.legend(loc='upper left')
    plt.xlabel("Progenitor Mass ($M_\odot$)")                        
    plt.ylabel("$M_{dust}$ ($M_\odot$)")
    plt.title("PISNe (mixed)")
    plt.show()
	
def interpolation_functions(): #interpolation of carbon and silicon destruction efficiency, returns the calculated e_c and e_s
    #velocities here in km/s
    v_s = np.array([0., 50., 75., 100., 125., 150., 175., 200.])
    e_c = np.array([0., 0.01,0.04,0.10,0.18,0.17,0.18,0.23])
    e_si = np.array([0., 0.02,0.09,0.23,0.41,0.40,0.42,0.40])
    
    carb_interpolation = interp1d(v_s, e_c, bounds_error=False, fill_value = 0.)
    silicon_interpolation = interp1d(v_s, e_si, bounds_error=False, fill_value = 0.)
    
    def zero_function(v):
    	return np.zeros(len(v));
    
    #list of interpolation functions for each species
    return (zero_function, zero_function, zero_function, zero_function, zero_function, zero_function, carb_interpolation, silicon_interpolation, carb_interpolation, silicon_interpolation, carb_interpolation, carb_interpolation, carb_interpolation, zero_function)

intf = interpolation_functions(); #calling this beforehand just so it's clear that it's set

def kroupa_imf(base_imf):
    coeff_0 = 1
    imf_final = np.zeros(len(base_imf))
    imf_final[base_imf < 0.08] = coeff_0 * base_imf[base_imf < 0.08]**-0.3
    coeff_1 = 2.133403503
    imf_final[(base_imf >= 0.08) & (base_imf < 0.5)] = coeff_1 * (base_imf/0.08)[(base_imf >= 0.08) & (base_imf < 0.5)]**-1.3
    coeff_2 = 0.09233279398
    imf_final[(base_imf >= 0.5)] = coeff_2 * coeff_1 * (base_imf/0.5)[(base_imf >= 0.5)]**-2.3 #fattened tail from 2.3!
    
    return (imf_final)
        
def luminosity_relation(base_imf, imf, lifetime=0):
    exp_radius = np.zeros(len(base_imf))
    coeff_radius = np.zeros(len(base_imf))
    exp_luminosity = np.zeros(len(base_imf))
    coeff_luminosity = np.zeros(len(base_imf))
    
    exp_radius[base_imf < 1.66] = 0.945
    exp_radius[base_imf >= 1.66] = 0.555
    coeff_radius[base_imf < 1.66] = 1.06
    coeff_radius[base_imf >= 1.66] = 1.2916594319
    
    exp_luminosity[base_imf < 0.7] = 2.62
    exp_luminosity[base_imf >= 0.7] = 3.92
    coeff_luminosity[base_imf < 0.7] = 0.35
    coeff_luminosity[base_imf >= 0.7] = 1.02
    
    #dimf = np.append(base_imf[0], np.diff(base_imf))
    
    luminosity_relation = coeff_luminosity * base_imf**exp_luminosity
    lifetime_relation = coeff_luminosity**(-1) * base_imf**(1 - exp_luminosity)
    lifetime_relation[lifetime_relation < 3.6e-4] = 3.6e-4
    #per Murray (2011), http://iopscience.iop.org/article/10.1088/0004-637X/729/2/133/meta,
    #stars have a minimum lifespan of about 3.6 Myr
    if lifetime == 0:
        return luminosity_relation
    if lifetime == 1:
        return lifetime_relation

def planck_law(temp, wl, dwl):
    emission = np.nan_to_num((2 * h * c**2 / wl**5)/(np.exp(h * c/(wl * k * temp)) - 1))
    emission[0] = 0
    #print emission
    norm_constant = np.sum(emission * dwl)
    return np.nan_to_num(emission/norm_constant)

def overall_spectrum(base_imf, imf):
    exp_radius = np.zeros(len(base_imf))
    coeff_radius = np.zeros(len(base_imf))
    exp_luminosity = np.zeros(len(base_imf))
    coeff_luminosity = np.zeros(len(base_imf))
    
    exp_radius[base_imf < 1.66] = 0.945
    exp_radius[base_imf >= 1.66] = 0.555
    coeff_radius[base_imf < 1.66] = 1.06
    coeff_radius[base_imf >= 1.66] = 1.2916594319
    
    exp_luminosity[base_imf < 0.7] = 2.62
    exp_luminosity[base_imf >= 0.7] = 3.92
    coeff_luminosity[base_imf < 0.7] = 0.35
    coeff_luminosity[base_imf >= 0.7] = 1.02
    
    dimf = 1.
    
    luminosity_relation = coeff_luminosity * base_imf**exp_luminosity
    temperature_relation = (coeff_luminosity/coeff_radius)**(1./4.) * base_imf**((exp_luminosity - exp_radius)/4.)
    wl = np.logspace(np.log10(wien/t_solar) - 5, np.log10(wien/t_solar) + 3, 10000)
    dwl = np.append(wl[0], np.diff(wl))
    planck_final = np.zeros(len(wl))
    
    for u in range(len(base_imf)):
        pl = np.nan_to_num(planck_law(temperature_relation[u] * t_solar, wl, dwl))
        #plt.plot(wl, pl, alpha=0.2)
        planck_final += np.nan_to_num(pl * imf[u] * luminosity_relation[u])
    
    total_luminosity = np.sum(imf * luminosity_relation * dimf)
    total_mass = np.sum(base_imf * imf * dimf)
    
    planck_final /= np.sum(dwl * planck_final)
    
    planck_final *= total_luminosity/total_mass * solar_luminosity/solar_mass
    
    return wl, planck_final, dwl

#### GRAVITATIONAL MODELING ####
#A VERY SIMPLE n-BODY SIMULATION THAT BREAKS THE CLOUD INTO A NUMBER OF BINS
#WHICH ARE INTENDED TO BE OF APPROXIMATELY EQUAL MASS. EACH OF THESE BINS THEN EXERTS A
#GRAVITATIONAL FORCE (dampened by the length scale of each bin) ON ALL OTHER ITEMS.

def grav_force_calculation(mass, points, sizes):
#NO LONGER USES A GRID METHOD
#DYNAMICALLY GENERATES SUB-CLUSTERS WHICH INTERACT WITH EACH OTHER USING A SMOOTHED GRAVITATIONAL FORCE
#WHILE INTERNAL FORCES ARE COMPUTED JUST AS N^2 INTERACTIONS
	pts_added = []
	pts_init = np.arange(len(points))
	kdt = spatial.cKDTree(points)
	uniq = 0
	for vwx in pts_init:
		#kdt = spatial.cKDTree(points[pts_init])
		distances, nbrs = kdt.query(points[vwx], int(np.sqrt(len(np.append(pts_init, 1))) + 1), distance_upper_bound = d)
		red_nbrs = np.array(nbrs)[np.array(distances) < d]
	
		pts_init_base = np.unique(np.append(pts_init, red_nbrs))
		pts_init_2 = np.unique(np.setdiff1d(pts_init_base, red_nbrs))
		uniq += len(pts_init) - len(pts_init_2)
		#print uniq
	
		pts_added.append(np.setdiff1d(pts_init, pts_init_2))
	
		pts_init = pts_init_2
	
	clusters = np.array(pts_added)[np.array([len(aide) for aide in pts_added]) != 0]
	n_parts = np.array([len(tree_el) for tree_el in clusters])
	total_mass = []
	center_of_mass = []
	squared_distance_com = []
	mean_size = []
	for xkz in clusters:
		total_mass.append(np.sum(mass[xkz]))
		center_of_mass.append(np.sum((points[xkz].T * mass[xkz]), axis=1)/np.sum(mass[xkz]))
		squared_distance_com.append(np.sum((points[xkz].T**2 * mass[xkz]), axis=1)/np.sum(mass[xkz]))
		mean_size.append(np.median(sizes[xkz]**3)**(1./3.))

	squared_distance_com = np.array(squared_distance_com)
	center_of_mass = np.array(center_of_mass)
	mean_size = np.array(mean_size)
	total_mass = np.array(total_mass)
	
	#use square_distance_com to evaluate smoothing length scale
	#smoothing_scale = (np.sum(squared_distance_com - center_of_mass**2, axis=1)**2 + mean_size**4)**0.25
	smoothing_scale = mean_size # * n_parts**(1./3.)
		
	grav_accels = np.zeros(shape=(len(points), 3))
	for int_cluster in np.arange(len(clusters)):
		cluster_el = clusters[int_cluster]
		dist_norm = (points - center_of_mass[int_cluster]).T/(np.sum((points - center_of_mass[int_cluster])**2, axis=1) + np.average(smoothing_scale**0.25)**8)**(3./2.)
		grav_accels.T[:] += -G * dist_norm * total_mass[int_cluster]
		grav_accels[cluster_el] += np.sum(G * dist_norm * mass, axis=1)
		#handles internal acceleration as well as a harmonic oscillator potential
	
	return grav_accels, center_of_mass
	
def bin_generator(masses, positions, subdivisions):
    #posmin = np.where(mass == min(mass))[0][0]
    positional_masses = np.array([masses[np.argsort(positions.T[b])] for b in range(len(subdivisions))])
    positional_masses = np.cumsum(positional_masses, axis = 1)
    
    subdivision_cutoffs = [np.sort(positions.T[c])[np.in1d(positional_masses[c], stats.scoreatpercentile(positional_masses[c], list(np.linspace(0., 100., subdivisions[c] + 1)), interpolation_method = 'lower'))] for c in range(len(subdivisions))]
    subdivision_intervals = [np.vstack([u[:-1], u[1:]]).T for u in subdivision_cutoffs]
    length_scale = np.sqrt(np.sum([np.average(np.diff(u)**2) for u in subdivision_cutoffs]))
    
    subdivision_boxes = [s for s in itertools.product(*subdivision_intervals)]
    object_indices = [[] for s in subdivision_boxes]
    
    position_indices = np.array(range(len(positions)))
    
    for q in range(len(subdivision_boxes)):
        for_removal = []
        for r in range(len(position_indices)):
            in_range = np.sum((np.array(subdivision_boxes[q]).T - positions[position_indices[r]]) < 0, axis=1)
            if np.array_equal(in_range, np.array([0, 3])) or np.array_equal(in_range, np.array([3, 0])):
                #print(r)
                object_indices[q].append(position_indices[r])
                for_removal.append(r)
        position_indices = np.delete(position_indices, np.array(for_removal))
        #print len(position_indices)
        
    com = []
    grid_masses = []
    for v in range(len(object_indices)):
    	if len(object_indices[v]) > 0:
    		com.append(np.sum(masses[np.array(object_indices[v]).astype('int')] * positions[np.array(object_indices[v]).astype('int')].T, axis=1)/np.sum(masses[np.array(object_indices[v]).astype('int')]))
    		grid_masses.append(np.sum(masses[np.array(object_indices[v]).astype('int')]))
    	else:
    		com.append(np.array([0,0,0]))
    		grid_masses.append(0)
    #com = [np.sum(masses[np.array(object_indices[v]).astype('int')] * positions[np.array(object_indices[v]).astype('int')].T, axis=1)/np.sum(masses[np.array(object_indices[v]).astype('int')]) for v in range(len(object_indices))]
    com = np.array(com)
    #grid_masses = np.array([np.sum(masses[np.array(object_indices[v]).astype('int')]) for v in range(len(object_indices))])
    grid_masses = np.array(grid_masses)
    
    return com, grid_masses, length_scale, np.array(subdivision_boxes), object_indices

def compute_gravitational_force(particle_positions, grid_com, grid_masses, length_scale):
    forces = 0;
    for i in range(len(grid_com)):
        forces += ((np.sum((particle_positions - grid_com[i]).T**2, axis=0) + (length_scale)**2)**-1.5 * -(particle_positions - grid_com[i]).T * grid_masses[i] * G)
        
    return (forces)

#### SMOOTHED PARTICLE HYDRODYNAMICS ####
#THESE FUNCTIONS ARE USED TO SIMULATE THE HYDRODYNAMICS OF OUR SYSTEM. THEY ARE BASED BOTH ON THE ESTABLISHED
#LITERATURE IN THE FIELD (Monaghan 1993 for artificial viscosity, for instance) AND SOME ORIGINAL CONTRIBUTIONS
#TO HANDLE DUST.

def neighbors(points, dist):
    kdt = spatial.cKDTree(points)  
    qbp = kdt.query_ball_point(points, dist, p=2, eps=0.1)
    
    return qbp

def Weigh2(x, x_0, m, d):
    norms_sq = np.sum((x - x_0)**2, axis=1)
    W = m * 315*(m_0/m)**3*((m/m_0)**(2./3.)*d**2-norms_sq)**3/(64*np.pi*d**9)
    return(W)

def Weigh2_dust(x, x_0, m, d, dustsize):
    norms_sq = np.sum((x - x_0)**2, axis=1)
    W = m * 315*(dustsize**2-norms_sq)**3/(64*np.pi*dustsize**9)
    return(W)
    
def grad_weight(x, x_0, m, d, type_particle):
    vec = (x - x_0)
    norms_sq = np.sum((x - x_0)**2, axis=1)
    
    W = -315 * 6 * (m_0/m)**3 / (64*np.pi*d**9) * ((m/m_0)**(2./3.)*d**2-norms_sq)**2 * (type_particle == 0) * vec.T
    
    #checks if SPH particles intersect or not
    return((np.nan_to_num(W.astype('float') * ((m/m_0)**(2./3.)*d**2-norms_sq > 0))).T)

def density(points,mass,particle_type,neighbor):
	final_density = np.zeros(len(points))
	for j in range(len(neighbor)):
		x_0 = points[j]
		x = np.array(points[np.array(neighbor[j])])
		m = np.array(mass[np.array(neighbor[j])])
	
		rho = Weigh2(x, x_0, m, d) * (particle_type[np.array(np.array(neighbor[j]))] == 0)
		final_density[j] += np.sum(rho[rho > 0])
	return final_density
    
def dust_density(points,mass,neighbor,particle_type,sizes):
	#evaluating dust density at each gas particle
	#including only neighboring points so that self-density of dust particles
	#is not counted
	final_dust_density = np.zeros(len(points))
	for j in range(len(neighbor)):
		x_0 = points[j]
		x = points[np.array(neighbor[j])]
		m = mass[np.array(neighbor[j])]
		ds = sizes[np.array(neighbor[j])]
	
		rho = Weigh2_dust(x, x_0, m, d, ds) * (particle_type[np.array(neighbor[j])] == 2)
		final_dust_density[j] += np.sum(rho[rho > 0])
	return final_dust_density
    
def net_impulse(points,mass,sizes,velocities,particle_type,neighbor,f_un):
	meff = grain_mass(mineral_densities, mrn_constants)
	seff = sigma_effective(mineral_densities, mrn_constants, mu_specie)
	
	accel_onto = np.zeros(points.shape)
	accel_reaction = np.zeros(points.shape)
	
	for j in range(len(neighbor)):
		mean_grainmass = np.sum(meff * f_un[np.array(neighbor[j])], axis=1)
		mean_cross = np.sum(seff * f_un[np.array(neighbor[j])], axis=1)
		net_vels = velocities[np.array(neighbor[j])] - velocities[j]
	
		x_0 = points[j]
		x = points[np.array(neighbor[j])]
		m = mass[np.array(neighbor[j])]
		ds = sizes[np.array(neighbor[j])]
	
		weight_factor = Weigh2_dust(x, x_0, m, d, ds)
		#gives units of m/s^2, which is exactly what we want
		accel_net = weight_factor/mean_grainmass * mean_cross * np.sum((net_vels.T)**2, axis=0)**0.5 * net_vels.T * (particle_type[np.array(neighbor[j])] == 2) * (weight_factor > 0)
		#need to account for relative weights over here!
		accel_onto[j] += np.sum(accel_net.T,axis=0)
		accel_reaction[neighbor[j]] += -accel_net.T
	return accel_onto, accel_reaction	
    
def num_dens(mass, points, mu_array, neighbor):
	number_density = np.zeros(len(points))
	for j in range(len(neighbor)):
		x_0 = points[j]
		x = np.array(points[np.array(neighbor[j])])
		m = np.array(mass[np.array(neighbor[j])])

		n_dens = Weigh2(x, x_0, m, d)/(mu_array[np.array(neighbor[j])] * m_h)
		number_density[j] += np.sum(n_dens[n_dens > 0])
	return number_density
    
def del_pressure(points,mass,particle_type,neighbor,E_internal, gamma_array): #gradient of the pressure
	grad_pressure = np.zeros(points.shape)
	for i in range(len(neighbor)):
		if (particle_type[i] == 0):
			x_0 = points[i]
			x = np.array(points[np.array(neighbor[i])])
			m = np.array(mass[np.array(neighbor[i])])
			pt = np.array(particle_type[neighbor[i]])
		
			#symmetrizing pressure gradient
			gw = (grad_weight(x, x_0, m, d, pt) + grad_weight(x, x_0, m[0], d, pt))/2.
			#print gw
		
			del_pres = np.sum((gw.T * E_internal[np.array(neighbor[i])]/gamma_array[np.array(neighbor[i])]).T, axis=0)
			grad_pressure[i] += del_pres
	#print grad_pressure
	return grad_pressure

def crossing_time(neighbor, velocities, sizes, particle_type):
	max_relvel = np.zeros(len(neighbor))
	neighbor_lengths = np.array([len(a) for a in neighbor])
	j_range = np.arange(len(neighbor))
	max_relvel[neighbor_lengths > 0] = np.array([np.max(np.sum((velocities[neighbor[j]] - velocities[j])**2, axis=1))**0.5 for j in j_range[neighbor_lengths > 0]])
	
	crossing_time = np.nan_to_num(sizes/max_relvel) * (particle_type == 0)
	if len(crossing_time[crossing_time != 0]) == 0:
		return dt_0/10.
	else:
		return min(crossing_time[crossing_time != 0]) + 0.0001

def artificial_viscosity(neighbor, points, particle_type, sizes, mass, densities, velocities, T, gamma_array, mu_array):
	css_base = np.nan_to_num((gamma_array * k * T/(mu_array * amu))**0.5)
	visc_accel = np.zeros((len(points),3))
	visc_heat = np.zeros(len(points))
	for j in np.arange(len(neighbor))[particle_type[np.arange(len(neighbor))] == 0]:
		if (np.sum(particle_type[neighbor[j]] == 0) > 0):
			w_ij = np.sum((velocities[neighbor[j]] - velocities[j]) * (points[neighbor[j]] - points[j]), axis=1)/np.sum((points[neighbor[j]] - points[j])**2, axis=1)**0.5
			w_ij[w_ij > 0] = 0
			w_ij = np.nan_to_num(w_ij)
			c_sound_sum = css_base[neighbor[j]] + css_base[j]
			vsig_ij = c_sound_sum - 3 * w_ij
		
			rho_ij = (densities[neighbor[j]] + densities[j])/2.
		
			PI_ij = -1./2. * vsig_ij * w_ij/rho_ij
			#symmetrizing pressure
			gw0 = grad_weight(points[neighbor[j]], points[j], mass[j], d, particle_type[neighbor[j]])
			gw1 = grad_weight(points[neighbor[j]], points[j], mass[neighbor[j]], d, particle_type[neighbor[j]])
			gw = (gw0 + gw1)/2.
			accel_ij = (mass[neighbor[j]] * PI_ij * gw.T).T
			heat_ij = 0.5 * mass[neighbor[j]] * PI_ij * np.sum((velocities[neighbor[j]] - velocities[j]) * gw, axis=1)
		
			accel_i = np.sum(accel_ij[(particle_type[neighbor[j]] == 0)],axis=0)
			heat_i = np.sum(heat_ij[particle_type[neighbor[j]] == 0])
			visc_accel[j] = accel_i
			visc_heat[j] = heat_i
	
	return visc_accel, visc_heat
    
def supernova_explosion(mass,points,velocities,E_internal,supernova_pos, f_un):
    #should make parametric later, and should include Nozawa 03 material
    #supernova_pos = np.arange(len(star_ages))[(star_ages > luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1) * year * 10e10)]
    gas_release = f_un[supernova_pos]
    grsum = np.sum(gas_release)
    
    dust_release = np.array([u(mass[supernova_pos]/solar_mass) for u in int_supernova]).T #int_supernova is the already called supernova interpolation from Nozawa
    drsum = np.sum(dust_release)
    #dust_release[0] = 1e-15
    #dust_release[1] = 1e-15
    
    total_release = dust_release+gas_release
    trsum = np.sum(total_release)
	
    gas_release /= grsum
    dust_release /= drsum
    
    dust_masses = (mass[supernova_pos] - 2. * solar_mass) * drsum/trsum
    gas_masses = (mass[supernova_pos] - 2. * solar_mass) * grsum/trsum
    
    supernova_dust_len = int(np.sum(dust_masses)/(0.05 * solar_mass))
    print supernova_dust_len
    
    dust_comps = (np.vstack([dust_release] * supernova_dust_len).T).T
    gas_comps = (np.vstack([gas_release] * len(supernova_pos)).T).T
    star_comps = (np.vstack([gas_release] * len(supernova_pos)).T).T
    
    dust_mass = np.ones(supernova_dust_len) * np.sum(dust_masses)/supernova_dust_len
    gas_mass = gas_masses
    stars_mass = (np.ones(len(supernova_pos)) * 2 * solar_mass)
    
    dustpoints = np.vstack([points[supernova_pos][0]] * supernova_dust_len) + (np.random.normal(size = (supernova_dust_len,3))) * d_0
    newpoints = points[supernova_pos] + (np.random.normal(size = (len(points[supernova_pos]),3))) * d_0
    dustvels = np.vstack([velocities[supernova_pos][0]] * supernova_dust_len)
    newvels = velocities[supernova_pos]
    #newaccel = accel[supernova_pos]
    
    newgastype = np.zeros(len(supernova_pos))
    newdusttype = np.ones(supernova_dust_len) * 2
    
    new_eint_stars = np.zeros(len(supernova_pos))
    new_eint_dust = np.zeros(supernova_dust_len)
    new_eint_gas = E_internal[supernova_pos]/2.
    
    return dust_comps, gas_comps, star_comps, dust_mass, gas_mass, stars_mass, newpoints, newvels, newgastype, newdusttype, new_eint_stars, new_eint_dust, new_eint_gas, supernova_pos, dustpoints, dustvels

#### RADIATIVE TRANSFER ####
#THE MOST CHALLENGING PART TO CODE---USES THE SIMPLIFYING ASSUMPTION THAT ABSORPTION CROSS-SECTIONS
#OF EACH SPECIES ARE CONSTANT ACROSS ALL TEMPERATURE, BUT DOES DYNAMICALLY ALTER THE COMPOSITION OF
#SPH PARTICLES TO CHANGE THEIR OVERALL OPACITIES AS STARS HEAT THEM UP.

#THE ENERGY/DUST PRODUCTION IMPARTED BY SUPERNOVAE ARE ALSO PLACED HERE.

def rad_heating(positions, ptypes, masses, sizes, cross_array, f_un, supernova_pos, mu_array, T, dt):
    random_stars = positions[ptypes == 1]
    rs2 = np.array([])
    rs2 = random_stars
    if len(random_stars > 0):
        star_selection = (np.random.rand(len(random_stars)) < (masses[ptypes == 1]/(4 * solar_mass)).astype('float') + len(random_stars)**-0.5 ) #selecting all stars over 4 solar masses for proper H II regions
        rs2 = random_stars[star_selection]
        if len(rs2) <= 0:
        	star_selection = np.arange(0, len(random_stars))
        	rs2 = random_stars
    
    suppos = np.zeros(len(masses))
    suppos[supernova_pos] += 1            
    lum_base = luminosity_relation(masses[ptypes == 1][star_selection]/solar_mass, 1) + 0.264 * supernova_energy/solar_luminosity/dt * suppos[ptypes == 1][star_selection]
    tot_luminosity = np.sum(luminosity_relation(masses[ptypes == 1]/solar_mass, 1))
    
    luminosities = lum_base/np.sum(lum_base) * tot_luminosity
    
    random_gas = positions[ptypes != 1]
    rg2 = np.array([])
    if len(random_gas > 0):
        gas_selection = (np.random.rand(len(random_gas)) < len(random_gas)**-0.5)
        rg2 = random_gas[gas_selection]
        if len(rs2) == 0:
            rg2 = random_gas
            
    blocked = np.zeros((len(rs2), len(rg2)))
    star_distance = np.zeros((len(rs2), len(rg2)))
    for i in np.arange(0, len(rs2)):
        for j in np.arange(0, len(rg2)):
            ray = np.array([rs2[i], rg2[j]])
            dists = (np.sum(np.cross(positions - ray[0], ray[1] - ray[0])**2, axis=1)/(np.sum((ray[1] - ray[0])**2)))
            blocking_kernel = np.sum((W6_constant * sizes[dists < sizes**2]**(-2) * cross_array[dists < sizes**2] * masses[dists < sizes**2]/(mu_array[dists < sizes**2] * amu)))
            
            blocked[i][j] += blocking_kernel
            star_distance[i][j] = np.sum((ray[1] - ray[0])**2)**0.5
    
    star_distance_2 = np.zeros((len(rs2), len(random_gas)))
    gas_distance = np.zeros((len(rg2), len(random_gas)))
    
    star_unit_norm = []
    
    for q in np.arange(0, len(rg2)):
        gas_distance[q] = np.sum((random_gas - rg2[q])**2, axis = 1)**0.5
    
    for r in np.arange(0, len(rs2)):
        star_distance_2[r] = np.sum((random_gas - rs2[r])**2, axis = 1)**0.5
        star_unit_norm.append(((random_gas - rs2[r]).T/star_distance_2[r]).T)
    
    lum_factor = []
    for ei in range(len(star_distance)):
        lum_factor.append(np.nan_to_num(star_distance_2[ei] * np.sum(((gas_distance + sizes[ptypes != 1]/2).T**-2/star_distance[ei] * blocked[ei]).T, axis=0)/np.sum((gas_distance + sizes[ptypes != 1]/2)**-2, axis=0)))
    
    lum_factor = np.array(lum_factor)
    extinction = (W6_constant * masses[ptypes != 1]/(mu_array[ptypes != 1] * amu) * cross_array[ptypes != 1]) * sizes[ptypes != 1]**(-2)
    
    exponential = np.exp(-np.nan_to_num(lum_factor))
    distance_factor = (np.nan_to_num(star_distance_2)**2 + np.average(sizes[ptypes != 1])**2 * np.ones(np.shape(star_distance_2))) * 4. * np.pi
    a_intercepted = (np.pi * sizes**2)[ptypes != 1]
    
    lum_factor_2 = ((exponential/distance_factor).T * luminosities).T * a_intercepted * (np.ones(np.shape(extinction)) - np.exp(-extinction))
    lum_factor_2 = np.nan_to_num(lum_factor_2)
    
    lf2 = np.sum(lum_factor_2, axis=0) * dt * solar_luminosity
    
    momentum = (np.sum(np.array([star_unit_norm[ak].T * lum_factor_2[ak] for ak in range(len(lum_factor_2))]), axis=0)/masses[ptypes != 1]).T * dt/c
    
    overall = overall_spectrum(masses[ptypes == 1]/solar_mass, np.ones(len(masses[ptypes == 1])))
    dest_wavelengths = destruction_energies**(-1) * (h * c)
    frac_dest = np.array([np.sum((overall[1] * overall[2] * overall[0])[overall[0] < dest_wavelengths[ai]]) for ai in np.arange(len(dest_wavelengths))])
    #frac_dest += np.array([np.sum((overall[1] * overall[2] * (overall[0] - dest_wavelengths[ai]))[overall[0] > dest_wavelengths[ai]]) for ai in np.arange(len(dest_wavelengths))])
    total_em = np.sum(overall[1] * overall[2] * overall[0])
    
    frac_dest /= total_em
    
    #Only gas particles suffer from ionization, not dust particles
    n_photons = np.sum(overall[0] * overall[1] * overall[2]/(h * c))/(np.sum(overall[1] * overall[2])) * lf2
    mols = masses[ptypes != 1]/np.sum((f_un[ptypes != 1] * mu_specie * amu), axis=1)
    
    weighted_interception = ((f_un * cross_sections).T/np.sum(cross_sections * f_un, axis=1)).T
    atoms_destroyed = ((weighted_interception * frac_dest)[ptypes != 1].T * n_photons).T
    atoms_total = (f_un[ptypes != 1].T * mols).T
    
    frac_destroyed_0 = atoms_destroyed/atoms_total
    #print frac_destroyed_0
    frac_destroyed_1 = np.exp(np.nan_to_num(-atoms_destroyed/atoms_total))
    frac_destroyed_by_species = 1. - frac_destroyed_1
    #print frac_destroyed_by_species
    
    new_fun = f_un.T
    new_fun2 = copy.deepcopy(new_fun)
    
    #forward reactions
    new_fun2[2][ptypes != 1] += new_fun[0][ptypes != 1] * frac_destroyed_by_species.T[0] * 2.
    new_fun2[3][ptypes != 1] += new_fun[2][ptypes != 1] * frac_destroyed_by_species.T[2]
    new_fun2[4][ptypes != 1] += new_fun[1][ptypes != 1] * frac_destroyed_by_species.T[1]
    
    new_fun2[5][ptypes != 1] += new_fun[2][ptypes != 1] * frac_destroyed_by_species.T[2]
    new_fun2[5][ptypes != 1] += new_fun[1][ptypes != 1] * frac_destroyed_by_species.T[1]
    
    new_fun2[0][ptypes != 1] -= new_fun[0][ptypes != 1] * frac_destroyed_by_species.T[0]
    new_fun2[1][ptypes != 1] -= new_fun[1][ptypes != 1] * frac_destroyed_by_species.T[1]
    new_fun2[2][ptypes != 1] -= new_fun[2][ptypes != 1] * frac_destroyed_by_species.T[2]
    
    new_fun2 /= np.sum(new_fun2,axis=0)    
    #energy, composition change, impulse
    return lf2, new_fun2.T, momentum

def rad_cooling(positions, particle_type, masses, sizes, cross_array, f_un, neighbor, mu_array, T, dt):
	#first, calculate an SPH density of electrons * temperature factor, then apply the model
	#of Draine (2011)
	
	final_comp = copy.deepcopy(np.nan_to_num(f_un.T))
	rec_array = copy.deepcopy(np.nan_to_num(f_un.T)) * 0
	rel_array = sizes * 0.
	energy_array = copy.deepcopy(np.nan_to_num(f_un.T)) * 0
	jarr = np.arange(len(neighbor))[particle_type[np.arange(len(neighbor))] == 0]
	for j in jarr:
		if np.sum(particle_type[neighbor[j]] == 0) > 0: #making sure that there are gaseous neighbors
			x = positions[neighbor[j]]
			m = masses[neighbor[j]]
			x_0 = positions[j]
			comps = f_un[neighbor[j]]
			comp_0 = f_un[j]
			temps = T[neighbor[j]]
			muloc = mu_array[neighbor[j]]
			t4 = (temps/10000.)
			
			#Can linearize the Draine model with SPH for the temperature * coeff * electron number density
			#but independently acting on each particle. This works because electron thermal velocity is much
			#faster than that of anything else
	
			n_e = np.nan_to_num(Weigh2(x, x_0, m, d)/(muloc * m_h) * comps.T[5] * (particle_type[neighbor[j]] == 0))
			n_e *= n_e > 0 #times electron concentration
			
			rel_weights = np.nan_to_num(Weigh2(x, x_0, m, d)) * (particle_type[neighbor[j]] == 0)
			rel_weights *= (rel_weights > 0)/Weigh2(x, x, m, d)
			
			n_H_plus =  np.nan_to_num(Weigh2(x, x_0, m, d)/(muloc * m_h) * comps.T[3] * (particle_type[neighbor[j]] == 0))
			n_H_plus *= (n_H_plus > 0)
			
			n_He_plus =  np.nan_to_num(Weigh2(x, x_0, m, d)/(muloc * m_h) * comps.T[4] * (particle_type[neighbor[j]] == 0))
			n_He_plus *= (n_He_plus > 0)
			
			n_H_neutral = np.nan_to_num(Weigh2(x, x_0, m, d)/(muloc * m_h) * comps.T[2] * (particle_type[neighbor[j]] == 0))
			n_H_neutral *= (n_H_neutral > 0)		
			
			#tot_e = Weigh2(x, x_0, m, d)/(muloc * m_h) * (particle_type[neighbor[j]] == 0) #total number density
			H_effect = np.nan_to_num(4.13e-19 * t4**(-0.7131 - 0.0115 * np.log(t4)) * (particle_type[neighbor[j]] == 0))
			He_effect = np.nan_to_num(2.72e-19 * t4**(-0.789) * (particle_type[neighbor[j]] == 0))
			H2_effect = np.nan_to_num(7.3e-23 * 0.5 * (temps/100)**0.5 * (particle_type[neighbor[j]] == 0))
			energy_coeff_H = np.nan_to_num((0.684 - 0.0416 * np.log(t4/1) + 0.54 * t4**(0.37)) * k * temps * (particle_type[neighbor[j]] == 0))
			energy_coeff_He = np.nan_to_num((0.684 - 0.0416 * np.log(t4/4)) * k * temps * (particle_type[neighbor[j]] == 0))
			#print energy_coeff_H, energy_coeff_He
			num_e = np.sum(n_e[n_e > 0])
			num_H_plus = np.sum(n_H_plus[n_H_plus > 0])
			num_He_plus = np.sum(n_He_plus[n_He_plus > 0])
			num_H_neut = np.sum(n_H_neutral[n_H_neutral > 0])
			
			#print num_e, num_H_plus, num_He_plus
			
			H_effect_rec = np.sum((H_effect * n_e)[n_e > 0])
			He_effect_rec = np.sum((He_effect * n_e)[n_e > 0])
			H_neut_effect_rec = np.sum((H2_effect * n_H_neutral)[n_H_neutral > 0])
			
			#print H_effect_rec, He_effect_rec
			
			H_effect_energy = np.sum((H_effect * n_e * energy_coeff_H)[n_e > 0])
			He_effect_energy= np.sum((He_effect * n_e * energy_coeff_He)[n_e > 0])
			#print H_effect_energy, He_effect_energy
			
			frac_rec_e = (H_effect_rec * num_H_plus + He_effect_rec * num_He_plus)/num_e * dt
			frac_rec_e = min(frac_rec_e, 0.999)
			
			#print frac_rec_e
			frac_rec_H = frac_rec_e * np.nan_to_num((H_effect_rec)/(H_effect_rec + He_effect_rec))
			frac_rec_He = frac_rec_e * np.nan_to_num((He_effect_rec)/(H_effect_rec + He_effect_rec))
			#print frac_rec_e, frac_rec_H, frac_rec_He
			
			frac_rec_H_neut = H_neut_effect_rec * num_H_neut/num_H_neut * dt
			frac_rec_H_neut = np.min(frac_rec_H_neut, 0.999)
			
			energy_rec_H = np.nan_to_num(H_effect_energy * dt)
			energy_rec_He = np.nan_to_num(He_effect_energy * dt)
			
			energy_array[3][neighbor[j]] += np.nan_to_num(energy_rec_H * rel_weights * (rel_weights > 0)/np.sum(rel_weights * (rel_weights > 0)))
			energy_array[4][neighbor[j]] += np.nan_to_num(energy_rec_He * rel_weights * (rel_weights > 0)/np.sum(rel_weights * (rel_weights > 0)))
						
			rec_array[2][neighbor[j]] += np.nan_to_num(frac_rec_H_neut) * (n_H_neutral > 0) * rel_weights * (rel_weights > 0)
			rec_array[3][neighbor[j]] += np.nan_to_num(frac_rec_H) * (n_e > 0) * rel_weights * (rel_weights > 0)
			rec_array[4][neighbor[j]] += np.nan_to_num(frac_rec_He) * (n_e > 0) * rel_weights * (rel_weights > 0)
			rec_array[5][neighbor[j]] += np.nan_to_num(frac_rec_e) * (n_e > 0) * rel_weights * (rel_weights > 0)
			
			rel_array[neighbor[j]] += rel_weights * (rel_weights > 0)
	
	
	rec_array /= (rel_array + 1e-90)
	energy_array /= (rel_array + 1e-90)
	#rec_array /= np.sum(rec_array/0.999, axis=0)/2
	rec_array = np.nan_to_num(rec_array)
	#print np.min(rec_array), np.max(rec_array)
	
	H2_plus_frac = f_un.T[2] * rec_array[2]
	H_plus_frac = f_un.T[5] * rec_array[3]
	He_plus_frac = f_un.T[5] * rec_array[4]
	elec_frac = f_un.T[5] * rec_array[5]
	
	energy = np.nan_to_num(np.sum(final_comp * energy_array, axis=0))
	
	final_comp[0] += H2_plus_frac/2.
	final_comp[1] += He_plus_frac
	final_comp[2] += H_plus_frac - H2_plus_frac
	final_comp[3] += -H_plus_frac
	final_comp[4] += -He_plus_frac
	final_comp[5] += -elec_frac
	
	final_comp /= np.sum(final_comp, axis=0)
	
	return final_comp.T, energy

def supernova_impulse(points, masses, supernova_pos, ptypes):
	mass_displaced = 194.28 * solar_mass #calculated from paper
	s_basis = np.sum((points - points[supernova_pos])**2, axis=1)
	ptype_s = ptypes[np.argsort(s_basis)]
	sorted_points = np.argsort(s_basis)[ptype_s == 0]
	selected_points = sorted_points[(np.cumsum(masses[sorted_points]) < mass_displaced)]
	true_mass_displaced = np.sum(masses[selected_points])
	
	vel = (2 * supernova_energy * 0.736/true_mass_displaced)**0.5
	d_vels = (points[selected_points] - points[supernova_pos]).T/np.sum((points[selected_points] - points[supernova_pos])**2, axis=1)**0.5 * vel
	#thermal energy should be added by radiative transfer
	
	return d_vels.T, selected_points

#### PHYSICS OF DUST ####
# VERY SIMILAR IN PURPOSE TO THE SPH SECTION, BUT WITH A SPECIAL EMPHASIS ON DUST.			 								 
def supernova_destruction(points, velocities, neighbor, mass, f_un, mu_array, sizes, densities, particle_type):
	#Indexes over all gas particles and sees if they intersect a dust,
	#rather than the other way around as previously, because gas particles
	#are much smaller than dust particles, by design
	
	#Calculate an old and new mass, then normalize f_un. Without relative changes in mass
	#this is pointless
	frac_destruction = copy.deepcopy(f_un * 0.)
	frac_reuptake = copy.deepcopy(f_un * 0.)
	jarr = np.arange(len(neighbor))[particle_type[np.arange(len(neighbor))] == 0] #setting up array where all particles are gas
	for j in jarr:
		if (np.sum(particle_type[np.array(neighbor[j])] == 2) > 0): #making sure that this gas particle has dusty neighbors!
			#no need to append j to this, because j is included as its own neighbor since it has a distance of 0 from itself
			x = points[neighbor[j]]
			m = mass[neighbor[j]]
			x_0 = points[j]
			comps = f_un[j]
			dustsize = sizes[neighbor[j]]
			dens = densities[j]
			
			#Density of dust at the center of the gas particle
			w2d = Weigh2_dust(x, x_0, m, d, dustsize)
			#density of dust at the center of the dust particles
			w2_max = Weigh2_dust(x, x, m, d, dustsize)
			
			#density of dust at the selected gas particle
			rho = w2d * (w2d > 0) * (particle_type[neighbor[j]] == 2);
			rho_base = w2_max
			if np.sum(rho) > 0:
				#Obtaining relative velocities between gas/dust, and destruction efficiency
				vels = np.sum((velocities[neighbor[j]] - velocities[j])**2, axis=1)**0.5 * (particle_type[np.array(neighbor[j])] == 2)
				dest_fracs = (np.array([u(vels/1000.) for u in intf]) * np.nan_to_num(rho/rho_base)).T
				#Distributing dust destruction over all intersecting dust particles
				final_fracs = dens/critical_density * dest_fracs.T #fraction destroyed
				final_fracs[final_fracs >= 0.99] = 0.99
				
				N_dust = mass[neighbor[j]]/(mu_array[neighbor[j]] * amu)
				
				#what relative fraction of refractory species are created in gas particle j? Summed over because only one particle
				#Conversely, how much dust is fractionally lost from each intersecting gas particle?
				dust_lost = (final_fracs * f_un[neighbor[j]].T * N_dust).T
				refractory_fracs = np.sum(dust_lost,axis=0)
				#print np.nan_to_num(dust_lost/refractory_fracs)
				#Dust lost in each dust particle, which is taken up as refractory gas by the gas particle
				frac_destruction[neighbor[j]] += dust_lost * (mass[j] > crit_mass)
				frac_reuptake[j] += refractory_fracs * (mass[j] > crit_mass)
				
	frac_destruction = np.nan_to_num((frac_destruction.T/(mass/(mu_array * amu))).T)
	frac_reuptake = np.nan_to_num((frac_reuptake.T/(mass/(mu_array * amu))).T)
				
	return frac_destruction, frac_reuptake			
			
def chemisputtering_2(points, neighbor, mass, f_un, mu_array, sizes, T, particle_type):
	num_particles = (f_un.T * (mass/mu_array)).T
	jarr = np.arange(len(neighbor))[particle_type == 2]
	for j in jarr:
		if (np.sum(particle_type[neighbor[j]] == 0) > 0):
			m = mass[neighbor[j]]
			x = points[neighbor[j]]
			x_0 = points[j]
			comps = f_un[neighbor[j]]
			dustsize = sizes[neighbor[j]]
			mu_local = mu_array[neighbor[j]]
			T_local = T[neighbor[j]]
			T_local[particle_type[neighbor[j]] == 1] = 2.73
			
			w2g_num = Weigh2(x, x_0, m, d)/(mu_local)
			w2d = Weigh2_dust(x, x_0, m, d, dustsize)/(mu_local)
			
			rel_w2g = Weigh2(x, x_0, m, d)/Weigh2(x, x, m, d)
			rel_w2d = Weigh2_dust(x, x_0, m, d, dustsize)/Weigh2_dust(x, x, m, d, dustsize)
			
			w2g_num *= (w2g_num > 0) * (particle_type[neighbor[j]] == 0)
			w2d *= (w2d > 0) * (particle_type[neighbor[j]] == 2)
			
			rel_w2g *= (w2g_num > 0) * (particle_type[neighbor[j]] == 0)
			rel_w2d *= (w2d > 0) * (particle_type[neighbor[j]] == 2)
			if np.sum(w2g_num) > 0:
				sph_indiv_composition = (w2g_num * comps.T).T * mu_specie
				sph_composition_density = np.sum((w2g_num * comps.T).T,axis=0) * mu_specie #SPH density by composition of GAS
			
				sph_temperature = np.sum((w2g_num * T_local))/np.sum(w2g_num)
			
				dust_indiv_composition = (w2d * comps.T).T * mu_specie
				dust_composition = np.sum((w2d * comps.T).T,axis=0) * mu_specie #SPH density of DUST
			
				J_u = -np.diff(mrn_constants**-0.5)/np.diff(mrn_constants**0.5)/(3 * mineral_densities)
				J_u *= (k * sph_temperature/(2 * np.pi * mu_specie * amu))**0.5
			
				Y_H = min(max(0.5 * np.exp(-4600/sph_temperature), 1e-7),1e-3) * sputtering_yields/max(sputtering_yields)
				Y_He = min(max(5 * np.exp(-4600/sph_temperature), 1e-6),1e-2) * sputtering_yields/max(sputtering_yields)
				#cannot sputter more mass than exists!
				sput_y = sph_composition_density[3] * Y_H + sph_composition_density[4] * Y_He
				sput_y[sput_y > dust_composition] = dust_composition[sput_y > dust_composition]
				K_u = sph_composition_density + dust_composition - sput_y
				
				L_u = sph_composition_density + dust_composition * np.exp(K_u * J_u * dt) - sput_y
				#yields are for ions!
			
				F_sput = K_u/L_u
				F_sput *= np.exp(K_u * J_u * dt)
				F_sput[np.isnan(F_sput)] = 1.
				
				#print F_sput - 1
				#effective_mass = -(sph_indiv_composition - np.outer(sph_indiv_composition.T[3], Y_H) - np.outer(sph_indiv_composition.T[4], Y_He))
				effective_mass = sph_indiv_composition
						
				reuptake_length = np.sum((w2g_num > 0) & (particle_type[neighbor[j]] == 0))
				reuptake_weight = effective_mass/np.sum(effective_mass,axis=0)
				reuptake_weight[np.isnan(reuptake_weight)] = 1./reuptake_length
				reuptake_weight = (reuptake_weight.T * ((w2g_num > 0) & (particle_type[neighbor[j]] == 0))).T
				
				new_particles = (((F_sput - 1.) * num_particles[neighbor[j]]).T * (w2d > 0)).T
				new_particles[new_particles < 0.] = 0.
				particle_loss = np.sum(new_particles, axis=0) * reuptake_weight
				ploss = copy.deepcopy(particle_loss)
				ploss[ploss > num_particles[neighbor[j]]] = num_particles[neighbor[j]][ploss > num_particles[neighbor[j]]]
				
				scale_f = np.sum(ploss, axis=0)/np.sum(new_particles, axis=0)
				scale_f[np.isnan(scale_f)] = 1.
				
				num_particles[neighbor[j]] += np.nan_to_num(new_particles * scale_f - ploss)
	
	mass_new = np.sum(num_particles * mu_specie,axis=1)
	f_un_new = (num_particles.T/np.sum(num_particles,axis=1)).T
	f_un_new[f_un_new < 0] == 0
	
	return mass_new, f_un_new
	
#### ARBITRARY INTERPOLATIONS #####
#To be used after all methods are verified to obtain very high-resolution
#graphs for publication. Extremely slow and time-consuming, so should not
#be used in real time---these are just one-shots.										 
def neighbors_arb(points, arb_points):
    kdt = spatial.cKDTree(points)  
    qbp = kdt.query_ball_point(arb_points, d, p=2, eps=0.1)
    
    return qbp

def density_arb(points, arb_points, mass, particle_type, narb):
    density_array = []
    #narb = neighbors_arb(points, arb_points)
    for j in range(len(arb_points)):
        if len(narb[j]) > 1:
            x_0 = arb_points[j]
            #print np.array(narb[j])
            x = points[np.array(narb[j])]
            m = mass[np.array(narb[j])]
            rho = Weigh2(x, x_0, m, d) * (particle_type[np.array(narb[j])] == 0)
        
            density_array.append(np.sum(rho[rho > 0]))
        else:
            density_array.append(0)
            
    return np.array(density_array)

def dust_density_arb(points, arb_points, mass, particle_type, sizes, narb):
    density_array = []
    #narb = neighbors_arb(points, arb_points)
    for j in range(len(arb_points)):
        if len(narb[j]) > 1:
            x_0 = arb_points[j]
            #print np.array(narb[j])
            x = points[np.array(narb[j])]
            m = mass[np.array(narb[j])]
            ds = sizes[np.array(narb[j])]
            rho = Weigh2_dust(x, x_0, m, d, ds) * (particle_type[np.array(narb[j])] == 2)
        
            density_array.append(np.sum(rho[rho > 0]))
        else:
            density_array.append(0)
            
    return np.array(density_array)

def temperature_arb(points, arb_points, mass, particle_type, T, narb):
    density_array = []
    #narb = neighbors_arb(points, arb_points)
    for j in range(len(arb_points)):
        if len(narb[j]) > 1:
            x_0 = arb_points[j]
            #print np.array(narb[j])
            x = points[np.array(narb[j])]
            m = mass[np.array(narb[j])]
            temps = T[np.array(narb[j])]
        
            temp_sum = Weigh2(x, x_0, m, d) * (particle_type[np.array(narb[j])] == 0) * temps
            temp_w = Weigh2(x, x_0, m, d) * (particle_type[np.array(narb[j])] == 0)
            
            #print temp_sum
            #print temp_w
            
            tsgo = np.nan_to_num(np.sum(temp_sum[temp_sum > 0])/np.sum(temp_w[temp_sum > 0]))
            
            density_array.append(tsgo)
        else:
            density_array.append(0)
            
    return np.array(density_array)

def dust_temperature_arb(points, arb_points, mass, particle_type, sizes, T, narb):
    density_array = []
    #narb = neighbors_arb(points, arb_points)
    for j in range(len(arb_points)):
        if len(narb[j]) > 1:
            x_0 = arb_points[j]
            #print np.array(narb[j])
            x = points[np.array(narb[j])]
            m = mass[np.array(narb[j])]
            ds = sizes[np.array(narb[j])]
            temps = T[np.array(narb[j])]
            #print temps
            #weighted_temp = Weigh2_dust(x, x_0, m, d, ds) * (particle_type[np.array(narb[j])] == 2) * temps
            rho = Weigh2_dust(x, x_0, m, d, ds) * (particle_type[np.array(narb[j])] == 2)
            #print np.sum((rho * temps) * (rho > 0)), np.sum(rho * (rho > 0))
        
            density_array.append(np.nan_to_num(np.sum((rho * temps)[rho > 0])/np.sum(rho[rho > 0])))
        else:
            density_array.append(0)
            
    return np.array(density_array)

def photoionization_arb(points, arb_points, mass, N_PART, photoionization, particle_type, narb):
    density_array = []
    #narb = neighbors_arb(points, arb_points)
    for j in range(len(arb_points)):
        if len(narb[j]) > 1:
            x_0 = arb_points[j]
            #print np.array(narb[j])
            x = points[np.array(narb[j])]
            m = mass[np.array(narb[j])]
            numpart_loc = N_PART[np.array(narb[j])]
            photio = photoionization[np.array(narb[j])]
            rho = Weigh2(x, x_0, m, d)/Weigh2(x, x, m, d) * (particle_type[np.array(narb[j])] == 0) * numpart_loc
        
            density_array.append(np.sum((rho * np.nan_to_num(photio))[rho > 0])/np.sum(rho[rho > 0]))
        else:
            density_array.append(0)
            
    return np.array(density_array)
    
#END ARBITRARY INTERPOLATIONS.

#SIMULATION IS RUN FROM ANOTHER FILE#