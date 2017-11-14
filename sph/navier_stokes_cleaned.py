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
mu_specie = np.array([2.0159,4.0026,1.0079,1.0074,4.0021,0.0005,140.69,60.08,12.0107,28.0855,55.834,100.39,131.93])
cross_sections = np.array([6.65e-29, 6.65e-29, 6.3e-28, 5e-26, 5e-26, 0., 0., 0., 0., 0., 0., 0., 0.])
destruction_energies = np.array([7.2418e-19, 3.93938891e-18, 2.18e-18, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000])
mineral_densities = np.array([1.e19, 1e19,1e19,1e19,1e19,1e19, 3320,2260,2266,2329,7870,3250,3250.])
sputtering_yields = np.array([0,0,0,0,0,0,0.137,0.295,0.137,0.295,0.137,0.137,0.137])
f_u = np.array([[.86,.14,0,0,0,0,0,0,0,0,0,0,0]]) #relative abundance for species in each SPH particle, an array of arrays
gamma = np.array([7./5,5./3,5./3,5./3,5./3,5./3,15.6354113,4.913,1.0125,2.364,3.02,10.,10.])#the polytropes of species in each SPH, an array of arrays
supernova_base_release = np.array([[.86 * (1. - 0.19),.14 + 0.19 * 0.86/2.,0.,0.,0.,0.,0.025,0.025,0.025,0.025,0.025,0.025,0.025]])
W6_constant = (3 * np.pi/80)

mrn_constants = np.array([50e-10, 5000e-10]) #minimum and maximum radii for MRN distribution

#DUMMY VALUES
DIAMETER = 1e6 * AU
N_PARTICLES = 15000
N_INT_PER_PARTICLE = 100
V = (DIAMETER)**3
d = (V/N_PARTICLES * N_INT_PER_PARTICLE)**(1./3.)
d_0 = 1e5 * AU
d_sq = d**2
specie_fraction_array = np.array([.86,.14,0,0,0,0,0,0,0,0,0,0,0]) 
	
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

def interpolation_functions(): #interpolation of carbon and silicon destruction efficiency, returns the calculated e_c and e_s
    #v_new = v/1000. #in km/s rather than m/s
    #v_new[(v_new < 0)] = 0
    #v_new[v_new > 200.] = 200.
    v_s = np.array([0., 50., 75., 100., 125., 150., 175., 200.])
    e_c = np.array([0., 0.01,0.04,0.10,0.18,0.17,0.18,0.23])
    e_si = np.array([0., 0.02,0.09,0.23,0.41,0.40,0.42,0.40])
    
    carb_interpolation = interp1d(v_s, e_c, bounds_error=False, fill_value = 0.)
    silicon_interpolation = interp1d(v_s, e_si, bounds_error=False, fill_value = 0.)
    
    def zero_function(v):
    	return np.zeros(len(v));
    
    #list of interpolation functions for each species
    return (zero_function, zero_function, zero_function, zero_function, zero_function, zero_function, carb_interpolation, silicon_interpolation, carb_interpolation, silicon_interpolation, carb_interpolation, carb_interpolation, carb_interpolation)

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

def Weigh2(x, x_0, m):
    norms_sq = np.sum((x - x_0)**2, axis=1)
    W = m * 315*(m_0/m)**3*((m/m_0)**(2./3.)*d**2-norms_sq)**3/(64*np.pi*d**9)
    return(W)

def Weigh2_dust(x, x_0, m, dustsize):
    norms_sq = np.sum((x - x_0)**2, axis=1)
    W = m * 315*(dustsize**2-norms_sq)**3/(64*np.pi*dustsize**9)
    return(W)
    
def grad_weight(x, x_0, m, type_particle):
    vec = (x - x_0)
    norms_sq = np.sum((x - x_0)**2, axis=1)
    
    W = -315 * 6 * (m_0/m)**3 / (64*np.pi*d**9) * ((m/m_0)**(2./3.)*d**2-norms_sq)**2 * (type_particle == 0) * vec.T
    
    #checks if SPH particles intersect or not
    return((np.nan_to_num(W.astype('float') * ((m/m_0)**(2./3.)*d**2-norms_sq > 0))).T)

def mu_j(j): #mean molecular weight of the SPH particle
    #species = 13
    mu = 0
    for u in range(len(mu_specie)):
        mu = mu + f_u[j][u]/mu_specie[u]
    if(mu == 0):
        return(0)
    else:
        mu = 1/(m_h*mu)
        return(mu)

def density(j):
    x_0 = points[j]
    x = np.array(points[np.array(neighbor[j])])
    m = np.array(mass[np.array(neighbor[j])])
    
    rho = Weigh2(x, x_0, m) * (particle_type[np.array(np.array(neighbor[j]))] == 0)
    return np.sum(rho[rho > 0])
    
def dust_density(j):
	#evaluating dust density at each gas particle
	#including only neighboring points so that self-density of dust particles
	#is not counted
    x_0 = points[j]
    x = points[np.array(neighbor[j])]
    m = mass[np.array(neighbor[j])]
    ds = sizes[np.array(neighbor[j])]
    
    rho = Weigh2_dust(x, x_0, m, ds) * (particle_type[np.array(neighbor[j])] == 2)
    return np.sum(rho[rho > 0])
    
def net_impulse(j):
	meff = grain_mass(mineral_densities, mrn_constants)
	seff = sigma_effective(mineral_densities, mrn_constants, mu_specie)
	mean_grainmass = np.sum(meff * f_un[np.array(neighbor[j])], axis=1)
	mean_cross = np.sum(seff * f_un[np.array(neighbor[j])], axis=1)
	net_vels = velocities[np.array(neighbor[j])] - velocities[j]
	
	x_0 = points[j]
	x = points[np.array(neighbor[j])]
	m = mass[np.array(neighbor[j])]
	ds = sizes[np.array(neighbor[j])]
	
	weight_factor = Weigh2_dust(x, x_0, m, ds)
	#gives units of m/s^2, which is exactly what we want
	accel_net = weight_factor/mean_grainmass * mean_cross * np.sum((net_vels.T)**2, axis=0)**0.5 * net_vels.T * (particle_type[np.array(neighbor[j])] == 2) * (weight_factor > 0)
	return np.sum(accel_net.T,axis=0), -accel_net, neighbor[j]		
    
def num_dens(j):
    x_0 = points[j]
    x = np.array(points[np.array(neighbor[j])])
    m = np.array(mass[np.array(neighbor[j])])
    
    n_dens = Weigh2(x, x_0, m)/(mu_array[np.array(neighbor[j])] * m_h)
    return np.sum(n_dens[n_dens > 0])
    
def del_pressure(i): #gradient of the pressure
    if (particle_type[i] != 0):
        return(np.zeros(3))
    else:
        x_0 = points[i]
        x = np.append([x_0], points[np.array(neighbor[i])],axis=0)
        m = np.append(mass[i], mass[np.array(neighbor[i])])
        pt = np.append(particle_type[i], particle_type[neighbor[i]])
        
        #symmetrizing pressure gradient
        gw = (grad_weight(x, x_0, m, pt) + grad_weight(x, x_0, m[0], pt))/2.
        
        del_pres = np.sum((gw.T * E_internal[np.append(i, neighbor[i])]/gamma_array[np.append(i, neighbor[i])]).T, axis=0)
        
        return del_pres

def crossing_time(neighbor, velocities, sizes, particle_type):
	max_relvel = np.zeros(len(neighbor))
	neighbor_lengths = np.array([len(a) for a in neighbor])
	j_range = np.arange(len(neighbor))
	max_relvel[neighbor_lengths > 0] = np.array([np.max(np.sum((velocities[neighbor[j]] - velocities[j])**2, axis=1))**0.5 for j in j_range[neighbor_lengths]])
	
	crossing_time = np.nan_to_num(sizes/max_relvel) * (particle_type == 0)
	return min(crossing_time[crossing_time != 0])

def artificial_viscosity(neighbor, points, particle_type, sizes, mass, densities, velocities, T):
	css_base = (gamma_array * k * T/(mu_array * amu))**0.5
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
			gw0 = grad_weight(points[neighbor[j]], points[j], mass[j], particle_type[neighbor[j]])
			gw1 = grad_weight(points[neighbor[j]], points[j], mass[neighbor[j]], particle_type[neighbor[j]])
			gw = (gw0 + gw1)/2.
			accel_ij = (mass[neighbor[j]] * PI_ij * gw.T).T
			heat_ij = 0.5 * mass[neighbor[j]] * PI_ij * np.sum((velocities[neighbor[j]] - velocities[j]) * gw, axis=1)
		
			accel_i = np.sum(accel_ij[(particle_type[neighbor[j]] == 0)],axis=0)
			heat_i = np.sum(heat_ij[particle_type[neighbor[j]] == 0])
			visc_accel[j] = accel_i
			visc_heat[j] = heat_i
	
	return visc_accel, visc_heat
    
def supernova_explosion():
    #supernova_pos = np.arange(len(star_ages))[(star_ages > luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1) * year * 10e10)]
    total_release = supernova_base_release
    trsum = np.sum(total_release)

    gas_release = f_u
    grsum = np.sum(gas_release)
    
    dust_release = total_release - gas_release
    drsum = np.sum(dust_release)
    
    #total_release /= trsum
    gas_release /= grsum
    dust_release /= drsum
    
    dust_masses = mass[supernova_pos] * drsum/trsum
    gas_masses = mass[supernova_pos] * grsum/trsum
    
    supernova_dust_len = int((np.sum(dust_masses)/solar_mass - 2.)/0.05)
    print supernova_dust_len
    
    dust_comps = (np.vstack([dust_release] * supernova_dust_len).T).T
    gas_comps = (np.vstack([gas_release] * len(supernova_pos)).T).T
    star_comps = (np.vstack([gas_release] * len(supernova_pos)).T).T
    
    dust_mass = np.ones(supernova_dust_len) * 0.05 * solar_mass
    #dust_mass = (mass[supernova_pos] - 2 * solar_mass) * drsum/trsum
    gas_mass = (mass[supernova_pos] - 2 * solar_mass) * drsum/grsum
    stars_mass = (np.ones(len(supernova_pos)) * 2 * solar_mass)
    
    dustpoints = np.vstack([points[supernova_pos][0]] * supernova_dust_len) + (np.random.rand(supernova_dust_len,3) - 0.5) * d_0
    newpoints = points[supernova_pos] + (np.random.rand(len(supernova_pos),3) - 0.5) * d_0
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

def rad_heating(positions, ptypes, masses, sizes, cross_array, f_un, supernova_pos):
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
            dists = (np.sum(np.cross(points - ray[0], ray[1] - ray[0])**2, axis=1)/(np.sum((ray[1] - ray[0])**2)))
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
        lum_factor.append(np.nan_to_num(star_distance_2[ei] * np.sum(((gas_distance + sizes[particle_type != 1]/2).T**-2/star_distance[ei] * blocked[ei]).T, axis=0)/np.sum((gas_distance + sizes[particle_type != 1]/2)**-2, axis=0)))
    
    lum_factor = np.array(lum_factor)
    extinction = (W6_constant * masses[ptypes != 1]/(mu_array[ptypes != 1] * amu) * cross_array[ptypes != 1]) * sizes[ptypes != 1]**(-2)
    
    exponential = np.exp(-np.nan_to_num(lum_factor))
    distance_factor = (np.nan_to_num(star_distance_2)**2 + np.average(sizes[particle_type != 1])**2 * np.ones(np.shape(star_distance_2))) * 4. * np.pi
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
    
    frac_destroyed_by_species = atoms_destroyed/atoms_total
    frac_destroyed_by_species = 1. - np.nan_to_num(np.exp(-atoms_destroyed/atoms_total))
    
    new_fun = f_un.T

    new_fun[2][ptypes != 1] += new_fun[0][ptypes != 1] * frac_destroyed_by_species.T[0] * 2.
    new_fun[3][ptypes != 1] += new_fun[2][ptypes != 1] * frac_destroyed_by_species.T[2]
    new_fun[4][ptypes != 1] += new_fun[1][ptypes != 1] * frac_destroyed_by_species.T[1]
    
    new_fun[5][ptypes != 1] += new_fun[2][ptypes != 1] * frac_destroyed_by_species.T[2]
    new_fun[5][ptypes != 1] += new_fun[1][ptypes != 1] * frac_destroyed_by_species.T[1]
    
    new_fun[0][ptypes != 1] -= new_fun[0][ptypes != 1] * frac_destroyed_by_species.T[0]
    new_fun[1][ptypes != 1] -= new_fun[1][ptypes != 1] * frac_destroyed_by_species.T[1]
    new_fun[2][ptypes != 1] -= new_fun[2][ptypes != 1] * frac_destroyed_by_species.T[2]
    
    new_fun /= np.sum(new_fun,axis=0)
    
    subt = new_fun[3][ptypes != 1] * new_fun[5][ptypes != 1] * (3 * k * T[ptypes != 1])**0.5/(c**2 * mu_specie[3]**0.5 * mu_specie[5]**0.5 * amu)**0.5
    subt2= new_fun[4][ptypes != 1] * new_fun[5][ptypes != 1] * (3 * k * T[ptypes != 1])**0.5/(c**2 * mu_specie[5]**0.5 * mu_specie[4]**0.5 * amu)**0.5
    
    new_fun[2][ptypes != 1] += subt
    new_fun[3][ptypes != 1] -= subt
    new_fun[4][ptypes != 1] -= subt2
    new_fun[5][ptypes != 1] -= subt + subt2
    
    new_fun /= np.sum(new_fun,axis=0)
    
    #energy, composition change, impulse
    return lf2, new_fun.T, momentum
    
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
def dust_accretion(j,u,dt,T): #index of each dust particle, specie index, timestep, and Temperature. Returns accreated dust mass
    #Still need to add n_h, sputtering yield, mineral density, initial number density, etc.
    n_H = 1;
    num_dens_ref = dust_comps[j]*mass[j]/m_ref_species[u] #initial number density of species, n_u_0
    #K_u calculation divided into 3 for convenience 
    K_u = (min_radius_array[u]**(-0.5)-max_radius_array[u]**(-0.5))/(3*mineral_density[u]*(max_radius_array[u]**(0.5)-min_radius_array[u]**(0.5)))
    K_u = K_u*(k*T/(2*np.pi*m_ref_species[u]))**0.5 
    K_u = K_u*(m_ref_species[u]*num_dens_ref+mineral_density[u]-n_H*m_ref_species[u]*sputtering_yield[u])
    rho = mineral_density[u]*K_u*np.exp(K_u*dt)/(m_ref_species[u]*num_dens_ref+np.exp(K_u*dt)*mineral_density[u]-n_H*m_ref_species[u]*sputtering_yield[u])     
    n_u = num_dens_ref-(rho-mineral_density[u])/mol_weights[u]                                              
    return(rho,n_u)
        
def num_dens_array(x)
	num_dens_ref = np.array([])
	indices = neighbors(x,d)
	for j in range(len(indices)):
		i = indices[j]
		factor = f_un[i].T*(mass[i]/(mu_species*amu))
		num_dens_ref += factor*weigh2(x,points[i],mass[i])
	return(num_dens_ref) #retuns n_u_0 for chemisputtering, not sure if it is correct
		
def chemical_sputtering_yield(i,x,dt): #for a particle i, returns F_sput array for all species
	F_sput = np.zeros(13) #for all the 13 species
	for u in range(len(F_sput)):
		y_H = min[max[10**(-7),0.5*np.exp(-4600/T[i])],10**(-3)]*(k*T[i]/(2*np.pi*m_h)**0.5 #for carbon, Fe, etc.
		y_He = min[max[10**(-6),0.5*np.exp(-4600/T[i])],10**(-2)]*(k*T[i]/(2*np.pi*mu_specie[1]*amu)**0.5 #for carbon, Fe, etc.								   
		if (u == 9): #if the element is a silicon
			y_H = y_H/0.464
			y_He = y_He/0.464						     
		num_dens_ref = num_dens_array(x)
		a_min = mrn_constants[0]
		a_max = mrn_constants[1]
		K_u = mu_specie[u]*num_dens_ref[u]+mineral_densities[u]-num_dens_ref[3]*y_H*mu_specie[u]-num_dens_ref[4]*y_He[u]*mu_specie[u]
		J_u = (a_min**-0.5-a_max**-0.5)/(3*dens_u(i,u)*(a_max**0.5-a_min**0.5))*(k*T[i]/(2*np.pi*mu_specie[u])**0.5*factor
		F_sput[u] = K_u*np.exp(K_u*J_u*dt)/(mu_specie[u]*num_dens_ref[u]-num_dens_ref[3]*mu_specie[u]*y_H+mineral_densities[u]*np.exp(K_u*dt)-num_dens_ref[4]*mu_specie[u]*y_He)
	return(F_sput)									 
''''
def nearest_gas(i): #returns the nearest gas particles for a particle i
	gas_particle_array = np.array([])									 
	indices = neighbor[i]
	for j in range(len(indices)):
		if (particle_type[indices[j]] == 0):
			gas_particle_array = np.append(gas_particle_array,indices[j])
	return(gas_particle_array)
										 
def chemical_sputtering(i,u,dt):
	y_H = min[max[10**(-7),0.5*np.exp(-4600*k/T[i])],10**(-3)]*(k*T[i]/(2*np.pi*m_h)**0.5 #for carbon, Fe, etc.
	y_He = min[max[10**(-6),0.5*np.exp(-4600*k/T[i])],10**(-2)]*(k*T[i]/(2*np.pi*mu_specie[1]*amu)**0.5 #for carbon, Fe, etc.								   
	if (u == 9): #if the element is a silicon
		y_H = y_H/0.464
		y_He = y_He/0.464
	total_diff_mass_u = 0 #total difference in mass for each dust particle i									 
	indices = neighbors(x,d)
	normalization = 0 #normalization factor for dust particles
	normalization_gas = 0 #normalization factor for gas particles									 
	all_intersecting_gas = np.array([]) #the array of all the gas particles that intersect the dust particles									 
	for k in range(len(indices)):
		jj = indices[k]
		if(particle_type[jj] !=1): #if it is not dust
			continue:
		else:
			total_N = mass[jj]/np.dot(f_un[jj],mu_specie)
			mass_u = total_N*f_un[jj][u]*mu_specie[u]									      
			normalization += mass_u*weigh2(x,points[jj],mass[jj])										      
	for j in range(len(indices)): #this is where I consider the destruction of dust of specie u in a dust particle
		i = indices[j]
		total_diff_mass_u								 
		if(particle_type[i] !=1): #if it is not dust
			continue:										      
 		else:   #accretion/sputtering 
			total_N = mass[i]/np.dot(f_un[i],mu_specie)
			mass_u = total_N*f_un[i][u]*mu_specie[u] #mass of specie u contained in the dust
			F_u = chemical_sputtering_yield(i,u,x,dt) #returns the intersecting gas particle indices, for each dust particle i										      
			gas_particle_arrray = nearest_gas(i)
			all_intersecting_gas = np.append(all_intersecting_gas,gas_particle_array)
			diff_mass_u = (F_u-1)*mass_u**2*weigh2(x,points[i],mass[i])/normalization #mass change in the dust particle
			total_diff_mass_u += diff_mass_u 
			mass[i] = mass[i]+diff_mass_u							 
			num_u_molecules = diff_mass_u/mu_specie[u] #number of gas particles precipitating into dust
			total_N = total_N + num_u_molecules							 
			f_un[i][u] = (mass_u+diff_mass_u)/(mu_specie[u]*total_N) #the new fraction of species (in terms of ratio of number of molecules)									 
	#this is where I consider the addition of dust to gas particles
	for ii in range(len(all_intersecting_gas)):
		index = all_intersecting_gas[ii]
		total_N = mass[index]/np.dot(f_un[index],mu_specie)
		new_mass = mu_specie[u]*total_N*f_un[index][u]-total_N*f_un[index][3]*mu_specie[u]*y_H - y_He*total_N*f_un[index][4]*mu_specie[u] #this is m_u_j'								 
		normalization_2 += new_mass*weigh2(x,points[index],new_mass)	#normalization factor for gas particles							 
	for zz in range(len(all_intersecting_gas)):
		index = all_intersecting_gas[zz]								 
		total_N = mass[index]/np.dot(f_un[index],mu_specie)
		mass_u = total_N*f_un[index][u]*mu_specie[u] #mass of specie u contained in the gas
		new_mass = mu_specie[u]*total_N*f_un[index][u]-total_N*f_un[index][3]*mu_specie[u]*y_h_u - y_he_u*total_N*f_un[index][4]*mu_specie[u]							 
		diff_mass_gas_u = -1*new_mass*weigh2(x,points[index],new_mass)/normalization_2*total_diff_mass_u #corresponding change in gas
		mass[index] = mass[index]+diff_mass_u	#new mass						 
		num_u_molecules = diff_mass_gas_u/mu_specie[u] #number of gas molecules of specie u that get added
		total_N = total_N + num_u_molecules							 
		f_un[index][u] = (mass_u+diff_mass_gas_u)/(mu_specie[u]*total_N) #the new fraction of species (in terms of ratio of number of molecules)								 
	return(1)	'''									 

def supernova_destruction_2(points, velocities, neighbor, mass, f_un, mu_array, sizes, densities, particle_type):
	#Indexes over all gas particles and sees if they intersect a dust,
	#rather than the other way around as previously, because gas particles
	#are much smaller than dust particles, by design
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
			w2d = Weigh2_dust(x, x_0, m, dustsize)
			#density of dust at the center of the dust particles
			w2_max = Weigh2_dust(x, x, m, dustsize)
			
			#density of dust at the selected gas particle
			rho = w2d * (w2d > 0) * (particle_type[neighbor[j]] == 2);
			rho_base = w2_max
			if np.sum(rho) > 0:
				#Obtaining relative velocities between gas/dust, and destruction efficiency
				vels = np.sum((velocities[neighbor[j]] - velocities[j])**2, axis=1)**0.5 * (particle_type[np.array(neighbor[j])] == 2)
				dest_fracs = (np.array([u(vels/1000.) for u in intf]) * np.nan_to_num(rho/rho_base)).T
				#Distributing dust destruction over all intersecting dust particles
				loss_relative = rho/np.sum(rho)
				final_fracs = dens/critical_density * dest_fracs.T #fraction destroyed
				final_fracs[final_fracs >= 1.] = 1.
				
				N_dust = mass[neighbor[j]]/mu_array[neighbor[j]]
				N_self = mass[j]/mu_array[j]
				
				#what relative fraction of refractory species are created in gas particle j? Summed over because only one particle
				refractory_fracs = np.sum((final_fracs * N_dust/N_self).T * f_un[neighbor[j]], axis=0).astype('float64')
				#Conversely, how much dust is fractionally lost from each intersecting gas particle?
				dust_lost = final_fracs.T * f_un[neighbor[j]]
				
				#Dust lost in each dust particle, which is taken up as refractory gas by the gas particle
				frac_destruction[neighbor[j]] += dust_lost
				frac_reuptake[j] += refractory_fracs
				#print refractory_fracs
				
	return frac_destruction, frac_reuptake													    

def chemisputtering_2(points, neighbor, mass, f_un, mu_array, sizes, densities, particle_type):
	#Indexes over all gas particles and sees if they intersect a dust, like supernova
	frac_destruction = copy.deepcopy(f_un * 0.)
	frac_reuptake = copy.deepcopy(f_un * 0.)
	jarr = np.arange(len(neighbor))[particle_type[np.arange(len(neighbor))] == 0] #setting up array where all particles are gas
	for j in jarr:
		if (np.sum(particle_type[np.array(neighbor[j])] == 2) > 0): #making sure that this gas particle has dusty neighbors!
			#no need to append j to this, because j is included as its own neighbor since it has a distance of 0 from itself
			x = points[neighbor[j]]
			N_total = mass[neighbor[j]]/np.dot(mu_array[neighborhood[j]],f_un[neigbhorhood[j]]) #the total number of molecules in each particle								 
			m = N_total*f_un[neighbor[j]]*mu_array[neighbor[j]] #mass of each specie u						 
			x_0 = points[j]
			comps = f_un[j]
			dustsize = sizes[neighbor[j]]
			dens = densities[j]
			
			#Density of dust at the center of the gas particle
			w2d = Weigh2_dust(x, x_0, m, dustsize)
			#density of dust at the center of the dust particles
			w2_max = Weigh2_dust(x, x, m, dustsize)
			
			#density of dust at the selected gas particle
			rho = w2d * (w2d > 0) * (particle_type[neighbor[j]] == 2);
			rho_base = w2_max
			if np.sum(rho) > 0:
				#Obtaining relative velocities between gas/dust, and destruction efficiency

				dest_fracs = np.array(chemical_sputtering_yield(x,j,dt_0)) * np.nan_to_num(rho/rho_base)).T #fraction destroyed
				#Distributing dust destruction over all intersecting dust particles
				loss_relative = rho/np.sum(rho)
				final_fracs = dest_fracs.T #fraction destroyed
				final_fracs[final_fracs >= 1.] = 1.
				
				N_dust = mass[neighbor[j]]/mu_array[neighbor[j]]
				
				y_H = min[max[10**(-7),0.5*np.exp(-4600/T[j])],10**(-3)]*(k*T[j]/(2*np.pi*m_h)**0.5 #for carbon, Fe, etc.
				y_He = min[max[10**(-6),0.5*np.exp(-4600/T[j])],10**(-2)]*(k*T[j]/(2*np.pi*mu_specie[1]*amu)**0.5 #for carbon, Fe, etc.						   
				
				N_self_old = mass[j]/mu_array[j]
				mass_new = N_self_old * f_un[j] * mu_specie - N_self_old*f_un[j]*mu_specie*y_H - y_He*total_N*f_un[j]*mu_specie #M_u_j, probably not correct
				N_self = mass_new/mu_array[j]							   
				#what relative fraction of refractory species are created in gas particle j? Summed over because only one particle
				refractory_fracs = np.sum((final_fracs * N_dust/N_self).T * f_un[neighbor[j]], axis=0).astype('float64')
				#Conversely, how much dust is fractionally lost from each intersecting gas particle?
				dust_lost = final_fracs.T * f_un[neighbor[j]]
				
				#Dust lost in each dust particle, which is taken up as refractory gas by the gas particle
				frac_destruction[neighbor[j]] += dust_lost
				frac_reuptake[j] += refractory_fracs
				#print refractory_fracs
				
	return frac_destruction, frac_reuptake	
										 
def neighbors_arb(points, arb_points):
    kdt = spatial.cKDTree(points)  
    qbp = kdt.query_ball_point(arb_points, d, p=2, eps=0.1)
    
    return qbp

def density_arb(arb_points):
    density_array = []
    narb = neighbors_arb(points, arb_points)
    for j in range(len(arb_points)):
        if len(narb[j]) > 1:
            x_0 = arb_points[j]
            #print np.array(narb[j])
            x = points[np.array(narb[j])]
            m = mass[np.array(narb[j])]
            rho = Weigh2(x, x_0, m) * (particle_type[np.array(narb[j])] == 0)
        
            density_array.append(np.sum(rho[rho > 0]))
        else:
            density_array.append(0)
            
    return np.array(density_array)

def temperature_arb(arb_points):
    density_array = []
    narb = neighbors_arb(points, arb_points)
    for j in range(len(arb_points)):
        if len(narb[j]) > 1:
            x_0 = arb_points[j]
            #print np.array(narb[j])
            x = points[np.array(narb[j])]
            m = mass[np.array(narb[j])]
            temps = T[np.array(narb[j])]
        
            temp_sum = Weigh2(x, x_0, m) * (particle_type[np.array(narb[j])] == 0) * temps
            temp_w = Weigh2(x, x_0, m) * (particle_type[np.array(narb[j])] == 0)
            
            #print temp_sum
            #print temp_w
            
            tsgo = np.nan_to_num(np.sum(temp_sum[temp_sum > 0])/np.sum(temp_w[temp_sum > 0]))
            
            density_array.append(tsgo)
        else:
            density_array.append(0)
            
    return np.array(density_array)
    
#END ARBITRARY INTERPOLATIONS.

#### AND NOW THE FUN BEGINS! THIS IS WHERE THE SIMULATION RUNS HAPPEN. ####
#SETTING VALUES OF BASIC SIMULATION PARAMETERS HERE (TO REPLACE DUMMY VALUES AT BEGINNING)
DIAMETER = 0.75e6 * AU
N_PARTICLES = 2000
N_INT_PER_PARTICLE = 100
V = (DIAMETER)**3
d = (V/N_PARTICLES * N_INT_PER_PARTICLE)**(1./3.)
d_sq = d**2
dt = dt_0
DUST_FRAC = 0.050000
DUST_MASS = 0.05
N_RADIATIVE = 100
MAX_AGE = 4e7 * year
#relative abundance for species in each SPH particle,  (H2, H, H+,He,He+Mg2SiO4,SiO2,C,Si,Fe,MgSiO3,FeSiO3)in that order
specie_fraction_array = np.array([.86,.14,0,0,0,0,0,0,0,0,0,0,0])
supernova_base_release = np.array([[.86,.14,0.,0.,0.,0.,0.1,0.1,0.1,0.1,0.1,0.1,0.1]])
dust_base_frac = (specie_fraction_array - supernova_base_release)
dust_base = dust_base_frac/np.sum(dust_base_frac)
cross_sections += sigma_effective(mineral_densities, mrn_constants, mu_specie)

base_imf = np.logspace(np.log10(0.1),np.log10(40.), 200)
d_base_imf = np.append(base_imf[0], np.diff(base_imf))
imf = kroupa_imf(base_imf) * d_base_imf
imf /= np.sum(imf)
points = np.random.normal(size=(N_PARTICLES, 3)) * DIAMETER
points2 = copy.deepcopy(points)
neighbor = neighbors(points, d)
#print(nbrs)
#print(points)
velocities = np.random.normal(size=(N_PARTICLES, 3)) * 10.
total_accel = np.random.rand(N_PARTICLES, 3) * 0.
mass = np.random.choice(base_imf, N_PARTICLES, p = imf) * solar_mass
sizes = (mass/m_0)**(1./3.) * d

particle_type = np.zeros([N_PARTICLES]) #0 for gas, 1 for star, 2 for dust

mu_array = np.zeros([N_PARTICLES])#array of all mu
E_internal = np.zeros([N_PARTICLES]) #array of all Energy
#copy of generate_E_array
#fills in E_internal array specified at the beginning
T = 20 * np.ones([N_PARTICLES]) #5 kelvins

#fills the f_u array
dust_fracs = ((np.random.rand(N_PARTICLES) > DUST_FRAC))
dust_fracs = dust_fracs.astype('bool')
particle_type[dust_fracs == False] = 2
fgas = np.array([specie_fraction_array] * N_PARTICLES)
fdust = np.array([dust_base[0]] * N_PARTICLES)
f_un = (fgas.T * dust_fracs + fdust.T * (1 - dust_fracs)).T
mass[particle_type == 2] = DUST_MASS * solar_mass
sizes[particle_type == 2] = d

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

densities = np.array([density(j) for j in range(len(neighbor))])
delp = np.array([del_pressure(j) for j in range(len(neighbor))])
num_densities = np.array([num_dens(j) for j in range(len(neighbor))])

star_ages = np.ones(len(points)) * -1.
age = 0

time_coord = np.array([])
dust_temps = np.array([])
star_frac = np.array([])

print("Simulation time: " + str(MAX_AGE/year) + " y")
print("Estimated free fall time: " + str(T_FF) + " y")
plt.ion()
#RUNNING SIMULATION FOR SPECIFIED TIME!
while (age < MAX_AGE):
    #timestep reset here
    ct = crossing_time(neighbor, velocities, sizes, particle_type)
    dt = min(dt_0, ct)
    if np.sum(particle_type[particle_type == 1]) > 0:
        supernova_pos = np.where(star_ages/luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1)/(year * 1e10) > 1.)[0]
        rh = rad_heating(points, particle_type, mass, sizes, cross_array, f_un,supernova_pos)
        f_un0 = f_un
        N_RADIATIVE = int(50 + np.average(np.nan_to_num(T))**(2./3.))
        area = (4 * np.pi * sizes**2)
        N_PART = mass/(m_h * mu_array)
        W6_integral = 9./area #evaluating integral of W6 kernel
        optd = 1. - np.exp(-optical_depth * W6_integral)
        T[particle_type == 2] = (rh[0][particle_type[particle_type != 1] == 2]/(sb * 4 * np.pi * optd[particle_type == 2] * sizes[particle_type == 2]**2 * dt * 4e-6 * 1))**(1./6.) + t_cmb
        E_internal[particle_type == 2] = (N_PART * k * T * gamma_array)[particle_type == 2]
        for nrad in range(N_RADIATIVE):
        	optd = 1. - np.exp(-optical_depth * W6_integral)
        	#dust particles function as if always in thermal equilibrium, F proportional to T^6. Last two variables are obtained from integration. http://www.astronomy.ohio-state.edu/~pogge/Ast871/Notes/Dust.pdf pg. 27
        	E_internal[particle_type == 0] *= np.nan_to_num((((sb * optd * W6_integral**(-1) * dt/N_RADIATIVE)/(N_PART * gamma_array * k)) * T**3 + 1.)**(-1./3.))[particle_type == 0]
        	E_internal[particle_type == 0] += rh[0][particle_type[particle_type != 1] == 0]/N_RADIATIVE
        	E_internal[E_internal < t_cmb * (gamma_array * mass * k)/(mu_array * m_h)] = (t_cmb * (gamma_array * mass * k)/(mu_array * m_h))[E_internal < t_cmb * (gamma_array * mass * k)/(mu_array * m_h)]
        	T[T < t_cmb] = t_cmb
        	
        	f_un = ((N_RADIATIVE - nrad) * f_un0 + nrad * rh[1])/N_RADIATIVE	
        	mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(f_un, axis=1)
        	gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(f_un, axis=1)
        	cross_array = np.sum(f_un * cross_sections, axis = 1)/np.sum(f_un, axis=1)
        	cross_array[particle_type == 1] == 1.e-90
        	optical_depth = mass/(m_h * mu_array) * cross_array
        	
        f_un = rh[1]
        velocities[particle_type != 1] += rh[2]
        
        #on supernova event--- add new dust particle (particle_type == 2)
        #in future, want to spew multiple low-mass dust particles
        #rather than one single very large one
        
        print (np.max(star_ages/luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1)/(year * 1e10)))
        print (np.max(mass[particle_type == 1]/solar_mass))
        
        print (len(supernova_pos))
        #print (star_ages/luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1)/(year * 1e10))[supernova_pos]
        if len(supernova_pos) > 0:
        	for ku in supernova_pos:
        		impulse, indices = supernova_impulse(points, mass, ku, particle_type)
        		velocities[indices] += impulse
			dust_comps, gas_comps, star_comps, dust_mass, gas_mass, stars_mass, newpoints, newvels, newgastype, newdusttype, new_eint_stars, new_eint_dust, new_eint_gas, supernova_pos, dustpoints, dustvels = supernova_explosion()
			E_internal[supernova_pos] = new_eint_stars
			f_un[supernova_pos] = star_comps
			mass[supernova_pos] = stars_mass
			E_internal = np.concatenate((E_internal, new_eint_dust, new_eint_gas))
			particle_type = np.concatenate((particle_type, newdusttype, newgastype))
			mass = np.concatenate((mass, dust_mass, gas_mass))
			f_un = np.vstack([f_un, dust_comps, gas_comps])
			velocities = np.concatenate((velocities, dustvels, newvels))
			star_ages = np.concatenate((star_ages, np.ones(len(dustpoints))* (-2), np.ones(len(supernova_pos))* (-2)))
			points = np.vstack([points, dustpoints, newpoints])
			sizes = np.zeros(len(points))
			sizes[particle_type == 0] = (mass[particle_type == 0]/m_0)**(1./3.) * d
			sizes[particle_type == 2] = d
			Tnew = np.zeros(len(sizes));
			Tnew[:len(T)] += T
			T = Tnew

			neighbor = neighbors(points, d)
            
        #specie_fraction_array's retention is deliberate; number densities are in fact increasing
        #so we want to divide by the same base
        mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(f_un, axis=1)
        gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(f_un, axis=1)
        cross_array = np.sum(f_un * cross_sections, axis = 1)/np.sum(f_un, axis=1)
        optical_depth = mass/(m_h * mu_array) * cross_array
        
    neighbor = neighbors(points, d)#find neighbors in each timestep
    num_neighbors = np.array([len(adjoining) for adjoining in neighbor])
    bg = bin_generator(mass, points, [4, 4, 4]); age += dt
    com = bg[0] #center of masses of each bin
    grav_accel = compute_gravitational_force(points, bg[0], bg[1], bg[2]).T #gravity is always acting, thus no cutoff distance introduced for gravity
    
    densities = np.array([density(j) for j in range(len(neighbor))])
    delp = np.array([del_pressure(j) for j in range(len(neighbor))])
    num_densities = np.array([num_dens(j) for j in range(len(neighbor))])
    dust_densities = np.array([dust_density(j) for j in range(len(neighbor))])
    #viscous force causing dust to accelerate/decelerate along with gas
    dust_net_impulse = np.array([net_impulse(j)[0] for j in range(len(neighbor))])
    #artificial viscosity to ensure proper blast wave
    av = artificial_viscosity(neighbor, points, particle_type, sizes, mass, densities, velocities, T)
    
    drag_accel_dust = np.zeros((len(neighbor), 3))
    for j in range(len(neighbor)):
    	drag_accel_dust[neighbor[j]] += net_impulse(j)[1].T
    
    pressure_accel = -np.nan_to_num((delp.T/densities * (particle_type == 0).astype('float')).T)
    #drag is a very small factor, only seems to matter at the solar-system scale
    drag_accel_gas = np.nan_to_num(((dust_net_impulse.T) * dust_densities/densities * (particle_type == 0).astype('float')).T)
    drag_accel_dust = -np.nan_to_num(drag_accel_dust)
    
    #leapfrog integration
    old_accel = copy.deepcopy(total_accel)
    visc_accel = drag_accel_gas + drag_accel_dust + av[0]
    #want perturbations of artificial viscosity to die out in time
    visc_accel[np.sum(velocities**2, axis=1)**0.5 - np.sum(visc_accel**2, axis=1)**0.5 * dt < 0] = (-velocities/dt)[np.sum(velocities**2, axis=1)**0.5 - np.sum(visc_accel**2, axis=1)**0.5 * dt < 0]
    
    total_accel = grav_accel + pressure_accel + visc_accel
        
    points += ((total_accel * (dt)**2)/2.) + velocities * dt
    if np.shape(total_accel) == np.shape(old_accel):
    	dv = (total_accel + old_accel)/2. * dt
    else:
    	dv = total_accel * dt
    velocities += dv
    
    E_internal = np.nan_to_num(E_internal) + np.nan_to_num(av[1] * dt)
    T = np.nan_to_num(E_internal * (mu_array * m_h)/(gamma_array * mass * k))

    star_ages[(particle_type == 1) & (star_ages > -2)] += dt
    probability = base_sfr * (densities/critical_density)**(1.6) * ((dt/year)/T_FF)
    diceroll = np.random.rand(len(probability))
    particle_type[(particle_type == 0) & (num_neighbors > 1)] = ((diceroll < probability).astype('float'))[(particle_type == 0) & (num_neighbors > 1)]
    #this helps ensure that lone SPH particles don't form stars at late times in the simulation
    #ideally, there is an infinite number of SPH particles, each with infinitesimal density
    #that only has any real physical effects in conjunction with other particles
        
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
    sizes[particle_type == 0] = (mass[particle_type == 0]/m_0)**(1./3.) * d
    sizes[particle_type == 2] = d
    
    dist_sq = np.sum(points**2,axis=1)
    '''
    min_dist = np.percentile(dist_sq[vel_condition < 80000**2], 0)
    max_dist = np.percentile(dist_sq[vel_condition < 80000**2], 90)
    
    xpts = points.T[1:][0][particle_type == 0]/AU
    ypts = points.T[1:][1][particle_type == 0]/AU
    
    xstars = points.T[1:][0][particle_type == 1]/AU
    ystars = points.T[1:][1][particle_type == 1]/AU
    sstars = (mass[particle_type == 1]/solar_mass) * 2.
    
    xdust = points.T[1:][0][particle_type == 2]/AU
    ydust = points.T[1:][1][particle_type == 2]/AU
    
    #colors = (f_un.T[5]/np.sum(f_un, axis=1))[particle_type == 0]
    colors = np.log10(T[particle_type == 0])
    col_dust = np.log10(T[particle_type == 2])
    
    plt.clf()
    plt.axis('equal')
    
    max_val = max(max(xpts[(dist_sq[particle_type == 0] < max_dist * 11./9.)]), max(ypts[(dist_sq[particle_type == 0] < max_dist * 11./9.)]))
    min_val = min(min(xpts[(dist_sq[particle_type == 0] < max_dist * 11./9.)]), min(ypts[(dist_sq[particle_type == 0] < max_dist * 11./9.)]))
    
    plt.scatter(np.append(xdust[(dist_sq[particle_type == 2] < max_dist * 11./9.)],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))]), np.append(ydust[(dist_sq[particle_type == 2] < max_dist * 11./9.)],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))]), c=np.append(col_dust[(dist_sq[particle_type == 2] < max_dist * 11./9.)],[0,7]), s = np.append(100 * np.ones(len(col_dust[(dist_sq[particle_type == 2] < max_dist * 11./9.)])), [0.01, 0.01]), alpha=0.25)
    plt.scatter(np.append(xpts[(dist_sq[particle_type == 0] < max_dist * 11./9.)],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))]), np.append(ypts[(dist_sq[particle_type == 0] < max_dist * 11./9.)],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))]), c=np.append(colors[(dist_sq[particle_type == 0] < max_dist * 11./9.)],[0,7]),s=np.append((sizes[(dist_sq[particle_type == 0] < max_dist * 11./9.)]/d) * 100, [0.01, 0.01]), edgecolor='none', alpha=0.1)
    #plt.scatter([max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))],c=[1,0],s=0.01,alpha=0.1)
    #plt.scatter(np.append(xdust[(dist_sq[particle_type == 2] < max_dist * 11./9.)],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))]), np.append(ydust[(dist_sq[particle_type == 2] < max_dist * 11./9.)],[max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val))]), c=np.append(col_dust[(dist_sq[particle_type == 2] < max_dist * 11./9.)],[0,7]), s = np.append((m_0/solar_mass) * np.ones(len(col_dust[(dist_sq[particle_type == 2] < max_dist * 11./9.)])), [0.01, 0.01]), alpha=0.25)
    plt.colorbar()
    plt.scatter(xstars[(dist_sq[particle_type == 1] < max_dist * 11./9.)], ystars[(dist_sq[particle_type == 1] < max_dist * 11./9.)], c='black', s=sstars[(dist_sq[particle_type == 1] < max_dist * 11./9.)])
    #plt.scatter(xdust[(dist_sq[particle_type == 2] < max_dist * 11./9.)], ydust[(dist_sq[particle_type == 2] < max_dist * 11./9.)], c=col_dust[(dist_sq[particle_type == 2] < max_dist * 11./9.)], s = (m_0/solar_mass), alpha=0.25)
    plt.xlabel('Position (astronomical units)')
    plt.ylabel('Position (astronomical units)')
    plt.title('Temperature in H II region (t = ' + str(age/year/1e6) + ' Myr)')
    plt.pause(1)
    '''
    time_coord = np.append(time_coord, [age] * len(T[particle_type == 2]))
    dust_temps = np.append(dust_temps, (E_internal/optical_depth)[particle_type == 2])
    star_frac = np.append(star_frac, [float(np.sum(mass[particle_type == 1]))/np.sum(mass)] * len(T[particle_type == 2]))
    
    print ('age=', age/year)
    #print (d/AU)
    print ('stars/total = ', float(np.sum(mass[particle_type == 1]))/np.sum(mass))
    print ('==================================')
    
'''
utime = np.unique(time_coord)
dustt = np.array([np.average(dust_temps[time_coord == med]) for med in np.unique(time_coord)])
dusts = np.array([np.std(dust_temps[time_coord == med]) for med in np.unique(time_coord)])
starf = np.array([np.average(star_frac[time_coord == med]) for med in np.unique(time_coord)])

plt.scatter(np.log10(time_coord/year), np.log10(dust_temps), alpha=0.2, c='grey', s = 10, edgecolor='none')
plt.plot(np.log10(utime/year), np.log10(dustt), c='maroon', alpha=0.5)
plt.plot(np.log10(utime/year), np.log10(dustt + 2 * dusts), c='maroon', alpha=0.25)
plt.plot(np.log10(utime/year), np.log10(dustt - 2 * dusts), c='maroon', alpha=0.25)

plt.scatter((time_coord/year), (dust_temps), alpha=0.2, c='grey', s = 10, edgecolor='none')
plt.plot((utime/year), (dustt), c='maroon', alpha=0.5)
plt.plot((utime/year), (dustt + 2 * dusts), c='maroon', alpha=0.25)
plt.plot((utime/year), (dustt - 2 * dusts), c='maroon', alpha=0.25)


VARIOUS FORMS OF PLOTTING THAT ONE CAN USE
TO REPRESENT THE FINAL STATE OF THE SIMULATION

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

INTERPOLATED PLOTTING:
arb_points = (np.random.rand(N_PARTICLES * 10, 3) - 0.5) * (max(ymax, xmax) - min(xmin, ymin))
#narb = neighbors_arb(points, arb_points)
darb = density_arb(arb_points)
tarb = temperature_arb(arb_points)
#[ax.scatter(points.T[0][particle_type == 0]/AU, points2.T[1][particle_type == 0]/AU, points2.T[2][particle_type == 0]/AU, alpha=0.1)]
[plt.scatter(arb_points.T[0]/AU, arb_points.T[1]/AU, c = np.log10(darb/critical_density), s=8, alpha=0.7, edgecolor='none'), plt.colorbar()]
[plt.scatter(points.T[1:][0][particle_type == 1]/AU, points.T[1:][1][particle_type == 1]/AU, c = 'black', s=(mass[particle_type == 1]/solar_mass) * 2, alpha=1)]
[plt.axis('equal'), plt.show()]'''
