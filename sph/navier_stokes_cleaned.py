'''
Created on Aug 21, 2017

@author: umbut, dhruv
'''
"""VISCOSITY HASN'T BEEN VECTORIZED YET, SO WE SHOULD DO THAT"""
"""MOST OTHER FUNCTIONS SEEM TO WORK AND OVERALL SPH SIMULATION IS MUCH FASTER THAN BEFORE"""
import numpy as np
import random
from scipy import constants
from scipy.integrate import odeint
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
dt_0 = 60. * 60. * 24. * 365 * 100000. # 100000 years
year = 60. * 60. * 24. * 365.
#properties for species in each SPH particle, (H2, He, H,H+,He+,e-,Mg2SiO4,SiO2,C,Si,Fe,MgSiO3,FeSiO3)in that order
mu_specie = np.array([2.0159,4.0026,1.0079,1.0074,4.0021,0.0005,140.69,60.08,12.0107,28.0855,55.834,100.39,131.93])
cross_sections = np.array([3.34e-30, 3.34e-30, 6.3e-28, 5e-26, 5e-26, 0., 0., 0., 0., 0., 0., 0., 0.])
destruction_energies = np.array([7.2418e-19, 3.93938891e-18, 2.18e-18, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000])
mineral_densities = np.array([1.e19, 1e19,1e19,1e19,1e19,1e19, 3320,2260,2266,2329,7870,3250,3250.])
sputtering_yields = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
f_u = np.array([[.86,.14,0,0,0,0,0,0,0,0,0,0,0]]) #relative abundance for species in each SPH particle, an array of arrays
gamma = np.array([7./5,5./3,5./3,5./3,5./3,5./3,7.09,4.91,1.03,2.41,3.022,1.,1.])#the polytropes of species in each SPH, an array of arrays
supernova_base_release = np.array([[.86,.14,0.,0.,0.,0.,0.1,0.1,0.1,0.1,0.1,0.1,0.1]])

mrn_constants = np.array([50e-10, 2500e-10]) #minimum and maximum radii for MRN distribution


heat_capacity_array = np.array([]) #heat capacities of refractory species

#DUMMY VALUES
DIAMETER = 1e6 * AU
N_PARTICLES = 15000
N_INT_PER_PARTICLE = 100
V = (DIAMETER)**3
d = (V/N_PARTICLES * N_INT_PER_PARTICLE)**(1./3.)
d_sq = d**2
specie_fraction_array = np.array([.86,.14,0,0,0,0,0,0,0,0,0,0,0]) 
	
def sigma_effective(mineral_densities, mrn_constants,mu_specie):
	#computes effective cross-sectional area per particle
	#assuming an MRN distribution. This gives a measure of
	#the optical activity of dust
	seff = mu_specie * amu/mineral_densities * (3./4.) * -np.diff(mrn_constants**-0.5)/np.diff(mrn_constants**0.5)
	return seff

def kroupa_imf(base_imf):
    coeff_0 = 1
    imf_final = np.zeros(len(base_imf))
    imf_final[base_imf < 0.08] = coeff_0 * base_imf[base_imf < 0.08]**-0.3
    coeff_1 = 2.133403503
    imf_final[(base_imf >= 0.08) & (base_imf < 0.5)] = coeff_1 * (base_imf/0.08)[(base_imf >= 0.08) & (base_imf < 0.5)]**-1.3
    coeff_2 = 0.09233279398
    imf_final[(base_imf >= 0.5)] = coeff_2 * coeff_1 * (base_imf/0.5)[(base_imf >= 0.5)]**-2.2 #fattened tail from 2.3!
    
    return (imf_final)

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

def laplacian_weight(x, x_0, m):
    sq_norms = np.sum((x - x_0)**2, axis=1)
    const = -945/(32*np.pi*d**9)
    

def Laplacian_Weight(x):#laplacian of the weight for the viscosity function
    x_norm = x[0]*x[0]+x[1]*x[1]+x[2]*x[2]
    const = -945/(32*np.pi*d**9)#d^9replaced by iterations of *d for computational efficiency
    lap_W = const*(d*d-x_norm)*(3*d*d-7*x_norm)
    return(lap_W)

def viscosity(j): #viscosity introduced to make the simulation more stable
    visc_coeff = 0.5 #vicsosity coefficient
    f_visc = 0
    x = points[j]
    v = velocities[j]
    indices_nearest = neighbor[j]
    for i in range(len(indices_nearest)):
        if(density(i)==0):
            continue
        f_visc = f_visc + (velocities[indices_nearest[i]]-v)*mass[indices_nearest[i]]*Laplacian_Weight(x-points[indices_nearest[i]])/density(i)
        #print(density(i))
    f_visc = f_visc/visc_coeff
    return(f_visc)
    
def bin_generator(masses, positions, subdivisions):
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
        
    com = [np.sum(masses[np.array(object_indices[v])] * positions[np.array(object_indices[v])].T, axis=1)/np.sum(masses[np.array(object_indices[v])]) for v in range(len(object_indices))]
    com = np.array(com)
    grid_masses = np.array([np.sum(masses[np.array(object_indices[v])]) for v in range(len(object_indices))])
    
    return com, grid_masses, length_scale, np.array(subdivision_boxes), object_indices


def compute_gravitational_force(particle_positions, grid_com, grid_masses, length_scale):
    forces = 0;
    for i in range(len(grid_com)):
        forces += ((np.sum((particle_positions - grid_com[i]).T**2, axis=0) + (length_scale)**2)**-1.5 * -(particle_positions - grid_com[i]).T * grid_masses[i] * G)
        
    return (forces)

def Weigh2(x, x_0, m):
    norms_sq = np.sum((x - x_0)**2, axis=1)
    W = m * 315*(m_0/m)**3*((m/m_0)**(2./3.)*d**2-norms_sq)**3/(64*np.pi*d**9)
    return(W)

def density(j):
    x_0 = points[j]
    x = np.append([x_0], points[np.array(neighbor[j])],axis=0)
    m = np.append(mass[j], mass[np.array(neighbor[j])])
    
    rho = Weigh2(x, x_0, m) * (particle_type[np.append(j, np.array(neighbor[j]))] == 0)
    return np.sum(rho[rho > 0])
    
def num_dens(j):
    x_0 = points[j]
    x = np.append([x_0], points[np.array(neighbor[j])],axis=0)
    m = np.append(mass[j], mass[np.array(neighbor[j])])
    
    n_dens = Weigh2(x, x_0, m)/(mu_array[np.append(j, neighbor[j])] * m_h)
    return np.sum(n_dens[n_dens > 0])

   
def grad_weight(x, x_0, m, type_particle):
    vec = (x - x_0)
    norms_sq = np.sum((x - x_0)**2, axis=1)
    
    W = -315 * 6 * (m_0/m)**3 / (64*np.pi*d**9) * ((m/m_0)**(2./3.)*d**2-norms_sq)**2 * (type_particle == 0) * vec.T
    
    return((np.nan_to_num(W.astype('float') * ((m/m_0)**(2./3.)*d**2-norms_sq > 0))).T)

def del_pressure(i): #gradient of the pressure
    if (particle_type[i] != 0):
        return(np.zeros(3))
    else:
        x_0 = points[i]
        x = np.append([x_0], points[np.array(neighbor[i])],axis=0)
        m = np.append(mass[i], mass[np.array(neighbor[i])])
        pt = np.append(particle_type[i], particle_type[neighbor[i]])
        
        gw = grad_weight(x, x_0, m, pt)
        
        del_pres = np.sum((gw.T * E_internal[np.append(i, neighbor[i])]/gamma_array[np.append(i, neighbor[i])]).T, axis=0)
        
        return del_pres

def neighbors(points, dist):
    kdt = spatial.cKDTree(points)  
    qbp = kdt.query_ball_point(points, dist, p=2, eps=0.1)
    
    return qbp
    
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
    
    dust_comps = (np.vstack([dust_release] * len(supernova_pos)).T).T
    gas_comps = (np.vstack([gas_release] * len(supernova_pos)).T).T
    star_comps = (np.vstack([gas_release] * len(supernova_pos)).T).T
    
    dust_mass = (mass[supernova_pos] - 2 * solar_mass) * drsum/trsum
    gas_mass = (mass[supernova_pos] - 2 * solar_mass) * drsum/grsum
    stars_mass = (np.ones(len(supernova_pos)) * 2 * solar_mass)
    
    newpoints = points[supernova_pos] + np.random.rand(3) * d
    newvels = velocities[supernova_pos]
    #newaccel = accel[supernova_pos]
    
    newgastype = np.zeros(len(supernova_pos))
    newdusttype = np.ones(len(supernova_pos)) * 2
    
    new_eint_stars = np.zeros(len(supernova_pos))
    new_eint_dust = E_internal[supernova_pos]/2.
    new_eint_gas = E_internal[supernova_pos]/2.
    
    return dust_comps, gas_comps, star_comps, dust_mass, gas_mass, stars_mass, newpoints, newvels, newgastype, newdusttype, new_eint_stars, new_eint_dust, new_eint_gas, supernova_pos
    
    #for cross-sections: use integrals in paper of the MRN distribution
    # to find cross-sectional area per species particle; use relative mass
    #fractions to figure out total cross-sectional area intercepted
    #by dust grains
    
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

def rad_heating(positions, ptypes, masses, sizes, cross_array, f_un):
    random_stars = positions[ptypes == 1]
    rs2 = np.array([])
    rs2 = random_stars
    if len(random_stars > 0):
        star_selection = (np.random.rand(len(random_stars)) < (masses[ptypes == 1]/solar_mass > 4.).astype('float') + len(random_stars)**-0.5 ) #selecting all stars over 4 solar masses for proper H II regions
        rs2 = random_stars[star_selection]
        if len(rs2) < 0:
            rs2 = random_stars
            
    lum_base = luminosity_relation(masses[ptypes == 1][star_selection]/solar_mass, 1)
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
            blocking_kernel = np.sum(((315./512.) * sizes[dists < sizes**2]**(-2) * masses[dists < sizes**2]/(mu_array[dists < sizes**2] * amu) * cross_array[dists < sizes**2]))
            
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
        lum_factor.append(np.nan_to_num(star_distance_2[ei] * np.sum(((gas_distance + 1).T**-2/star_distance[ei] * blocked[ei]).T, axis=0)/np.sum((gas_distance + 1)**-2, axis=0)))
    
    lum_factor = np.array(lum_factor)
    extinction = ((315./512.) * masses[ptypes != 1]/(mu_array[ptypes != 1] * amu) * cross_array[ptypes != 1]) * sizes[ptypes != 1]**(-2)
    
    exponential = np.exp(-np.nan_to_num(lum_factor))
    distance_factor = (np.nan_to_num(star_distance_2)**2 + np.ones(np.shape(star_distance_2)))
    a_intercepted = (np.pi * sizes**2)[ptypes != 1]
    
    lum_factor_2 = ((exponential/distance_factor).T * luminosities).T * a_intercepted * (np.ones(np.shape(extinction)) - np.exp(-extinction))
    lum_factor_2 = np.nan_to_num(lum_factor_2)
    
    lf2 = np.sum(lum_factor_2, axis=0) * dt_0 * solar_luminosity
    
    momentum = (np.sum(np.array([star_unit_norm[ak].T * lum_factor_2[ak] for ak in range(len(lum_factor_2))]), axis=0)/masses[ptypes != 1]).T * dt_0/c
    
    overall = overall_spectrum(masses[ptypes == 1]/solar_mass, np.ones(len(masses[ptypes == 1])))
    dest_wavelengths = destruction_energies**(-1) * (h * c)
    frac_dest = np.array([np.sum((overall[1] * overall[2] * overall[0])[overall[0] < dest_wavelengths[ai]]) for ai in np.arange(len(dest_wavelengths))])
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
    
    subt = new_fun[3][ptypes != 1] * new_fun[5][ptypes != 1] * min(cross_sections[3] * c * dt_0 * 1e3,1.)
    subt2= new_fun[4][ptypes != 1] * new_fun[5][ptypes != 1] * min(cross_sections[4] * c * dt_0 * 1e3,1.)
    
    new_fun[2][ptypes != 1] += subt
    new_fun[3][ptypes != 1] -= subt
    new_fun[4][ptypes != 1] -= subt2
    new_fun[5][ptypes != 1] -= subt + subt2
    
    #energy, composition change, impulse
    return lf2, new_fun.T, momentum

DIAMETER = 1e6 * AU
N_PARTICLES = 3000
N_INT_PER_PARTICLE = 100
V = (DIAMETER)**3
d = (V/N_PARTICLES * N_INT_PER_PARTICLE)**(1./3.)
d_sq = d**2
base_sfr = 0.02
#relative abundance for species in each SPH particle,  (H2, H, H+,He,He+Mg2SiO4,SiO2,C,Si,Fe,MgSiO3,FeSiO3)in that order
specie_fraction_array = np.array([.86,.14,0,0,0,0,0,0,0,0,0,0,0])
supernova_base_release = np.array([[.86,.14,0.,0.,0.,0.,0.1,0.1,0.1,0.1,0.1,0.1,0.1]])
cross_sections += sigma_effective(mineral_densities, mrn_constants, mu_specie)

base_imf = np.logspace(-1,1.7, 200)
d_base_imf = np.append(base_imf[0], np.diff(base_imf))
imf = kroupa_imf(base_imf) * d_base_imf
imf /= np.sum(imf)
points = (np.random.rand(N_PARTICLES, 3) - 0.5) * DIAMETER
points2 = copy.deepcopy(points)
neighbor = neighbors(points, d)
#print(nbrs)
#print(points)
velocities = (np.random.rand(N_PARTICLES, 3) - 0.5) * 0.0
mass = np.random.choice(base_imf, N_PARTICLES, p = imf) * solar_mass
sizes = (mass/m_0)**(1./3.) * d

particle_type = np.zeros([N_PARTICLES]) #0 for gas, 1 for star, 2 for dust

mu_array = np.zeros([N_PARTICLES])#array of all mu
E_internal = np.zeros([N_PARTICLES]) #array of all Energy
#copy of generate_E_array
#fills in E_internal array specified at the beginning
T = 5 * np.ones([N_PARTICLES]) #5 kelvins
T_FF = 3000000. #years
#fills the f_u array
f_un = np.array([specie_fraction_array] * N_PARTICLES)
mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(specie_fraction_array)
gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(specie_fraction_array)
cross_array = np.sum(f_un * cross_sections, axis = 1)/np.sum(specie_fraction_array)
E_internal = gamma_array * mass * k * T/(mu_array * m_h)
optical_depth = mass/(m_h * mu_array) * cross_array

#copy of generate_mu_array

critical_density = 1000*amu*10**6 #critical density of star formation

densities = np.array([density(j) for j in range(len(neighbor))])
delp = np.array([del_pressure(j) for j in range(len(neighbor))])
num_densities = np.array([num_dens(j) for j in range(len(neighbor))])
print("Begin SPH")
star_ages = np.ones(len(points)) * -1.
age = 0
plt.ion()
for iq in range(400): 
    if np.sum(particle_type[particle_type == 1]) > 0:
        rh = rad_heating(points, particle_type, mass, sizes, cross_array, f_un)
        E_internal[particle_type != 1] += rh[0]
        E_internal[particle_type == 0] *= np.nan_to_num((((sb * optical_depth * dt_0)/(gamma_array * (mass/(mu_array * m_h)) * k)) + T**-3)**(-1./3.)/T)[particle_type == 0]
        E_internal[particle_type == 2] *= np.nan_to_num((((4 * sb * optical_depth * dt_0)/(gamma_array * (mass/(mu_array * m_h)) * k)) + T**-3)**(-1./3.)/T)[particle_type == 2]
        E_internal[E_internal < t_cmb * (gamma_array * mass * k)/(mu_array * m_h)] == t_cmb * (gamma_array * mass * k)/(mu_array * m_h)
        f_un = rh[1]
        velocities[particle_type != 1] += rh[2]
        
        #on supernova event--- add new dust particle (particle_type == 2)
        
        print (np.max(star_ages/luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1)/(year * 1e10)))
        print (np.max(mass[particle_type == 1]/solar_mass))
        
        supernova_pos = np.where(star_ages/luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1)/(year * 1e10) > 1.)[0]
        print (len(supernova_pos))
        #print (star_ages/luminosity_relation(mass/solar_mass, np.ones(len(mass)), 1)/(year * 1e10))[supernova_pos]
        if len(supernova_pos) > 0:
        	for ku in supernova_pos:
        		impulse, indices = supernova_impulse(points, mass, ku, particle_type)
        		velocities[indices] += impulse
			dust_comps, gas_comps, star_comps, dust_mass, gas_mass, stars_mass, newpoints, newvels, newgastype, newdusttype, new_eint_stars, new_eint_dust, new_eint_gas, supernova_pos = supernova_explosion()
			E_internal[supernova_pos] = new_eint_stars
			f_un[supernova_pos] = star_comps
			mass[supernova_pos] = stars_mass
			E_internal = np.concatenate((E_internal, new_eint_dust, new_eint_gas))
			particle_type = np.concatenate((particle_type, newdusttype, newgastype))
			mass = np.concatenate((mass, dust_mass, gas_mass))
			f_un = np.vstack([f_un, dust_comps, gas_comps])
			velocities = np.concatenate((velocities, newvels, newvels))
			star_ages = np.concatenate((star_ages, np.ones(len(supernova_pos))* (-2), np.ones(len(supernova_pos))* (-2)))
			points = np.vstack([points, newpoints, newpoints])

			neighbor = neighbors(points, d)
            
        #specie_fraction_array's retention is deliberate; number densities are in fact increasing
        #so we want to divide by the same base
        mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(specie_fraction_array)
        gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(specie_fraction_array)
        cross_array = np.sum(f_un * cross_sections, axis = 1)/np.sum(specie_fraction_array)
        optical_depth = mass/(m_h * mu_array) * cross_array
        
    neighbor = neighbors(points, d)#find neighbors in each timestep
    num_neighbors = np.array([len(adjoining) for adjoining in neighbor])
    bg = bin_generator(mass, points, [4, 4, 4]); age += dt_0
    com = bg[0] #center of masses of each bin
    grav_accel = compute_gravitational_force(points, bg[0], bg[1], bg[2]).T #gravity is always acting, thus no cutoff distance introduced for gravity
    
    densities = np.array([density(j) for j in range(len(neighbor))])
    delp = np.array([del_pressure(j) for j in range(len(neighbor))])
    num_densities = np.array([num_dens(j) for j in range(len(neighbor))])
    
    pressure_accel = -np.nan_to_num((delp.T/densities * (particle_type == 0).astype('float')).T)
    
    total_accel = grav_accel + pressure_accel
        
    points += ((total_accel * (dt_0)**2)/2.) + velocities * dt_0
    velocities += (total_accel * dt_0)
    
    T = np.nan_to_num(E_internal * (mu_array * m_h)/(gamma_array * mass * k))
    E_internal = np.nan_to_num(E_internal)

    star_ages[(particle_type == 1) & (star_ages > -2)] += dt_0
    probability = base_sfr * (densities/critical_density)**(1.4) * ((dt_0/year)/T_FF)
    diceroll = np.random.rand(len(probability))
    particle_type[(particle_type == 0) & (num_neighbors > 1)] = ((diceroll < probability).astype('float'))[(particle_type == 0) & (num_neighbors > 1)]
    #this helps ensure that lone SPH particles don't form stars at late times in the simulation
    #ideally, there is an infinite number of SPH particles, each with infinitesimal density
    #that only has any real physical effects in conjunction with other particles
    
    xmin = np.percentile(points.T[0], 10)/AU
    xmax =  np.percentile(points.T[0], 90)/AU
    ymin =  np.percentile(points.T[1], 10)/AU
    ymax =  np.percentile(points.T[1], 90)/AU
    zmin = np.percentile(points.T[2], 10)/AU
    zmax = np.percentile(points.T[2], 90)/AU

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
    d = (V_new/len(points)/(0.9 - 0.1) * N_INT_PER_PARTICLE)**(1./3.)
    d_sq = d**2 
    sizes = (mass/m_0)**(1./3.) * d
    
    dist_sq = np.sum(points**2,axis=1)
    min_dist = np.percentile(dist_sq, 0)
    max_dist = np.percentile(dist_sq, 90)
    
    xpts = points.T[1:][0][particle_type == 0]/AU
    ypts = points.T[1:][1][particle_type == 0]/AU
    
    xstars = points.T[1:][0][particle_type == 1]/AU
    ystars = points.T[1:][1][particle_type == 1]/AU
    sstars = (mass[particle_type == 1]/solar_mass) * 2.
    
    colors = (f_un.T[5]/np.sum(f_un, axis=1))[particle_type == 0]
    
    
    plt.clf()
    plt.axis('equal')
    
    max_val = max(max(xpts[(dist_sq[particle_type == 0] < max_dist * 11./9.)]), max(ypts[(dist_sq[particle_type == 0] < max_dist * 11./9.)]))
    min_val = min(min(xpts[(dist_sq[particle_type == 0] < max_dist * 11./9.)]), min(ypts[(dist_sq[particle_type == 0] < max_dist * 11./9.)]))
    
    plt.scatter(xpts[(dist_sq[particle_type == 0] < max_dist * 11./9.)], ypts[(dist_sq[particle_type == 0] < max_dist * 11./9.)], c=colors[(dist_sq[particle_type == 0] < max_dist * 11./9.)],s=20, edgecolor='none', alpha=0.1)
    plt.colorbar()
    plt.scatter(xstars[(dist_sq[particle_type == 1] < max_dist * 11./9.)], ystars[(dist_sq[particle_type == 1] < max_dist * 11./9.)], c='black', s=sstars[(dist_sq[particle_type == 1] < max_dist * 11./9.)])
    plt.scatter(max(max_val, np.abs(min_val)),max(max_val, np.abs(min_val)),alpha=0.001)
    plt.scatter(-max(max_val, np.abs(min_val)),-max(max_val, np.abs(min_val)),alpha=0.001)
    plt.xlabel('Position (astronomical units)')
    plt.ylabel('Position (astronomical units)')
    plt.title('Ionization fraction in H II region (t = ' + str(age/year/1e6) + ' Myr)')
    plt.pause(1)
    
    print ('age=', age/year)
    #print (d/AU)
    print ('stars/total = ', float(np.sum(mass[particle_type == 1]))/np.sum(mass))
    print ('==================================')
    
'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
[ax.scatter(points.T[0][particle_type == 0]/AU, points.T[1][particle_type == 0]/AU, points.T[2][particle_type == 0]/AU, alpha=0.1)]
[ax.scatter(points.T[0][particle_type == 1]/AU, points.T[1][particle_type == 1]/AU, points.T[2][particle_type == 1]/AU, alpha=0.2)]
[ax.set_xlim3d(-DIAMETER/2/AU, DIAMETER/2/AU)]
[ax.set_ylim3d(-DIAMETER/2/AU, DIAMETER/2/AU)]
[ax.set_zlim3d(-DIAMETER/2/AU, DIAMETER/2/AU)]
[plt.show()]

[plt.scatter(points.T[0][particle_type == 0]/AU, points.T[1][particle_type == 0]/AU, c = np.log10(densities/critical_density)[particle_type == 0], s=30, edgecolor='none', alpha=0.1)]
[plt.scatter(points.T[0][particle_type == 0]/AU, points.T[2][particle_type == 0]/AU, c = np.log10(densities/critical_density)[particle_type == 0], s=30, edgecolor='none', alpha=0.1)]
[plt.scatter(points.T[1][particle_type == 0]/AU, points.T[2][particle_type == 0]/AU, c = np.log10(densities/critical_density)[particle_type == 0], s=30, edgecolor='none', alpha=0.1)]
[plt.colorbar()]
[plt.scatter(points.T[0][particle_type == 1]/AU, points.T[1][particle_type == 1]/AU, c = 'black', s=(mass[particle_type == 1]/solar_mass), alpha=1)]
[plt.axis('equal'), plt.show()]
plt.xlabel('Position (astronomical units)')
plt.ylabel('Position (astronomical units)')
plt.title('Density in H II region')

arb_points = (np.random.rand(N_PARTICLES * 10, 3) - 0.5) * (max(ymax, xmax) - min(xmin, ymin))
#narb = neighbors_arb(points, arb_points)
darb = density_arb(arb_points)
tarb = temperature_arb(arb_points)
plt.ion()
#[ax.scatter(points.T[0][particle_type == 0]/AU, points2.T[1][particle_type == 0]/AU, points2.T[2][particle_type == 0]/AU, alpha=0.1)]
[plt.scatter(arb_points.T[0]/AU, arb_points.T[1]/AU, c = np.log10(darb/critical_density), s=8, alpha=0.7, edgecolor='none'), plt.colorbar()]
[plt.scatter(points.T[1:][0][particle_type == 1]/AU, points.T[1:][1][particle_type == 1]/AU, c = 'black', s=(mass[particle_type == 1]/solar_mass) * 2, alpha=1)]
[plt.axis('equal'), plt.show()]'''