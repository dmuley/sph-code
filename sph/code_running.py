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

from navier_stokes_cleaned import *

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
    
    f_un = f_un/np.sum(f_un, axis=1) #normalizing composition
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

INTERPOLATED PLOTTING:
arb_points = (np.random.rand(N_PARTICLES * 10, 3) - 0.5) * (max(ymax, xmax) - min(xmin, ymin))
#narb = neighbors_arb(points, arb_points)
darb = density_arb(arb_points)
tarb = temperature_arb(arb_points)
#[ax.scatter(points.T[0][particle_type == 0]/AU, points2.T[1][particle_type == 0]/AU, points2.T[2][particle_type == 0]/AU, alpha=0.1)]
[plt.scatter(arb_points.T[0]/AU, arb_points.T[1]/AU, c = np.log10(darb/critical_density), s=8, alpha=0.7, edgecolor='none'), plt.colorbar()]
[plt.scatter(points.T[1:][0][particle_type == 1]/AU, points.T[1:][1][particle_type == 1]/AU, c = 'black', s=(mass[particle_type == 1]/solar_mass) * 2, alpha=1)]
[plt.axis('equal'), plt.show()]'''