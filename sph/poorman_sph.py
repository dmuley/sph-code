import numpy as np
from scipy import constants
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D

G = constants.G
k = constants.Boltzmann
sb = constants.Stefan_Boltzmann
AU = constants.au
amu = constants.physical_constants['atomic mass constant'][0]
h = constants.h
c = constants.c
wien = constants.Wien

solar_mass = 1.989e30 #kilograms
solar_luminosity = 3.846e26 #watts
solar_lifespan = 1e10 #years
t_cmb = 2.732
t_solar = 5776
mean_opacity = 1e-7

dt_0 = 60. * 60. * 24. * 365 * 25000. # 25000 years
year = 60. * 60. * 24. * 365.
base_sfr = 1.0e-13

#molecular hydrogen, helium, neutral hydrogen, H+, e-
#Later, add various gas-phase metals to this array
mol_weights = np.array([1.00794 * 2, 4.002602, 1.00794, 1.00727647, 548.579909e-6])
cross_sections = np.array([3.34e-30, 3.34e-30, 6.3e-25, 5e-23, 0.]) 
destruction_energies = np.array([7.2418e-19, 3.93938891e-18, 2.18e-18, 10000, 10000])
adiab_indices = (np.array([7./5., 5./3., 5./3., 5./3., 5./3.]) - 1)**(-1)

#Dust must be considered separately 
#SPACE TO CONSIDER DUST COMPOSITIONS AND RADII
#End dust

def kroupa_imf(base_imf):
	coeff_0 = 1
	imf_final = np.zeros(len(base_imf))
	imf_final[base_imf < 0.08] = coeff_0 * base_imf[base_imf < 0.08]**-0.3
	coeff_1 = 2.133403503
	imf_final[(base_imf >= 0.08) & (base_imf < 0.5)] = coeff_1 * (base_imf/0.08)[(base_imf >= 0.08) & (base_imf < 0.5)]**-1.3
	coeff_2 = 0.09233279398
	imf_final[(base_imf >= 0.5)] = coeff_2 * coeff_1 * (base_imf/0.5)[(base_imf >= 0.5)]**-2.3
	
	return imf_final

def luminosity_relation(base_imf, imf):
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
	
	dimf = np.append(base_imf[0], np.diff(base_imf))
	
	luminosity_relation = coeff_luminosity * base_imf**exp_luminosity
	
	return luminosity_relation

def planck_law(temp, wl, dwl):
	emission = np.nan_to_num((2 * h * c**2 / wl**5)/(np.exp(h * c/(wl * k * temp)) - 1))
	emission[0] = 0
	#print emission
	norm_constant = np.sum(emission * dwl)
	return np.nan_to_num(emission/norm_constant)

def bin_generator(masses, positions, subdivisions):
	#We want equal mass per axis in each subdivision
	positional_masses = np.array([masses[np.argsort(positions.T[b])] for b in range(len(subdivisions))])
	positional_masses = np.cumsum(positional_masses, axis = 1)
	
	subdivision_cutoffs = [np.zeros(a + 1) for a in subdivisions]
	subdivision_pairs = [[] for a in subdivisions]
	
	for i in range(len(subdivision_cutoffs)):
		for j in range(subdivisions[i]):
			subdivision_cutoffs[i][j + 1] = np.sort(positions.T[i])[positional_masses[i] <= positional_masses[i][-1]/(subdivisions[i]) * (j + 1)][-1]
		
	subdivision_pairs = [np.array([f[:-1], f[1:]]).T for f in subdivision_cutoffs]
		
	subdivision_boxes = [s for s in itertools.product(*subdivision_pairs)]
	com = []
	grid_masses = np.array([])
	length_scale = np.sqrt(np.min([np.average(np.diff(u)**2) for u in subdivision_cutoffs]))
	
	for wq in range(len(subdivision_boxes)):
		com_partial = np.array([])
		ranges = 0
		for l in range(len(subdivision_boxes[wq])):
			ir = (positions.T[l] >= subdivision_boxes[wq][l][0]).astype('int')
			ir2 = (positions.T[l] <= subdivision_boxes[wq][l][1]).astype('int')
			
			ranges += ir
			ranges += ir2
		
		#print max(ranges)
		ranges[ranges < max(ranges)] = 0
		ranges = ranges.astype('bool')
		
		grid_masses = np.append(grid_masses, np.sum(masses[ranges]))
		#particle_location.append(np.arange(len(masses))[ranges])
		
		for m in range(len(subdivision_boxes[wq])):
			com_partial = np.append(com_partial, np.sum(positions.T[m][ranges] * masses[ranges])/np.sum(masses[ranges]))
		
		com.append(com_partial)
	return np.array(com), grid_masses, length_scale

def compute_gravitational_force(particle_positions, grid_com, grid_masses, length_scale):
	forces = 0;
	for i in range(len(grid_com)):
		forces += ((np.sum((particle_positions - grid_com[i]).T**2, axis=0) + (length_scale)**2)**-1.5 * -(particle_positions - grid_com[i]).T * grid_masses[i] * G)
		
	return forces

N_PARTICLES = 10000
composition = np.array([0.86, 0.14, 0, 0, 0])

base_imf = np.logspace(-2,1.5, 200)
d_base_imf = np.append(base_imf[0], np.diff(base_imf))
imf = kroupa_imf(base_imf) * d_base_imf
imf /= np.sum(imf)

random_sph = (np.random.rand(N_PARTICLES, 3) - 0.5) * 1e6 * AU
velocities = 0
masses = np.random.choice(base_imf, N_PARTICLES, p = imf) * solar_mass

for iq in range(100):
	bg = bin_generator(masses, random_sph, [6, 6, 6])
	cgf = compute_gravitational_force(random_sph, bg[0], bg[1], bg[2])
	random_sph += ((cgf * (dt_0)**2)/2.).T + velocities * dt_0
	velocities += (cgf * dt_0).T
		
	print iq
	
	
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(random_sph.T[0]/AU, random_sph.T[1]/AU, random_sph.T[2]/AU, alpha=0.01)
'''
plt.plot(random_sph.T[0], velocities.T[0], '.', alpha=0.1)
plt.plot(random_sph.T[1], velocities.T[1], '.', alpha=0.1)
plt.plot(random_sph.T[2], velocities.T[2], '.', alpha=0.1)'''
