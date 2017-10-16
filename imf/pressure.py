import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

def optical_dimming(density, x, pressure, k):
	cum_density = np.append(0, cumtrapz(density, x = x));
	base_array = np.zeros(len(x))
	for u in range(1, len(base_array)):
		base_array[u] = np.sum(pressure[:u]**1.5 * density[:u]**0.5 * x[:u]**2 * np.exp(k * (-cum_density[u - 1] + cum_density[:u])) * (np.append(np.diff(x)[0], np.diff(x)))[:u] * k * -density[u] * np.diff(x)[:u])
		
	return base_array


#Throughout this process, energy is lost in collisions between hydrogen atoms
#and dust particles and so is immediately radiated away. Thus the collapse is
#ISOTHERMAL. Eventually, radiation begins to be trapped by the collapsing cloud
#and so the radiative contribution to internal energy must be considered. The 
#radiative component INCREASES the internal energy of the particles.

z = 10**np.linspace(-8, -2, 7)
G = 0.000000001
s_sb = 0.000000001


#plt.plot(density)
#colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

jeans_cutoffs = []


for w in range(len(z)):

	base = np.linspace(0,10,5001)
	b2 = np.linspace(0,10,5001)
	density = (base**6 + 1)**(-1)
	density_0 = (base**6 + 1)**(-1)

	N = ((base + 0.01)**3 - base**3) * density_0

	menc = np.append(0, cumtrapz(density * base**2, base))
	pbase = -menc * density/base**2; pbase[0] = 0	
	pressure = cumtrapz(pbase)
	pressure = np.append(0, pressure)
	pressure -= min(pressure)
	
	temperature = pressure/density
	
	jeans_cutoff_base = np.array([])
	
	for u in range(0, 3000):
		de_part = -density**0.5 * pressure**1.5 * z[w] * (1. - z[w]) * (1 - density)* 5 + temperature**3 * np.append(np.diff(temperature)[0], np.diff(temperature)) * s_sb
		de_part *= 0.02
		e_part = 1.5 * pressure + de_part
		
		pressure += de_part
		temperature *= e_part/(1.5 * pressure)
		
		dv = (N/density)
		b2_prov = np.cumsum(dv)**(1./3.)
		#pressure += np.nan_to_num(menc/b2 - menc/b2_prov * density)
	
		stability_proxy = np.append(0, np.diff(pressure)) - pbase
		stability_cutoff = np.where(np.diff(stability_proxy/np.abs(stability_proxy)) > 1.)
		
		#print stability_cutoff[0]
		
		jeans_cutoff_base = np.append(jeans_cutoff_base, menc[stability_cutoff[0]]/max(menc))
		
		#jeans_cutoff_base.append(menc[stability_cutoff])
			
		#plt.plot(menc[1:]/max(menc), np.abs(np.diff(pressure) - pbase[1:]), alpha=0.01)
		print u
	
	plt.plot(jeans_cutoff_base, np.log(np.ones(len(jeans_cutoff_base)) * z[w])/np.log(10.), '.', alpha=0.002)
	jeans_cutoffs.append(jeans_cutoff_base.astype('float'))
	

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

G = 0.0000001
k_pressure = 1.
s_sb = 0.00000001
dt = 0.01

base = np.linspace(0,10,5001)
d = (base**6 + 1)**(-1)
metallicity = 0.01
iterations = 10

def evolve_system_2(metallicity, base, d, iterations, k_pressure, s_sb, dt):
	density = d[1:]
	dV = (base[1:]**3 - base[:-1]**3)
	n_particles = density * dV
	
	menc = np.cumsum(n_particles)
	d_pressure = -menc * density/base[1:]**2
	d_pressure_hydrostatic = -menc * density/base[1:]**2
	pressure = np.cumsum(d_pressure); pressure -= min(pressure)
	
	temperature = pressure/density
	
	mencs = np.array([])
	
	for i in range(iterations):
		#simulate reduction of pressure and temperature due to metallicity
		#simulate reduction of pressure/temperature due to blackbody cooling
		
		dp_metallicity = -pressure**1.5 * density**0.5 * metallicity * k_pressure * dt
		dp_blackbody = np.append(0, np.diff(temperature**4)) * s_sb * dt
		
		#fractional_reduction = (dp_metallicity + dp_blackbody)/pressure
		#fractional_reduction[1] = np.average([fractional_reduction[0], fractional_reduction[2]])
		
		pressure -= dp_metallicity + dp_blackbody
		temperature -= (dp_metallicity + dp_blackbody)/density
		
		d_pressure = np.append(0, np.diff(pressure))
		menc_values = menc[np.diff(np.sign(d_pressure - d_pressure_hydrostatic)) != 0]
		
		mencs = np.append(mencs, menc_values[:20])
		
		plt.plot(menc, d_pressure - d_pressure_hydrostatic, alpha=0.002)
		
	return mencs, np.ones(len(mencs)) * metallicity
		


def evolve_system(metallicity, base, d, iterations):
	density = d[1:]
	dV = (base[1:]**3 - base[:-1]**3)
	n_particles = density * dV
	
	menc = np.cumsum(n_particles)
	d_pressure = -menc * density/base[1:]**2
	d_pressure_hydrostatic = -menc * density/base[1:]**2
	pressure = np.cumsum(d_pressure); pressure -= min(pressure)
	
	temperature = pressure/density
	
	for i in range(iterations):
		#simulate reduction of pressure and temperature due to metallicity
		#then simulate reduction of pressure/temperature due to blackbody cooling
		
		dp_metallicity = -pressure**1.5 * density**0.5 * metallicity * k_pressure * dt * np.exp(-density)
		dp_blackbody = np.append(0, np.diff(temperature**4)) * s_sb * dt
		
		fractional_reduction = (dp_metallicity + dp_blackbody)/pressure; 
		fractional_reduction[1] = np.average([fractional_reduction[0], fractional_reduction[2]])
		temperature *= (1 + fractional_reduction)
		pressure *= (1 + fractional_reduction)
		
		#handling contraction of cloud under gravity
		
		d_pressure = np.append(0, np.diff(pressure))
		d_pressure_hydrostatic = -menc * density/base[1:]**2
		
		accel = (d_pressure_hydrostatic - d_pressure)/density; accel[1] = np.average([accel[0], accel[2]])
		d_dist = -accel * (dt)**2; d_dist[-1] = d_dist[-2]
		
		dV_new = (base[1:] + d_dist)**3 - (base[:-1] + np.append(0, d_dist[:-1]))**3
		density_new = n_particles/dV_new

		d_energy_density = ((base[1:] + d_dist)**(-1) - base[1:]**-1) * menc * density_new
		#plt.plot(base[1:], d_energy_density, color='blue', alpha=0.002)
		plt.plot(base[1:], menc, color='red', alpha=0.002)
		
		pressure_new = pressure + d_energy_density
		temperature_new = temperature + d_energy_density/density_new
		
		pressure = pressure_new
		temperature = temperature_new
		base = np.append(0, base[1:] + d_dist)
		
		print i
		
		
############################################################################################################################################
############################################################################################################################################
############################ NEW IMPLEMENTATION BELOW HERE #################################################################################
############################################################################################################################################
############################################################################################################################################


#assuming the cloud oscillates from marginal instability to marginal stability
def cloud_model(base, d, metallicity, opacity, iterations, color='blue'):
	k_metallicity = 0.01
	dt = 0.1
	G = 0.00001
	c_ad = 5./3.
	
	density = copy.deepcopy(d[1:])
	dV = copy.deepcopy(base[1:]**3 - base[:-1]**3)
	n_particles = copy.deepcopy(density * dV)
	
	menc = copy.deepcopy(np.cumsum(n_particles))
	d_pressure = -menc * density/base[1:]**2
	d_pressure_hydrostatic = copy.deepcopy(-menc * density/base[1:]**2)
	pressure = np.cumsum(d_pressure) * base[1]; pressure -= min(pressure)
	
	temperature = pressure/density
	dust_temperature = np.zeros(len(temperature))
	
	#plt.plot(base[1:], temperature, color)
	
	for i in range(iterations):
		dp_metallicity = density**2 * np.sqrt(temperature) * (np.exp(-density * opacity)) * (temperature - dust_temperature) * k_metallicity * metallicity * dt
		dust_temperature += density * np.sqrt(temperature) * (np.exp(-density * opacity)) * (temperature - dust_temperature) * k_metallicity * dt 
		dp = dp_metallicity #will add more factors later
		
		
		frac_pressure = -np.exp(np.log(dp) - np.log(pressure)); frac_pressure[-1] = 0
		pressure *= (1 + frac_pressure)
		temperature *= (1 + frac_pressure)	
		
	plt.plot(menc/max(menc), np.cumsum(pressure * base[1:]**2)/500**2 - np.cumsum(menc * n_particles/base[1:]), alpha=0.2, color=color)
		
	plt.xlim(0, 10)
	
def cloud_model_2(base, d, metallicity, opacity, iterations, color='blue'):
	k_metallicity = 0.01
	k_pressure = 0.001
	dt = 0.00001
	G = 0.00001
	c_ad = 5./3.
	
	density = copy.deepcopy(d[1:])
	dV = copy.deepcopy(base[1:]**3 - base[:-1]**3) * 4 * np.pi/3.
	n_particles = copy.deepcopy(density * dV)
	
	menc = copy.deepcopy(np.cumsum(n_particles))
	
	d_pressure = -menc * density/base[1:]**2
	d_pressure_hydrostatic = copy.deepcopy(-menc * density/base[1:]**2)
	pressure = np.cumsum(d_pressure) * base[1]; pressure -= min(pressure)
	
	temperature = pressure/density/k_pressure
	dust_temperature = np.zeros(len(temperature))
	velocities = np.zeros(len(temperature))
		
	#plt.plot(base[1:], temperature, color)
	
	for i in range(iterations):
		dp_metallicity = density**2 * np.sqrt(temperature) * (np.exp(-density * opacity)) * (temperature - dust_temperature) * k_metallicity * metallicity * dt
		dust_temperature += density * np.sqrt(temperature) * (np.exp(-density * opacity)) * (temperature - dust_temperature) * k_metallicity * dt 
		dp = dp_metallicity #will add more factors later
		
		frac_pressure = -dp/pressure; frac_pressure[-1] = 0
		pressure *= (1 + frac_pressure)
		temperature *= (1 + frac_pressure)
		
		d_pressure = np.append(0, np.diff(pressure))/np.diff(base)/density
		accel = -(d_pressure + menc/base[1:]**2)
	
		accel[0] = accel[1]/base[2] * base[1]
		velocities += accel * dt
	
		base_new = base + np.append(0, (velocities * dt + accel * (dt)**2./2) * np.sign(base[1:]))
		dV_new = copy.deepcopy(base_new[1:]**3 - base_new[:-1]**3) * 4 * np.pi/3.
		density *= dV/dV_new
		temperature += pressure * (dV - dV_new)/(dV_new * density)/k_pressure
		pressure = k_pressure * temperature * density
		
		dV = dV_new
		base = base_new
		
	plt.plot(base[1:], density, alpha=0.2, color=color)

def cloud_model_3(base, d, metallicity, opacity, iterations, color='blue'):
	k_metallicity = 0.1
	k_pressure = 0.001
	dt = 0.00001
	G = 0.00001
	c_ad = 5./3.
	n = (np.sqrt(iterations) * metallicity**(-0.1) * (10)**(-0.1)).astype('int')
	print int(iterations/n)
	
	density = copy.deepcopy(d[1:])
	dV = copy.deepcopy(base[1:]**3 - base[:-1]**3) * 4 * np.pi/3.
	n_particles = copy.deepcopy(density * dV)
	
	menc = copy.deepcopy(np.cumsum(n_particles))
	
	d_pressure = -menc * density/base[1:]**2
	d_pressure_hydrostatic = copy.deepcopy(-menc * density/base[1:]**2)
	pressure = np.cumsum(d_pressure) * base[1]; pressure -= min(pressure)
	
	temperature = pressure/density/k_pressure
	dust_temperature = np.zeros(len(temperature))
	velocities = np.zeros(len(temperature))
	base_new = np.zeros(len(temperature))
	accel = np.zeros(len(temperature))
	
	temperature = temperature.astype('float128')
	dust_temperature = dust_temperature.astype('float128')
	dV = dV.astype('float128')
	density = density.astype('float128')
	pressure = pressure.astype('float128')
	base = base.astype('float128')
	base_new = base_new.astype('float128')
	velocities = velocities.astype('float128')
	accel = accel.astype('float128')
	n_particles = n_particles.astype('float128')
	
	
	for i in range(iterations):
		dp_metallicity = density**2 * np.sqrt(temperature) * (np.exp(-density * opacity)) * (temperature - dust_temperature) * k_metallicity * metallicity * dt
		dust_temperature += density * np.sqrt(temperature) * (np.exp(-density * opacity)) * (temperature - dust_temperature) * k_metallicity * dt 
		dp = dp_metallicity #will add more factors later

		frac_pressure = -dp/pressure; frac_pressure[-1] = 0
		pressure *= (1 + frac_pressure)
		temperature *= (1 + frac_pressure)
		
		if (i % n) == n - 1:
			d_pressure = np.append(0, np.diff(pressure)) * np.exp(-np.log(np.diff(base)) - np.log(density)) 
			grav_accel = np.exp(np.log(menc) - 2 * np.log(base[1:]))
			
			accel = -(d_pressure.astype('float128') + grav_accel.astype('float128'))
			accel[0] = accel[1]/base[2] * base[1]
			velocities += accel * dt

			base_new = base + np.append(0, velocities * dt * n)
			dV_new = copy.deepcopy(base_new[1:]**3 - base_new[:-1]**3) * 4 * np.pi/3.
			dV_new = dV_new.astype('float128')
			
			density = n_particles/dV_new
			#print np.where(dV/dV_new < 1)
			temperature += pressure * (dV - dV_new)/n_particles/k_pressure
			pressure = k_pressure * temperature * density
			
			dV = dV_new
			base = base_new
		
	jeans_proxy = np.cumsum(pressure * dV) * (1-0.9999**3) - np.cumsum(menc * n_particles * base[1:]**-1 * (0.9999**-1 - 1))
	return (menc/max(menc))[np.abs(np.diff(np.sign(jeans_proxy))) > 1.], metallicity * np.ones(len((menc/max(menc))[np.abs(np.diff(np.sign(jeans_proxy))) > 1.]))
	
	


	#plt.plot(menc/max(menc), np.cumsum(pressure * dV) * (1-0.9999**3) - np.cumsum(menc * n_particles * base[1:]**-1 * (0.9999**-1 - 1)), color=color, alpha=0.2)

for k in range(len(metallicities)):
	mencs = []
	iters = []
	for q in (np.append([0, 0.2, 0.5, 1, 2, 3], np.arange(2, 9)**2) * 50).astype('int'):
		cm = cloud_model_3(base, d, metallicities[k], 0.1, q); base = np.linspace(0, 4., 5001);
		mencs.extend(cm[0])
		iters.append(q)
		
	plt.plot(iters, mencs, color = colors[k % 7], alpha=0.2)		

for q in (np.append([0, 0.2, 0.5, 1, 2, 3], np.arange(2, 9)**2) * 50).astype('int'):
	mencs = []
	jeans_proxies = []
	metallicity_final = []
	for k in range(len(metallicities)):
		cm = cloud_model_3(base, d, metallicities[k], 0.1, q, colors[k % 7]); base = np.linspace(0, 4., 5001);
		mencs.extend(cm[0])
		metallicity_final.extend(cm[1])
	
	plt.plot(metallicities, mencs, color = colors[q/50 % 7], alpha=0.2)
	

################## THIS MODEL CONTAINS EVERYTHING. IT IS DESIGNED TO BE UNITALLY CORRECT ####################
################## PREVIOUS MODELS ARE ONLY CONCEPTUALLY WORKING. THIS ONE ACTUALLY WORKS PROPERLY ##########

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz	
from scipy import constants
import copy
from scipy.optimize import fsolve, root, newton

base = np.linspace(0,4,5001)
density = copy.deepcopy(((base**6 + 1)**-1)[1:])
iterations = 5000
gas_masses = np.array([1.00794, 4.002602])
nd = np.array([8.47996371e22 * 0.91, 8.47996371e22 * 0.09])
gas_number_densities = np.einsum('i, j', nd, density)
grain_densities = np.array([5, 6, 7])
grain_radius = np.array([1, 2, 3]) * 10**-4
metallicities = np.array([0.003, 0.002, 0.001])

def cm5(base, iterations, gas_masses, gas_number_densities, grain_densities, grain_radius, metallicities):
	#base: state this value in AU
	#gas_masses: state in AMU
	#gas_number_densities: state in m^-3
	#grain_densities: state in kg m^-3
	#grain_radius: state in m
	#metallicities: state as a decimal
	
	base = base.astype('float128')
	iterations = int(iterations)
	gas_masses = gas_masses.astype('float128')
	gas_number_densities = gas_number_densities.astype('float128')
	grain_densities = grain_densities.astype('float128')
	grain_radius = grain_radius.astype('float128')
	metallicities = metallicities.astype('float128')
	
	k = constants.Boltzmann
	G = constants.G
	AU = constants.au
	sb = constants.sigma
	t_cmb = 2.725
	mean_opacity = 1e-7 #in m^2/kg
	
	solar_mass = 1.989e30
	dt = 1.00
	amu = constants.physical_constants['atomic mass constant'][0]
	
	base *= AU
	gas_masses *= amu
	
	gas_mass_density = np.zeros(len(base[1:])).astype('float128')
	gas_number_density = np.zeros(len(base[1:])).astype('float128')
	grain_factor = 0
	
	#determining number and size of dust grains
	
	for i in range(len(gas_masses)):
		gas_mass_density += gas_masses[i] * gas_number_densities[i]
		gas_number_density += gas_number_densities[i]
		
	for j in range(len(metallicities)):
		grain_mass = grain_densities[j] * 4./3. * np.pi * grain_radius[j]**3
		grain_factor += metallicities[j]/grain_mass
		
	grain_number_density = grain_factor * gas_mass_density
	
	gas_mass_density = gas_mass_density.astype('float128')
	grain_number_density = grain_number_density.astype('float128')
	
	mass_density = gas_mass_density * (1 + np.sum(metallicities))
	dV = copy.deepcopy(base[1:]**3 - base[:-1]**3) * 4 * np.pi/3.
	dM = mass_density * dV
	menc = np.cumsum(mass_density * dV)
	
	d_pressure = -G * menc * mass_density/base[1:]**2
	pressure = np.cumsum(d_pressure) * base[1]; pressure -= min(pressure)
	#P = nkT; T = P/nK
	temperature = pressure/(np.sum(gas_number_densities, axis = 0)*k)
	dust_temperature = np.vstack([np.zeros(len(pressure))] * len(metallicities)).astype('longdouble')
	
	velocities = np.zeros(len(temperature)).astype('float128')
	accel = np.zeros(len(temperature)).astype('float128')
	
	accel_diff = (np.sum(metallicities)**-0.1 * np.sqrt(iterations) * 10**-0.1).astype('int')
	
	############
	
	print accel_diff
	
	for iter in range(iterations):
		#energy lost is now multilinear in number densities
		dp_metallicity = 0
		
		planck_part = 0;
		rosseland_part = 0;
		
		for gnd2 in range(len(gas_number_densities)):
			col_density = -np.cumsum(gas_number_densities[gnd2] * np.diff(base)) * gas_masses[gnd2]
			col_density -= min(col_density)
			
			rosseland_part += col_density**2 * mean_opacity
			planck_part += mean_opacity
			
		denominator = rosseland_part + planck_part**-1
		
		#print denominator
		
		for gnd in range(len(gas_number_densities)):
			for met in range(len(metallicities)):
				m_grain = (4./3. * np.pi * grain_radius[j]**3 * grain_densities[met])
				n_dust = (metallicities[met] * gas_mass_density)/m_grain
				cross_section = np.pi * grain_radius[met]**2
				v_gas = np.sqrt(3 * k * temperature/gas_masses[gnd])
				
				dtm = copy.deepcopy(dust_temperature[met])
				
				polynomials = np.vstack([np.zeros(len(temperature))] * 5)
				polynomials[0] += 4 * sb
				polynomials[3] += gas_number_densities[gnd] * cross_section * v_gas * (2 * k) * denominator
				polynomials[4] += -gas_number_densities[gnd] * cross_section * v_gas * (2 * k) * temperature * denominator - 4 * sb * t_cmb**4
				
				polynomials = np.nan_to_num(polynomials)
				
				dust_temperature[met] = np.array([max(np.roots(a)) for a in polynomials.T]).ravel().real

				e_lost = (4 * n_dust * sb * dust_temperature[met]**4)/(denominator) * dt

				dp_metallicity += e_lost.astype('float128')
		
		dp = -dp_metallicity * 2./3.
		frac_pressure = dp/pressure
		
		#print frac_pressure
		pressure *= (1 + frac_pressure)
		temperature *= (1 + frac_pressure)
		
		pressure = np.nan_to_num(pressure)
		temperature = np.nan_to_num(temperature)
		
		#gravitational contraction of the cloud
		if (iter % accel_diff) == accel_diff - 1:
			d_pressure = np.append(0, np.diff(pressure))/np.diff(base) * density**(-1)
			grav_accel = G * menc/base[1:]**2
			
			accel = -(d_pressure.astype('float128') + grav_accel.astype('float128'))
			accel[0] = accel[1]/base[2] * base[1]
			velocities += accel * dt

			base_new = base + np.append(0, velocities * dt * accel_diff)
			dV_new = copy.deepcopy(base_new[1:]**3 - base_new[:-1]**3) * 4 * np.pi/3.
			dV_new = dV_new.astype('float128')
			
			temperature += pressure * (dV - dV_new)/gas_number_density/k
			pressure = k * temperature * gas_number_density
			
			pressure = np.nan_to_num(pressure)
			temperature = np.nan_to_num(temperature)
			base = np.nan_to_num(base)
			
			for m in range(len(gas_number_densities)):
				gas_number_densities[m] *= dV/dV_new
				
			gas_mass_density *= dV/dV_new			
			
			dV = dV_new
			base = base_new		
		
			plt.plot(menc/solar_mass, pressure, alpha=0.2)
for gnd in range(1,2):
    cm5(base * 10, 200, gas_masses, gas_number_densities * (10)**-3, grain_densities, grain_radius, metallicities * gnd * 0.2)


#################### SIMPLIFIED MODEL, DOES NOT ACCOUNT FOR ENERGY ABSORPTION BY DUST GRAINS #################
#################### COMPUTATIONALLY LESS EXPENSIVE AND MORE TRACTABLE THAN THAT ABOVE #######################
#################### ACCOUNTS FOR ALL PHYSICAL CONSTANTS PROPERLY ############################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz	
from scipy import constants
import copy
from scipy.optimize import fsolve, root, newton
from pandas import rolling_mean

base = np.linspace(0,4,1001)
density = copy.deepcopy(((base**10 + 1)**-1)[1:])
iterations = 500
gas_masses = np.array([1.00794, 4.002602])
nd = np.array([8.47996371e22 * 0.91, 8.47996371e22 * 0.09])
gas_number_densities = np.einsum('i, j', nd, density)
grain_densities = np.array([5000, 6000, 7000])
grain_radius = np.array([1, 2, 3]) * 1e-5
specific_heats = np.array([0.8, 0.9, 1.]) * 1e3
metallicities = np.array([0.003, 0.002, 0.001])

def cm6(base, iterations, gas_masses, gas_number_densities, grain_densities, grain_radius, specific_heats, metallicities):
	#base: state this value in AU
	#gas_masses: state in AMU
	#gas_number_densities: state in m^-3
	#grain_densities: state in kg m^-3
	#grain_radius: state in m
	#metallicities: state as a decimal
	
	base = base.astype('float128')
	iterations = int(iterations)
	gas_masses = gas_masses.astype('float128')
	gas_number_densities = gas_number_densities.astype('float128')
	grain_densities = grain_densities.astype('float128')
	grain_radius = grain_radius.astype('float128')
	metallicities = metallicities.astype('float128')
	
	k = constants.Boltzmann
	G = constants.G
	AU = constants.au
	sb = constants.sigma
	c = constants.c
	t_cmb = 2.725
	mean_opacity = 1e-7 #in m^2/kg
	adiab_constant = 5./3.
	
	solar_mass = 1.989e30
	amu = constants.physical_constants['atomic mass constant'][0]
	
	base *= AU
	gas_masses *= amu
	
	gas_mass_density = np.zeros(len(base[1:])).astype('float128')
	gas_number_density = np.zeros(len(base[1:])).astype('float128')
	grain_factor = 0
	
	#determining number and size of dust grains
	
	for i in range(len(gas_masses)):
		gas_mass_density += gas_masses[i] * gas_number_densities[i]
		gas_number_density += gas_number_densities[i]
			
	gas_mass_density = gas_mass_density.astype('float128')
	
	mass_density = gas_mass_density * (1 + np.sum(metallicities))
	dV = copy.deepcopy(base[1:]**3 - base[:-1]**3) * 4 * np.pi/3.
	dM = mass_density * dV
	dN = gas_number_density * dV
	menc = np.cumsum(mass_density * dV)
	
	d_pressure = -G * menc * mass_density/base[1:]**2
	pressure = np.cumsum(d_pressure) * base[1]; pressure -= min(pressure)
	#P = nkT; T = P/nK
	temperature = pressure/(np.sum(gas_number_densities, axis = 0)*k)
	dust_temperature = np.vstack([np.zeros(len(pressure))] * len(metallicities)).astype('longdouble')
	dt_0 = 1e-8
	dt = dt_0 / max(mass_density)
	total_time = 0.
	velocities = np.zeros(len(temperature)).astype('float128')
	accel = np.zeros(len(temperature)).astype('float128')
	
	accel_diff = (np.sqrt(iterations)).astype('int')
	
	total_densities = []
	pressures = []
	
	
	############
	
	print accel_diff
	
	for iter in range(iterations):
		tau = 0;
		
		for gnd2 in range(len(gas_number_densities)):
			col_density = -np.cumsum(gas_number_densities[gnd2] * np.diff(base)) * gas_masses[gnd2]
			col_density -= min(col_density)
			
			tau += col_density * mean_opacity
		
		dtau = gas_number_densities[gnd2] * gas_masses[gnd2] * np.diff(base) * mean_opacity
		
		dp_metallicity = 0
		
		for gnd in range(len(gas_number_densities)):
			for met in range(len(metallicities)):
				m_grain = (4./3. * np.pi * grain_radius[met]**3 * grain_densities[met])
				n_dust = (metallicities[met] * gas_mass_density)/m_grain
				cross_section = np.pi * grain_radius[met]**2
				v_gas = np.sqrt(3 * k * temperature/gas_masses[gnd])
				
				e_lost = 0
				
				e_lost_0 = gas_number_densities[gnd] * n_dust * cross_section * v_gas * (2 * k) * (temperature - dust_temperature[met]) * dt * np.exp(-tau)
				#e_lost_0 += sb * dust_temperature[met]**4 * 4 * cross_section/(specific_heats[met]) * dt * np.exp(-tau)
				dust_temperature[met] -= sb * dust_temperature[met]**4 * 4 * cross_section * dt/(specific_heats[met])
				dust_temperature[met] += e_lost_0/(specific_heats[met] * n_dust * m_grain)
				
				dp_metallicity += e_lost_0.astype('float128')
		
		dp = -dp_metallicity * 2./3.
		#print dp
		frac_pressure = dp/pressure; frac_pressure[-1] = 0
		frac_pressure += np.ones(len(frac_pressure)).astype('float128')
		
		pressure *= frac_pressure
		temperature *= frac_pressure
		
		pressure = np.nan_to_num(pressure)
		temperature = np.nan_to_num(temperature)
		
		total_time += dt
		
		#plt.plot(menc/solar_mass, temperature - dust_temperature[met], 'maroon', alpha=0.2)
		
		#################### CONTRACTION OF THE GAS CLOUD AFTER LOSS OF PRESSURE ###################
		
		if (iter % accel_diff) == accel_diff - 1:
			p_radiation = (4 * sb)/(3 * c) * temperature**4 * 0
			d_pressure = -np.exp(np.log(np.append(0, -np.diff(pressure + p_radiation))) - np.log(np.diff(base)) - np.log(mass_density))
			#d_pressure = (np.append(0, np.diff(pressure)) * np.exp(-np.log(np.diff(base)) - np.log(mass_density))).astype('float128')
			#d_pressure *= 0
			grav_accel = G * np.exp(np.log(menc) - 2 * np.log(base[1:]))
			
			accel = -(d_pressure.astype('float128') + grav_accel.astype('float128'))
			accel[0] = accel[1]/base[2] * base[1]
			
			accel_fft = np.fft.fft(accel.astype('complex128'))
			

			velocities += accel * dt * accel_diff

			base_new = base + np.append(0, velocities * dt * accel_diff)
			dV_new = copy.deepcopy(base_new[1:]**3 - base_new[:-1]**3) * 4 * np.pi/3.
			dV_new = dV_new.astype('float128')
			
			gas_number_densities = np.array([a * np.exp(np.log(dV) - np.log(dV_new)) for a in gas_number_densities]).astype('float128')
			
			gas_number_density = np.zeros(len(gas_number_densities[0])).astype('longdouble')
			gas_mass_density = np.zeros(len(gas_number_densities[0])).astype('longdouble')
			
			for hj in range(len(gas_masses)):
				gas_mass_density += gas_masses[hj] * gas_number_densities[hj]
				gas_number_density += gas_number_densities[hj]
			
			mass_density = gas_mass_density * (1 + np.sum(metallicities))
			pressure *= (dV/dV_new)**adiab_constant
			temperature = (pressure)/(k * gas_number_density)
			
			dt = dt_0 / max(mass_density)
			#plt.plot(menc/solar_mass, pressure, alpha=0.1, color='maroon')
			pressures.append(pressure)
			total_densities.append(mass_density)
						
			dV = dV_new
			base = base_new
			
			print total_time/(60. * 60. * 24. * 365.), dt/(60. * 60. * 24. * 365.)

	return np.array(total_densities), np.array(pressures), total_time/(60. * 60. * 24. * 365.), dM
lrs = []
linsp_0 = -np.linspace(-2, 3, 20)
for rs in -linsp_0:
	lr_slopes = []
	mets = []
	for qr in np.linspace(-8, 1, 6):
		cm6_result = cm6(base * 10**rs, iterations, gas_masses, gas_number_densities * (10**rs)**-3, grain_densities, grain_radius, specific_heats, metallicities * 10**qr)
		densities = cm6_result[0].T
		pressures = cm6_result[1].T
		dM = cm6_result[3]
	
		lr = linregress(np.log10(densities[:-1]).ravel(), np.log10(pressures[:-1]).ravel())
		lr_slopes.append(lr[0])
		mets.append(np.sum(metallicities * 10**qr)/(1 + np.sum(metallicities * 10**qr)))
	
	lrs.append(lr_slopes)
	
################################# MODEL FOR GAS CLOUDS USING WHAT WAS DEVELOPED ABOVE #############################
################################# IGNORES DUST-MEDIATE COOLING OF THE CLOUD #######################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy import constants
import copy
from scipy.optimize import fsolve, root, newton
from pandas import rolling_mean

base = np.linspace(0,1e7,1001)
density = copy.deepcopy((((base/base[-1] * 2)**6 + 1)**-1)[1:])
iterations = 10000
gas_masses = np.array([1.00794, 4.002602])
nd = np.array([8.47996371e08 * 0.91, 8.47996371e08 * 0.09])
gas_number_densities = np.einsum('i, j', nd, density)

def cm7(base, iterations, gas_masses, gas_number_densities):
	base = base.astype('float128')
	iterations = int(iterations)
	gas_masses = gas_masses.astype('float128')
	gas_number_densities = gas_number_densities.astype('float128')
	
	k = constants.Boltzmann
	G = constants.G
	AU = constants.au
	sb = constants.sigma
	c = constants.c
	t_cmb = 2.725
	mean_opacity = 1e-7 #in m^2/kg
	polytropic_index = 1.007 #this is true because reasons
	
	solar_mass = 1.989e30
	amu = constants.physical_constants['atomic mass constant'][0]
	
	base *= AU
	gas_masses *= amu
	
	gas_mass_density = np.zeros(len(base[1:])).astype('float128')
	gas_number_density = np.zeros(len(base[1:])).astype('float128')
	grain_factor = 0
	accel_diff = int(np.sqrt(iterations))
	
	#determining number and size of dust grains
	
	for i in range(len(gas_masses)):
		gas_mass_density += gas_masses[i] * gas_number_densities[i]
		gas_number_density += gas_number_densities[i]
			
	gas_mass_density = gas_mass_density.astype('float128')
	
	dV = copy.deepcopy(base[1:]**3 - base[:-1]**3) * 4 * np.pi/3.
	dM_gas = gas_mass_density * dV
	dM_stars = np.zeros(len(dV))
	base_stars = base[1:]
	
	menc = np.cumsum(dM_stars + dM_gas)
	
	d_pressure = -G * menc * gas_mass_density/base[1:]**2
	pressure = np.cumsum(d_pressure) * base[1]; pressure -= min(pressure)
	#P = nkT; T = P/nK
	temperature = pressure/(gas_number_density*k)
	dt = 60. * 60. * 24. * 365.
	base_sfr = 3.170979198E-15
	accel = np.zeros(len(pressure))
	velocities = np.zeros(len(pressure))
	
	for i in range(iterations):
		dM_stars += base_sfr * dt * dM_gas
		gas_mass_density *= (1. - base_sfr * dt)
		gas_number_density *= (1. - base_sfr * dt)
		gas_number_densities *= (1. - base_sfr * dt)
		dM_gas *= (1. - base_sfr * dt)
		#temperature += (5000 - temperature) * (dM_stars/dM_gas * 0.01)
		
		pressure = gas_number_density * k * temperature
		dp_hydrostatic = -G * menc * gas_mass_density/base[1:]**2
		pressure_hydrostatic = np.cumsum(d_pressure) * base[1]; pressure -= min(pressure)
		
		
		#Using pressure computation in order to accelerate the molecular cloud after acceleration		
		#plt.plot(menc/solar_mass, velocities, alpha=0.2)
		
		if (i % int(accel_diff)) == int(accel_diff - 1):
			p_radiation = (4 * sb)/(3 * c) * temperature**4 * 0
			d_pressure = -np.exp(np.log(np.append(0, -np.diff(pressure + p_radiation))) - np.log(np.diff(base)) - np.log(gas_mass_density))

			grav_accel = G * np.exp(np.log(menc) - 2 * np.log(base[1:]))
			
			accel = (d_pressure.astype('float128') + grav_accel.astype('float128'))
			accel[0] = accel[1]/base[2] * base[1]
			
			velocities += accel * dt * accel_diff

			base_new = base + np.append(0, velocities * dt * accel_diff)
			dV_new = copy.deepcopy(base_new[1:]**3 - base_new[:-1]**3) * 4 * np.pi/3.
			dV_new = dV_new.astype('float128')
			
			gas_number_densities = np.array([a * np.exp(np.log(dV) - np.log(dV_new)) for a in gas_number_densities]).astype('float128')
			
			gas_number_density = np.zeros(len(gas_number_densities[0])).astype('longdouble')
			gas_mass_density = np.zeros(len(gas_number_densities[0])).astype('longdouble')
			
			for hj in range(len(gas_masses)):
				gas_mass_density += gas_masses[hj] * gas_number_densities[hj]
				gas_number_density += gas_number_densities[hj]

			pressure *= (dV/dV_new)**polytropic_index
			temperature = (pressure)/(k * gas_number_density)
			
			plt.plot(menc/solar_mass, pressure, alpha=0.1, color='maroon')
						
			dV = dV_new
			base = base_new
			
cm7(base, iterations, gas_masses, gas_number_densities)