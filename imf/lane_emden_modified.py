import numpy as np
from scipy import constants
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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

dt_0 = 60. * 60. * 24. * 365 * 250000. # 250000 years
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
	
	dimf = np.append(base_imf[0], np.diff(base_imf))
	
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
	
	return wl, planck_final

def planck_law(temp, wl, dwl):
	emission = np.nan_to_num((2 * h * c**2 / wl**5)/(np.exp(h * c/(wl * k * temp)) - 1))
	emission[0] = 0
	#print emission
	norm_constant = np.sum(emission * dwl)
	return np.nan_to_num(emission/norm_constant)

#rate at which photons with sufficient energy impact the input species to produce
#the output species
	
def momentum_imparted(wl, spectrum, R, dR, gas_number_densities, dM_star):
	dV = 4 * np.pi * (R[1:]**3 - R[:-1]**3)/3.
	dV = np.append(0, dV)
	
	dwl = np.append(wl[0], np.diff(wl))
	base_momentum = np.sum(spectrum * dwl/c)
	gas_number_densities.T[0] = gas_number_densities.T[1]
	gas_number_densities = np.nan_to_num(gas_number_densities)
	
	source_function = dM_star * base_momentum
	optical_depth_exponent = -np.cumsum(np.sum((cross_sections * gas_number_densities.T).T * dR, axis=0))
	differential_luminosity = np.zeros(len(source_function))
	for us in range(len(differential_luminosity)):
		differential_luminosity += source_function[us] * np.exp(-np.abs(optical_depth_exponent - optical_depth_exponent[us])) * (np.sign(optical_depth_exponent - optical_depth_exponent[us]) + 1)
		
	total_momentum = (np.cumsum(differential_luminosity) - differential_luminosity[0])/2.
	
	return total_momentum, np.exp(-optical_depth_exponent + optical_depth_exponent[-1])

def n_photons(wl, spectrum, R, dR, gas_number_densities, dM_star, threshold_energy):
	dV = 4 * np.pi * (R[1:]**3 - R[:-1]**3)/3.
	dV = np.append(0, dV)
	
	dwl = np.append(wl[0], np.diff(wl))
	base_luminosity = np.zeros(len(R))
	for qe in range(len(threshold_energy)):
		base_luminosity[qe] += np.sum((spectrum * wl/(h * c) * dwl)[wl < (h * c)/threshold_energy[qe]])
	
	gas_number_densities.T[0] = gas_number_densities.T[1]
	gas_number_densities = np.nan_to_num(gas_number_densities)	
	
	source_function = dM_star * base_luminosity
	optical_depth_exponent = -np.cumsum(np.sum((cross_sections * gas_number_densities.T).T * dR, axis=0))

	differential_luminosity = np.zeros(len(source_function))
	for us in range(len(differential_luminosity)):
		differential_luminosity += source_function[us] * np.exp(-np.abs(optical_depth_exponent - optical_depth_exponent[us]))

	luminosity = np.cumsum(differential_luminosity) - differential_luminosity[0]
	num_photons = luminosity/c/(dV/dR)

	#base luminosity is in units of solar luminosities per solar mass
	num_photons[0] = num_photons[1]
	
	return num_photons, np.exp(-optical_depth_exponent + optical_depth_exponent[-1])
	

#parameters specific to the simulation rather than existing
#from first principles

def setup_hydrostatic_equilibrium(R, dR, gas_masses, gas_number_densities, dM_stars):
	gas_mass_density = np.zeros(len(R)).astype('float128')
	gas_number_density = np.zeros(len(R)).astype('float128')
	
	#determining number and size of dust grains
	
	for i in range(len(gas_masses)):
		gas_mass_density += gas_masses[i] * gas_number_densities[i]
		gas_number_density += gas_number_densities[i]
			
	gas_mass_density = gas_mass_density.astype('float128')
	
	dV = 4 * np.pi * (R[1:]**3 - R[:-1]**3)/3.
	dV = np.append(0, dV)
	dM_gas = gas_mass_density * dV
	dM_stars = np.zeros(len(dV))
	
	menc_gas = np.cumsum(dM_gas)
	menc_stars = np.cumsum(dM_stars)
	
	menc = menc_gas + menc_stars
	
	d_pressure = np.nan_to_num(-G * menc * gas_mass_density/R**2)
	pressure = np.cumsum(d_pressure) * dR; pressure -= min(pressure)
	#P = nkT; T = P/nK
	temperature = pressure/(gas_number_density*k)
	
	return pressure, temperature, gas_number_density, gas_mass_density, menc_gas, dV
			
def accelerate(R, dR, pressure, menc_gas, menc_star, velocities, accel_diff, dV, gas_number_densities, gas_mass_density, gas_masses, adiab_index):
	#d_pressure = np.nan_to_num(np.exp(np.log(np.append(0, np.diff(pressure))) - np.log(dR) - np.log(gas_mass_density)))
	d_pressure = -np.nan_to_num(np.append(np.diff(pressure), 0)/dR/gas_mass_density) * 0.
	temperature = np.nan_to_num(pressure/np.sum(gas_number_densities, axis = 0))
	
	#d_prad = np.nan_to_num((-(4 * sb)/c * temperature**3 * np.append(0, np.diff(temperature))/dR)/gas_mass_density)
	
	grav_accel = -np.nan_to_num(G * (menc_gas + menc_star)/R**2)

	accel = ((d_pressure).astype('float128') + grav_accel.astype('float128'))
	#accel[1] = accel[2]/R[2] * R[1]
	
	velocities += accel * dt * accel_diff
	velocities[0] = 0
		
	R_new = R * np.exp(velocities * dt * accel_diff/R)
	
	R_new[0] = 0
	dR_new = np.append(dR[0], np.diff(R_new))
	dR_new[0] = 0
	dV_new = 4 * np.pi * (R_new[1:]**3 - R_new[:-1]**3)/3.
	dV_new = np.append(0, dV_new)

	dV_new = dV_new.astype('float128')

	gas_number_density = np.zeros(len(gas_number_densities[0])).astype('longdouble')
	gas_mass_density = np.zeros(len(gas_number_densities[0])).astype('longdouble')

	
	gas_number_densities = np.nan_to_num(np.array([a * dV/dV_new for a in gas_number_densities]).astype('float128'))
	for hj in range(len(gas_masses)):
		gas_mass_density += gas_masses[hj] * gas_number_densities[hj]
		gas_number_density += gas_number_densities[hj]
		
	gas_number_densities.T[0] = gas_number_densities.T[1]
	gas_number_density[0] = gas_number_density[1]
	gas_mass_density[0] = gas_number_density[1]
	
	pressure_factor = np.nan_to_num((dV/dV_new)**1.007)
	pressure_factor[pressure_factor == 0] = 1.007
	pressure *= pressure_factor
	temperature = pressure/(k * gas_number_density)
	
	return R_new, dR_new, dV_new, pressure, velocities, gas_number_densities, gas_number_density, gas_mass_density

	
def form_star_shell(R, gas_number_densities, gas_mass_density, pressure):
	dV = 4 * np.pi * (R[1:]**3 - R[:-1]**3)/3.
	dV = np.append(0, dV)

	dM_gas = np.nan_to_num(gas_mass_density * dV)
	
	frac_stars = 0.25
	
	star_selection = (np.random.rand(len(r)) < 0.25).astype('int')
	#star_selection[np.nan_to_num(dM_gas/dV) < rho_crit] = 0
	#frac_stars = np.sum(star_selection)/len(star_selection)
	stars_formed = np.exp(-base_sfr * dt * (gas_mass_density/rho_crit)**1.4)
	
	dM_gas_new = dM_gas * (stars_formed + (1 - stars_formed) * frac_stars)
	gas_number_densities_new = gas_number_densities * (stars_formed + (1 - stars_formed) * frac_stars)
	gas_mass_density_new = gas_mass_density * (stars_formed + (1 - stars_formed) * frac_stars)
	pressure_new = pressure * (stars_formed + (1 - stars_formed) * frac_stars)
	dM_stars = dM_gas * (1 - stars_formed) * star_selection
	
	return R[star_selection.astype('bool')], dM_stars[star_selection.astype('bool')], dM_gas_new, gas_number_densities_new, gas_mass_density_new, pressure_new
	
def calculate_star_mass_enclosed(R_stars, dM_stars, R_new):
	#orig_menc = np.cumsum(dM_stars)
	new_menc = []
	for q in R_new:
		new_menc.append(np.sum(dM_stars[R_stars <= q]))
		
	return np.array(new_menc)
	
#radiative transfer functions are below	
def drop_temperature(R, dR, gas_number_densities, temperature, cross_sections, indices):
	a = np.sum((indices * gas_number_densities.T).T, axis=0) * k * dR
	optical_depth_exponent = -np.cumsum(np.sum((cross_sections * gas_number_densities.T).T * dR, axis=0))[::-1]
	
	optical_depth = np.exp(optical_depth_exponent)
	individual_density = 1 - np.exp(-np.sum((cross_sections * gas_number_densities.T).T, axis=0) * dR)
	
	b = sb * optical_depth * individual_density
	
	temp_new = np.nan_to_num((b/(3 * a) * dt + np.nan_to_num((temperature - t_cmb)**(-3)))**(-1./3.)) + t_cmb
	
	return temp_new, temp_new * np.sum(gas_number_densities, axis=0) * k

#ultimately, superimpose multiple radiative transfers at different stellar ages (and thus with different IMFs)
	
def stellar_radiative_transfer(R, dR, gas_number_densities, temperature, cross_sections, incdices, wl, spectrum, dM_star):
	dV = 4 * np.pi * (R[1:]**3 - R[:-1]**3)/3.
	dV = np.append(0, dV)
	
	gas_number_densities.T[0] = gas_number_densities.T[1]
	
	dwl = np.append(wl[1], np.diff(wl))
	base_luminosity = np.sum(spectrum * dwl)

	optical_depth_exponent = -np.cumsum(np.nan_to_num(np.sum((cross_sections * gas_number_densities.T).T, axis=0)) * dR)
	individual_density = 1 - np.exp(-np.sum((cross_sections * gas_number_densities.T).T, axis=0) * dR)
	
	source_function = dM_star * base_luminosity
	#no internal singular source
	luminosity = np.exp(optical_depth_exponent) * np.cumsum(individual_density * source_function * np.exp(-optical_depth_exponent))
	energy_contained = np.nan_to_num(np.sum((indices * gas_number_densities.T).T, axis=0) * k * temperature * dV)
	
	temp_new = (1 + np.nan_to_num(luminosity * dt/energy_contained)) * temperature
	
	return temp_new, temp_new * np.sum(gas_number_densities, axis=0) * k

###########

'''mol_weights = np.array([1.00794 * 2, 4.002602, 1.00794, 1.00727647, 548.579909e-6]) #grams per mol
cross_sections = np.array([6.3e-30, 0, 6.3e-26, 5e-23, 0.]) # m^2
destruction_energies = np.array([7.2418e-19, 3.93938891e-18, 2.18e-18, 10000, 10000]) #joules
adiab_indices = (np.array([7./5., 5./3., 5./3., 5./3., 5./3.]) - 1)**(-1)
'''
#put these parameters for reference

r = np.linspace(0, 25e14, 2001)**0.5 * AU
dr = np.diff(r); dr = np.append(dr[0], dr)
composition = np.array([0.86, 0.14, 0, 0, 0])
nd = 1e06 * composition
density = (r**4/(AU * 2e7)**4 + 1)**-1
gas_number_densities = np.einsum('i, j', nd, density)
adiab_index = np.sum(adiab_indices * composition)/np.sum(composition)
mu = np.sum(mol_weights * composition)/np.sum(composition)
dt = 5000. * year

base_imf = np.logspace(-2, 1., 200)
imf = kroupa_imf(base_imf)

r = r.astype('longdouble')
dr = dr.astype('longdouble')

dM_stars = np.zeros(len(r))
velocities = np.zeros(len(r))

wl, spectrum = overall_spectrum(base_imf, imf)
hyd_eq = setup_hydrostatic_equilibrium(r, dr, mol_weights * amu, gas_number_densities, dM_stars)
droptemp = drop_temperature(r, dr, gas_number_densities, hyd_eq[1], cross_sections, adiab_indices)
temperature = droptemp[0]
pressure = droptemp[1]
gas_mass_density = hyd_eq[3]
rho_crit = copy.deepcopy(gas_mass_density[0]) * 1e02
dV = hyd_eq[-1]

menc_star = np.zeros(len(r))
dM_star = np.zeros(len(r))
star_pos = np.array([])
star_dM = np.array([])
menc_gas = hyd_eq[-2]

menc_0 = hyd_eq[-2][-1]/solar_mass

mencs_tot = []
mencs_star_tot = []
mencs_gas_tot = []
times_0 = []
#leakage = np.zeros((len(gas_number_densities), 0))

for vw in range(2000):
	'''if (vw % 10) == 9:
		srt = stellar_radiative_transfer(r, dr, gas_number_densities, temperature, cross_sections, adiab_indices, wl, spectrum, dM_star)
		print srt[0]/temperature
		pressure = srt[1]
		temperature = srt[0]
		
		#plt.plot(r/AU, temperature, alpha=0.2)

		droptemp = drop_temperature(r, dr, gas_number_densities, temperature, cross_sections, adiab_indices)
		pressure = droptemp[1]
		temperature = droptemp[0]'''
	if (vw % 10) == 8:
		fss = form_star_shell(r, gas_number_densities, gas_mass_density, pressure)
		gas_number_densities = fss[3]
		gas_mass_density = fss[4]
		gas_mass_density[0] = gas_mass_density[1]
		pressure = fss[5]
		
		star_pos = np.append(star_pos, fss[0])
		star_dM = np.append(star_dM, fss[1])
		menc_gas = np.cumsum(fss[2])
		
		menc_star = np.nan_to_num(calculate_star_mass_enclosed(star_pos, star_dM, r))
		dM_star = np.append(0, np.diff(menc_star))
		
		escape_energy = np.nan_to_num(2 * G * (menc_gas + menc_star)/r)
		escape_energy_0 = np.nan_to_num(2 * G * (menc_gas + menc_star)/r)
		escape_energy[r <= r[escape_energy == max(escape_energy)]] = max(escape_energy)
		
		gas_number_density *= 0
		gas_mass_density *= 0
		
		dur = np.cumsum(dt/year * np.ones(len(times_0)))
		lifespan = (luminosity_relation(base_imf, imf)/base_imf)**(-1) * solar_lifespan
		mencs_star_tot = np.array(mencs_star_tot)
		mst_diff = np.diff(np.append(0,mencs_star_tot))
		total_mass_function = np.zeros(len(imf))
		
		for uq in range(len(dur)):
			total_mass_function[lifespan - dur[uq] > 0] += (imf)[lifespan - dur[uq] > 0] * mst_diff[uq]
			
		wl, spectrum = overall_spectrum(base_imf, total_mass_function/total_mass_function[0])
		
		
		for u in range(len(mol_weights)):
			n_phot, optical_depth = n_photons(wl, spectrum, r, dr, gas_number_densities, dM_star, 0.5 * escape_energy * amu * mol_weights[u] * adiab_indices[u])
			additional_energy = np.cumsum(np.nan_to_num(0.5 * 2 * G * (menc_gas + menc_star)/dV * dr * dr))
			time_taken = np.cumsum((additional_energy * adiab_indices[0] + escape_energy)**-0.5 * dr)
			timesteps_required =(-(time_taken - time_taken[escape_energy_0 == max(escape_energy_0)])/dt)**(-1)
			timesteps_required[((0 > timesteps_required) | (1 <= timesteps_required))] = 1
		
			density_reduction = np.nan_to_num(n_phot * optical_depth * cross_sections[0] * c) * dt * timesteps_required
			gas_number_densities[u] *= np.exp(-density_reduction)
			gas_number_density += gas_number_densities[u]
			gas_mass_density += gas_number_densities[u] * mol_weights[u] * amu
		
		#creation of H II regions is going over here
		'''n_phot_2, optical_depth_2 = n_photons(wl, spectrum, r, dr, gas_number_densities, dM_star, np.ones(len(gas_number_densities[0])) * destruction_energies[0])
		n_phot_3, optical_depth_3 = n_photons(wl, spectrum, r, dr, gas_number_densities, dM_star, np.ones(len(gas_number_densities[0])) * destruction_energies[2])
		
		density_reduction_2 = np.nan_to_num(n_phot_2 * cross_sections[0] * c) * dt
		density_reduction_3 = np.nan_to_num(n_phot_3 * cross_sections[2] * c) * dt
		
		gnd_0_new = gas_number_densities[0] * np.exp(-density_reduction_2)
		gnd_2_new = gas_number_densities[2]
		gnd_2_new += 2 * gas_number_densities[0] * (1 - np.exp(-density_reduction_2))
		gnd_2_new *= np.exp(-density_reduction_3)

		gas_number_densities[0] = gnd_0_new
		gas_number_densities[2] = gnd_2_new		
		
		gnd_3_new = gas_number_densities[2] * (1 - np.exp(-density_reduction_3))		

		gas_number_densities[3] += gnd_3_new
		gas_number_densities[4] += gnd_3_new'''
		
		menc_gas = np.cumsum(gas_mass_density * dV)
		
		#plt.plot(r, gas_number_densities[0] * dV)
		
	moved = accelerate(r, dr, pressure, menc_gas, menc_star, velocities, 0.1, dV, gas_number_densities, gas_mass_density, mol_weights * amu, 1.007)
	r = moved[0]
	dr = moved[1]
	dV = moved[2]
	pressure = moved[3]
	velocities = moved[4]
	gas_number_densities = np.nan_to_num(moved[5])
	gas_number_density = np.nan_to_num(moved[6])
	gas_number_density[0] = np.nan_to_num(gas_number_density[1])
	gas_mass_density = moved[7]
	gas_mass_density[0] = gas_mass_density[1]
	
	if (vw % 10) == 9:
		#plt.plot(r, menc_gas + menc_star, alpha=0.2)
	
		menc_tot = menc_gas[-1] + np.sum(star_dM)
		menc_gas_tot = menc_gas[-1]
		menc_star_tot = np.sum(star_dM)
		
		mencs_tot = np.append(mencs_tot, menc_tot)
		mencs_gas_tot = np.append(mencs_gas_tot, menc_gas_tot)
		mencs_star_tot = np.append(mencs_star_tot, menc_star_tot)

		'''for qe in range(len(gas_number_densities)):
			leakage[qe] = np.append(leakage[qe], np.sum(np.nan_to_num(gas_number_densities[qe] * dV * (np.exp(density_reduction) - 1))))'''
		times_0.append(dt)
		dt = dt_0 * (1 - np.exp(-vw/100.)) + 5000 * year
	
		print times_0[-1]/year
	#print u
	
times = np.append(0, np.cumsum(times_0)/year)
mencs_tot = np.append(menc_0, np.array(mencs_tot)/solar_mass)
mencs_gas_tot = np.append(menc_0, np.array(mencs_gas_tot)/solar_mass)
mencs_star_tot = np.append(0, np.array(mencs_star_tot)/solar_mass)

plt.plot(times, mencs_tot, label='Total mass within cloud')
plt.plot(times, mencs_gas_tot, label = 'Mass of gas and dust in cloud')
plt.plot(times, mencs_star_tot, label= 'Mass of stars in cloud')
#plt.plot(times, np.sum(leakage, axis=0), label = 'Mass leakage rate by species')
'''
plt.plot(times, np.log10(mencs_tot), label='Total mass within cloud')
plt.plot(times, np.log10(mencs_gas_tot), label = 'Mass of gas and dust in cloud')
plt.plot(times, np.log10(mencs_star_tot), label= 'Mass of stars in cloud')'''

plt.title('Mass contained within molecular cloud')
plt.xlabel('Time (years)')
plt.ylabel('Mass (solar masses)')
plt.legend()
plt.show()