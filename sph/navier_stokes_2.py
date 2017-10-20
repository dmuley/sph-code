'''
Created on Aug 21, 2017

@author: umbut
'''
import numpy as np
import random
from scipy import constants
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial, stats

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
mean_opacity = 1e-7
m_0 = 10**1.5 * solar_mass #solar masses, maximum mass in the kroupa IMF
dt_0 = 60. * 60. * 24. * 365 * 25000. # 25000 years
year = 60. * 60. * 24. * 365.
base_sfr = 1.0e-13
mu_specie = np.array([2.0159,1.0079,1.0074,4.0026,4.0021,0.0005,140.69,60.08,12.0107,28.0855,55.834,100.39,131.93])
#molecular hydrogen, helium, neutral hydrogen, H+, e-
#Later, add various gas-phase metals to this array
mol_weights = np.array([1.00794 * 2, 4.002602, 1.00794, 1.00727647, 548.579909e-6])
cross_sections = np.array([3.34e-30, 3.34e-30, 6.3e-25, 5e-23, 0.]) 
destruction_energies = np.array([7.2418e-19, 3.93938891e-18, 2.18e-18, 10000, 10000])
adiab_indices = (np.array([7./5., 5./3., 5./3., 5./3., 5./3.]) - 1)**(-1)
f_u = np.array([[.86,.14,0,0,0,0,0,0,0,0,0,0,0]]) #relative abundance for species in each SPH particle, an array of arrays
gamma = np.array([7./5,5./3,5./3,5./3,5./3,0,5./3,5./3,5./3,5./3,5./3,0,0])#the polytropes of species in each SPH, an array of arrays

DIAMETER = 1e6 * AU
N_PARTICLES = 15000
N_INT_PER_PARTICLE = 100
V = (DIAMETER)**3
d = (V/N_PARTICLES * N_INT_PER_PARTICLE)**(1./3.)
d_sq = d**2
specie_fraction_array = np.array([.86,.14,0,0,0,0,0,0,0,0,0,0,0]) 

def kroupa_imf(base_imf):
    coeff_0 = 1
    imf_final = np.zeros(len(base_imf))
    imf_final[base_imf < 0.08] = coeff_0 * base_imf[base_imf < 0.08]**-0.3
    coeff_1 = 2.133403503
    imf_final[(base_imf >= 0.08) & (base_imf < 0.5)] = coeff_1 * (base_imf/0.08)[(base_imf >= 0.08) & (base_imf < 0.5)]**-1.3
    coeff_2 = 0.09233279398
    imf_final[(base_imf >= 0.5)] = coeff_2 * coeff_1 * (base_imf/0.5)[(base_imf >= 0.5)]**-2.3
    
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
'''
def generate_mu_array(pts):
    for j in range(len(pts)):
        mu_array = mu_array.append(mu_array,[mu_j(pts[j])],axis = 0)
'''


'''    
def generate_E_internal(pts): #size is the number of SPH particles
    T = 5 #5 kelvins
    m_h = 1.0008 #atomic weight of hydrogen
    E_array = np.array([])
    for j in range(len(pts)):
        E = gamma_j(pts[j])*mass[j]*k*T/(mu_array[j]*m_h)
        E_array = E_array.append(E_array,[E],axis = 0)
    return(E_array)
'''
        
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
		print len(position_indices)
		
	com = [np.sum(masses[np.array(object_indices[v])] * positions[np.array(object_indices[v])].T, axis=1)/np.sum(masses[np.array(object_indices[v])]) for v in range(len(object_indices))]
	com = np.array(com)
	grid_masses = np.array([np.sum(masses[np.array(object_indices[v])]) for v in range(len(object_indices))])
	
	return com, grid_masses, length_scale, np.array(subdivision_boxes), object_indices
	

'''def bin_generator(masses, positions, subdivisions):
    #We want equal mass per axis in each subdivision
    positional_masses = np.array([masses[np.argsort(positions.T[b])] for b in range(len(subdivisions))])
    positional_masses = np.cumsum(positional_masses, axis = 1)
    
    subdivision_cutoffs = [np.zeros(a + 1) for a in subdivisions]
    subdivision_pairs = [[] for a in subdivisions]
    
    for i in range(len(subdivision_cutoffs)):
        for j in range(subdivisions[i]):
            subdivision_cutoffs[i][j + 1] = np.sort(positions.T[i])[positional_masses[i] <= positional_masses[i][-1]/(subdivisions[i]) * (j + 1)][-1]
        
    subdivision_pairs = [np.array([f[:-1], f[1:]]).T for f in subdivision_cutoffs]
    
    #for q in range(len(subdivision_cutoffs)):
    #    for r in range(subdivisions[q]):
    #        subdivision_pairs[q].append([subdivision_cutoffs[q][r], subdivision_cutoffs[q][r + 1]])
    
    subdivision_boxes = [s for s in itertools.product(*subdivision_pairs)]
    com = []
    grid_masses = np.array([])
    length_scale = np.sqrt(np.sum([np.average(np.diff(u)**2) for u in subdivision_cutoffs]))
    
    for wq in range(len(subdivision_boxes)):
        com_partial = np.array([])
        ranges = 0
        #print subdivision_boxes[k]
        for l in range(len(subdivision_boxes[wq])):
            ir = (positions.T[l] >= subdivision_boxes[wq][l][0]).astype('int')
            ir2 = (positions.T[l] <= subdivision_boxes[wq][l][1]).astype('int')
            
            # & (positions.T[l] <= subdivision_boxes[wq][l][1]).astype('int')
            #print subdivision_boxes[wq][l][0]/AU, subdivision_boxes[wq][l][1]/AU
            #print min(positions.T[l][in_range])/(AU), max(positions.T[l][in_range_2])/(AU)
            ranges += ir
            ranges += ir2
        
        #print max(ranges)
        ranges[ranges < max(ranges)] = 0
        ranges = ranges.astype('bool')
        #plt.plot(positions.T[1][ranges.astype('bool')])
        #print np.average(position = s.T[0][ranges]), np.average(positions.T[1][ranges]), np.average(positions.T[2][ranges])
        
        grid_masses = np.append(grid_masses, np.sum(masses[ranges]))
        #particle_location.append(np.arange(len(masses))[ranges])
        
        for m in range(len(subdivision_boxes[wq])):
            com_partial = np.append(com_partial, np.sum(positions.T[m][ranges] * masses[ranges])/np.sum(masses[ranges]))
        
        com.append(com_partial)
    return np.array(com), grid_masses, length_scale'''


def compute_gravitational_force(particle_positions, grid_com, grid_masses, length_scale):
    forces = 0;
    for i in range(len(grid_com)):
        forces += ((np.sum((particle_positions - grid_com[i]).T**2, axis=0) + (length_scale)**2)**-1.5 * -(particle_positions - grid_com[i]).T * grid_masses[i] * G)
        
    return (forces)

'''
def Weigh_old(x,m):
    x_norm = np.sqrt(x[0]**2+x[1]**2+x[2]**2)
    W = 315*(m_0/m)**3*((m/m_0)**(2/3)*d**2-x_norm**2)**3/(64*np.pi*d**9)
    return(W)
'''

'''grad_scale = -945/(32*np.pi*d**9)
weigh_scale = 315/(64*np.pi*d**9)




def Weigh(x, m):
    x_norm_sq = x[0]*x[0] + x[1]*x[1] + x[2]*x[2]
    xsqdif = d_sq*(m/m_0)**(2/3)-x_norm_sq
    W = xsqdif*xsqdif*xsqdif
    return(weigh_scale*W*(m_0/m)*(m_0/m)*(m_0/m))'''

def Weigh2(x, x_0, m):
	norms_sq = np.sum((x - x_0)**2, axis=1)
	W = m * 315*(m_0/m)**3*((m/m_0)**(2./3.)*d**2-norms_sq)**3/(64*np.pi*d**9)
	return(W)

'''def density(j):
    x = points[j]
    rho = 0
    
    #tree = spatial.KDTree(points)
    #indices = tree.query_ball_point(x,d)
    indices = neighbor[j]
    #print('j:', j)
    #print('neighbors of j:', indices)
    r = len(indices)
    for i in range(r):
        rho = rho + mass[indices[i]]*Weigh(x-points[indices[i]],mass[i])
    return(rho)'''

def density(j):
	x_0 = points[j]
	x = np.append([x_0], points[np.array(neighbor[j])],axis=0)
	m = np.append(mass[j], mass[np.array(neighbor[j])])
	
	rho = Weigh2(x, x_0, m)
	return np.sum(rho[rho > 0])
	
def num_dens(j):
	x_0 = points[j]
	x = np.append([x_0], points[np.array(neighbor[j])],axis=0)
	m = np.append(mass[j], mass[np.array(neighbor[j])])
	
	n_dens = Weigh2(x, x_0, m)/(mu_array[np.append(j, neighbor[j])] * m_h)
	return np.sum(n_dens[n_dens > 0])

'''def num_dens_u(j,u):
    rho = 0
    indices = neighbor[j]
    x = points[j]
    #print(points)
    #assert False
    r = len(indices)
    for i in range(r):
        rho = rho + f_u[indices[i]][u]*mass[indices[i]]*Weigh(x-points[indices[i]], mass[indices[i]])/(mu_array[indices[i]]*m_h)
    return(rho)'''

'''def grad_weight(x,m):
    x_norm_sq = x[0]*x[0] + x[1]*x[1] + x[2]*x[2]
    xsqdif = d_sq*(m/m_0)**(2/3)-x_norm_sq
    del_weight = (xsqdif*xsqdif)*x
    return(grad_scale*del_weight*(m_0/m)*(m_0/m)*(m_0/m))'''
   
def grad_weight(x, x_0, m, type_particle):
	vec = (x - x_0)
	norms_sq = np.sum((x - x_0)**2, axis=1)
	
	W = -315 * 6 * (m_0/m)**3 / (64*np.pi*d**9) * ((m/m_0)**(2./3.)*d**2-norms_sq)**2 * (type_particle == 0) * vec.T
	#W = W.T
	
	return((np.nan_to_num(W.astype('float') * ((m/m_0)**(2./3.)*d**2-norms_sq > 0))).T)
 
'''
def pressure(x,x_array,m_array,E_internal): #x_array denotes the array of the position vectors of nearest neighbors
    pressure = 0
    r = len(x_array)
    for j in range(r):
        pressure = pressure + E_internal[j]*density(x_array[j], m_array, x_array)*Weigh(x-x_array[j],m_array[j])/(gamma_j(j)*m_array[j])
    return(pressure)
'''

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
'''	

    if (particle_type[i] == 1):
        return(0)
    pressure = 0
    x = points[i]
    indices = neighbor[i]
    r = len(indices)
    for j in range(r):
        pressure = pressure + E_internal[indices[j]]*density(indices[j])*grad_weight(x-points[indices[j]],mass[indices[j]])/(gamma_j(indices[j])*mass[indices[j]])
    return(pressure)'''

def gamma_j(j):
    species = 13
    gamma_j = 0
    if (num_dens(j) == 0):
        return(0)
    for u in range(species):
        gamma_j = gamma_j + num_dens_u(j,u)*gamma[u]
    gamma_j = gamma_j/num_dens(j)
    return(gamma_j)

def neighbors(points):
	kdt = spatial.cKDTree(points)  
	qbp = kdt.query_ball_point(points, d, p=2, eps=0.1)
	
	return qbp	

def compute_key(pt, d):
    return tuple([np.rint(pt[kk]/d) for kk in range(3)])
'''
def neighbors(pts):   #finds the neighbors of all particles in terms of indices, makes a list of lists containing these indices for each particle. 
    bins=dict()
#set the bins
    for p in range(N_PARTICLES):
        key = compute_key(pts[p],d)
        bins.setdefault(key,[]).append((pts[p],p))
    
    d2=d*d
    tot_pot_neighbors=0.0
    tot_neighbors=0.0
    nbrs = [[] for kk in range(N_PARTICLES)]
    for p in range(N_PARTICLES):
        pnt=pts[p]
        xx=pnt[0]
        yy=pnt[1]
        zz=pnt[2]
        key = compute_key(pnt,d)
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    nk = (key[0]+i, key[1]+j, key[2]+k)
                    nk_lst = bins.get(nk, [])
                    tot_pot_neighbors += len(nk_lst)
                    for nep,nind in nk_lst:
                        if nind == p:
                            continue #self
                        xd=xx-nep[0]
                        yd=yy-nep[1]
                        zd=zz-nep[2]

                        dst2=xd*xd+yd*yd+zd*zd
                        if dst2 <= d2:
                            tot_neighbors +=1
                        #print(p, ' and ', nind, ' are neighbors')
                            nbrs[p].append(nind)
    #print('neighbors of ', p, ':', nbrs[p])                   
                        #print(xx, yy, zz, 'bin:', key) 
    #print(nep[0], nep[1], nep[2], 'bin:', nk, 'rev_check:', compute_key(nep, min_coords, d) ) 
    #for jj in range(len(nbrs)):
    #    if(nbrs[jj]==[]):
    #        nbrs[jj] = [0]
    return(nbrs)'''
#N_PARTICLES = 100000
#def main():
    #composition = np.array([0.86, 0.14, 0, 0, 0])


'''f_un=np.zeros([N_PARTICLES-1, len(specie_fraction_array)])
for i in range(N_PARTICLES-1):
    f_un[i] = specie_fraction_array
f_u = np.append(f_u,f_un,axis = 0)'''

DIAMETER = 1e6 * AU
N_PARTICLES = 2500
N_INT_PER_PARTICLE = 100
V = (DIAMETER)**3
d = (V/N_PARTICLES * N_INT_PER_PARTICLE)**(1./3.)
d_sq = d**2
specie_fraction_array = np.array([.86,.14,0,0,0,0,0,0,0,0,0,0,0])
#relative abundance for species in each SPH particle,  (H2, H, H+,He,He+Mg2SiO4,SiO2,C,Si,Fe,MgSiO3,FeSiO3)in that order

base_imf = np.logspace(-2,1.5, 200)
d_base_imf = np.append(base_imf[0], np.diff(base_imf))
imf = kroupa_imf(base_imf) * d_base_imf
imf /= np.sum(imf)
points = (np.random.rand(N_PARTICLES, 3) - 0.5) * DIAMETER
neighbor = neighbors(points)
#print(nbrs)
#print(points)
velocities = (np.random.rand(N_PARTICLES, 3) - 0.5) * 1000
mass = np.random.choice(base_imf, N_PARTICLES, p = imf) * solar_mass

particle_type = np.zeros([N_PARTICLES]) #0 for gas, 1 for star, 2 for dust

mu_array = np.zeros([N_PARTICLES])#array of all mu
E_internal = np.zeros([N_PARTICLES]) #array of all Energy
#copy of generate_E_array
#fills in E_internal array specified at the beginning
T = 5 #5 kelvins


#fills the f_u array
f_un = np.array([specie_fraction_array] * N_PARTICLES)
mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(specie_fraction_array)
gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(specie_fraction_array)
E_internal = gamma_array * mass * k * T/(mu_array * m_h)
'''
for l in range(len(points)):#copy of generate_mu_array
    mu_array[l] = mu_j(l)
        

for jj in range(len(points)):
    E_internal[jj] = gamma_j(jj)*mass[jj]*k*T/(mu_array[jj]*m_h)
    '''
#copy of generate_mu_array

critical_density = 1000*amu*10**6 #critical density of star formation

for iq in range(25): 
    neighbor = neighbors(points)#find neighbors in each timestep
    bg = bin_generator(mass, points, [4, 4, 4])
    com = bg[0] #center of masses of each bin
    grav_accel = compute_gravitational_force(points, bg[0], bg[1], bg[2]).T #gravity is always acting, thus no cutoff distance introduced for gravity
    
    densities = np.array([density(j) for j in range(len(neighbor))])
    delp = np.array([del_pressure(j) for j in range(len(neighbor))])
    
    probability = 0.02 * (densities/critical_density)**(1.4)
    diceroll = np.random.rand(len(probability))
    particle_type[particle_type == 0] = ((diceroll < probability).astype('float'))[particle_type == 0]
    pressure_accel = -(delp.T/densities).T
    
    total_accel = grav_accel + pressure_accel
    
    '''
    for j in range(N_PARTICLES):
        #cgf = compute_gravitational_force(points, bg[0], bg[1], bg[2]).T[j] #gravity is always acting, thus no cutoff distance introduced for gravity
        if neighbor[j]:#if there are any neighbors, adds the gradient of pressure over density, which is F/m. If there are no neighbors, no pressure exrted
            #print('calling del_pressure')
            #delp = del_pressure(j)
            #print('done with del_pressure, now calling density')
            #dens= density(j)
            #print('done with density')
            #print('delp=', delp, ' dens=', dens)
            dens = density(j)
            probability = 0.02*(dens/critical_density)**(1.4)
            if random.random() <= probability:
                particle_type[j] = 1
            total_force[j] += -del_pressure(j)/dens
            total_force[j] += viscosity(j)/mass[j]
        #cgf = -del_pressure(j)/density(j)+compute_gravitational_force(points, bg[0], bg[1], bg[2]).T[j]
        #points[j] += ((cgf * (dt_0)**2)/2.) + velocities[j] * dt_0
        #velocities[j] += (cgf * dt_0)
        #print('cgf=',cgf)f'''
        
    points += ((total_accel * (dt_0)**2)/2.) + velocities[j] * dt_0
    velocities += (total_accel * dt_0)
    
    f_un = np.array([specie_fraction_array] * N_PARTICLES)
    mu_array = np.sum(f_un * mu_specie, axis=1)/np.sum(specie_fraction_array)
    gamma_array = np.sum(f_un * gamma, axis=1)/np.sum(specie_fraction_array)
    E_internal = gamma_array * mass * k * T/(mu_array * m_h)
    print ('iter count=', iq)

#print(points[0:len(points):20])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points.T[0]/AU, points.T[1]/AU, points.T[2]/AU, alpha=0.1)
ax.set_xlim3d(-DIAMETER/2, DIAMETER/2)
ax.set_ylim3d(-DIAMETER/2, DIAMETER/2)
ax.set_zlim3d(-DIAMETER/2, DIAMETER/2)
#plt.plot(points.T[0], velocities.T[0], '.', alpha=0.1)
#plt.plot(points.T[1], velocities.T[1], '.', alpha=0.1)
#plt.plot(points.T[2], velocities.T[2], '.', alpha=0.1)
plt.show()
