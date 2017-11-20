from navier_stokes import specie_fraction_array
import numpy
from numpy import linalg as LA
'''
Use this file to generate test data
'''
 
def data_gen(dust_mass_array): #generates test data for input array of (fractional) masses of different dust species for each particle 
    #arbitrary constants
    C = 1 #saturation point
    b = 0.5
    d = 3
    destruction_frac_array = C/(1+d*np.exp(-b*dust_mass_array))+ C*np.random.rand(N_PARTICLES,len(specie_fraction_array))
    #random function, the destruction_frac_array gives the destruction rate. Grows proportionally to dust_mass.
    #Becomes saturated eventually
    return(destruction_frac_array)

def interpolate_test_data(input_params,dust_mass_array): #input_params are where we want to evaulate the "true value"
    interp = 0 #the true value
    total_weight = 0
    destruction_frac = data_gen(dust_mass_array)
    for i in range(len(dust_mass_array)):
        interp+=destruction_frac[i]*LA.norm(dust_mass_array[i]-input_params)
        total_weight += LA.norm(dust_mass_array[i]-input_params) #used to divide the result by the total weight
    return(interp/total_weight) #interp is the value evaluated at a point in the input space, input_params
        
