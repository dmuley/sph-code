import numpy as np
import navier_stokes_cleaned as nsc
import os

absolute_path_to_nsc = os.path.dirname(os.path.abspath(nsc.__file__))
absolute_path_to_outputs = absolute_path_to_nsc + '/../savefiles/outputs'
absolute_path_to_config  = absolute_path_to_nsc + '/../savefiles/config'

config_files = os.listdir(absolute_path_to_config)
conf_number = np.array([am[7:-4] for am in config_files]).astype('int')
conf_number = np.append(conf_number, -1)
max_conf_file = (conf_number == max(conf_number))
TIMESTEP_NUMBER = max(conf_number)

#Run only initially to set up config_0.npz!
overall_gas_number_composition = np.array([ 0.86,  0.14,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
overall_dust_number_composition = np.array([.0,.0,0.,0.,0.,0.,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025])
overall_dust_number_composition /= np.sum(overall_dust_number_composition)
new_dustfrac = 0.000000001
OVERALL_AGE = 0
overall_AGB_metallicity = np.array([0.001, 0.001])
overall_AGB_time_until = np.array([1e20 * 60 * 60 * 24 * 365, 1e20 * 60 * 60 * 24 * 365])
overall_AGB_list = np.array([1.1, 1.1])
overall_AGB_composition = np.vstack([overall_gas_number_composition] * 2)

timestep_time_coord = np.array([0.])
timestep_dust_temps = np.array([0.])
timestep_star_frac = np.array([0.])
timestep_imf_measure = np.array([1.])
timestep_chems_error = np.array([0.])
timestep_sup_error = np.array([0.])

np.savez(unicode(absolute_path_to_config + '/config_' + str(int(TIMESTEP_NUMBER + 1))), specie_fraction_array = overall_gas_number_composition, dust_base_frac = overall_dust_number_composition, DUST_FRAC = new_dustfrac, OVERALL_AGE = OVERALL_AGE, overall_AGB_list = overall_AGB_list, overall_AGB_time_until = overall_AGB_time_until, overall_AGB_metallicity = overall_AGB_metallicity, overall_AGB_composition = overall_AGB_composition, timestep_time_coord = timestep_time_coord, timestep_dust_temps = timestep_dust_temps, timestep_star_frac = timestep_star_frac, timestep_imf_measure = timestep_imf_measure, timestep_chems_error = timestep_chems_error, timestep_sup_error = timestep_sup_error)