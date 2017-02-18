from sim_tools.connectors.mapping_funcs import row_col_to_input, \
                                               row_col_to_input_breakout
import numpy as np
frame_rate = 50
dir_delay = int(1000./frame_rate)

exc_cell = "IF_curr_exp"
exc_cell_params = { 'cm': 0.35,  # nF
                    'i_offset': 0.0,
                    'tau_m': 10.0,
                    'tau_refrac': 2.0,
                    'tau_syn_E': 2.,
                    'tau_syn_I': 2.,
                    'v_reset': -70.0,
                    'v_rest': -65.0,
                    'v_thresh': -55.4
                  }

inh_cell = "IF_curr_exp"
inh_cell_params = { 'cm': 0.35,  # nF
                    'i_offset': 0.0,
                    'tau_m': 10.0,
                    'tau_refrac': 1.0,
                    'tau_syn_E': 2.,
                    'tau_syn_I': 1.,
                    'v_reset': -70.0,
                    'v_rest': -65.0,
                    'v_thresh': -58.
                  }

wta_inh_cell = "IF_curr_exp"
wta_inh_cell_params = { 'cm': 0.3,  # nF
                        'i_offset': 0.0,
                        'tau_m': 4.0,
                        'tau_refrac': 2.0,
                        'tau_syn_E': 4.,
                        'tau_syn_I': 1.,
                        'v_reset': -70.0,
                        'v_rest': -65.0,
                        'v_thresh': -58.
                       }


g_w2s = 2.
inh_w2s = 2.
dir_w2s = 2.
ssamp_w2s = 3.
w2s_e = 1.

defaults_retina = {
                   'kernel_width': 3,
                   'kernel_exc_delay': 2.,
                   'kernel_inh_delay': 1.,
                   'corr_self_delay': 4.,
                   'corr_w2s_mult': 2.,
                   'row_step': 1, 'col_step': 1,
                   'start_row': 0, 'start_col': 0,
                   # 'gabor': {'num_divs': 2., 'freq': 5., 'std_dev': 5., 'width': 7,
                             # 'step': 3, 'start': 0},

                   'cs': {'std_dev': 0.8, 'sd_mult': 6.7, 'width': 3, 
                          'step': 1, 'start':0, 'w2s_mult':1.},
                   'cs_half': {'std_dev': 1.8664, 'sd_mult': 6.7, 'width': 7,
                               'step': 3, 'start':0, 'w2s_mult': 5.},
                   'cs_quart': {'std_dev': 4., 'sd_mult': 6.7, 'width': 15,
                                'step': 6, 'start': 0, 'w2s_mult': 6.},
                    #retina receives 1 spike per change, needs huge weights
                   'w2s': g_w2s*1.1, 
                   'inhw': inh_w2s,
                   'inh_cell': {'cell': inh_cell,
                               'params': inh_cell_params,
                               }, 
                   'exc_cell': {'cell': exc_cell,
                                'params': exc_cell_params,
                                },
                   'record': {'voltages': False, 
                              'spikes': False,
                             },
                             
                   'direction': {'keys': [
                                          'E', 
                                          # 'W',
                                          # 'N', 
                                          # 'S',
                                          #'NW', 'SW', 'NE', 'SE',
                                          #'east', 'south', 'west', 'north',
                                          #'south east', 'south west', 
                                          #'north east', 'north west'
                                         ],
                                 'div': 4,#6,
                                 'weight': dir_w2s,
                                 'delays': [1, 4, 6, 8],#, 3, 4 ],
                                 'subsamp': 1,#2,
                                 'w2s': ssamp_w2s,
                                 'angle': 11,
                                 'dist': 4,
                                 'delay_func': lambda dist: dir_delay*dist, 
                                               #20ms = 1000/framerate
                                 'step': 3,
                                 'start': 0,
                                 
                                },
                                
                  # 'input_mapping_func': row_col_to_input_breakout,
                  'input_mapping_func': row_col_to_input,
                  'row_bits': 6,
                  }


#######################################################################
####################         L G N          ###########################
#######################################################################

defaults_lgn = {'kernel_width': 3,
                'kernel_exc_delay': 2.,
                'kernel_inh_delay': 1.,
                'row_step': 1, 'col_step': 1,
                'start_row': 0, 'start_col': 0,
                'gabor': {'num_divs': 7., 'freq': 5., 'std_dev': 1.1},
                'ctr_srr': {'std_dev': 0.8, 'sd_mult': 6.7} ,
                'w2s': g_w2s*1.5,
                'inh_cell': {'cell': inh_cell,
                             'params': inh_cell_params
                            }, 
                'exc_cell': {'cell': exc_cell,
                             'params': exc_cell_params
                            },
                'record': {'voltages': False, 
                           'spikes': False,
                        },
                'lat_inh': False,
            }



#######################################################################
####################           V 1          ###########################
#######################################################################

#from A Statistical Analysis of Information-Processing Properties of 
#Lamina-SpecificCortical Microcircuit Models
column_conn = {'l2': {'inh': 0.2, 'exc': 0.8,
                      'exc2inh': 0.21, 'inh2exc': 0.16,
                      'exc2exc': 0.26, 'inh2inh': 0.25,
                      'exc2l5e': 0.55, 'inh2l5e': 0.20,
                      'exc2l4i': 0.8,
                     },
               'l4': {'inh': 0.2, 'exc': 0.8,
                      'exc2inh': 0.19, 'inh2exc': 0.10,
                      'exc2exc': 0.17, 'inh2inh': 0.50,
                      'exc2l2e': 0.28, 'inh2l2e': 0.50,
                     },
               'l5': {'inh': 0.2, 'exc': 0.8,
                      'exc2inh': 0.10, 'inh2exc': 0.12,
                      'exc2exc': 0.09, 'inh2inh': 0.60,
                      'exc2l2e': 0.03,
                     },
              }
input_conn = {'main':  {'l2e': 0.20,
                        'l4e': 0.80, 'l4i': 0.50,
                        'l5e': 0.10,
                       },
              'extra': {'l2e': 0.20,
                       }
             }
neurons_in_column = {'l2': 10,
                     'l4': 100,
                     'l6': 40}
defaults_v1 = { 'w2s': g_w2s,
                'column_conn': column_conn,
                'input_conn': input_conn,
                'neurons_in_column': neurons_in_column,
                'context_in_weight': 0.3,
                'context_to_context_weight': 0.5, 
                'context_to_simple_weight': 1., 
                'max_delay': 14.,
                'max_weight': 1.7,
                'wta_inh_cell': { 'cell': wta_inh_cell,
                                  'params': wta_inh_cell_params,
                                }, 
                'inh_cell': {'cell': inh_cell,
                             'params': inh_cell_params,
                            }, 
                'exc_cell': {'cell': exc_cell,
                             'params': exc_cell_params,
                            },
                'record': {'voltages': False, 
                           'spikes': False,
                          },
                'lat_inh': False,
                'stdp': {'tau_plus': 20,
                         'tau_minus': 20,
                         'w_max': 0.25,
                         'w_min': 0.,
                         'a_plus': 0.1,
                         'a_minus': 0.12,
                        },
                'in_receptive_width': 5,
                'in_receptive_step':  10,
                'in_receptive_start': 3,
                'min_scale_weight': 0.00001,
                'pix_in_weight': g_w2s*0.2,
                'readout_w': 0.5,
                'num_input_wta': 15,
                'num_liquid': 500,
                'num_output': 25,
                'in_to_liquid_exc_probability': 0.8,
                'in_to_liquid_inh_probability': 0.5,
                'col_weight_func': lambda dist: w2s_e*np.exp(-dist),
                'build_liquid': False,
                'build_readout': False,
               }



