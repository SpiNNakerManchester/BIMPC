from sim_tools.common import *
from column import V1MultiColumn
from liquid import Liquid
import sys

defaults = {'kernel_width': 3,
            'kernel_exc_delay': 2.,
            'kernel_inh_delay': 1.,
            'gabor': {'num_divs': 7., 'freq': 5., 'std_dev': 1.1},
            'ctr_srr': {'std_dev': 0.8, 'sd_mult': 6.7} ,
            'w2s': 1.7,
            'pix_in_weight': 0.001,
            'context_in_weight': 0.3,
            'context_to_context_weight': 0.5, 
            'context_to_simple_weight': 1., 
            'max_delay': 14.,
            'max_weight': 1.7,
            'wta_inh_cell': {'cell': "IF_curr_exp",
                             'params': {'cm': 0.3,  # nF
                                        'i_offset': 0.0,
                                        'tau_m': 10.0,
                                        'tau_refrac': 2.0,
                                        'tau_syn_E': 2.,
                                        'tau_syn_I': 8.,
                                        'v_reset': -70.0,
                                        'v_rest': -65.0,
                                        'v_thresh': -55.4
                                       }
                            }, 
            'inh_cell': {'cell': "IF_curr_exp",
                         'params': {'cm': 0.3,  # nF
                                    'i_offset': 0.0,
                                    'tau_m': 10.0,
                                    'tau_refrac': 2.0,
                                    'tau_syn_E': 2.,
                                    'tau_syn_I': 4.,
                                    'v_reset': -70.0,
                                    'v_rest': -65.0,
                                    'v_thresh': -55.4
                               }
                        }, 
            'exc_cell': {'cell': "IF_curr_exp",
                         'params': {'cm': 0.3,  # nF
                                    'i_offset': 0.0,
                                    'tau_m': 10.0,
                                    'tau_refrac': 2.0,
                                    'tau_syn_E': 2.,
                                    'tau_syn_I': 2.,
                                    'v_reset': -70.0,
                                    'v_rest': -65.0,
                                    'v_thresh': -55.4
                               }
                        },
            'record': {'voltages': False, 
                       'spikes': False,
                      },
            'lat_inh': False,
            'stdp': {'tau_plus': 15,
                     'tau_minus': 20,
                     'w_max': 0.25,
                     'w_min': 0.,
                     'a_plus': 0.01,
                     'a_minus': 0.02,
                    },
            'readout_w': 0.5,
            'num_input_wta': 25,
            'num_liquid': 500,
            'num_output': 25,
            'in_to_liquid_exc_probability': 0.8,
            'in_to_liquid_inh_probability': 0.5,
            
           }

class V1():
    
    def __init__(self, sim, lgn, learning_on,
                 in_width, in_height, unit_receptive_width, 
                 cfg=defaults):
        
        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]

        self.sim = sim
        self.cfg = cfg
        self.lgn = lgn
        self.learn_on = learning_on
        self.in_width = in_width
        self.in_height = in_height
        self.unit_recpt_width = unit_receptive_width
        self.pix_key   = 'cs'
        self.feat_keys = [k for k in lgn.pops.keys() if k != 'cs']
        self.num_in_ctx = len(self.feat_keys)
        self.simples_per_liquid = 9 # 3x3 -> [[o o o][o (r,c) o][o o o]]
        print("Building V1...")
        self.build_units()
        self.connect_units()


    def build_units(self):
        cfg = self.cfg
        cols = []
        hlf_col_w = self.unit_recpt_width//2
        r_start = hlf_col_w
        r_end = self.in_height - hlf_col_w
        r_step = hlf_col_w# + 1
        
        c_start = hlf_col_w
        c_end = self.in_width  - hlf_col_w
        c_step = hlf_col_w# + 1
        total_cols = (r_end//r_step)*(c_end//c_step)
        num_steps = 20
        cols_to_steps = float(num_steps)/total_cols
        
        print("\t%d Columns..."%(total_cols))
        ### COLUMNS -------------------------------------------------
        prev_step = 0
        curr_col = 0
        simple = {}
        sys.stdout.write("\t\tSimple layer")
        sys.stdout.flush()

        for r in range(r_start, r_end, r_step):
            simple[r] = {}
            for c in range(c_start, c_end, c_step):
                coords = [r, c]
                mc = V1MultiColumn(self.sim, self.lgn, self.learn_on,
                                   self.in_width, self.in_height, 
                                   coords, self.unit_recpt_width, 
                                   cfg['num_input_wta'], 
                                   cfg=cfg)
                simple[r][c] = mc
                
                curr_col += 1
                curr_step = int(curr_col*cols_to_steps)
                if curr_step > prev_step:
                    prev_step = curr_step
                    sys.stdout.write("#")
                    sys.stdout.flush()
                
        sys.stdout.write("\n")
        self.simple = simple
        self.num_simple_rows = len(simple.keys())
        self.num_simple_per_row = len(simple[simple.keys()[0]])
        self.num_simple = self.num_rows*self.units_per_row
        
        ### LIQUID --------------------------------------------------
        prev_step = 0
        curr_col = 0
        liquid = {}
        readout = {}
        sys.stdout.write("\t\tLiquid and Readout layers")
        sys.stdout.flush()
        keys_r = self.simple.keys().sorted()
        for r in range(self.num_simple_rows)[1:-1:2]:
            kr = keys_r[r]
            liquid[r] = {}
            keys_c = self.simple[r].keys().sorted()
            for c in range(len(keys_c))[1:-1:2]:
                kc = keys_c[c]
                in_pops = self.liquid_in_units(r, c, keys_r, keys_c)
                coords = [kr, kc]
                l = Liquid(self.sim, in_pops, 
                           cfg['num_input_wta'], 
                           False, #learning_on
                           self.in_width, self.in_height, 
                           coords, self.simples_per_liquid, 
                           cfg['num_liquid'], cfg=self.cfg)

                liquid[r][c] = l
                
                l = 
                
                curr_col += 1
                curr_step = int(curr_col*cols_to_steps)
                if curr_step > prev_step:
                    prev_step = curr_step
                    sys.stdout.write("#")
                    sys.stdout.flush()
                
        sys.stdout.write("\n")
        self.liquid = liquid
        self.readout = readout
        
    def connect_units(self):
        pass

    def liquid_in_units(self, r, c, keys_r, keys_c):
        return [self.simple[keys_r[in_r]][keys_c[in_c]] \
                            for in_r in range(r-1, r+1) \
                            for in_c in range(c-1, c+1)]
