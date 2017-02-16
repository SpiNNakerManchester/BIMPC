from sim_tools.common import *
from column import V1MultiColumn
from liquid import Liquid
from default_config import defaults_v1 as defaults
import sys


class V1():
    
    def __init__(self, sim, lgn, learning_on,
                 cfg=defaults):
        
        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]

        self.sim = sim
        self.cfg = cfg
        self.lgn = lgn
        self.retina = lgn.retina
        self.learn_on = learning_on
        self.width   = lgn.width
        self.height  = lgn.height
        
        self.pix_key   = 'cs'
        self.feat_keys = [k for k in lgn.pops.keys() if k != 'cs']
        self.num_in_ctx = len(self.feat_keys)
        self.simples_per_liquid = 9 # 3x3 -> [[o o o][o (r,c) o][o o o]]
        print("Building V1...")
        self.build_units()
        self.connect_units()


    def build_units(self):
        cfg = self.cfg
        in_start = cfg['in_receptive_start']
        in_step  = cfg['in_receptive_step']
        cols = []
        total_cols = ((self.width-in_start)*(self.height-in_start))/(in_step**2)
        num_steps = 60
        cols_to_steps = float(num_steps)/total_cols
        
        print("\t%d Columns..."%(total_cols))
        ### COLUMNS (input interface)  ---------------------------------
        prev_step = 0
        curr_col = 0
        simple = {}
        sys.stdout.write("\t\tSimple layer")
        sys.stdout.flush()

        for r in range(in_start, self.width, in_step):
            simple[r] = {}
            for c in range(in_start, self.height, in_step):
                coords = [r, c]
                mc = V1MultiColumn(self.sim, self.lgn, self.learn_on,
                                   self.width, self.height, coords, 
                                   cfg['num_input_wta'], cfg=cfg)
                simple[r][c] = mc
                
                curr_col += 1
                curr_step = int(curr_col*cols_to_steps)
                if curr_step > prev_step:
                    prev_step = curr_step
                    sys.stdout.write(".")
                    sys.stdout.flush()
                
        sys.stdout.write("\n")
        self.simple = simple
        self.num_simple = cfg['num_input_wta']*self.width*self.height
        
        if cfg['build_liquid']:
            ### LIQUID (memory/hi-dim unfolding) ---------------------------
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

                    liquid[r][c] = Liquid(self.sim, in_pops, 
                                          cfg['num_input_wta'], 
                                          False, #learning_on
                                          self.width, self.height, 
                                          coords, self.simples_per_liquid, 
                                          cfg['num_liquid'], cfg=self.cfg)

                    curr_col += 1
                    curr_step = int(curr_col*cols_to_steps)
                    if curr_step > prev_step:
                        prev_step = curr_step
                        sys.stdout.write("#")
                        sys.stdout.flush()
                    
            sys.stdout.write("\n")
            self.liquid = liquid
            self.readout = readout
            
        if cfg['build_readout']:
            pass
            
    def connect_units(self):
        pass

    def liquid_in_units(self, r, c, keys_r, keys_c):
        return [self.simple[keys_r[in_r]][keys_c[in_c]] \
                            for in_r in range(r-1, r+1) \
                            for in_c in range(c-1, c+1)]
