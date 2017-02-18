from sim_tools.common import *
from sim_tools.kernels import center_surround as krn_cs, gabor as krn_gbr
from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std, \
                                 mapping_funcs as mapf
from scipy.signal import convolve2d, correlate2d

from default_config import defaults_lgn as defaults


class LGN():
    def __init__(self, simulator, retina, cfg=defaults):
        
        print("Building LGN...")
        
        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]
        
        self.cfg      = cfg
        self.sim      = simulator
        self.retina   = retina
        self.channels = retina.channels
        self.shapes   = retina.shapes
        self.width    = retina.width
        self.height   = retina.height


        print("\tBuilding kernels...")
        self.build_kernels()
        print("\t\tdone!")
        
        print("\tBuilding connectors...")
        self.build_connectors()
        print("\t\tdone!")
        
        print("\tBuilding populations...")
        self.build_populations()
        print("\t\tdone!")
        
        print("\tBuilding projections...")
        self.build_projections()
        print("\t\tdone!")

    def _right_key(self, key):
        if 'gabor' in key:
            return 'gabor'
        elif 'dir' in key:
            return 'dir'
        else:
            return key
        
    def pop_size(self, key):
        return self.shapes[self._right_key(key)]['size']
    
    def pop_width(self, key):
        return self.shapes[self._right_key(key)]['width']
    
    def pop_height(self, key):
        return self.shapes[self._right_key(key)]['height']

    def sample_step(self, key):
        return self.shapes[self._right_key(key)]['step']
    
    def sample_start(self, key):
        return self.shapes[self._right_key(key)]['start']

    def output_keys(self):
        return self.pops[self.channels[0]].keys()
        
    def build_kernels(self):
        cfg = self.cfg
        self.cs = krn_cs.center_surround_kernel(cfg['kernel_width'],
                                                cfg['ctr_srr']['std_dev'], 
                                                cfg['ctr_srr']['sd_mult'])
        self.cs *= cfg['w2s']
        
        self.split_cs = krn_cs.split_center_surround_kernel(cfg['kernel_width'],
                                                            cfg['ctr_srr']['std_dev'], 
                                                            cfg['ctr_srr']['sd_mult'])
        for i in range(len(self.split_cs)):
            self.split_cs[i] *= cfg['w2s']


    def build_connectors(self):
        cfg = self.cfg
        conns = {}
        
        for k in retina.pops[retina.channels[0]]:
            width, height = self.pop_width(k), self.pop_height(k)
            conns[p] = {}
            exc, inh = conn_krn.full_kernel_connector(self.width,
                                                      self.height,
                                                      self.split_cs[EXC],
                                                      cfg['kernel_exc_delay'],
                                                      cfg['kernel_inh_delay'],
                                                      cfg['col_step'], 
                                                      cfg['row_step'],
                                                      cfg['start_col'], 
                                                      cfg['start_row'],
                                                      map_to_src=mapf.row_major,
                                                      row_bits=self.width)
            
            tmp, inh[:] = conn_krn.full_kernel_connector(self.width,
                                                         self.height,
                                                         self.split_cs[INH],
                                                         cfg['kernel_exc_delay'],
                                                         cfg['kernel_inh_delay'],
                                                         cfg['col_step'], 
                                                         cfg['row_step'],
                                                         cfg['start_col'], 
                                                         cfg['start_row'],
                                                         map_to_src=mapf.row_major,
                                                         row_bits=self.width,
                                                         remove_inh_only=False)

            conns['split'] = [exc, inh]

        exc, inh = conn_krn.full_kernel_connector(self.width2,
                                                  self.height2,
                                                  self.split_cs[EXC],
                                                  cfg['kernel_exc_delay'],
                                                  cfg['kernel_inh_delay'],
                                                  cfg['col_step'], 
                                                  cfg['row_step'],
                                                  cfg['start_col'], 
                                                  cfg['start_row'],
                                                  map_to_src=mapf.row_major,
                                                  row_bits=self.width2)
        
        tmp[:], inh[:] = conn_krn.full_kernel_connector(self.width2,
                                                        self.height2,
                                                        self.split_cs[INH],
                                                        cfg['kernel_exc_delay'],
                                                        cfg['kernel_inh_delay'],
                                                        cfg['col_step'], 
                                                        cfg['row_step'],
                                                        cfg['start_col'], 
                                                        cfg['start_row'],
                                                        map_to_src=mapf.row_major,
                                                        row_bits=self.width2,
                                                        remove_inh_only=False)
        conns['split2'] = [exc, inh]
        
        exc, inh = conn_krn.full_kernel_connector(self.width4,
                                                  self.height4,
                                                  self.split_cs[EXC],
                                                  cfg['kernel_exc_delay'],
                                                  cfg['kernel_inh_delay'],
                                                  cfg['col_step'], 
                                                  cfg['row_step'],
                                                  cfg['start_col'], 
                                                  cfg['start_row'],
                                                  map_to_src=mapf.row_major,
                                                  row_bits=self.width4)
        
        tmp[:], inh[:] = conn_krn.full_kernel_connector(self.width4,
                                                        self.height4,
                                                        self.split_cs[INH],
                                                        cfg['kernel_exc_delay'],
                                                        cfg['kernel_inh_delay'],
                                                        cfg['col_step'], 
                                                        cfg['row_step'],
                                                        cfg['start_col'], 
                                                        cfg['start_row'],
                                                        map_to_src=mapf.row_major,
                                                        row_bits=self.width4,
                                                        remove_inh_only=False)
        conns['split4'] = [exc, inh]

        self.conns = conns
        
    def build_populations(self):
        sim = self.sim
        cfg = self.cfg
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        inh_cell = getattr(sim, cfg['inh_cell']['cell'], None)
        inh_parm = cfg['inh_cell']['params']
        
        pops = {}
        for c in self.channels:
            pops[c] = {}
            for k in self.retina.get_output_keys():
                # print('lgn - populations - key: %s'%k)
                if k == 'cs4':
                    popsize = self.size4
                elif k == 'cs2':
                    popsize = self.size2
                else:
                    popsize = self.size

                pops[c][k] = {}
                pops[c][k]['inter']   = sim.Population(popsize,
                                                    inh_cell, inh_parm,
                                                    label='LGN inter %s %s'%(c, k))
                pops[c][k]['output']  = sim.Population(popsize,
                                                    exc_cell, exc_parm,
                                                    label='LGN output %s %s'%(c, k))

                if cfg['record']['voltages']:
                    pops[c][k]['inter'].record_v()
                    pops[c][k]['output'].record_v()

                if cfg['record']['spikes']:
                    pops[c][k]['inter'].record()
                    pops[c][k]['output'].record()
            
        self.pops = pops


    def build_projections(self):
        sim = self.sim
        cfg = self.cfg
        projs = {}
        for c in self.channels:
            projs[c] = {}
            for k in self.retina.get_output_keys():
                # print('lgn - projections - key: %s'%k)
                
                projs[c][k] = {}
                o2o = sim.OneToOneConnector(weights=cfg['w2s'],
                                            delays=cfg['kernel_inh_delay'])
                # print("src size: %d"%self.retina.pops['off'][k]['ganglion'].size) 
                # print("dst size: %d"%self.pops[k]['inter'].size)
                projs[c][k]['inter'] = sim.Projection(self.retina.pops[c][k]['ganglion'],
                                                      self.pops[c][k]['inter'], o2o,
                                                      target='excitatory')

                if k == 'cs4':
                    split = self.conns['split4']
                elif k == 'cs2':
                    split = self.conns['split2']
                else:
                    split = self.conns['split']

                flc = sim.FromListConnector(split[EXC])
                projs[c][k]['exc'] = sim.Projection(self.retina.pops[c][k]['ganglion'], 
                                                    self.pops[c][k]['output'], flc,
                                                    target='excitatory')

                cntr_c = 'off' if c == 'on' else 'on'
                flc = sim.FromListConnector(split[INH]) #conns['cs']?
                projs[c][k]['inh'] = sim.Projection(self.pops[cntr_c][k]['inter'], 
                                                    self.pops[c][k]['output'], flc,
                                                    target='inhibitory')

            self.projs = projs

        
