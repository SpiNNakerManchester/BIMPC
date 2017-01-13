from sim_tools.common import *
from sim_tools.kernels import center_surround as krn_cs, gabor as krn_gbr
from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std
from sim_tools.connectors import direction_connectors as dir_conn
from sim_tools.connectors import mapping_funcs as mapf

from scipy.signal import convolve2d, correlate2d
MERGED, SPLIT = 0, 1
dvs_modes = ['merged', 'split']
defaults = {'kernel_width': 3,
            'kernel_exc_delay': 2.,
            'kernel_inh_delay': 1.,
            'corr_self_delay': 4.,
            'row_step': 1, 'col_step': 1,
            'start_row': 0, 'start_col': 0,
            'gabor': {'num_divs': 4., 'freq': 5., 'std_dev': 5., 'width': 3},
            
            'ctr_srr': {'std_dev': 0.8, 'sd_mult': 6.7, 'width': 3, 
                        'step': 1, 'start':0, 'w2s_mult':1.},
            'ctr_srr_half': {'std_dev': 1.8664, 'sd_mult': 6.7, 'width': 7,
                             'step': 3, 'start':0, 'w2s_mult': 4.},
            'ctr_srr_quarter': {'std_dev': 4., 'sd_mult': 6.7, 
                                'width': 15,
                                'step': 6, 'start': 0, 'w2s_mult': 8.},

            'w2s': 1.7,
            'inhw': 2.,
            'inh_cell': {'cell': "IF_curr_exp",
                         'params': {'cm': 0.3,  # nF
                                    'i_offset': 0.0,
                                    'tau_m': 10.0,
                                    'tau_refrac': 1.0,
                                    'tau_syn_E': 2.,
                                    'tau_syn_I': 6.,
                                    'v_reset': -70.0,
                                    'v_rest': -65.0,
                                    'v_thresh': -58.
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
            'direction': {'keys': ['east', 'south', 'west', 'north',
                                   'south east', 'south west', 
                                   'north east', 'north west'
                                  ],
                          'div': 5,
                          'weight': 1.1,
                          'delays': [1, 1, 2, 2, 3, 4 ],
                          'subsamp': 2,
                          'w2s': 3.,
                         },
            'row_bits': 8,
           }





class Retina():
    
    def __init__(self, simulator, camera_pop, width, height, dvs_mode, 
                 cfg=defaults):
        
        print("Building Retina (%d x %d)"%(width, height))
        
        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]
        
        self.width = width
        self.height = height
        self.dvs_mode = dvs_mode
        if self.dvs_mode == dvs_modes[0]:
            self.on_idx = 0
            self.off_idx = width*height
        else:
            self.on_idx = 0
            self.off_idx = 0
        
        self.filter_width  = ((width  - cfg['start_col'])//cfg['col_step'])
        self.filter_height = ((height - cfg['start_row'])//cfg['row_step']) 
        self.filter_size   = self.filter_width*self.filter_height

        self.filter_width2  = ((width  - cfg['ctr_srr_half']['start'])//\
                              cfg['ctr_srr_half']['step'])
        self.filter_height2 = ((height - cfg['ctr_srr_half']['start'])//\
                              cfg['ctr_srr_half']['step'])
        self.filter_size2   = self.filter_width2*self.filter_height2

        self.filter_width4  = ((width  - cfg['ctr_srr_quarter']['start'])//\
                              cfg['ctr_srr_quarter']['step'])
        self.filter_height4 = ((height - cfg['ctr_srr_quarter']['start'])//\
                              cfg['ctr_srr_quarter']['step'])
        self.filter_size4   = self.filter_width4*self.filter_height4
        

        self.cam = {'on':  camera_pop if dvs_mode==dvs_modes[0] else camera_pop[ON],
                    'off': camera_pop if dvs_mode==dvs_modes[0] else camera_pop[OFF],
                   }
        
        self.subsamp_width = width//cfg['direction']['subsamp']
        self.subsamp_height = height//cfg['direction']['subsamp']
        self.subsamp_size = self.subsamp_width*self.subsamp_height
        
        self.sim = simulator
        
        self.cfg = cfg
        
        
        if cfg['gabor']:
            self.ang_div = deg2rad(180./cfg['gabor']['num_divs'])
            self.angles = [i*self.ang_div for i in range(cfg['gabor']['num_divs'])]
        
        print("\tBuilding kernels...")
        self.build_kernels()
        print("\t\tdone!")
        
        print("\tBuilding connectors...")
        self.build_connectors()
        print("\t\tdone!")
        
        print("\tBuilding populations...")
        self.build_populations()
        # import pprint
        # pprint.pprint(self.pops)
        print("\t\tdone!")
        
        print("\tBuilding projections...")
        self.build_projections()
        # import pprint
        # pprint.pprint(self.projs)
        print("\t\tdone!")
    
    
    
    def get_output_keys(self):
        return [k for k in self.pops['on'] if k is not 'cam_inter']
    
    
    def build_kernels(self):
        def a2k(a):
            return 'gabor_%d'%( int( a ) )
            
        cfg = self.cfg
        self.cs = krn_cs.center_surround_kernel(cfg['ctr_srr']['width'],
                                                cfg['ctr_srr']['std_dev'], 
                                                cfg['ctr_srr']['sd_mult'])
        self.cs *= cfg['w2s']*cfg['ctr_srr']['w2s_mult']

        cfg = self.cfg
        self.cs2 = krn_cs.center_surround_kernel(cfg['ctr_srr_half']['width'],
                                                 cfg['ctr_srr_half']['std_dev'], 
                                                 cfg['ctr_srr_half']['sd_mult'])
        self.cs2 *= cfg['w2s']*cfg['ctr_srr_half']['w2s_mult']

        cfg = self.cfg
        self.cs4 = krn_cs.center_surround_kernel(cfg['ctr_srr_quarter']['width'],
                                                 cfg['ctr_srr_quarter']['std_dev'], 
                                                 cfg['ctr_srr_quarter']['sd_mult'])
        self.cs4 *= cfg['w2s']*cfg['ctr_srr_quarter']['w2s_mult']

        
        if cfg['gabor']:
            angles = self.angles
            gab = krn_gbr.multi_gabor(cfg['gabor']['width'], 
                                      angles, 
                                      cfg['gabor']['std_dev'], 
                                      cfg['gabor']['freq'])
            self.gab = {a2k(k): gab[k]*cfg['w2s'] for k in gab.keys()}

        

    def build_connectors(self):
        cfg = self.cfg
        self.conns = {'off': {}, 'on':{}}
        self.lat_conns = {'off': {}, 'on':{}}
        row_col_to_input = self.row_col_to_input
        
        self.conns['off']['cs'] = conn_krn.full_kernel_connector(self.width,
                                            self.height, self.cs,
                                            exc_delay=cfg['kernel_exc_delay'],
                                            inh_delay=cfg['kernel_inh_delay'],
                                            map_to_src=row_col_to_input,
                                            on_path=False)

        self.conns['on']['cs']  = conn_krn.full_kernel_connector(self.width,
                                            self.height, self.cs,
                                            exc_delay=cfg['kernel_exc_delay'],
                                            inh_delay=cfg['kernel_inh_delay'],
                                            map_to_src=row_col_to_input,
                                            on_path=True)
                                                                 
        self.conns['on']['cs2'] =  conn_krn.full_kernel_connector(self.width,
                                            self.height, self.cs2,
                                            cfg['kernel_exc_delay'],
                                            cfg['kernel_inh_delay'],
                                            col_step=cfg['ctr_srr_half']['step'], 
                                            row_step=cfg['ctr_srr_half']['step'],
                                            col_start=cfg['ctr_srr_half']['start'], 
                                            row_start=cfg['ctr_srr_half']['start'], 
                                            map_to_src=row_col_to_input,
                                            on_path=True)
                                                                  
        self.conns['off']['cs2'] = conn_krn.full_kernel_connector(self.width,
                                            self.height, self.cs2,
                                            cfg['kernel_exc_delay'],
                                            cfg['kernel_inh_delay'],
                                            col_step=cfg['ctr_srr_half']['step'], 
                                            row_step=cfg['ctr_srr_half']['step'],
                                            col_start=cfg['ctr_srr_half']['start'], 
                                            row_start=cfg['ctr_srr_half']['start'], 
                                            map_to_src=row_col_to_input,
                                            on_path=False)

        self.conns['on']['cs4'] =  conn_krn.full_kernel_connector(self.width,
                                            self.height, self.cs4,
                                            cfg['kernel_exc_delay'],
                                            cfg['kernel_inh_delay'],
                                            col_step=cfg['ctr_srr_quarter']['step'], 
                                            row_step=cfg['ctr_srr_quarter']['step'],
                                            col_start=cfg['ctr_srr_quarter']['start'], 
                                            row_start=cfg['ctr_srr_quarter']['start'], 
                                            map_to_src=row_col_to_input,
                                            on_path=True)
        
        self.conns['off']['cs4'] = conn_krn.full_kernel_connector(self.width,
                                            self.height, self.cs4,
                                            cfg['kernel_exc_delay'],
                                            cfg['kernel_inh_delay'],
                                            col_step=cfg['ctr_srr_quarter']['step'], 
                                            row_step=cfg['ctr_srr_quarter']['step'],
                                            col_start=cfg['ctr_srr_quarter']['start'], 
                                            row_start=cfg['ctr_srr_quarter']['start'], 
                                            map_to_src=row_col_to_input,
                                            on_path=False)



        if cfg['gabor']:
            for k in self.gab.keys():
                krn = self.gab[k]
                
                self.conns['off'][k] = conn_krn.full_kernel_connector(self.width,
                                                        self.height, krn,
                                                        cfg['kernel_exc_delay'],
                                                        cfg['kernel_inh_delay'],
                                                        col_step=cfg['col_step'], 
                                                        row_step=cfg['row_step'],
                                                        col_start=cfg['start_col'], 
                                                        row_start=cfg['start_row'], 
                                                        map_to_src=row_col_to_input,
                                                        on_path=False)

                self.conns['on'][k] = conn_krn.full_kernel_connector(self.width,
                                                        self.height, krn,
                                                        cfg['kernel_exc_delay'],
                                                        cfg['kernel_inh_delay'],
                                                        col_step=cfg['col_step'], 
                                                        row_step=cfg['row_step'],
                                                        col_start=cfg['start_col'], 
                                                        row_start=cfg['start_row'], 
                                                        map_to_src=row_col_to_input,
                                                        on_path=True)

        ssamp_div = cfg['direction']['subsamp']
        
        for dk in cfg['direction']['keys']:
            k = "dir: %s"%dk
            conns =  dir_conn.direction_connection(dk, \
                                              self.subsamp_width,
                                              self.subsamp_height,
                                              cfg['direction']['div'],
                                              cfg['direction']['delays'],
                                              cfg['direction']['weight'])
            self.conns['on'][k], self.conns['off'][k] = conns

######################################

        # for c in self.conns['on'][dk]:
            # print(c)
        
        self.extra_conns = {}
        #cam to inh-version of cam
        if self.dvs_mode == dvs_modes[0]:
            conns = conn_std.one2one(self.width*self.height*2,
                                     weight=cfg['inhw'], 
                                     delay=cfg['kernel_inh_delay'])
        else:
            conns = conn_std.one2one(self.width*self.height,
                                     weight=cfg['inhw'], 
                                     delay=cfg['kernel_inh_delay'])
        
        self.extra_conns['o2o'] = conns
        
        self.extra_conns['subsamp'] = dir_conn.subsample_connection(\
                                        self.filter_width,
                                        self.filter_height, 
                                        cfg['direction']['subsamp'],
                                        cfg['direction']['w2s'],
                                        self.row_col_to_input_breakout)
        
        #bipolar to interneuron 
        conns = conn_std.one2one(self.filter_size,
                                 weight=cfg['inhw'], 
                                 delay=cfg['kernel_inh_delay'])
        self.extra_conns['inter'] = conns
        
        conns = conn_std.one2one(self.filter_size2,
                                 weight=cfg['inhw'], 
                                 delay=cfg['kernel_inh_delay'])
        self.extra_conns['inter2'] = conns
        
        conns = conn_std.one2one(self.filter_size4,
                                 weight=cfg['inhw'], 
                                 delay=cfg['kernel_inh_delay'])
        self.extra_conns['inter4'] = conns

        conns = conn_std.one2one(self.subsamp_size,
                                 weight=cfg['inhw'], 
                                 delay=cfg['kernel_inh_delay'])
        self.extra_conns['interSS'] = conns

        
        #bipolar/interneuron to ganglion
        conns = conn_krn.full_kernel_connector(self.filter_width, 
                                               self.filter_height,
                                               self.cs,
                                               cfg['kernel_exc_delay']+1,
                                               cfg['kernel_inh_delay'])
        self.extra_conns['cs'] = conns

        conns = conn_krn.full_kernel_connector(self.filter_width2, 
                                               self.filter_height2,
                                               self.cs,
                                               cfg['kernel_exc_delay']+1,
                                               cfg['kernel_inh_delay'])
        self.extra_conns['cs2'] = conns

        conns = conn_krn.full_kernel_connector(self.filter_width4, 
                                               self.filter_height4,
                                               self.cs,
                                               cfg['kernel_exc_delay']+1,
                                               cfg['kernel_inh_delay'])
        self.extra_conns['cs4'] = conns
        
        conns = conn_krn.full_kernel_connector(self.subsamp_width, 
                                               self.subsamp_height,
                                               self.cs,
                                               cfg['kernel_exc_delay']+1,
                                               cfg['kernel_inh_delay'])
        self.extra_conns['csSS'] = conns


                                            

    
    def build_populations(self):
        self.pops = {}
        sim = self.sim
        cfg = self.cfg
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        inh_cell = getattr(sim, cfg['inh_cell']['cell'], None)
        inh_parm = cfg['inh_cell']['params']

        if self.dvs_mode == dvs_modes[0]:
            cam_inter = sim.Population(self.width*self.height*2,
                                       inh_cell, inh_parm,
                                       label='cam_inter')
            if cfg['record']['voltages']:
                cam_inter.record_v()

            if cfg['record']['spikes']:
                cam_inter.record()

            for k in self.conns.keys():
                self.pops[k] = {}
                self.pops[k]['cam_inter'] = cam_inter
        else:
            for k in self.conns.keys(): 
                self.pops[k] = {}
                self.pops[k]['cam_inter'] = sim.Population(self.width*self.height,
                                                           inh_cell, inh_parm,
                                                           label='cam_inter_%s'%k)
                if cfg['record']['voltages']:
                   self.pops[k]['cam_inter'].record_v()

                if cfg['record']['spikes']:
                    self.pops[k]['cam_inter'].record()
        
        ssp_div = cfg['direction']['subsamp']
        self.ssamp = {}
        for k in self.conns.keys():
            self.ssamp[k] = sim.Population( self.subsamp_size,
                                            exc_cell, exc_parm,
                                            label='subsample channel %s'%k )
            if cfg['record']['voltages']:
               self.ssamp[k].record_v()

            if cfg['record']['spikes']:
                self.ssamp[k].record()
        
        for k in self.conns.keys():
            for p in self.conns[k].keys():
                if p == 'cs4':
                    filter_size = self.filter_size4
                elif p == 'cs2':
                    filter_size = self.filter_size2
                elif p.startswith('dir'):
                    filter_size = self.subsamp_size
                else:
                    filter_size = self.filter_size
                
                self.pops[k][p] = {'bipolar': sim.Population(filter_size,
                                                             exc_cell, exc_parm,
                                                             label='bipolar_%s_%s'%(k, p)),
                                                             
                                   'inter':   sim.Population(filter_size,
                                                             inh_cell, inh_parm,
                                                             label='inter_%s_%s'%(k, p)),
                                                             
                                   'ganglion':  sim.Population(filter_size,
                                                               exc_cell, exc_parm,
                                                               label='ganglion_%s_%s'%(k, p)),
                                  } 
                if cfg['record']['voltages']:
                   self.pops[k][p]['bipolar'].record_v()
                   self.pops[k][p]['inter'].record_v()
                   self.pops[k][p]['ganglion'].record_v()

                if cfg['record']['spikes']:
                    self.pops[k][p]['bipolar'].record()
                    self.pops[k][p]['inter'].record()
                    self.pops[k][p]['ganglion'].record()


    def build_projections(self):
        self.projs = {}
        cfg = self.cfg
        sim = self.sim
        
        #on/off photoreceptors interneuron projections (for inhibition)
        if self.dvs_mode == dvs_modes[0]: 
            conn = self.extra_conns['o2o']
            exc = sim.Projection(self.cam['on'], 
                                 self.pops['on']['cam_inter'],
                                 sim.FromListConnector(conn),
                                 target='excitatory')
            
            for k in self.conns.keys():
                self.projs[k] = {}
                self.projs[k]['cam_inter'] = {}
                self.projs[k]['cam_inter']['cam2intr'] = [exc]
        else:
            for k in self.conns.keys():
                self.projs[k] = {}
                self.projs[k]['cam_inter'] = {}
                
                conn = self.extra_conns['o2o']
                exc = sim.Projection(self.cam[k], 
                                     self.pops[k]['cam_inter'],
                                     sim.FromListConnector(conn),
                                     target='excitatory')            
                self.projs[k]['cam_inter']['cam2intr'] = [exc]
        
        #subsampling
        flc = sim.FromListConnector(self.extra_conns['subsamp'][ON])
        self.projs['on']['subsamp'] = sim.Projection(self.cam[k],
                                                     self.ssamp['on'],
                                                     flc,
                                                     target='excitatory')
        flc = sim.FromListConnector(self.extra_conns['subsamp'][OFF])
        self.projs['off']['subsamp'] = sim.Projection(self.cam[k],
                                                      self.ssamp['off'],
                                                      flc,
                                                      target='excitatory')
        
        #bipolar, interneurons and ganglions
        for k in self.conns.keys():
            for p in self.conns[k].keys():
                # print("\t\t%s channel - %s filter"%(k, p))
                self.projs[k][p] = {}
                exc_src = self.ssamp[k] if p.startswith('dir') \
                          else self.cam[k]
                          
                conn = self.conns[k][p][EXC] 
                exc = sim.Projection(exc_src, 
                                     self.pops[k][p]['bipolar'],
                                     sim.FromListConnector(conn),
                                     target='excitatory')
                
                conn = self.conns[k][p][INH]
                if (conn is not None) and (len(conn) > 0):
                    inh = sim.Projection(self.pops[k]['cam_inter'], 
                                         self.pops[k][p]['bipolar'],
                                         sim.FromListConnector(conn),
                                         target='inhibitory')
                
                    self.projs[k][p]['cam2bip'] = [exc, inh]
                else:
                    self.projs[k][p]['cam2bip'] = [exc]
                    
                if p == 'cs4':
                    inter  = self.extra_conns['inter4']
                    cs_exc = self.extra_conns['cs4'][EXC]
                    cs_inh = self.extra_conns['cs4'][INH]
                elif p == 'cs2':
                    inter  = self.extra_conns['inter2']
                    cs_exc = self.extra_conns['cs2'][EXC]
                    cs_inh = self.extra_conns['cs2'][INH]
                elif p.startswith('dir'):
                    inter  = self.extra_conns['interSS']
                    cs_exc = self.extra_conns['csSS'][EXC]
                    cs_inh = self.extra_conns['csSS'][INH]
                else:
                    inter  = self.extra_conns['inter']
                    cs_exc = self.extra_conns['cs'][EXC]
                    cs_inh = self.extra_conns['cs'][INH]
                    
                exc = sim.Projection(self.pops[k][p]['bipolar'], 
                                     self.pops[k][p]['inter'],
                                     sim.FromListConnector(inter),
                                     target='excitatory')
                
                self.projs[k][p]['bip2intr'] = [exc]

                exc = sim.Projection(self.pops[k][p]['bipolar'], 
                                     self.pops[k][p]['ganglion'],
                                     sim.FromListConnector(cs_exc),
                                     target='excitatory')
                inh = sim.Projection(self.pops[k][p]['inter'], 
                                     self.pops[k][p]['ganglion'],
                                     sim.FromListConnector(cs_inh),
                                     target='inhibitory')
                
                self.projs[k][p]['bip2gang'] = [exc, inh]

    def row_col_to_input(self, row, col, is_on_input, width, height):
        return mapf.row_col_to_input(row, col, is_on_input, width, 
                                     height, self.cfg['row_bits'])

    def row_col_to_input_breakout(self, row, col, is_on_input):
        return mapf.row_col_to_input_breakout(row, col, is_on_input,
                                              self.cfg['row_bits'])

    def row_col_to_input_subsamp(self, row, col, is_on_input):
        return mapf.row_col_to_input_subsamp(row, col, is_on_input, 
                                             self.cfg['row_bits'])
    
    
