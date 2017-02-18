from sim_tools.common import *
from sim_tools.kernels import center_surround as krn_cs, gabor as krn_gbr
from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std, \
                                 direction_connectors as dir_conn, \
                                 mapping_funcs as mapf

from scipy.signal import convolve2d, correlate2d

from default_config import defaults_retina as defaults

import sys






class Retina():
    
    def __init__(self, simulator, camera_pop, width, height, dvs_mode, 
                 cfg=defaults):
        
        print("Building Retina (%d x %d)"%(width, height))
        
        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]

        self.sim = simulator
        self.cfg = cfg
        
        self.width = width
        self.height = height
        self.dvs_mode = dvs_mode
        if self.dvs_mode == dvs_modes[0]:
            self.on_idx = 0
            self.off_idx = width*height
        else:
            self.on_idx = 0
            self.off_idx = 0
        
        self.shapes = {}
        ### // <- integer div
        self.css = ['cs', 'cs2', 'cs4']
        self.shapes['cs']  = self.gen_shape(width, height, cfg['cs']['step'],
                                            cfg['cs']['start'])
                             
        self.shapes['cs2'] = self.gen_shape(width, height, 
                                            cfg['cs_half']['step'],
                                            cfg['cs_half']['start'])

        self.shapes['cs4'] = self.gen_shape(width, height, 
                                            cfg['cs_quart']['step'],
                                            cfg['cs_quart']['start'])
        
        if 'direction' in cfg:
            self.shapes['dir'] = self.gen_shape(width, height, 
                                                cfg['direction']['step'],
                                                cfg['direction']['start'])
        
        if 'gabor' in cfg and cfg['gabor']:
            self.shapes['gabor'] = self.gen_shape(width, height, 
                                                  cfg['gabor']['step'],
                                                  cfg['gabor']['start'])
                                
            self.ang_div = deg2rad(180./cfg['gabor']['num_divs'])
            self.angles = [i*self.ang_div for i in range(cfg['gabor']['num_divs'])]
        
        # print(self.shapes)
        
        self.channels = ['on', 'off']

        self.cam = {'on':  camera_pop if dvs_mode==dvs_modes[0] else camera_pop[ON],
                    'off': camera_pop if dvs_mode==dvs_modes[0] else camera_pop[OFF],
                   }
        
        self.mapping_f = cfg['input_mapping_func']
        
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
    
    
    def gen_shape(self, width, height, step, start):
        w = subsamp_size(start, width,  step)
        h = subsamp_size(start, height,  step)
        sh = {'width': w, 'height': h, 'size': w*h,
              'start': start, 'step': step}
        return sh
    
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


    def get_output_keys(self):
        return [k for k in self.pops['on'] if k is not 'cam_inter']
    
    
    def build_kernels(self):
        def a2k(a):
            return 'gabor_%d'%( int( a ) )
        krns = {}
        corr = {}
        cfg = self.cfg
        self.cs = krn_cs.center_surround_kernel(cfg['cs']['width'],
                                                cfg['cs']['std_dev'], 
                                                cfg['cs']['sd_mult'])
        corr['cs'] = correlate2d(self.cs, self.cs, mode='same')
        corr['cs'] *= cfg['w2s']*cfg['corr_w2s_mult']
        self.cs *= cfg['w2s']*cfg['cs']['w2s_mult']
        krns['cs'] = self.cs
        
        cfg = self.cfg
        self.cs2 = krn_cs.center_surround_kernel(cfg['cs_half']['width'],
                                                 cfg['cs_half']['std_dev'], 
                                                 cfg['cs_half']['sd_mult'])
        self.cs2 *= cfg['w2s']*cfg['cs_half']['w2s_mult']
        krns['cs2'] = self.cs2
        
        cfg = self.cfg
        self.cs4 = krn_cs.center_surround_kernel(cfg['cs_quart']['width'],
                                                 cfg['cs_quart']['std_dev'], 
                                                 cfg['cs_quart']['sd_mult'])
        self.cs4 *= cfg['w2s']*cfg['cs_quart']['w2s_mult']
        krns['cs4'] = self.cs4
        
        
        if 'gabor' in cfg and cfg['gabor']:
            angles = self.angles
            gab = krn_gbr.multi_gabor(cfg['gabor']['width'], 
                                      angles, 
                                      cfg['gabor']['std_dev'], 
                                      cfg['gabor']['freq'])
            self.gab = {a2k(k): gab[k]*cfg['w2s'] for k in gab.keys()}
            
            # for k in self.gab:
            #     self.corr[k] = convolve2d(self.gab[k], self.gab[k], mode='same')

        self.kernels = krns
        self.corr = corr

    def build_connectors(self):
        cfg = self.cfg
        self.conns = {ch: {} for ch in self.channels}
        self.lat_conns = {ch: {} for ch in self.channels}
        mapping_f = self.mapping_f
        css = self.css
        krn_conn = conn_krn.full_kernel_connector
        for c in self.conns:
            for k in css:
                on_path = (c == 'on')
                step = self.sample_step(k)
                start = self.sample_start(k)
                self.conns[c][k] = krn_conn( self.width, self.height, 
                                             self.kernels[k],
                                             exc_delay=cfg['kernel_exc_delay'],
                                             inh_delay=cfg['kernel_inh_delay'],
                                             col_step=step, row_step=step,
                                             col_start=start, row_start=start, 
                                             map_to_src=mapping_f,
                                             row_bits=cfg['row_bits'],
                                             on_path=on_path )

        if 'gabor' in cfg and cfg['gabor']:
            for c in self.conns:
                for k in self.gab.keys():
                    krn = self.gab[k]
                    on_path = (c == 'on')
                    step = self.sample_step(k)
                    start = self.sample_start(k)
                    self.conns[c][k] = krn_conn(self.width, self.height, 
                                                krn,
                                                cfg['kernel_exc_delay'],
                                                cfg['kernel_inh_delay'],
                                                col_step=step, row_step=step,
                                                col_start=start, row_start=start, 
                                                map_to_src=mapping_f,
                                                row_bits=cfg['row_bits'],
                                                on_path=on_path)

        
        if 'direction' in cfg:
            for dk in cfg['direction']['keys']:
                k = "%s_dir"%dk
                step = self.sample_step(k)
                start = self.sample_start(k)
                if '2' in dk  or  dk.isupper():
                    dir_cn = dir_conn.direction_connection_angle
                    conns = dir_cn(dk, 
                                   cfg['direction']['angle'],
                                   cfg['direction']['dist'], 
                                   self.width, self.height, 
                                   mapping_f,
                                   start=start, step=step,
                                   exc_delay=cfg['kernel_exc_delay'],
                                   inh_delay=cfg['kernel_inh_delay'],
                                   delay_func=cfg['direction']['delay_func'], 
                                   weight=cfg['direction']['weight'],
                                   row_bits=cfg['row_bits'],)
                else:
                    dir_cn = dir_conn.direction_connection
                    conns = dir_cn(dk, \
                                   self.width, self.height,
                                   cfg['direction']['div'],
                                   cfg['direction']['delays'],
                                   cfg['direction']['weight'],
                                   mapping_f)
                # print(conns)
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
        
        #bipolar to interneuron 
        self.extra_conns['inter'] = {}
        for k in css:
            size = self.pop_size(k)
            conns = conn_std.one2one(size,
                                     weight=cfg['inhw'], 
                                     delay=cfg['kernel_inh_delay'])
            self.extra_conns['inter'][k] = conns
        
        if 'gabor' in cfg:
            size = self.pop_size('gabor')
            conns = conn_std.one2one(size,
                                     weight=cfg['inhw'], 
                                     delay=cfg['kernel_inh_delay'])
            self.extra_conns['inter']['gabor'] = conns
        
        if 'direction' in cfg:
            size = self.pop_size('dir')
            conns = conn_std.one2one(size,
                                     weight=cfg['inhw'], 
                                     delay=cfg['kernel_inh_delay'])
            self.extra_conns['inter']['dir'] = conns
        
        #bipolar/interneuron to ganglion (use row-major mapping)
        self.extra_conns['ganglion'] = {}
        for k in css:
            w, h = self.pop_width(k), self.pop_height(k)
            conns = conn_krn.full_kernel_connector(w, h,
                                                   self.corr['cs'],
                                                   cfg['kernel_exc_delay'],
                                                   cfg['kernel_inh_delay'],
                                                   map_to_src=mapf.row_major,
                                                   pop_width=w)
            self.extra_conns['ganglion'][k] = conns

        if 'gabor' in cfg:
            w, h = self.pop_width('gabor'), self.pop_height('gabor')
            conns = conn_krn.full_kernel_connector(w, h,
                                                   self.corr['cs'],
                                                   cfg['kernel_exc_delay'],
                                                   cfg['kernel_inh_delay'],
                                                   map_to_src=mapf.row_major,
                                                   pop_width=w)
            self.extra_conns['ganglion']['gabor'] = conns
        
        if 'direction' in cfg:
            w, h = self.pop_width('dir'), self.pop_height('dir')
            conns = conn_krn.full_kernel_connector(w, h,
                                                   self.corr['cs'],
                                                   cfg['kernel_exc_delay'],
                                                   cfg['kernel_inh_delay'],
                                                   map_to_src=mapf.row_major,
                                                   pop_width=w)
            self.extra_conns['ganglion']['dir'] = conns
        


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
        

        for k in self.conns.keys():
            for p in self.conns[k].keys():
                filter_size = self.pop_size(p)
                self.pops[k][p] = {'bipolar': sim.Population(filter_size,
                                                             exc_cell, exc_parm,
                                                             label='Retina: bipolar_%s_%s'%(k, p)),
                                                             
                                   'inter':   sim.Population(filter_size,
                                                             inh_cell, inh_parm,
                                                             label='Retina: inter_%s_%s'%(k, p)),
                                                             
                                   'ganglion':  sim.Population(filter_size,
                                                               exc_cell, exc_parm,
                                                               label='Retina: ganglion_%s_%s'%(k, p)),
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

        #bipolar, interneurons and ganglions
        for k in self.conns.keys():
            for p in self.conns[k].keys():
                # print("\t\t%s channel - %s filter"%(k, p))
                self.projs[k][p] = {}
                exc_src = self.cam[k]
                conn = self.conns[k][p][EXC] 
                exc = sim.Projection(exc_src, 
                                     self.pops[k][p]['bipolar'],
                                     sim.FromListConnector(conn),
                                     target='excitatory')
                
                
                if self.conns[k][p][INH]:
                    inh = sim.Projection(self.pops[k]['cam_inter'], 
                                         self.pops[k][p]['bipolar'],
                                         sim.FromListConnector(conn),
                                         target='inhibitory')
                
                    self.projs[k][p]['cam2bip'] = [exc, inh]
                else:
                    self.projs[k][p]['cam2bip'] = [exc]
                
                
                if 'cs' in p:
                    inter  = self.extra_conns['inter'][p]
                    cs_exc = self.extra_conns['ganglion'][p][EXC]
                    cs_inh = self.extra_conns['ganglion'][p][INH]
                elif 'gabor' in p:
                    inter  = self.extra_conns['inter']['gabor']
                    cs_exc = self.extra_conns['ganglion']['gabor'][EXC]
                    cs_inh = self.extra_conns['ganglion']['gabor'][INH]
                elif 'dir' in p:
                    inter  = self.extra_conns['inter']['dir']
                    cs_exc = self.extra_conns['ganglion']['dir'][EXC]
                    cs_inh = self.extra_conns['ganglion']['dir'][INH]
                    
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

    # def row_col_to_input(self, row, col, is_on_input, row_bits):
        # return mapf.row_col_to_input(row, col, is_on_input, self.cfg['row_bits'])
# 
    # def row_col_to_input_breakout(self, row, col, is_on_input):
        # return mapf.row_col_to_input_breakout(row, col, is_on_input,
                                              # self.cfg['row_bits'])
# 
    # def row_col_to_input_subsamp(self, row, col, is_on_input):
        # return mapf.row_col_to_input_subsamp(row, col, is_on_input, 
                                             # self.cfg['row_bits'])
    
    
