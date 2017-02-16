from ..common import *
import numpy as np
import sys

def unique_rows(a):
    # print(len(a), len(a[0]))
    a = np.ascontiguousarray(a)
    # print(a.shape, a.dtype)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    shape = (unique_a.shape[0], a.shape[1])
    return unique_a.view(a.dtype).reshape(shape)


def direction_connection_angle(direction, max_angle, max_dist, 
                               width, height, mapfunc,
                               start=0, step=1,
                               exc_delay=2, inh_delay=1,
                               delay_func=lambda x: x, 
                               weight_func=lambda d, a, w: w,
                               weight=2., inh_weight_mult=1.5,
                               row_bits=8, map_width=None):
    row_bits = row_bits if map_width is None else map_width
    
    # print("direction_connection_angle")
    dcah = direction_connection_angle_helper
    exc_conns_on  = []
    inh_conns_on  = []
    exc_conns_off = []
    inh_conns_off = []
    exc = []
    inh = []
    # sys.stdout.write("\t\tPercent %03d"%0)
    dang = 0
    if   direction == 'left2right' or direction == 'E':
        dang = 0
    elif direction == 'bl2tr'      or direction == 'SE':
        dang = 45
    elif direction == 'bottom2top' or direction == 'S':
        dang = 90
    elif direction == 'br2tl'      or direction == 'SW':
        dang = 135
    elif direction == 'right2left' or direction == 'W':
        dang = 180
    elif direction == 'tr2bl'      or direction == 'NW':
        dang = 225
    elif direction == 'top2bottom' or direction == 'N':
        dang = 270
    elif direction == 'tl2br'      or direction == 'NE':
        dang = 315
    else:
        raise Exception("Not a valid direction - %s -"%(direction))

    on_chan = True
    for y in range(int(start), int(height), int(step)):
        # sys.stdout.write( "\r\t\tPercent %03d"%( (r*100.)/height ) ) 
        # sys.stdout.flush() 
        for x in range(int(start), int(width), int(step)):
            on_chan = True
            exc[:], inh[:] = dcah(dang, max_angle, max_dist, x, y, 
                                  width, height, mapfunc, on_chan, 
                                  exc_delay=exc_delay, 
                                  inh_delay=inh_delay,
                                  start=start,
                                  step=step,
                                  delay_func=delay_func, 
                                  weight_func=weight_func,
                                  weight=weight,
                                  inh_weight_mult=inh_weight_mult,
                                  row_bits=row_bits)  
            if exc and inh:
                exc_conns_on += exc
                inh_conns_on += inh
                
            on_chan = False
            exc[:], inh[:] = dcah(dang, max_angle, max_dist, x, y, 
                                  width, height, mapfunc, on_chan,
                                  exc_delay=exc_delay, 
                                  inh_delay=inh_delay,
                                  start=start,
                                  step=step,
                                  delay_func=delay_func,
                                  weight_func=weight_func,
                                  weight=weight,
                                  inh_weight_mult=inh_weight_mult,
                                  row_bits=row_bits)  
            if exc and inh:
                exc_conns_off += exc
                inh_conns_off += inh
    # print(" - Done!")
    return [[exc_conns_on,  inh_conns_on], \
            [exc_conns_off, inh_conns_off]]


def direction_connection_angle_helper(dang, max_angle, max_dist, 
                                      start_x, start_y, width, height,
                                      mapfunc, is_on_channel,
                                      start=0, step=1,
                                      exc_delay=2, inh_delay=1,
                                      delay_func=lambda d: d, 
                                      weight_func=lambda d, a, w: w,
                                      weight=2., inh_weight_mult=1.5,
                                      row_bits=8,
                                     ):
    deg2rad = np.pi/180.

    
    chan = int(is_on_channel)
    coord = {}
    e = []; i = []
    dst_width = subsamp_size(start, width, step)
    dst_height = subsamp_size(start, height, step)
    dst_y = subsamp_size(start, start_y, step)
    dst_x = subsamp_size(start, start_x, step)
    dst = int(dst_y*dst_width + dst_x)

    for a in range(-max_angle, max_angle + 1):
        for d in range(1, max_dist+1):
            new_c = True
            
            xd = int(np.round( d*np.cos((a+dang+180)*deg2rad) ))
            yd = int(np.round( d*np.sin((a+dang+180)*deg2rad) ))

            delay = delay_func( np.abs(xd) + np.abs(yd) )
            
            if xd in coord:
                if yd in coord[xd]:
                    new_c = False
                coord[xd][yd] = 0
            else:
                coord[xd] = {yd: 0}

            if new_c:
                xe, ye = start_x + xd, start_y + yd
                src = mapfunc(ye, xe, chan, row_bits)
                
                if 0 <= xe < width and 0 <= ye < height:
                    w = weight_func(d, a, weight)
                    e.append( (src, dst, w, exc_delay+delay) )
                
                xd = int(np.round( d*np.cos((a+dang)*deg2rad) ))
                yd = int(np.round( d*np.sin((a+dang)*deg2rad) ))
                xi, yi = start_x + xd, start_y + yd
                src = mapfunc(yi, xi, chan, row_bits)
                
                if 0 <= xi < width and 0 <= yi < height:
                    w = weight_func(d, a, -weight*inh_weight_mult)
                    # i.append( (src, dst, w, inh_delay) )
                    i.append( (src, dst, w, inh_delay+delay) )
                    
    if e:
        src = mapfunc(start_y, start_x, chan, row_bits)
        delay = delay_func(0)
        w = weight_func(0, 0, weight)
        e.append( (src, dst, w, exc_delay+delay) )
        w = weight_func(0, 0, -weight*inh_weight_mult)
        i.append( (src, dst, w, inh_delay+delay) )

        
    # return [unique_rows(e), unique_rows(i)]
    return [e, i]




def direction_connection_full_res(direction, x_res, y_res, div, delays, weight,
                                  mapfunc):
    
    # subY_BITS=int(np.ceil(np.log2(y_res)))
    connection_list_on  = []
    connection_list_off = []
    connection_list_inh_on = []
    connection_list_inh_off = []
    add_exc = False
    src_on = 0
    src_off = 0
    #direction connections
    for j in range(y_res):
        for i in range(x_res):
            for k in range(div):
                Delay=delays[k]
                dst = j*x_res + i
                if direction=="south east":
                     #south east connections  
                    #check targets are within range
                    if( ((i+k) < x_res) and ((j+k) < y_res) ):
                        add_exc = True
                        src_on  = mapfunc((j+k), i+k, 1)
                        src_off = mapfunc((j+k), i+k, 0)
                
                elif direction=="south west":
                    #south west connections
                    #check targets are within range
                    if((i-k)>=0 and ((j+k)<=(y_res-1))):   
                        add_exc = True
                        src = (j+k)*x_res + i-k
                
                elif direction=="north east":
                    #north east connections
                    #check targets are within range
                    if(((i+k)<=(x_res-1)) and ((j-k)>=0)):  
                        add_exc = True 
                        src_on  = mapfunc((j-k), i+k, 1)
                        src_off = mapfunc((j-k), i+k, 0)
                                        
                elif direction=="north west":
                    #north east connections
                    #check targets are within range
                    if((i-k)>=0 and ((j-k)>=0)):   
                        add_exc = True
                        src_on  = mapfunc((j-k), i-k, 1)
                        src_off = mapfunc((j-k), i-k, 0)
                                        
                elif direction=="north":
                    #north connections
                    #check targets are within range
                    if((j-k)>=0):   
                        add_exc = True
                        src_on  = mapfunc((j-k), i, 1)
                        src_off = mapfunc((j-k), i, 0)
                
                elif direction=="south":
                    #north connections
                    #check targets are within range
                    if((j+k)<=(y_res-1)):   
                        add_exc = True
                        src_on  = mapfunc((j+k), i, 1)
                        src_off = mapfunc((j+k), i, 0)
                        
                elif direction=="east":
                    #north connections
                    #check targets are within range
                    if((i+k)<=(x_res-1)):   
                        add_exc = True
                        src_on  = mapfunc(j, i+k, 1)
                        src_off = mapfunc(j, i+k, 0)
                        
                elif direction=="west":
                    #north connections
                    #check targets are within range
                    if((i-k)>=0):   
                        add_exc = True
                        src_on  = mapfunc(j, i-k, 1)
                        src_off = mapfunc(j, i-k, 0)
                        
                else:
                    raise Exception( "Not a valid direction: %s"%direction )

                #ON channels
                connection_list_on.append((src_on, dst, weight, Delay))
                #OFF channels
                connection_list_off.append((src_off, dst, weight, Delay))
                add_exc = False

    return [connection_list_on, connection_list_inh_on], \
            [connection_list_off, connection_list_inh_off]



def direction_connection(direction, x_res, y_res, div, delays, weight, 
                         coord_map_func):
    
    # subY_BITS=int(np.ceil(np.log2(y_res)))
    connection_list_on  = []
    connection_list_off = []
    connection_list_inh_on = []
    connection_list_inh_off = []
    add_exc = False
    #direction connections
    for j in range(y_res):
        for i in range(x_res):
            for k in range(div):
                Delay=delays[k]
                dst = j*x_res + i
                if direction=="south east":
                     #south east connections  
                    #check targets are within range
                    if( ((i+k) < x_res) and ((j+k) < y_res) ):
                        add_exc = True
                        on_src=coord_map_func(j+k,i+k,1,x_res)
                        off_src=coord_map_func(j+k,i+k,0,x_res)
                        #src = (j+k)*x_res + i+k
                
                elif direction=="south west":
                    #south west connections
                    #check targets are within range
                    if((i-k)>=0 and ((j+k)<=(y_res-1))):   
                        add_exc = True
                        on_src=coord_map_func(j+k,i-k,1,x_res)
                        off_src=coord_map_func(j+k,i-k,0,x_res)
                        #src = (j+k)*x_res + i-k
                
                elif direction=="north east":
                    #north east connections
                    #check targets are within range
                    if(((i+k)<=(x_res-1)) and ((j-k)>=0)):  
                        add_exc = True 
                       # src = (j-k)*x_res + i+k
                        on_src=coord_map_func(j-k,i+k,1,x_res)                       
                        off_src=coord_map_func(j-k,i+k,0,x_res)                       
                                        
                elif direction=="north west":
                    #north east connections
                    #check targets are within range
                    if((i-k)>=0 and ((j-k)>=0)):   
                        add_exc = True
                        #src = (j-k)*x_res + i-k
                        on_src=coord_map_func(j-k,i-k,1,x_res)                       
                        off_src=coord_map_func(j-k,i-k,0,x_res)                                                               
                elif direction=="north":
                    #north connections
                    #check targets are within range
                    if((j-k)>=0):   
                        add_exc = True
                        #src = (j-k)*x_res + i
                        on_src=coord_map_func(j-k,i,1,x_res) 
                        off_src=coord_map_func(j-k,i,0,x_res)                                          
                
                elif direction=="south":
                    #north connections
                    #check targets are within range
                    if((j+k)<=(y_res-1)):   
                        add_exc = True
                        #src = (j+k)*x_res + i
                        on_src=coord_map_func(j+k,i,1,x_res)                   
                        off_src=coord_map_func(j+k,i,0,x_res)                     
                        
                elif direction=="east":
                    #north connections
                    #check targets are within range
                    if((i+k)<=(x_res-1)):   
                        add_exc = True
                        #src = j*x_res + i+k
                        on_src=coord_map_func(j,i+k,1,x_res)
                        off_src=coord_map_func(j,i+k,0,x_res)                       
                        
                elif direction=="west":
                    #north connections
                    #check targets are within range
                    if((i-k)>=0):   
                        add_exc = True
                        #src = j*x_res + i-k
                        on_src=coord_map_func(j,i-k,1,x_res)     
                        off_src=coord_map_func(j,i-k,0,x_res)    
                else:
                    raise Exception( "Not a valid direction: %s"%direction )

                #ON channels
                connection_list_on.append((on_src, dst, weight, Delay))
                #OFF channels
                connection_list_off.append((off_src, dst, weight, Delay))
                add_exc = False

#    return [connection_list_on, connection_list_inh_on], \
#            [connection_list_off, connection_list_inh_off]
    return connection_list_on, connection_list_off


def subsample_connection(x_res, y_res, subsamp_factor_x,subsamp_factor_y,weight, 
                         coord_map_func):
    
    # subY_BITS=int(np.ceil(np.log2(y_res/subsamp_factor)))
    connection_list_on=[]
    connection_list_off=[]
    
    sx_res = int(x_res)//int(subsamp_factor_x)
    
    for j in range(int(y_res)):
        for i in range(int(x_res)):
            si = i//subsamp_factor_x
            sj = j//subsamp_factor_y
            #ON channels
            subsampidx = sj*sx_res + si
            connection_list_on.append((coord_map_func(j, i,1, x_res), 
                                       subsampidx, weight, 1.))
            #OFF channels only on segment borders 
            #if((j+1)%(y_res/subsamp_factor)==0 or (i+1)%(x_res/subsamp_factor)==0 or j==0 or i==0):
            connection_list_off.append((coord_map_func(j, i,0, x_res),
                                        subsampidx, weight, 1.))    
            
    return connection_list_on, connection_list_off
    

def paddle_connection(x_res,paddle_row, subsamp_factor_x, weight, coord_map_func):
    connection_list_on=[]
    connection_list_off=[]
    
    for i in range(int(x_res)):
        idx = i//subsamp_factor_x
        connection_list_on.append((coord_map_func(paddle_row, i,1, x_res),
                                   idx, weight, 1.))  
        connection_list_off.append((coord_map_func(paddle_row, i,0, x_res),
                                    idx, weight, 1.))    
    
    return connection_list_on, connection_list_off
