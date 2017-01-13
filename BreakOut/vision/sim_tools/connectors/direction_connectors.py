import numpy as np

def direction_connection(direction, x_res, y_res, div, delays, weight):
    
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
                        src = (j+k)*x_res + i+k
                
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
                        src = (j-k)*x_res + i+k
                                        
                elif direction=="north west":
                    #north east connections
                    #check targets are within range
                    if((i-k)>=0 and ((j-k)>=0)):   
                        add_exc = True
                        src = (j-k)*x_res + i-k
                                        
                elif direction=="north":
                    #north connections
                    #check targets are within range
                    if((j-k)>=0):   
                        add_exc = True
                        src = (j-k)*x_res + i
                
                elif direction=="south":
                    #north connections
                    #check targets are within range
                    if((j+k)<=(y_res-1)):   
                        add_exc = True
                        src = (j+k)*x_res + i
                        
                elif direction=="east":
                    #north connections
                    #check targets are within range
                    if((i+k)<=(x_res-1)):   
                        add_exc = True
                        src = j*x_res + i+k
                        
                elif direction=="west":
                    #north connections
                    #check targets are within range
                    if((i-k)>=0):   
                        add_exc = True
                        src = j*x_res + i-k
                        
                else:
                    raise Exception( "Not a valid direction: %s"%direction )

                #ON channels
                connection_list_on.append((src, dst, weight, Delay))
                #OFF channels
                connection_list_off.append((src, dst, weight, Delay))
                add_exc = False

    return [connection_list_on, connection_list_inh_on], \
            [connection_list_off, connection_list_inh_off]


def subsample_connection(x_res, y_res, subsamp_factor, weight, 
                         row_col_to_input):
    
    # subY_BITS=int(np.ceil(np.log2(y_res/subsamp_factor)))
    connection_list_on=[]
    connection_list_off=[]
    
    sx_res = int(x_res)//int(subsamp_factor)
    
    for j in range(int(y_res)):
        for i in range(int(x_res)):
            si = i//subsamp_factor
            sj = j//subsamp_factor
            #ON channels
            subsampidx = sj*sx_res + si
            connection_list_on.append((row_col_to_input(j, i,1), 
                                       subsampidx, weight, 1.))
            #OFF channels
            connection_list_off.append((row_col_to_input(j, i,0), 
                                        subsampidx, weight, 1.))    
            
    return connection_list_on, connection_list_off
    

