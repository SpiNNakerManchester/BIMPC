def direction_connection(direction, x_res, y_res, div, delays, weight, row_col_to_input):
    
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
                        on_src=row_col_to_input(j+k,i+k,1,x_res)
                        off_src=row_col_to_input(j+k,i+k,0,x_res)
                        #src = (j+k)*x_res + i+k
                
                elif direction=="south west":
                    #south west connections
                    #check targets are within range
                    if((i-k)>=0 and ((j+k)<=(y_res-1))):   
                        add_exc = True
                        on_src=row_col_to_input(j+k,i-k,1,x_res)
                        off_src=row_col_to_input(j+k,i-k,0,x_res)
                        #src = (j+k)*x_res + i-k
                
                elif direction=="north east":
                    #north east connections
                    #check targets are within range
                    if(((i+k)<=(x_res-1)) and ((j-k)>=0)):  
                        add_exc = True 
                       # src = (j-k)*x_res + i+k
                        on_src=row_col_to_input(j-k,i+k,1,x_res)                       
                        off_src=row_col_to_input(j-k,i+k,0,x_res)                       
                                        
                elif direction=="north west":
                    #north east connections
                    #check targets are within range
                    if((i-k)>=0 and ((j-k)>=0)):   
                        add_exc = True
                        #src = (j-k)*x_res + i-k
                        on_src=row_col_to_input(j-k,i-k,1,x_res)                       
                        off_src=row_col_to_input(j-k,i-k,0,x_res)                                                               
                elif direction=="north":
                    #north connections
                    #check targets are within range
                    if((j-k)>=0):   
                        add_exc = True
                        #src = (j-k)*x_res + i
                        on_src=row_col_to_input(j-k,i,1,x_res) 
                        off_src=row_col_to_input(j-k,i,0,x_res)                                          
                
                elif direction=="south":
                    #north connections
                    #check targets are within range
                    if((j+k)<=(y_res-1)):   
                        add_exc = True
                        #src = (j+k)*x_res + i
                        on_src=row_col_to_input(j+k,i,1,x_res)                   
                        off_src=row_col_to_input(j+k,i,0,x_res)                     
                        
                elif direction=="east":
                    #north connections
                    #check targets are within range
                    if((i+k)<=(x_res-1)):   
                        add_exc = True
                        #src = j*x_res + i+k
                        on_src=row_col_to_input(j,i+k,1,x_res)
                        off_src=row_col_to_input(j,i+k,0,x_res)                       
                        
                elif direction=="west":
                    #north connections
                    #check targets are within range
                    if((i-k)>=0):   
                        add_exc = True
                        #src = j*x_res + i-k
                        on_src=row_col_to_input(j,i-k,1,x_res)     
                        off_src=row_col_to_input(j,i-k,0,x_res)    
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
                         row_col_to_input):
    
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
            connection_list_on.append((row_col_to_input(j, i,1, x_res), 
                                       subsampidx, weight, 1.))
            #OFF channels only on segment borders 
            #if((j+1)%(y_res/subsamp_factor)==0 or (i+1)%(x_res/subsamp_factor)==0 or j==0 or i==0):
            connection_list_off.append((row_col_to_input(j, i,0, x_res),subsampidx, weight, 1.))    
            
    return connection_list_on, connection_list_off

def paddle_connection(x_res,paddle_row, subsamp_factor_x, weight, row_col_to_input):
    connection_list_on=[]
    connection_list_off=[]
    
    for i in range(int(x_res)):
        idx = i//subsamp_factor_x
        connection_list_on.append((row_col_to_input(paddle_row, i,1, x_res),idx, weight, 1.))  
        connection_list_off.append((row_col_to_input(paddle_row, i,0, x_res),idx, weight, 1.))    
    
    return connection_list_on, connection_list_off
