import numpy as np

def row_col_to_input(row, col, is_on_input, width, height, row_bits):
    # row_bits = np.uint8(np.ceil(np.log2(height)))
    idx = np.uint32(0)
    
    if is_on_input:
        idx = idx | 1
    
    idx = idx | (row << 1)
    idx = idx | (col << (row_bits + 1))
    
    return idx


def row_col_to_input_breakout(row, col, is_on_input, x_res=160):
    row_bits = np.uint32(8)
    idx = np.uint32(0)
    
    if is_on_input:
        idx = idx | 1
    
    idx = idx | (row << 1)#colour bit
    idx = idx | (col << (row_bits + 1))
    
    #add two to allow for special event bits
    idx=idx+2
    return idx


def row_col_to_input_subsamp(row, col,is_on_input,x_res):
    idx = np.uint32(0)
    idx = row*x_res + col
    return idx
                
