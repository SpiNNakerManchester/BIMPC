import numpy as np
import pylab as plt
import pickle
import os
import cv2

HEIGHT, WIDTH = 0, 1

base_name = 'retina_dir--__%s_ganglion.m4v'
fnames = [base_name%'SW', base_name%'S', base_name%'SE',
          base_name%'W', 'retina_cs_ganglion.m4v',   base_name%'E',
          base_name%'NW', base_name%'N', base_name%'NE',]

vids = [cv2.VideoCapture(fn) for fn in fnames]
print("Read video files")
frames = []
frame_shape = {WIDTH: 160, HEIGHT: 128}
total_shape = {WIDTH: frame_shape[WIDTH]*3, HEIGHT: frame_shape[HEIGHT]*3}
out_frame = np.zeros((total_shape[HEIGHT], total_shape[WIDTH], 3), dtype=np.uint8)
scan_frames = True
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
title = 'retina_output_tiled'
fps = 40
# frame_t = int(1000./fps)
vid_out = cv2.VideoWriter("%s.m4v"%title, fourcc, fps, 
                          (total_shape[WIDTH], total_shape[HEIGHT]))

fcount = 0
while scan_frames:
    print(fcount)
    fcount += 1
    frames[:] = [ vid.read() for vid in vids ]
    # print(len(frames))
    out_frame[:] = 0
    i = 0
    some_right = False
    for (ret, frm) in frames:
        fr = (i//3)*frame_shape[HEIGHT]
        tr = fr + frame_shape[HEIGHT]
        fc = (i%3)*frame_shape[WIDTH]
        tc = fc + frame_shape[WIDTH]
        i += 1
        # print(i//3, i%3)
        
        if not ret:
            continue


        # print("rows(%d->%d), cols(%d->%d)"%(fr, tr, fc, tc))
        out_frame[fr:tr, fc:tc, :] = cv2.resize(frm, 
                                        (frame_shape[WIDTH], frame_shape[HEIGHT]), 
                                        interpolation=cv2.INTER_NEAREST)
        cv2.imshow('frame', out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        vid_out.write(out_frame)
        some_right = some_right or ret
    if not some_right:
        break

vid_out.release()
