import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from scipy import misc

def subsamp_size(start, end, step):
    return ( (end - start - 1)//step ) + 1


base_dir = sys.argv[1]

NO_CONN = np.array([50, 0, 0])
weight_to_pix = (254./(4.8*2.))
width  = 160
height = 128
scale = 10.
cs_w = subsamp_size(1, width, 2)
cs_h = subsamp_size(1, height, 2)
cs2_w = subsamp_size(3, width, 4)
cs2_h = subsamp_size(3, height, 4)

print(cs_w, cs_h)
print(cs2_w, cs2_h)

weight_fnames = glob.glob(os.path.join(base_dir, "weights_loop_*.txt"))

weight_fnames.sort()

files = []
for fname in weight_fnames:
    files.append(open(fname, mode='r'))

img_cs = [np.zeros((cs_h, cs_w, 3), dtype='uint8') for _ in range(100)]
img_cs2 = [np.zeros((cs2_h, cs2_w, 3), dtype='uint8') for _ in range(100)]


w_list = []
done_reading = False
read_comment = False
read_title   = False
finish_conn  = False
is_cs  = False
is_cs2 = False
title = ""

weight_count = 0
weight_sum = 0.
for file_idx, file in enumerate(files):
    for line in file:
        if line.startswith("--- Weights"):
            print(line)
            title = line.replace("-", "")
            title = title.strip()
            title = title.replace("  ", " ")
            title = title.replace(" ", "_")
            title = title.replace("\n", "")

            if "cs2" in line:
                is_cs2 = True; is_cs  = False
            else:
                is_cs  = True; is_cs2 = False

            for i in range(len(img_cs)):
                img_cs[i][:, :] = NO_CONN

            for i in range(len(img_cs2)):
                img_cs2[i][:, :] = NO_CONN

            continue

        elif line.startswith("---"):
            finish_conn = True

            if not os.path.isdir(os.path.join(base_dir, title)):
                os.makedirs(os.path.join(base_dir, title))

            if is_cs:
                for idx, img in enumerate(img_cs):
                    sys.stdout.write("\r%s\t%05d\t%05d"%(title, file_idx, idx))
                    sys.stdout.flush()
                    misc.imsave(os.path.join(base_dir, title,
                                        "%s_%05d_%05d.png"%(title, idx, file_idx)),
                                misc.imresize(img, float(scale), interp='nearest'))
                print("")
            elif is_cs2:
                for idx, img in enumerate(img_cs2):
                    sys.stdout.write("\r%s\t%05d\t%05d"%(title, file_idx, idx))
                    sys.stdout.flush()
                    misc.imsave(os.path.join(base_dir, title,
                                        "%s_%05d_%05d.png"%(title, idx, file_idx)),
                                misc.imresize(img, float(scale), interp='nearest'))
                print("")
            is_cs2 = False
            is_cs = False
            continue

        elif line.startswith("\n"):
            continue
        else:
            spl = line.split(", ")
            src = int(spl[0])
            dst = int(spl[1])
            w   = np.uint8(float(spl[2])*weight_to_pix)

            if is_cs:
                row = src//cs_w
                col = src%cs_w
                img_cs[dst][row, col, :] = w
            elif is_cs2:
                row = src//cs2_w
                col = src%cs2_w
                img_cs2[dst][row, col, :] = w
            else:
                continue


file_list = glob.glob(os.path.join(base_dir, "*"))
dir_list = []
for fname in file_list:
    if os.path.isdir(fname):
        dir_list.append()