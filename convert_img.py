import cv2
import os.path
from os.path import join, isdir
from os import listdir, path
import glob
import numpy as np
from PIL import Image

def convertPNG(pngfile,outdir):
    # READ THE DEPTH
    im_depth = cv2.imread(pngfile)
    #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
    im_color=cv2.applyColorMap(cv2.convertScaleAbs(im_depth,alpha=1),cv2.COLORMAP_JET)
    #convert to mat png
    im=Image.fromarray(im_color)
    #save image
    print(outdir)
    print(os.path.basename(pngfile))
    im.save(os.path.join(outdir,os.path.basename(pngfile)))
root = os.environ["AVOIDBENCH_PATH"] + "/../mavrl/saved/LSTM_0/Reconstruction/Sequence_0"
dirs = [
    join(root, sd)
    for sd in listdir(root) if isdir(join(root, sd))]
print(dirs)
for d in dirs:
    files = [join(d, f) for f in listdir(d)]
    dir_color = os.makedirs(d+"/color", exist_ok=True)
    for file in files:
        convertPNG(file, d+"/color")
