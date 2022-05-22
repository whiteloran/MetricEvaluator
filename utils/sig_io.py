import imp
import os
import re
import imageio

def batcher(folderpath,):
    return folderpath, os.listdir(folderpath) 

def saver(savename,thefile):
    imageio.imsave(savename,thefile)

def array2save(array,fapath):  # sourcery skip: use-fstring-for-concatenation
    if not os.path.exists(fapath):
        os.mkdir(fapath)
    for idx, i in enumerate(array):
        saver(fapath + '/' + str(idx).rjust(5,'0')+'.png', i)