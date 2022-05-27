import numpy as np
import cv2
import re

from utils.sig_io import batcher,saver


class img_dic(object):

    def __init__(self,thepath,img_h=80,img_w=80,theli=[], mixflag=None,**kwargs) -> None:
        self.path = thepath
        self.mix_flag = mixflag

        self.height = img_h
        self.width = img_w
        self.piclist = []
        self.sublist = self.readlist(theli)
        self.subdic = self.readic(theli)

        self.mat = []      
        self.final_mat = []
        self.dic = {}      
        self.final_dic = {}

    def readlist(self,theli=[]):
        _, subjects = batcher(self.path)
        return [sub for sub in sorted(subjects) if sub in theli or theli == []]

    def readic(self,theli=[]):
        _, subjects = batcher(self.path)
        return {sub: f'{self.path}/{sub}' for sub in subjects if sub in theli or theli == []}

    def list_pics(self,tpath):
        _, pics = batcher(tpath)
        return list(sorted(pics))

    def read_subject_mat(self,tpath):
        return self.read_subject_dual_list(tpath) if self.mix_flag else self.read_subject_list(tpath)
    
    def subject_dic(self,liste):
        p = {}
        for idx,l in enumerate(liste):
            k = self.subject_dual_list(idx) if self.mix_flag else self.subject_list(idx)
            p[l] = k
        return p
    
    def read_subject_list(self,tpath):
        k = []
        lister = self.list_pics(tpath)
        for imgp in lister:
            img = self.loader(f'{tpath}/{imgp}')
            img = cv2.resize(img,(self.height,self.width), interpolation = cv2.INTER_CUBIC)
            k.append(img)
        return np.array(k)
    
    def read_subject_dual_list(self,tpath):
        k = []
        img1 = []
        img2 = []
        lister = self.list_pics(tpath)
        #print(lister)
        for imgp in lister:
            index,flag = self.namecutter(imgp)
            if flag == '00.png':
                img = self.loader(f'{tpath}/{imgp}') 
                img = cv2.resize(img,(self.height,self.width), interpolation = cv2.INTER_CUBIC)
                img1.append(img)  
            if flag == '01.png':
                img = self.loader(f'{tpath}/{imgp}') 
                img = cv2.resize(img,(self.height,self.width), interpolation = cv2.INTER_CUBIC)
                img2.append(img)
        for i in zip(img1,img2):
            reseted = self.mix_pic(i[0],i[1])
            #reseted = self.mix_pic(i[1],i[0])
            k.append(reseted)
        return np.array(k)

    def one_hot(self,arra):
        k = []
        for i in range(len(arra)):
            p = []
            uni = np.unique(arra[i])
            for u in uni:
                ori_img = np.copy(arra[i])
                t_img = np.copy(arra[i])
                t_img[ori_img != u] = 0
                t_img[ori_img == u] = 1
                p.append(t_img)
            k.append(p)
        return k

    def one_hot_ring2cycle(self,arra):
        #print(len(arra))
        k = []
        for i in range(len(arra)):
            p = []
            uni = np.unique(arra[i])[1:]
            #print(uni)
            for idx,u in enumerate(uni):
                ori_img = np.copy(arra[i])
                t_img = np.copy(arra[i])
                r_img = np.copy(arra[i])
                t_img[ori_img != u] = 0
                t_img[ori_img == u] = 1
                p.append(t_img)
                if idx == 1:
                    r_img[ori_img != u] = 0
                    r_img[ori_img == u] = 1
                    r_img[ori_img == uni[idx-1]] = 1
                    p.append(r_img)
            k.append(p)
        return k

    def read_one_hot_ring2cycle(self,arra):
        #print(len(arra))
        p = []
        uni = np.unique(arra)[1:]
        #print(uni)
        for idx,u in enumerate(uni):
            ori_img = np.copy(arra)
            t_img = np.copy(arra)
            r_img = np.copy(arra)
            t_img[ori_img != u] = 0
            t_img[ori_img == u] = 1
            p.append(t_img)
            if idx == 1:
                r_img[ori_img != u] = 0
                r_img[ori_img == u] = 1
                r_img[ori_img == uni[idx-1]] = 1
                p.append(r_img)
        return p

    def one_hot_ring2cycle_dic(self,arra):
        k = {}
        for i in arra.keys():
            p = []
            uni = np.unique(arra[i])[1:]
            for idx,u in enumerate(uni):
                ori_img = np.copy(arra[i])
                t_img = np.copy(arra[i])
                t_img[ori_img != u] = 0
                t_img[ori_img == u] = 1
                if idx == 2:
                    t_img[ori_img == 1] = 1
                p.append(t_img)
            k[i] = p
        return k

    def namecutter(self,filename):
        out1 = re.split(r'_',filename)
        return out1[-3], out1[-1]

    def mix_pic(self,img1,img2):
        # change this if category changes
        k = np.copy(img1)
        k[img1 == np.unique(img1)[-1]] = 2
        #k[img1 == np.unique(img1)[-1]] = 1
        k[img2 == np.unique(img2)[-1]] = 1
        return k

    def loader(self,tpath):
        return cv2.imread(tpath, cv2.IMREAD_UNCHANGED)
    
    def get_sublist(self):
        return self.sublist

    def get_subdic(self):
        return self.subdic

    def get_single_mat(self,p):
        self.mat = self.read_subject_mat(p)  
        self.mat = self.one_hot_ring2cycle(self.mat)
        return np.array(self.mat)
    
    def get_path(self):
        return self.path

    def get_subnum(self):
        return len(self.sublist)