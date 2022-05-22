import torch
import numpy as np
import cv2
import re

from utils.sig_io import batcher,saver


class img_dic(object):

    def __init__(self,thepath,img_h=256,img_w=256,theli=[], mixflag=False,**kwargs) -> None:
        self.path = thepath
        self.mix_flag = mixflag

        self.height = img_h
        self.width = img_w
        self.piclist = []
        self.sublist = self.readlist(theli)

        self.mat = []      
        self.final_mat = []
        self.dic = {}      
        self.final_dic = {}


    def readlist(self,theli=[]):
        _, subjects = batcher(self.path)
        return [sub for sub in sorted(subjects) if sub in theli or theli == []]

    def iter_subject(self,idx):
        return f'{self.path}/{self.sublist[idx]}'

    def list_pics(self,tpath):
        _, pics = batcher(tpath)
        return list(sorted(pics))

    def subject_mat(self,liste):
        p = []
        for idx,l in enumerate(liste):
            k = self.subject_dual_list(idx) if self.mix_flag else self.subject_list(idx)
            p.append(k)
        return p
    
    def subject_dic(self,liste):
        p = {}
        for idx,l in enumerate(liste):
            k = self.subject_dual_list(idx) if self.mix_flag else self.subject_list(idx)
            p[l] = k
        return p

    def subject_list(self,idx):
        k = []
        lister = self.list_pics(self.iter_subject(idx))
        for imgp in lister:
            img = self.loader(f'{self.iter_subject(idx)}/{imgp}')
            img = cv2.resize(img,(self.height,self.width), interpolation = cv2.INTER_CUBIC)
            k.append(img)
        return torch.from_numpy(np.array(k))

    def subject_dual_list(self,idx):
        k = []
        img1 = []
        img2 = []
        lister = self.list_pics(self.iter_subject(idx))
        #print(lister)
        for imgp in lister:
            index,flag = self.namecutter(imgp)
            if flag == '00.png':
                img = self.loader(f'{self.iter_subject(idx)}/{imgp}') 
                img = cv2.resize(img,(self.height,self.width), interpolation = cv2.INTER_CUBIC)
                img1.append(img)  
            if flag == '01.png':
                img = self.loader(f'{self.iter_subject(idx)}/{imgp}') 
                img = cv2.resize(img,(self.height,self.width), interpolation = cv2.INTER_CUBIC)
                img2.append(img)
        for i in zip(img1,img2):
            reseted = self.mix_pic(i[0],i[1])
            k.append(reseted)
        return torch.from_numpy(np.array(k))

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
        k = np.copy(img1)
        k[img1 == np.unique(img1)[-1]] = 2
        k[img2 == np.unique(img2)[-1]] = 1
        return k

    def loader(self,tpath):
        return cv2.imread(tpath, cv2.IMREAD_UNCHANGED)
    
    def get_sublist(self):
        return self.sublist

    def get_mat(self):
        self.mat = self.subject_mat(self.sublist)  
        return self.mat

    def get_dic(self):
        self.dic = self.subject_dic(self.sublist)  
        return self.dic
    
    def get_final_mat(self):
        self.get_mat()
        self.final_mat = self.one_hot_ring2cycle(self.mat)
        return self.final_mat
    
    def get_final_dic(self):
        self.get_dic()
        self.final_dic = self.one_hot_ring2cycle_dic(self.dic)
        return self.final_dic
    
    def get_path(self):
        return self.path

    def get_subnum(self):
        return len(self.sublist)