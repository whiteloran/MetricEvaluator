from cmath import pi

import torch
import numpy as np
import os
import cv2
import re

from utils.metrics import DiceAverage,IouAverage,HDAverage,RVDAverage,RMSEAverage

from utils.sig_io import batcher


class img_dic(object):

    def __init__(self,thepath,theli=[], mixflag=False) -> None:
        self.path = thepath
        self.mix_flag = mixflag
        self.piclist = []
        self.sublist = self.readlist(theli)
        self.mat = self.subject_mat(self.sublist)        
        self.final_mat = self.one_hot_ring2cycle(self.mat)

    def readlist(self,theli=[]):
        _, subjects = batcher(self.path)
        #print(subjects)
        return [sub for sub in sorted(subjects,key=lambda x:int(str(x))) if sub in theli or theli == []]

    def iter_subject(self,idx):
        return f'{self.path}/{self.sublist[idx]}'

    def list_pics(self,tpath):
        self.piclist = []
        _, pics = batcher(tpath)
        self.piclist.extend(f'{tpath}/{p}' for p in pics)

    def subject_mat(self,liste):
        #self.mat = []
        p = []
        for idx,l in enumerate(liste):
            k = self.subject_dual_list(idx) if self.mix_flag else self.subject_list(idx)
            p.append(k)
        return p

    def subject_list(self,idx):
        k = []
        self.list_pics(self.iter_subject(idx))
        for imgp in self.piclist:
            img = self.loader(imgp)
            k.append(img)
        return torch.from_numpy(np.array(k))

    def subject_dual_list(self,idx):
        k = []
        img1 = []
        img2 = []
        self.list_pics(self.iter_subject(idx))
        for imgp in self.piclist:
            index,flag = self.namecutter(imgp)
            if flag == '00.png':
                img = self.loader(imgp) 
                img1.append(img)  
            if flag == '01.png':
                img = self.loader(imgp) 
                img2.append(img) 
        for idx,i in enumerate(zip(img1,img2)):
            reseted = self.mix_pic(i[1],i[0])
            k.append(reseted)
        return torch.from_numpy(np.array(k))

    def one_hot(self,arra):
        #print(len(arra))
        k = []
        for i in range(len(arra)):
            p = []
            uni = np.unique(arra[i])
            #print(arra[i].shape)
            for idx,u in enumerate(uni):
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
            uni = np.unique(arra[i])
            #print(arra[i].shape)
            for idx,u in enumerate(uni):
                ori_img = np.copy(arra[i])
                t_img = np.copy(arra[i])
                t_img[ori_img != u] = 0
                t_img[ori_img == u] = 1
                if idx == 2:
                    t_img[ori_img == 1] = 1
                p.append(t_img)
            k.append(p)
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
        return self.mat
    
    def get_final_mat(self):
        return self.final_mat
    
    def get_path(self):
        return self.path

    def get_subnum(self):
        print(len(self.sublist))


def run_eva(gt_path,pred_path):

    pred_dic = img_dic(pred_path,mixflag=True)
    gt_dic = img_dic(gt_path,pred_dic.get_sublist())

    pred_dic.get_subnum()
    gt_dic.get_subnum()
    print('Load complete.')

    getdice = DiceAverage(2)
    getiou = IouAverage(2)
    gethd = HDAverage(2)
    getrmse = RMSEAverage(2)
    getrvd = RVDAverage(2)

    print('Metrics initialized.')

    for idx,sub in enumerate(zip(gt_dic.get_final_mat(),pred_dic.get_final_mat())):

        gt_mat = torch.stack((torch.from_numpy(sub[0][1]),torch.from_numpy(sub[0][2])),0)
        pred_mat = torch.stack((torch.from_numpy(sub[1][1]),torch.from_numpy(sub[1][2])),0)
        
        getdice.update(gt_mat[:,0:pred_mat.shape[1],:,:],pred_mat)
        getiou.update(gt_mat[:,0:pred_mat.shape[1],:,:],pred_mat)
        gethd.update(gt_mat[:,0:pred_mat.shape[1],:,:],pred_mat)
        getrmse.update(gt_mat[:,0:pred_mat.shape[1],:,:],pred_mat)
        getrvd.update(gt_mat[:,0:pred_mat.shape[1],:,:],pred_mat)

    print(f'Dice:\t{getdice.avg}')
    print(f'Iou:\t{getiou.avg}')
    print(f'HD:\t{gethd.avg}')
    print(f'RMSE:\t{getrmse.avg}')
    print(f'RVD:\t{getrvd.avg}')

    print('Complete.')

if __name__ == '__main__':

    gt_path = r''
    pred_path = r''
    run_eva(gt_path,pred_path)