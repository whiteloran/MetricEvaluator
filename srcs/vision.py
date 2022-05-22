import contextlib
import sys
import numpy as np
import cv2

from utils.imgdic import img_dic
from utils.sig_io import *


# todo transparant mode
def sig_array(array1,array2):
    if array1.shape != array2.shape:
        return None
    
    array1,array2 = array1.astype(np.uint8),array2.astype(np.uint8)
    
    array3 = np.copy(array1)
    stacked_array = np.stack((array3,)*3, axis=-1)
    
    stacked_array[array2==1] = [128,0,0]
    stacked_array[array2==2] = [0,128,0]

    return stacked_array


def sig_contour_2(array1,array2,array3,color1,color2,color3,color4,color5,color6):
    k = []
    if array1.shape != array2.shape != array3.shape:
        return None

    (x, y, z) = array1.shape

    array1,array2,array3 = array1.astype(np.uint8),array2.astype(np.uint8),array3.astype(np.uint8)

    for i in range(x):
        img1 = array1[i,:,:].astype(np.uint8)
        img2 = array2[i,:,:].astype(np.uint8)
        img3 = array3[i,:,:].astype(np.uint8)

        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)

        mask1,mask2 = np.where(img2 == 1),np.where(img2 == 2)
        mask_1 = np.zeros((y,z), dtype="uint8")
        mask_1[mask1]=255
        mask_2 = np.zeros((y,z), dtype="uint8")
        mask_2[mask2]=255

        _, binary_g_1 = cv2.threshold(mask_1, 127, 255, cv2.THRESH_BINARY)
        mask_ground_1, _ = cv2.findContours(binary_g_1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
        cv2.drawContours(img1, mask_ground_1, -1, color1, 1) 

        _, binary_g_2 = cv2.threshold(mask_2, 127, 255, cv2.THRESH_BINARY)
        mask_ground_2, _ = cv2.findContours(binary_g_2, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[-2:]
        cv2.drawContours(img1, mask_ground_2[0], -1, color2, 1)
        with contextlib.suppress(Exception):
            cv2.drawContours(img1, mask_ground_2[1], -1, color5, 1)
        mask3,mask4 = np.where(img3 == 1),np.where(img3 == 2)
        mask_3 = np.zeros((y,z), dtype="uint8")
        mask_3[mask3]=255
        mask_4 = np.zeros((y,z), dtype="uint8")
        mask_4[mask4]=255

        _, binary_g_3 = cv2.threshold(mask_3, 127, 255, cv2.THRESH_BINARY)
        mask_ground_3, _ = cv2.findContours(binary_g_3, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
        cv2.drawContours(img1, mask_ground_3, -1, color3, 1) 


        _, binary_g_4 = cv2.threshold(mask_4, 127, 255, cv2.THRESH_BINARY)
        mask_ground_4,_ = cv2.findContours(binary_g_4, cv2.cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[-2:]

        try: 
            cv2.drawContours(img1, mask_ground_4[0], -1, color4, 1) 
        except Exception:
            cv2.drawContours(img1, mask_ground_4, -1, color4, 1) 

        with contextlib.suppress(Exception):
            cv2.drawContours(img1, mask_ground_4[1], -1, color6, 1)
        k.append(img1)

    return np.array(k)

def sig_contour(array1,array2,color1,color2):
    k = []
    if array1.shape != array2.shape:
        return None

    (x, y, z) = array1.shape

    array1,array2 = array1.astype(np.uint8),array2.astype(np.uint8)

    for i in range(x):
        img1 = array1[i,:,:].astype(np.uint8)
        img2 = array2[i,:,:].astype(np.uint8)
        

        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        
        mask1,mask2 = np.where(img2 == 1),np.where(img2 == 2)  
        mask_1 = np.zeros((y,z), dtype="uint8")
        mask_1[mask1]=255
        mask_2 = np.zeros((y,z), dtype="uint8")
        mask_2[mask2]=255

        ret_1, binary_g_1 = cv2.threshold(mask_1, 127, 255, cv2.THRESH_BINARY)
        mask_ground_1, hierarchy_1 = cv2.findContours(binary_g_1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
        cv2.drawContours(img1, mask_ground_1, -1, color1, 1) 

        ret_2, binary_g_2 = cv2.threshold(mask_2, 127, 255, cv2.THRESH_BINARY)
        mask_ground_2, hierarchy_2 = cv2.findContours(binary_g_2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
        cv2.drawContours(img1, mask_ground_2, -1, color2, 1) 

        k.append(img1)

    return np.array(k)



def show_cont_vision(img_path,gt_path,pred_path,savepath,**kwargs):
    # sourcery skip: remove-dict-keys

    pred_dic = img_dic(pred_path,mixflag=True)
    gt_dic = img_dic(gt_path,pred_dic.get_sublist())
    ori_dic = img_dic(img_path,pred_dic.get_sublist())

    if gt_dic.get_subnum() != pred_dic.get_subnum() != ori_dic.get_subnum:
        sys.exit('unmatch gt and pred')

    print(f"Total example:{gt_dic.get_subnum()}")

    print('Load complete.')
    
    m = dict(zip(pred_dic.get_sublist(),pred_dic.get_mat()))
    n = dict(zip(gt_dic.get_sublist(),gt_dic.get_mat()))
    p = dict(zip(ori_dic.get_sublist(),ori_dic.get_mat()))

    for name in m.keys():

    #for idx,sub in enumerate(zip(ori_dic.get_mat(),gt_dic.get_mat(),pred_dic.get_mat())):

        k = sig_contour_2(p[name].numpy(),n[name].numpy(),m[name].numpy(),(60,0,0),(200,0,0),(0,60,0),(0,200,0),(128,0,0),(0,128,0))
        

        array2save(k,f'{savepath}/{name}')
    
    print('Save vision complete.')


def show_fill_vision(img_path,gt_path,pred_path,savepath,**kwargs):
    pred_dic = img_dic(pred_path,mixflag=True)
    gt_dic = img_dic(gt_path,pred_dic.get_sublist())
    ori_dic = img_dic(img_path,pred_dic.get_sublist())

    if gt_dic.get_subnum() != pred_dic.get_subnum() != ori_dic.get_subnum:
        sys.exit('unmatch gt and pred')

    print(f"Total example:{gt_dic.get_subnum()}")

    print('Load complete.')

    for idx,sub in enumerate(zip(ori_dic.get_mat(),gt_dic.get_mat(),pred_dic.get_mat())):


        k = sig_array(sub[0].numpy(),sub[1].numpy())
        array2save(k,f'{savepath}/{pred_dic.get_sublist()[idx]}')
    print('Save vision complete.')
