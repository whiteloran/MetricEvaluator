import sys
import torch
import numpy as np

from utils.imgdic import img_dic
from utils.metrics import DiceAverage,IouAverage,HDAverage,RVDAverage,RMSEAverage

def ini_metrics(classes):
    return DiceAverage(classes),IouAverage(classes),HDAverage(classes),RMSEAverage(classes),RVDAverage(classes)

def old_run_eva(gt_path,pred_path,classes,**kwargs):
    # sourcery skip: remove-dict-keys, remove-redundant-pass

    '''
    every subject have the equal contribution of the final metric.
    '''

    k = []
    d = []

    pred_dic = img_dic(pred_path,mixflag=True)
    gt_dic = img_dic(gt_path,theli=pred_dic.get_sublist())

    if gt_dic.get_subnum() != pred_dic.get_subnum():
        sys.exit(f'unmatch gt {gt_dic.get_subnum()} and pred {pred_dic.get_subnum()} ')

    print(f"Total example:{gt_dic.get_subnum()}")
    print('Load complete.')

    getdice,getiou,gethd,getrmse,getrvd = ini_metrics(classes)
    print('Metrics initialized.')

    m = dict(zip(pred_dic.get_sublist(),pred_dic.get_final_mat()))
    n = dict(zip(gt_dic.get_sublist(),gt_dic.get_final_mat()))

    print('Dictionary sucessfully built.')

    for name in m.keys():

        try:
            gt_mat = torch.stack([torch.from_numpy(i) for i in n[name]],0)
            pred_mat = torch.stack([torch.from_numpy(i) for i in m[name]],0)

            print(gt_mat.shape)
            print(pred_mat.shape)
            
            getdice.update(logits=gt_mat,targets=pred_mat,class_num = classes)
            getiou.update(logits=gt_mat,targets=pred_mat,class_num = classes)
            gethd.update(logits=gt_mat,targets=pred_mat,class_num = classes)
            getrmse.update(logits=gt_mat,targets=pred_mat,class_num = classes)
            getrvd.update(logits=gt_mat,targets=pred_mat,class_num = classes)


            if np.mean(getdice.value) <= 0.7:
                print(f'Low dice {getdice.value}: example {name}')


            k.append([getdice.value,getiou.value,gethd.value,getrmse.value,getrvd.value])

        except Exception as e:
            print(f'Skip example:{name} \twith:  {e}')
            d.append(name)
            pass

    dimet = np.array(k)

    print(f'Unsucceed example:\t{len(d)}')

    print('Metric:')
    print(f'Indices:\t    [Endo Myo Epi]; \t    [Endo Myo Epi]')
    print(f'Dice:\tmean:{getdice.avg}; \tvar:{[np.around(dimet[:,0,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'Iou:\tmean:{getiou.avg}; \tvar:{[np.around(dimet[:,1,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'HD:\tmean:{gethd.avg}; \tvar:{[np.around(dimet[:,2,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'RMSE:\tmean:{getrmse.avg}; \tvar:{[np.around(dimet[:,3,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'RVD:\tmean:{getrvd.avg}; \tvar:{[np.around(dimet[:,4,i].var(),4) for i in range(dimet.shape[2])]}') 



def run_eva(gt_path,pred_path,classes,**kwargs):
    # sourcery skip: remove-dict-keys, remove-redundant-pass

    '''
    every subject have the equal contribution of the final metric.
    '''

    k = []
    d = []

    pred_dic = img_dic(pred_path,mixflag=True)
    gt_dic = img_dic(gt_path,theli=pred_dic.get_sublist())

    if gt_dic.get_subnum() != pred_dic.get_subnum():
        sys.exit(f'unmatch gt {gt_dic.get_subnum()} and pred {pred_dic.get_subnum()} ')

    print(f"Total example:{gt_dic.get_subnum()}")
    print('Load complete.')

    getdice,getiou,gethd,getrmse,getrvd = ini_metrics(classes)
    print('Metrics initialized.')

    m = pred_dic.get_subdic()
    n = gt_dic.get_subdic()

    print('Dictionary sucessfully built.')

    for name in m.keys():

        try:
            if len(gt_dic.get_single_mat(n[name]).shape) == 4 and len(pred_dic.get_single_mat(m[name]).shape) == 4:
                gt_mat = torch.from_numpy(np.moveaxis(gt_dic.get_single_mat(n[name]),0,1))
                pred_mat = torch.from_numpy(np.moveaxis(pred_dic.get_single_mat(m[name]),0,1))
            else:
                gt_mat = torch.from_numpy(gt_dic.get_single_mat(n[name]))
                pred_mat = torch.from_numpy(pred_dic.get_single_mat(m[name]))
                gt_mat = gt_mat.unsqueeze(0)
                pred_mat = pred_mat.unsqueeze(0)

            #print(gt_mat.shape)
            #print(pred_mat.shape)
            
            
            getdice.update(logits=gt_mat,targets=pred_mat,class_num = classes)
            getiou.update(logits=gt_mat,targets=pred_mat,class_num = classes)
            gethd.update(logits=gt_mat,targets=pred_mat,class_num = classes)
            getrmse.update(logits=gt_mat,targets=pred_mat,class_num = classes)
            getrvd.update(logits=gt_mat,targets=pred_mat,class_num = classes)


            if np.mean(getdice.value) <= 0.7:
                print(f'Low dice {getdice.value}: example {name}')


            k.append([getdice.value,getiou.value,gethd.value,getrmse.value,getrvd.value])

        except Exception as e:
            print(f'Skip example:{name} \twith:  {e}')
            d.append(name)
            pass

    dimet = np.array(k)

    print(f'Unsucceed example:\t{len(d)}')

    print('Metric:')
    print(f'Indices:\t    [Endo Myo Epi]; \t    [Endo Myo Epi]')
    print(f'Dice:\tmean:{getdice.avg}; \tvar:{[np.around(dimet[:,0,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'Iou:\tmean:{getiou.avg}; \tvar:{[np.around(dimet[:,1,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'HD:\tmean:{gethd.avg}; \tvar:{[np.around(dimet[:,2,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'RMSE:\tmean:{getrmse.avg}; \tvar:{[np.around(dimet[:,3,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'RVD:\tmean:{getrvd.avg}; \tvar:{[np.around(dimet[:,4,i].var(),4) for i in range(dimet.shape[2])]}') 


def old_run_eva_single(gt_path,pred_path,classes,**kwargs):
    # sourcery skip: remove-dict-keys, remove-redundant-pass
    '''
    every slice have the equal contribution of the final metric.
    '''

    k = []
    d = []

    # mixflag = 1 returned 2 normal
    pred_dic = img_dic(pred_path,mixflag=True)
    #pred_dic = img_dic(pred_path,mixflag=False)
    gt_dic = img_dic(gt_path,theli=pred_dic.get_sublist())

    if gt_dic.get_subnum() != pred_dic.get_subnum():
        sys.exit(f'unmatch gt {gt_dic.get_subnum()} and pred {pred_dic.get_subnum()}')

    print(f"Total example:{gt_dic.get_subnum()}")
    print('Load complete.')

    # set flagger to single class or multi class
    getdice,getiou,gethd,getrmse,getrvd = ini_metrics(classes)
    print('Metrics initialized.')

    m = dict(zip(pred_dic.get_sublist(),pred_dic.get_final_mat()))
    n = dict(zip(gt_dic.get_sublist(),gt_dic.get_final_mat()))

    print('Dictionary sucessfully built.')

    for name in m.keys():

        try:
            gt_mat = torch.stack([torch.from_numpy(i) for i in n[name]],0)
            pred_mat = torch.stack([torch.from_numpy(i) for i in m[name]],0)

            #print(gt_mat.shape)
            #print(pred_mat.shape)

            for i in range(gt_mat.shape[1]):
                getdice.update(logits=gt_mat[:,i,:,:],targets=pred_mat[:,i,:,:],class_num = classes)
                getiou.update(logits=gt_mat[:,i,:,:],targets=pred_mat[:,i,:,:],class_num = classes)
                gethd.update(logits=gt_mat[:,i,:,:],targets=pred_mat[:,i,:,:],class_num = classes)
                getrmse.update(logits=gt_mat[:,i,:,:],targets=pred_mat[:,i,:,:],class_num = classes)
                getrvd.update(logits=gt_mat[:,i,:,:],targets=pred_mat[:,i,:,:],class_num = classes)

                if np.mean(getdice.value) <= 0.7:
                    print(f'Low dice{getdice.value}: example {name} with slice {i}')

                k.append([getdice.value,getiou.value,gethd.value,getrmse.value,getrvd.value])

        except Exception as e:
            print(f'Skip example:{name} \t with:  {e}')
            d.append(name)
            pass

    dimet = np.array(k)

    print(f'Unsucceed example:\t{len(d)}')

    print('Metric:')
    
    print(f'Dice:\tmean:{getdice.avg}; \tvar:{[np.around(dimet[:,0,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'Iou:\tmean:{getiou.avg}; \tvar:{[np.around(dimet[:,1,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'HD:\tmean:{gethd.avg}; \tvar:{[np.around(dimet[:,2,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'RMSE:\tmean:{getrmse.avg}; \tvar:{[np.around(dimet[:,3,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'RVD:\tmean:{getrvd.avg}; \tvar:{[np.around(dimet[:,4,i].var(),4) for i in range(dimet.shape[2])]}') 


def run_eva_single(gt_path,pred_path,classes,**kwargs):
    # sourcery skip: remove-dict-keys, remove-redundant-pass
    '''
    every slice have the equal contribution of the final metric.
    '''

    k = []
    d = []

    # mixflag = 1 returned 2 normal
    pred_dic = img_dic(pred_path,mixflag=True)
    #pred_dic = img_dic(pred_path,mixflag=False)
    gt_dic = img_dic(gt_path,theli=pred_dic.get_sublist())

    if gt_dic.get_subnum() != pred_dic.get_subnum():
        sys.exit(f'unmatch gt {gt_dic.get_subnum()} and pred {pred_dic.get_subnum()}')

    print(f"Total example:{gt_dic.get_subnum()}")
    print('Load complete.')

    # set flagger to single class or multi class
    getdice,getiou,gethd,getrmse,getrvd = ini_metrics(classes)
    print('Metrics initialized.')

    m = pred_dic.get_subdic()
    n = gt_dic.get_subdic()

    print('Dictionary sucessfully built.')


    for name in m.keys():

        try:

            if len(gt_dic.get_single_mat(n[name]).shape) == 4 and len(pred_dic.get_single_mat(m[name])) == 4:
                gt_mat = torch.from_numpy(np.moveaxis(gt_dic.get_single_mat(n[name]),0,1))
                pred_mat = torch.from_numpy(np.moveaxis(pred_dic.get_single_mat(m[name]),0,1))
            else:
                gt_mat = torch.from_numpy(gt_dic.get_single_mat(n[name]))
                pred_mat = torch.from_numpy(pred_dic.get_single_mat(m[name]))
                gt_mat = gt_mat.unsqueeze(0)
                pred_mat = pred_mat.unsqueeze(0)

            #print(gt_mat.shape)
            #print(pred_mat.shape)


            if len(gt_mat.shape) <= 3:
                gt_mat = gt_mat.unsqueeze(0)

            if len(pred_mat.shape) <= 3:
                pred_mat = pred_mat.unsqueeze(0)

            for i in range(gt_mat.shape[1]):
                getdice.update(logits=gt_mat[:,i,:,:],targets=pred_mat[:,i,:,:],class_num = classes)
                getiou.update(logits=gt_mat[:,i,:,:],targets=pred_mat[:,i,:,:],class_num = classes)
                gethd.update(logits=gt_mat[:,i,:,:],targets=pred_mat[:,i,:,:],class_num = classes)
                getrmse.update(logits=gt_mat[:,i,:,:],targets=pred_mat[:,i,:,:],class_num = classes)
                getrvd.update(logits=gt_mat[:,i,:,:],targets=pred_mat[:,i,:,:],class_num = classes)

                if np.mean(getdice.value) <= 0.7:
                    print(f'Low dice{getdice.value}: example {name} with slice {i}')

                k.append([getdice.value,getiou.value,gethd.value,getrmse.value,getrvd.value])

        except Exception as e:
            print(f'Skip example:{name} \t with:  {e}')
            d.append(name)
            pass

    dimet = np.array(k)

    print(f'Unsucceed example:\t{len(d)}')

    print('Metric:')
    
    print(f'Dice:\tmean:{getdice.avg}; \tvar:{[np.around(dimet[:,0,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'Iou:\tmean:{getiou.avg}; \tvar:{[np.around(dimet[:,1,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'HD:\tmean:{gethd.avg}; \tvar:{[np.around(dimet[:,2,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'RMSE:\tmean:{getrmse.avg}; \tvar:{[np.around(dimet[:,3,i].var(),4) for i in range(dimet.shape[2])]}')
    print(f'RVD:\tmean:{getrvd.avg}; \tvar:{[np.around(dimet[:,4,i].var(),4) for i in range(dimet.shape[2])]}') 



# todo: unfinished function
def run_eva_dic(gt_path,pred_path,**kwargs):

    k = []

    # mixflag = 1 returned 2 normal
    pred_dic = img_dic(pred_path,mixflag=True)
    gt_dic = img_dic(gt_path,pred_dic.get_sublist())

    if gt_dic.get_subnum() != pred_dic.get_subnum():
        sys.exit('unmatch gt and pred')

    print(f"Total example:{gt_dic.get_subnum()}")
    print('Load complete.')

    getdice,getiou,gethd,getrmse,getrvd = ini_metrics(2)
    print('Metrics initialized.')

    
    q = pred_dic.get_final_dic()
    p = gt_dic.get_final_dic()

    for x in q.keys():

        pred = q[x]
        gt = p[x]
    
        gt_mat = torch.stack((torch.from_numpy(gt[1]),torch.from_numpy(gt[2])),0)
        pred_mat = torch.stack((torch.from_numpy(pred[1]),torch.from_numpy(pred[2])),0)

        getdice.update(gt_mat,pred_mat)
        getiou.update(gt_mat,pred_mat)
        gethd.update(gt_mat,pred_mat)
        getrmse.update(gt_mat,pred_mat)
        getrvd.update(gt_mat,pred_mat)

        k.append([getdice.value,getiou.value,gethd.value,getrmse.value,getrvd.value])

    #print(f'Succeed example: {}')
    dimet = np.array(k)

    #print(f'Unsucceed example:\t{len(d)}')

    print('Metric:')
    print(f'Dice:\tmean:{getdice.avg}; \tstd:{[np.around(dimet[:,0,i].std(),4) for i in range(dimet.shape[2])]}')
    print(f'Iou:\tmean:{getiou.avg}; \tstd:{[np.around(dimet[:,1,i].std(),4) for i in range(dimet.shape[2])]}')
    print(f'HD:\tmean:{gethd.avg}; \tstd:{[np.around(dimet[:,2,i].std(),4) for i in range(dimet.shape[2])]}')
    print(f'RMSE:\tmean:{getrmse.avg}; \tstd:{[np.around(dimet[:,3,i].std(),4) for i in range(dimet.shape[2])]}')
    print(f'RVD:\tmean:{getrvd.avg}; \tstd:{[np.around(dimet[:,4,i].std(),4) for i in range(dimet.shape[2])]}') 

    