
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
from numpy.core.umath_tests import inner1d


def ring2cycle(mask_ring):
    mask_ring = torch.squeeze(mask_ring)
    [ h, a, b] = np.shape(mask_ring)  # (80,80, patch_size)
    new_mask = np.zeros_like(mask_ring)


    for t in range(h):
        img_tmp = mask_ring[ t,:,:]
        M = np.where(img_tmp != 0)
        c = 0
        nonzeros_num = np.zeros(len(M[0]))
        for i in range(a):
            li = []
            for j in range(b):
                if img_tmp[ i, j] != 0:
                    nonzeros_num[c] = img_tmp[ i, j]
                    v = nonzeros_num[c]
                    c = c + 1
                    li.append(j)
                    # print("c:", c, "v:", v, "coor:", [i, j])
            myli = np.array(li)
            # print("myli:", myli)
            if len(myli) != 0:
                first = myli[0]
                last = myli[len(myli) - 1]
                for j in range(b):
                    if (j > first and j < last):
                        img_tmp[i, j] = 1
        new_mask[t,:,:] = img_tmp

    new_mask = torch.from_numpy(new_mask)
    # new_mask= torch.unsqueeze(new_mask, 0)
    return new_mask



def dim3_4(logits,targets,class_num):
    if class_num != 1:
        if len(logits.shape) <= 3:
            logits = logits.unsqueeze(1)
        if len(targets.shape) <= 3:
            targets = targets.unsqueeze(1)
    elif class_num == 1:
        if len(logits.shape) <= 3:
            logits = logits.unsqueeze(0)
        if len(targets.shape) <= 3:
            targets = targets.unsqueeze(0)
    return logits, targets

def dim2_3(logits,targets):
    if len(logits.shape) <= 2:
        logits = logits.unsqueeze(0)
    if len(targets.shape) <= 2:
        targets = targets.unsqueeze(0)
    return logits, targets

def dim3_4_1(tens):
    return tens.unsqueeze(1)

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)

class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0
        self.std = 0

    def update(self, logits, targets, class_num):
        self.value = DiceAverage.get_dices(self,logits=logits, targets=targets,class_num = class_num)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(self, logits, targets, class_num):
        dices = []
        smooth = 1e-5
        logits = torch.squeeze(logits)
        targets = torch.squeeze(targets)
        
        logits,targets = dim2_3(logits,targets)
        logits,targets = dim3_4(logits,targets,class_num)

        #print(logits.shape)
        #print(targets.shape)
        
        for class_index in range(targets.size()[0]):
  
            inter = (logits[class_index, :, :, :] * targets[class_index, :, :, :]).sum()
            union = torch.sum(logits[class_index, :, :, :]) + torch.sum(targets[ class_index, :, :, :])
            
            dice = 2. * (inter + smooth) / (union + smooth)

            dices.append(dice.item())
        return np.asarray(dices)


class IouAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets, class_num):
        self.value = IouAverage.get_ious(logits, targets, class_num)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)


    @staticmethod
    def get_ious(logits, targets, class_num):
        ious = []
        smooth = 1e-5
        logits = torch.squeeze(logits)
        targets = torch.squeeze(targets)

        logits,targets = dim2_3(logits,targets)
        logits,targets = dim3_4(logits, targets, class_num)
    
        for class_index in range(targets.size()[0]):

            inter = torch.sum(logits[class_index, :, :, :] * targets[class_index, :, :, :])
            union = torch.sum(logits[ class_index, :, :, :]) + torch.sum(targets[ class_index, :, :, :]) -inter
            iou = ( inter + smooth) / (union + smooth)
            ious.append(iou.item())
        return np.asarray(ious)


class HDAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets, class_num):
        self.value = HDAverage.get_hds(logits, targets, class_num)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_hds(logits, targets, class_num):
        hds = []
        dhs = []
        smooth = 1e-5
        logits = torch.squeeze(logits)
        targets = torch.squeeze(targets)

        logits,targets = dim2_3(logits,targets)
        logits,targets = dim3_4(logits, targets, class_num)

        for class_index in range(targets.size()[0]):
    
            for slice_index in range(targets.size()[1]):
                D_mat = np.sqrt(inner1d(logits[ class_index, slice_index,:, :], logits[ class_index, slice_index, :, :])[np.newaxis].T
                                + inner1d(targets[ class_index, slice_index, :, :], targets[ class_index, slice_index, :, :])
                                - 2 * (np.dot(logits[class_index, slice_index, :, :], targets[ class_index, slice_index,:, :].T)))

                D_mat = D_mat.numpy()
                dh = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))
                dhs.append(dh)
            dh_avg = np.mean(dhs)
            hds.append(dh_avg.item())
        return np.asarray(hds)



class RMSEAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets, class_num):
        self.value = RMSEAverage.get_rmses(logits, targets, class_num)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)


    @staticmethod
    def get_rmses(logits, targets, class_num):
        rmses = []
        mses = []
        smooth = 1e-5
        logits = torch.squeeze(logits)
        targets = torch.squeeze(targets)

        logits,targets = dim2_3(logits, targets)
        logits,targets = dim3_4(logits, targets, class_num)

        for class_index in range(targets.size()[0]):
        
            for slice_index in range(targets.size()[1]):
                mse = ((targets[ class_index, slice_index, :, :] - logits[ class_index, slice_index, :, :]) ** 2).sum() / float(
                    targets.shape[2] * targets.shape[3])
                mses.append(mse)

            mse_avg = np.mean(mses)
            rmse = np.sqrt(mse_avg)
            rmses.append(rmse.item())
        return np.asarray(rmses)


class RVDAverage(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets, class_num):
        self.value = RVDAverage.get_rvds(logits, targets, class_num)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)

    @staticmethod
    def get_rvds(logits, targets, class_num):
        rvds = []
        smooth = 1e-5

        logits = torch.squeeze(logits)
        targets = torch.squeeze(targets)

        logits,targets = dim2_3(logits, targets)
        logits,targets = dim3_4(logits, targets, class_num)
        
        for class_index in range(targets.size()[0]):
            l_v = (logits[ class_index, :, :, :]).sum()
            t_v = (targets[ class_index, :, :, :]).sum()

            rvd = abs((abs(t_v - l_v)+ smooth) / (abs(l_v) + smooth))
            rvds.append(rvd.item())

        return np.asarray(rvds)












