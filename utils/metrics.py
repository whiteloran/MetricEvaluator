import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from numpy.core.umath_tests import inner1d


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

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        smooth = 1e-5
        logits = torch.squeeze(logits)
        targets = torch.squeeze(targets)
        
        for class_index in range(targets.size()[0]):

            inter = (logits[class_index, :, :, :] * targets[class_index, :, :, :]).sum()
            union = torch.sum(logits[class_index, :, :, :]) + torch.sum(targets[ class_index, :, :, :])
            dice = (2. * inter + smooth) / (union + smooth)
            
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

    def update(self, logits, targets):
        self.value = IouAverage.get_ious(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)    

    @staticmethod
    def get_ious(logits, targets):
        ious = []
        smooth = 1e-5
        logits = torch.squeeze(logits)
        targets = torch.squeeze(targets)
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

    def update(self, logits, targets):
        self.value = HDAverage.get_hds(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_hds(logits, targets):
        hds = []
        dhs = []
        smooth = 1e-5
        logits = torch.squeeze(logits)
        targets = torch.squeeze(targets)
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

    def update(self, logits, targets):
        self.value = RMSEAverage.get_rmses(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
    

    @staticmethod
    def get_rmses(logits, targets):
        rmses = []
        mses = []
        smooth = 1e-5
        logits = torch.squeeze(logits)
        targets = torch.squeeze(targets)
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

    def update(self, logits, targets):
        self.value = RVDAverage.get_rvds(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_rvds(logits, targets):
        rvds = []
        logits = torch.squeeze(logits)
        targets = torch.squeeze(targets)

        smooth = 1e-5
        for class_index in range(targets.size()[0]):
            
            l_v = (logits[ class_index, :, :, :]).sum()
            t_v = (targets[ class_index, :, :, :]).sum()

            rvd = (abs(t_v - l_v)+ smooth) / (abs(l_v) + smooth)
            rvds.append(rvd.item())

        return np.asarray(rvds)












