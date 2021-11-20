# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

# @author ZhangMinghui,Southeast University
#
# Module Function: Calculate Loss Between Prediction
# and GroundTruth,then return loss for backpropagation.

"""
Module includes:
        __all__ = ['BinaryDiceLoss',
                   'DiceLoss',
                   'CombinedLoss',
                   'CE_Dynamicweighted',
                   'CombinedLoss_naive',
                   'FocalLoss']
"""


##############################################################################
# Different Loss Function
##############################################################################
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1e-5, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict)+torch.sum(target)+ self.smooth
        dice = num / den
        loss = 1 - dice
        return loss


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    @zmh Annotation
    target and prediction must Align the dimensions
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, sigmoid_normalization,weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.sigmoid_normalization = sigmoid_normalization
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0.0
        if not self.sigmoid_normalization:
            predict = torch.softmax(predict, dim=1)
        else:
            predict = torch.sigmoid(predict)
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss
        if self.ignore_index is not None:
            return total_loss / (target.shape[1]-1)
        else:
            return total_loss / target.shape[1]


class CombinedLoss(nn.Module):
    """
    A combination of dice  and cross entropy loss
    """

    def __init__(self,sigmoid_normalization,ignore_index=None,weight=None):
        super(CombinedLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.flag = True
        self.dice_loss = DiceLoss(sigmoid_normalization=sigmoid_normalization,ignore_index=self.ignore_index)
        self.crossentropy_loss1 = nn.CrossEntropyLoss(weight=self.weight)
        self.crossentropy_loss2 = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target,target_ce,flag,coeff_weight=None):
        """
        Forward pass
        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxCxHxW)
        :param coeff_weight: torch.tensor C*1
        :return: scalar
        """
        y_2 = self.dice_loss(input, target)
        if (flag == True):
            y_1 =  self.crossentropy_loss1(input,target_ce.long())
        else:
            for i in range(1,target.shape[1]):
                if (i == 1):
                    weight = torch.mul(target[:,i,...],coeff_weight[i-1])
                else:
                    weight += torch.mul(target[:,i,...],coeff_weight[i-1])
            y_1 = torch.mean(torch.mul(self.crossentropy_loss2(input,target_ce.long()),weight))
        return y_1 + y_2


class CE_Dynamicweighted(nn.Module):
    """
    A combination of dice  and cross entropy loss
    """

    def __init__(self,sigmoid_normalization,ignore_index=None,weight=None):
        super(CE_Dynamicweighted, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.flag = True
        self.crossentropy_loss1 = nn.CrossEntropyLoss(weight=self.weight)
        self.crossentropy_loss2 = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target,target_ce,flag,coeff_weight=None):
        """
        Forward pass
        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxCxHxW)
        :param coeff_weight: torch.tensor C*1
        :return: scalar
        """
        if (flag == True):
            y_1 =  self.crossentropy_loss1(input,target_ce.long())
        else:
            print(coeff_weight)
            for i in range(1,target.shape[1]):
                if (i == 1):
                    weight = torch.mul(target[:,i,...],coeff_weight[i-1])
                else:
                    weight += torch.mul(target[:,i,...],coeff_weight[i-1])
            y_1 = torch.mean(torch.mul(self.crossentropy_loss2(input,target_ce.long()),weight))
        return y_1


class CombinedLoss_naive(nn.Module):
    """
    A combination of dice  and cross entropy loss
    """

    def __init__(self,sigmoid_normalization,ignore_index=None,weight=None):
        super(CombinedLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.flag = True
        self.dice_loss = DiceLoss(sigmoid_normalization=sigmoid_normalization,ignore_index=self.ignore_index)
        self.crossentropy_loss = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, input, target,target_ce):
        """
        Forward pass
        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxCxHxW)
        :param coeff_weight: torch.tensor C*1
        :return: scalar
        """
        y_2 = self.dice_loss(input, target)
        y_1 = self.crossentropy_loss(input,target_ce)
        return y_1 + y_2       


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection
    """

    def __init__(self, gamma=2, alpha=None, size_average=True):

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """Forward pass
        :param input: shape = NxCxHxW
        :type input: torch.tensor
        :param target: shape = NxHxW
        :type target: torch.tensor
        :return: loss value
        :rtype: torch.tensor
        """

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = (logpt.data.exp()).clone().detach().requires_grad_(True)

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at.clone().detach().requires_grad_(True)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
