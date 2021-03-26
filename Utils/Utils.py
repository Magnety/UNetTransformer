import torch
from torch import nn
from .eva import dc, hd95, asd, obj_asd,hd,sensitivity,precision,specificity,jc
import numpy


class DiceLoss(nn.Module):
    """
    define the dice loss
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1.
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)

        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
def dice_ratio(seg, gt):
    """
    define the dice ratio
    :param seg: segmentation result
    :param gt: ground truth
    :return:
    """
    seg = seg.flatten()
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0

    gt = gt.flatten()
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0

    tp = (seg * gt).sum()

    dice = 2*float(tp)/float(gt.sum() + seg.sum()+0.000001)

    return dice
def jaccard_ratio(seg, gt):
    """
    define the jaccard ratio
    :param seg: segmentation result
    :param gt: ground truth
    :return:
    """
    seg = seg.flatten()
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0

    gt = gt.flatten()
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0

    tp = (seg * gt).sum() #jiao
    bing = gt.sum()+seg.sum()-tp


    jaccard = float(tp)/float(bing+0.000001)

    return jaccard
def precision_ratio(seg, gt):
    """
    define the jaccard ratio
    :param seg: segmentation result
    :param gt: ground truth
    :return:
    """
    seg = seg.flatten()
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0

    gt = gt.flatten()
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0

    tp = (seg * gt).sum() #jiao
    fp = seg.sum()-tp


    precision = float(tp)/float(fp+0.000001)

    return precision
def recall_ratio(seg, gt):
    """
    define the jaccard ratio
    :param seg: segmentation result
    :param gt: ground truth
    :return:
    """
    seg = seg.flatten()
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0

    gt = gt.flatten()
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0

    tp = (seg * gt).sum() #jiao
    fn = gt.sum()-tp


    recall = float(tp)/float(fn+0.000001)

    return recall
def allmetrics_ratio(seg, gt):
    """
    define the jaccard ratio
    :param seg: segmentation result
    :param gt: ground truth
    :return:
    """
    seg = seg.flatten()
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0

    gt = gt.flatten()
    sum = gt.sum()
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0

    tp = (seg * gt).sum() #jiao
    fp = seg.sum()-tp
    fn = gt.sum()-tp
    tn = sum - fp-tp-fn

    dice = 2*float(tp)/float(gt.sum() + seg.sum()+0.000001)
    jaccard= float(tp)/float(fp+fn+tp+0.000001)
    precision = float(tp) / float(fp+tp+ 0.000001)
    recall = float(tp)/float(fn+tp+0.000001) #= sensitivity
    FNs = float(fp)/float(tn+fp+0.000001)

    return dice,jaccard,precision,recall,FNs
def test_allmetrics_ratio(seg, gt):
    """
    define the jaccard ratio
    :param seg: segmentation result
    :param gt: ground truth
    :return:
    """
    smooth = 1.
    iflat = seg.contiguous().view(-1)
    tflat = gt.contiguous().view(-1)
    tp = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    fp = A_sum-tp
    fn = B_sum-tp


    dice = 2*float(tp+ smooth)/float(A_sum+B_sum+smooth)
    jaccard= float(tp+ smooth)/float(fp+fn+tp+smooth)
    precision = float(tp+ smooth) / float(fp+tp+ smooth)
    recall = float(tp+ smooth)/float(fn+tp+ smooth) #= sensitivity


    return dice,jaccard,precision,recall
def metrics(seg, gt):
    """
    define the jaccard ratio
    :param seg: segmentation result
    :param gt: ground truth
    :return:
    """
    #seg = seg.data.cpu().numpy()
    #gt = gt.data.cpu().numpy()


    return dc(seg,gt),hd95(seg,gt), jc(seg,gt), precision(seg,gt), sensitivity(seg,gt),specificity(seg,gt)