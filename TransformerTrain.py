from Dataset.DataOperate import MySet,MySet_npy, get_data_list
from Utils.Utils import DiceLoss,metrics
from Utils.focal_loss import focal_loss
from Model.DMFNet_16x import My_DFMNet,New_DMFNet,New_DMFNet_NewRecons,New_Att_DMFNet,DMF_FPN,Deep_Attentive_DMFNet
from Model.transformer.vit_seg_modeling import VisionTransformer
import time
import os
import numpy as np
import cv2
import SimpleITK as itk
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

def get_k_fold_data(k, i, X):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = len(X) // k  # 每份的个数:数据总条数/折数（组数）
    train_list, val_list = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        part = X[idx]
        if j == i:  ###第i折作valid
            val_list = part
        elif train_list is None:
            train_list = part
        else:
            train_list = train_list + part  # dim=0增加行数，竖着连接
    return train_list, val_list
def k_fold(k, X, train_log, val_seg_log, val_cls_log, val_path, train_path,tensorboard_dir):
    for i in range(k):
        writer = SummaryWriter(tensorboard_dir+'/fold%d'%i)
        total = 0
        positive = 0
        torch.cuda.empty_cache()
        net =VisionTransformer()
        net = torch.nn.DataParallel(net, device_ids).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4,weight_decay=0.05)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.75)
        train_list, val_lsit = get_k_fold_data(k, i, X)  # 获取k折交叉验证的训练和验证数据
        train_set = MySet_npy(train_list)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_set = MySet_npy(val_lsit)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        for batch_idx, (image, mask, label, name) in enumerate(train_loader):
            label = Variable(label[0].cuda())
            l_class = label.data.cpu().numpy()[0]
            if l_class == 1:
                positive += 1
            total += 1
        positive_weight = positive / total
        criterion_mse = focal_loss(alpha=positive_weight)
        criterion_bce = torch.nn.BCELoss()
        l2_loss = torch.nn.MSELoss()
        criterion_dice = DiceLoss()
        ### 每份数据进行训练,体现步骤三####
        train(i, net, train_loader, val_loader, optimizer, criterion_bce, criterion_dice, criterion_mse,l2_loss, scheduler,
              train_log, val_seg_log, val_cls_log, val_path,train_path,writer)


def train(k, net, train_loader, val_loader, optimizer, criterion_bce, criterion_dice, criterion_mse,l2_loss, scheduler,train_log,val_seg_log,val_cls_log, val_path,train_path,writer):
    for epoch in range(0, 80):
        best_dice = 0.
        best_recall = 0.
        epoch_start_time = time.time()
        print("Epoch: {}".format(epoch))
        epoch_loss = 0.
        total_num = 1
        true_num = 0
        p_num=0
        p_thresh=0
        p_min = 1
        for batch_idx, (image,mask,label,name) in enumerate(train_loader):
            print(name)
            start_time = time.time()
            image = Variable(image.cuda())
            mask = Variable(mask.cuda())
            label = Variable(label.cuda())
            
            output= net(image)

            output = F.sigmoid(output)
            #recons1_l2loss = l2_loss(recons,image)
            loss0_bce = criterion_bce(output[0], mask)
            loss0_dice = criterion_dice(output, mask)

            #loss = 0.8 * loss0_dice + 4.0 * cls_loss + 0.3 * loss0_bce + 0.3 * loss1_dice + 0.2 * loss1_bce
            loss = 1.0 * loss0_dice + 0.5 * loss0_bce
            #loss = 1.0 * loss0_dice + 4.0 * cls_loss + 0.5 * loss0_bce

            epoch_loss += loss.item()
            print('Fold: {} | Epoch: {} | Batch: {} | Patient: {:20}----->Train loss: {:4f} | Cost Time: {}\n'.format(k, epoch, batch_idx,name[0],loss.item(),time.time() - start_time))
            open(train_log, 'a').write('Fold: {} | Epoch: {} | Batch: {} | Patient: {:30}----->Train loss: {:4f} | Cost Time: {}\n'.format(k, epoch, batch_idx,name[0],loss.item(),time.time() - start_time))
            loss.backward()
            optimizer.step()
        print("Fold: {} | Epoch: {} | Loss: {} | Acc: {} | Time: {}\n".format(k, epoch, epoch_loss / (batch_idx + 1),true_num/total_num,time.time() - epoch_start_time))
        open(train_log, 'a').write("Fold: {} | Epoch: {} | Loss: {} | Acc: {} | Time: {}\n".format(k, epoch, epoch_loss / (batch_idx + 1),true_num/total_num,time.time() - epoch_start_time))
        writer.add_scalar('train_loss', epoch_loss / (batch_idx + 1), global_step=epoch)
        writer.add_scalar('train_acc', true_num/total_num, global_step=epoch)
        scheduler.step()
        # begin to eval
        net.eval()
        total_num = 1
        true_num = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        true_num1 = 0
        tp1 = 0
        fp1 = 0
        tn1 = 0
        fn1 = 0
        dice = 0.
        jaccard = 0.
        seg_precision = 0.
        seg_sensitivity = 0.
        seg_specificity = 0.
        hd95 = 0.
        with torch.no_grad():
            for batch_idx, (image, mask, label, name) in enumerate(val_loader):
                image = Variable(image.cuda())
                mask = Variable(mask.cuda())
                label = Variable(label.cuda())
                #val_predict1, val_predict, cls_out = net(image)
                val_predict = net(image)

                val_predict = F.sigmoid(val_predict)
                mask = mask.data.cpu().numpy()
                val_predict = val_predict.data.cpu().numpy()
                image = image.data.cpu().numpy()
                save_predict = np.squeeze(val_predict)
                val_seg = np.zeros_like(save_predict)
                val_seg[save_predict > 0.4] = 1

                seg_output = itk.GetImageFromArray(val_seg)
                posibility_output = itk.GetImageFromArray(save_predict)
                image_output = itk.GetImageFromArray(image)
                mask_output = itk.GetImageFromArray(mask)

                save_dir = val_path + '/' + "fold_%d" % k + '/' + name[0]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_seg_dir = save_dir + "/%s_seg.nii" % epoch
                itk.WriteImage(seg_output, save_seg_dir)
                save_posibility_dir = save_dir + "/%s_posibility.nii" % epoch
                itk.WriteImage(posibility_output, save_posibility_dir)
                save_mask_dir = save_dir + "/label.nii"
                itk.WriteImage(mask_output, save_mask_dir)
                save_image_dir = save_dir + "/image.nii"
                itk.WriteImage(image_output, save_image_dir)
                print("val_seg.shape")
                print(val_seg.shape)
                print("mask.shape")
                print(mask[0][0].shape)
                dice_tmp, hd_tmp, jaccard_tmp, seg_precision_tmp, seg_sensitivity_tmp, seg_specificity_tmp = metrics(
                    val_seg, mask[0][0])
                open(val_seg_log, 'a').write(
                    "Fold: {} | Epoch: {} | Patient: {:20} | Dice: {:5f} | HD95: {:5f} | Jaccard: {:5f} | Seg_precision: {:5f} | Seg_sensitivity: {:5f} | Seg_specificity: {:5f}\n".format(
                        k, epoch, name[0], dice_tmp, hd_tmp, jaccard_tmp, seg_precision_tmp, seg_sensitivity_tmp,
                        seg_specificity_tmp))
                dice = dice + dice_tmp
                jaccard = jaccard + jaccard_tmp
                seg_precision = seg_precision + seg_precision_tmp
                seg_sensitivity = seg_sensitivity + seg_sensitivity_tmp
                hd95 = hd_tmp + hd95
                seg_specificity = seg_specificity_tmp + seg_specificity




        #epoch cls metrics

        # epoch seg metrics
        dice = dice / (1 + batch_idx)
        jaccard = jaccard / (1 + batch_idx)
        seg_precision = seg_precision / (1 + batch_idx)
        seg_sensitivity = seg_sensitivity / (1 + batch_idx)
        hd95 = hd95 / (1 + batch_idx)
        seg_specificity = seg_specificity / (1 + batch_idx)

        print("Fold {} Epoch {} dice: {} hd95: {} jaccard: {} seg_precision: {} seg_sensitivity: {} seg_specificity: {}\n".format(
                k, epoch, dice, hd95, jaccard, seg_precision, seg_sensitivity, seg_specificity))
        open(val_seg_log, 'a').write(
            "Fold {} Epoch {} dice: {} hd95: {} jaccard: {} seg_precision: {} seg_sensitivity: {} seg_specificity: {}\n".format(
                k, epoch, dice, hd95, jaccard, seg_precision, seg_sensitivity, seg_specificity))

        writer.add_scalar('dice', dice, global_step=epoch)
        writer.add_scalar('jaccard', jaccard, global_step=epoch)
        writer.add_scalar('hd95', hd95, global_step=epoch)
        writer.add_scalar('val_seg_precision', seg_precision, global_step=epoch)
        writer.add_scalar('val_seg_sensitivity', seg_sensitivity, global_step=epoch)
        writer.add_scalar('val_seg_specificity', seg_specificity, global_step=epoch)


        if dice > best_dice:
            best_dice = dice
            torch.save(net.state_dict(), check_point + '/Best_Dice.pth')

        torch.save(net.state_dict(), check_point + '/model_fold{}_epoch{}.pth'.format(k, epoch))


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device_ids = [0]
train_list = get_data_list("/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/miccai_data_64*256*256_patch", ratio=0.8)
train_log = './Transformer_0325/train/trainLog.txt'
val_seg_log = './Transformer_0325/valid/valsegLog.txt'
val_cls_log = './Transformer_0325/valid/valclsLog.txt'
val_path = './Transformer_0325/valid'
train_path = './Transformer_0325/train'
check_point = './Transformer_0325/checkpoints'
tensorboard_dir = './Transformer_0325/tensorboard_log'

if not os.path.exists(check_point):
    os.makedirs(check_point)
if not os.path.exists(val_path):
    os.makedirs(val_path)
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
information_line = '=' * 20 + ' Transformer_0325-TRAIN ' + '=' * 20 + '\n'
open(train_log, 'w').write(information_line)
information_line = '=' * 20 + ' Transformer_0325-VAL-SEG ' + '=' * 20 + '\n'
open(val_seg_log, 'w').write(information_line)
information_line = '=' * 20 + ' Transformer_0325-VAL-CLS ' + '=' * 20 + '\n'
open(val_cls_log, 'w').write(information_line)
K = 5
k_fold(K, train_list, train_log,val_seg_log, val_cls_log, val_path,train_path,tensorboard_dir)

