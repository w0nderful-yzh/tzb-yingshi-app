# ============================================================
# 训练与验证主流程 (Solver_Sep)
# 第一阶段：对比学习训练两个模型分支
# 第二阶段：用训练好的模型提取特征 → XGBoost回归预测PHQ-8
# ============================================================

import pickle
import numpy as np
import torch
import torch.nn as nn
import os
import numpy as np

from ContrastiveLoss import CustomSCLLoss
from train_seprate.Train_One import Classifier_One
from train_seprate.Train_Two import Classifier_Two
from train_seprate.regression import log_regression_train, log_regression_val
from dataloaders.depression_dataset import get_dataloaders
from sklearn import metrics
from utils import *


def create_model(args):
    """创建两个模型分支"""
    model1 = Classifier_One(args)  # 分支1: 多线索直接融合
    model2 = Classifier_Two(args)  # 分支2: 多线索+多时间片段
    return model1, model2


class Solver_Sep(object):
    def __init__(self, args):
        super(Solver_Sep, self).__init__()
        self.args = args

        # ---------- 设置设备（GPU/CPU）----------
        if len(self.args.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.args.gpu_ids[0])
            self.device = torch.device('cuda:%d' % self.args.gpu_ids[0])
        else:
            self.device = torch.device('cpu')

        # ---------- 设置随机种子 ----------
        seed = self.args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # ---------- 初始化数据加载器 ----------
        if args.dataset == 'DAIC':
            dataloaders = get_dataloaders(args)
            self.train_dataloader = dataloaders['train']
            self.test_dataloader = dataloaders['validation']

        # ---------- 损失函数 ----------
        self.mse = nn.MSELoss().to(self.device)

        # ---------- 初始化两个模型 ----------
        self.model1, self.model2 = create_model(self.args)
        self.model1.to(self.device)
        self.model2.to(self.device)

        # ---------- 对比学习损失 ----------
        self.two_scl_loss = CustomSCLLoss(args).to(self.device)

        # ---------- 优化器和学习率调度 ----------
        self.optimizer1 = torch.optim.AdamW(
            self.model1.parameters(), lr=self.args.lr, eps=self.args.eps,
            weight_decay=self.args.weight_decay)
        self.scheduler1 = build_scheduler(self.args, self.optimizer1, len(self.train_dataloader))

        self.optimizer2 = torch.optim.AdamW(
            self.model2.parameters(), lr=self.args.lr, eps=self.args.eps,
            weight_decay=self.args.weight_decay)
        self.scheduler2 = build_scheduler(self.args, self.optimizer2, len(self.train_dataloader))

    def run(self):
        """主训练循环"""
        best_global_val_mae = 100
        best_result = [100, 100]  # [MAE, RMSE]
        best_epoch = 0
        patience = 50
        no_improve = 0

        # ---------- 仅推理模式 ----------
        if self.args.inference == '1':
            self.model1 = torch.load(os.path.join('checkpoint/', self.args.dataset, 'current_model1'), map_location=self.device, weights_only=False)
            self.model1.to(self.device)
            self.model2 = torch.load(os.path.join('checkpoint/', self.args.dataset, 'current_model2'), map_location=self.device, weights_only=False)
            self.model2.to(self.device)
            val_result, regressor, val_loss = validate(
                self.model1, self.model2, self.train_dataloader,
                self.test_dataloader, self.args, 276)
            return

        # ---------- 训练循环 ----------
        for epoch in range(self.args.start_epoch + 1, self.args.epochs + 1):
            print('********************' + str(epoch) + '********************')

            # 每个epoch: 先训练对比学习，再验证
            train_loss = self.train(epoch)
            val_result, regressor, val_loss = validate(
                self.model1, self.model2, self.train_dataloader,
                self.test_dataloader, self.args, epoch)
            print('val_mae={:.4f}'.format(val_result[0]))

            # 早停
            if best_global_val_mae > val_result[0]:
                best_global_val_mae = val_result[0]
                best_result = val_result
                best_epoch = epoch
                no_improve = 0
                save_path = os.path.join(self.args.output_path, self.args.dataset)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                torch.save(self.model1, os.path.join(save_path, 'current_model1'))
                torch.save(self.model2, os.path.join(save_path, 'current_model2'))
                print("will update:mae=" + str(best_result[0]) + ',rmse=' + str(best_result[1]) + '\n')
            else:
                no_improve += 1

            print("Current best [epoch {}] mae={:.4f},rmse={:.4f}".format(
                best_epoch, best_result[0], best_result[1]))

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}: no improvement for {patience} epochs")
                break

        print("Final best [epoch {}] val mae={:.4f},rmse={:.4f}".format(
            best_epoch, best_result[0], best_result[1]))

    def train(self, epoch):
        """
        对比学习训练阶段
        输入：4种线索特征（3D关键点、注视、姿态、AU）
        训练目标：通过对比学习让分支1和分支2的表征一致
        """
        all_loss = 0
        self.model1.train()
        self.model2.train()

        for i, (features, target, binary) in enumerate(self.train_dataloader):
            print("Training epoch \t{}: {}\\{}".format(epoch, i + 1, len(self.train_dataloader)), end='\r')

            # 将所有特征移到GPU
            for v in range(len(features)):
                features[v] = features[v].to(self.device)
            target = target.to(self.device).to(torch.float32)
            if self.args.dataset == 'DAIC':
                target = target[:, 2]  # DAIC-WOZ取第3列(PHQ-8总分)

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()

            # ----- 两分支前向传播 -----
            feature_tensor1 = self.model1(features, epoch, 'train')  # 分支1输出: [B, 256]
            feature_tensor2 = self.model2(features, epoch, 'train')  # 分支2输出: [B, 256]

            # ----- 对比学习损失（让两个分支输出相似）-----
            loss = self.two_scl_loss(feature_tensor1, feature_tensor2).to(torch.float32)
            all_loss = all_loss + loss

            # 反向传播
            loss.backward()
            self.optimizer1.step()
            self.scheduler1.step_update(epoch * len(self.train_dataloader) + i)
            self.optimizer2.step()
            self.scheduler2.step_update(epoch * len(self.train_dataloader) + i)

        loss = all_loss / len(self.train_dataloader)
        print('train loss: ' + str(loss))
        return loss


def validate(model1, model2, tra_dataloader, val_dataloader, args, epoch):
    """
    验证阶段：
    1. 用训练好的模型提取所有样本的特征
    2. 用训练集特征训练XGBoost回归器
    3. 在验证集上评估 MAE 和 RMSE
    """
    two_scl_loss = CustomSCLLoss(args).to(args.device)

    with torch.no_grad():
        model1.eval()
        model2.eval()

        rep_tra1_all = []  # 分支1的训练集特征
        rep_tra2_all = []  # 分支2的训练集特征
        rep_val1_all = []  # 分支1的验证集特征
        rep_val2_all = []  # 分支2的验证集特征
        audio_tra_all = []  # 训练集COVAREP统计特征
        audio_val_all = []  # 验证集COVAREP统计特征
        target_tra_all = []
        target_val_all = []

        # ---------- 提取训练集特征 ----------
        for i, (fea_tra_ori, target_tra, _) in enumerate(tra_dataloader):
            for v in range(len(fea_tra_ori)):
                fea_tra_ori[v] = fea_tra_ori[v].to(args.device)
            target_tra = target_tra.to(args.device).to(torch.float32)
            if args.dataset == 'DAIC':
                target_tra = target_tra[:, 2]

            rep_tra1 = model1(fea_tra_ori, epoch, 'val_trainset')
            rep_tra2 = model2(fea_tra_ori, epoch, 'val_trainset')

            rep_tra1_all.append(rep_tra1)
            rep_tra2_all.append(rep_tra2)
            # 音频统计特征在索引4，移到CPU避免影响后续
            audio_tra_all.append(fea_tra_ori[4].cpu())
            target_tra_all.append(target_tra)

        rep_tra1_all = torch.cat(rep_tra1_all, dim=0)
        rep_tra2_all = torch.cat(rep_tra2_all, dim=0)
        audio_tra_all = torch.cat(audio_tra_all, dim=0)
        target_tra_all = torch.cat(target_tra_all, dim=0)

        # ---------- 提取验证集特征 ----------
        all_loss = 0
        for i, (fea_val_ori, target_val, _) in enumerate(val_dataloader):
            for v in range(len(fea_val_ori)):
                fea_val_ori[v] = fea_val_ori[v].to(args.device)
            target_val = target_val.to(args.device).to(torch.float32)
            if args.dataset == 'DAIC':
                target_val = target_val[:, 2]

            rep_val1 = model1(fea_val_ori, epoch, 'val')
            rep_val2 = model2(fea_val_ori, epoch, 'val')

            loss = two_scl_loss(rep_val1, rep_val2).to(torch.float32)
            all_loss = all_loss + loss
            rep_val1_all.append(rep_val1)
            rep_val2_all.append(rep_val2)
            audio_val_all.append(fea_val_ori[4].cpu())
            target_val_all.append(target_val)

        rep_val1_all = torch.cat(rep_val1_all, dim=0)
        rep_val2_all = torch.cat(rep_val2_all, dim=0)
        audio_val_all = torch.cat(audio_val_all, dim=0)
        target_val_all = torch.cat(target_val_all, dim=0)
        loss = all_loss / len(val_dataloader)

        # ---------- 训练XGBoost并评估（视觉特征 + COVAREP统计特征拼接）----------
        regressor = log_regression_train(
            rep_tra1_all, rep_tra2_all, target_tra_all,
            rep_val1_all, rep_val2_all, target_val_all,
            audio_tra_all, audio_val_all, args)
        val_result, pred = log_regression_val(
            regressor, rep_val1_all, rep_val2_all, target_val_all,
            audio_val_all, args)

        return val_result, regressor, loss
