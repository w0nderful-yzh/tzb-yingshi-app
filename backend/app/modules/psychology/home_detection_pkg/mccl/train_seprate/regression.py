# ============================================================
# XGBoost 回归器
# 自动检测GPU/CPU
# ============================================================

import numpy as np
import xgboost as xgb
import torch
from sklearn import metrics

# CuPy可选导入
try:
    import cupy as cp
    _HAS_CUPY = True
except:
    _HAS_CUPY = False


def log_regression_train(z_tra1, z_tra2, target_tra, z_val1, z_val2, target_val,
                         audio_tra, audio_val, args):
    """
    训练XGBoost回归器。
    特征 = 两个分支拼接(256维) + COVAREP统计特征(148维) = 404维
    """
    # 拼接两个分支的特征（128+128=256维）+ 音频统计(148维) = 404维
    z_train = torch.cat((z_tra1, z_tra2), dim=1).detach().cpu().numpy()
    z_train = np.concatenate([z_train, audio_tra.numpy()], axis=1)
    target_tra = target_tra.cpu().numpy()
    target_val = target_val.cpu().numpy()
    z_val = torch.cat((z_val1, z_val2), dim=1).detach().cpu().numpy()
    z_val = np.concatenate([z_val, audio_val.numpy()], axis=1)

    # XGBoost回归器配置
    xgb_device = args.device if _HAS_CUPY and torch.cuda.is_available() else 'cpu'
    regressor = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.xg_lr,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        max_depth=args.max_depth,
        gamma=args.gamma,
        n_jobs=args.n_jobs,
        device=xgb_device,
        tree_method=args.tree_method,
    )

    # 训练
    if _HAS_CUPY and torch.cuda.is_available():
        with cp.cuda.Device(args.gpu_ids[0]):
            regressor.fit(cp.array(z_train), cp.array(target_tra))
    else:
        regressor.fit(z_train, target_tra)

    return regressor


def log_regression_val(regressor, z_val1, z_val2, target_val, audio_val, args):

    target_val = target_val.cpu().numpy()
    z_val = torch.cat((z_val1, z_val2), dim=1).detach().cpu().numpy()
    z_val = np.concatenate([z_val, audio_val.numpy()], axis=1)

    # 预测
    if _HAS_CUPY and torch.cuda.is_available():
        with cp.cuda.Device(args.gpu_ids[0]):
            z_val = cp.array(z_val)
            y_pred = regressor.predict(z_val)
    else:
        y_pred = regressor.predict(z_val)

    # 计算指标
    mae = metrics.mean_absolute_error(target_val, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(target_val, y_pred))
    print("In log_regression_val:mae=" + str(mae) + ',rmse=' + str(rmse))

    return [mae, rmse], y_pred
