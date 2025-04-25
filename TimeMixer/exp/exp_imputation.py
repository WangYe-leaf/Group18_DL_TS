
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0
                mask[mask > self.args.mask_rate] = 1
                inp = batch_x.masked_fill(mask == 0, 0)
                outputs = self.model(inp, batch_x_mark, None, None, mask)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x.detach()
                mask = mask.detach()
                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, portion=1.0):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0
                mask[mask > self.args.mask_rate] = 1
                inp = batch_x.masked_fill(mask == 0, 0)
                outputs = self.model(inp, batch_x_mark, None, None, mask)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]
                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def vali_multi_mask_detailed(self, vali_data, vali_loader, criterion, mask_rates=[0.1, 0.2, 0.3, 0.4]):
        self.model.eval()
        results = {}
        detailed_metrics = {}
        with torch.no_grad():
            for mask_rate in mask_rates:
                print(f"Testing with mask rate: {mask_rate}")
                preds, trues, masks = [], [], []
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    B, T, N = batch_x.shape
                    mask = torch.rand((B, T, N)).to(self.device)
                    mask[mask <= mask_rate] = 0
                    mask[mask > mask_rate] = 1
                    inp = batch_x.masked_fill(mask == 0, 0)

                    outputs = self.model(inp, batch_x_mark, None, None, mask)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]
                    mask = mask[:, :, f_dim:]

                    outputs = outputs.detach().cpu().numpy()
                    true = batch_x.detach().cpu().numpy()
                    mask = mask.detach().cpu().numpy()

                    preds.append(outputs)
                    trues.append(true)
                    masks.append(mask)

                preds = np.concatenate(preds, 0)
                trues = np.concatenate(trues, 0)
                masks = np.concatenate(masks, 0)

                mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
                results[mask_rate] = mse
                detailed_metrics[mask_rate] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'mspe': mspe
                }
                print(f"Mask Rate {mask_rate:.1f}: MSE = {mse:.6f}, MAE = {mae:.6f}, RMSE = {rmse:.6f}, MAPE = {mape:.6f}, MSPE = {mspe:.6f}")
        self.model.train()
        return detailed_metrics
    
    def vali_multi_noise(self, vali_data, vali_loader, criterion, noise_levels=[0.01, 0.05, 0.1, 0.2]):
        self.model.eval()
        results = {}
        with torch.no_grad():
            for noise_std in noise_levels:
                print(f"Testing with noise std: {noise_std}")
                all_pred = []
                all_true = []
                all_mask = []

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    # 添加噪声
                    noise = torch.randn_like(batch_x) * noise_std
                    noisy_x = batch_x + noise

                    # 随机 mask
                    B, T, N = batch_x.shape
                    mask = torch.rand((B, T, N)).to(self.device)
                    mask[mask <= self.args.mask_rate] = 0
                    mask[mask > self.args.mask_rate] = 1
                    inp = noisy_x.masked_fill(mask == 0, 0)

                    outputs = self.model(inp, batch_x_mark, None, None, mask)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]
                    mask = mask[:, :, f_dim:]

                    all_pred.append(outputs.detach().cpu().numpy())
                    all_true.append(batch_x.detach().cpu().numpy())
                    all_mask.append(mask.detach().cpu().numpy())

                pred = np.concatenate(all_pred, axis=0)
                true = np.concatenate(all_true, axis=0)
                mask = np.concatenate(all_mask, axis=0)

                mae, mse, rmse, mape, mspe = metric(pred[mask == 0], true[mask == 0])
                results[noise_std] = (mae, mse, rmse, mape, mspe)

                print(f"Noise STD {noise_std:.2f} -> MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.6f}, MSPE: {mspe:.6f}")
        self.model.train()
        return results
    
    def evaluate_data_volume(self, setting_prefix='volume_eval', fractions=[0.1, 0.2, 0.5, 1.0]):
        """
        评估不同训练集大小对模型性能的影响
        """
        criterion = self._select_criterion()
        results = {}

        for frac in fractions:
            print(f"\n🔍 Training with {int(frac * 100)}% of training data...")

            setting = f"{setting_prefix}_{int(frac * 100)}"
            train_data, train_loader = self._get_data(flag='train', fraction=frac)
            val_data, val_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

            # 重建模型
            self.model = self._build_model().to(self.device)

            # 模型训练
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            self.train(setting)

            # 测试
            preds = []
            trues = []
            masks = []
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    B, T, N = batch_x.shape
                    mask = torch.rand((B, T, N)).to(self.device)
                    mask[mask <= self.args.mask_rate] = 0
                    mask[mask > self.args.mask_rate] = 1
                    inp = batch_x.masked_fill(mask == 0, 0)

                    outputs = self.model(inp, batch_x_mark, None, None, mask)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]
                    mask = mask[:, :, f_dim:]

                    preds.append(outputs.detach().cpu().numpy())
                    trues.append(batch_x.detach().cpu().numpy())
                    masks.append(mask.detach().cpu().numpy())

            preds = np.concatenate(preds, 0)
            trues = np.concatenate(trues, 0)
            masks = np.concatenate(masks, 0)

            # 评估
            mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
            results[frac] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'mspe': mspe,
            }

            print(f"✅ Data Fraction {frac:.2f} -> MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.6f}, MSPE: {mspe:.6f}")

        # 总结输出
        print("\n📊 Summary of data volume robustness evaluation:")
        for frac, metrics in results.items():
            print(f" - {int(frac * 100)}% data -> MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}, MAPE: {metrics['mape']:.6f}, MSPE: {metrics['mspe']:.6f}")
        
        return results

        

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                if i % 20 == 0:
                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                             pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        print('mse:{}, mae:{},rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return
    
   