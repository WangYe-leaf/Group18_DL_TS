from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv, visual_weights
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

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
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali_multi_noise(self, vali_data, vali_loader, criterion, noise_levels=[0.0, 0.01, 0.05, 0.1, 0.2]):
        self.model.eval()
        results = {}
        with torch.no_grad():
            for noise_level in noise_levels:
                print(f"Testing with input noise std: {noise_level}")
                total_loss = []
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    if self.args.down_sampling_layers == 0:
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    else:
                        dec_inp = None

                    noise = torch.randn_like(batch_x) * noise_level
                    noisy_x = batch_x + noise

                    if self.args.output_attention:
                        outputs = self.model(noisy_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(noisy_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    pred = outputs.detach()
                    true = batch_y.detach()
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())

                avg_loss = np.mean(total_loss)
                results[noise_level] = avg_loss
                print(f"Noise {noise_level:.2f} | Val Loss: {avg_loss:.6f}")

        self.model.train()
        return results

    # train(), test(), vali() remain unchanged

