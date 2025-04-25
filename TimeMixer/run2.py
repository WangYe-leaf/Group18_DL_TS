import argparse
import torch
import pandas as pd
from exp.exp_basic import Exp_Basic
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Volume Robustness Evaluation')

   
    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly, ms:milliseconds], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                        help='whether to use future_temporal_feature; True 1 False 0')

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--comment', type=str, default='none', help='com')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
        exp = Exp(args)

        import pandas as pd
        from torch.utils.data import Subset, DataLoader

        fractions = [0.1, 0.2, 0.5, 1.0]
        volume_results = {}

        for frac in fractions:
            print(f"\n🔍 Training with {int(frac * 100)}% of training data...")

            # 获取完整数据
            train_data, train_loader = exp._get_data(flag='train')
            vali_data, vali_loader = exp._get_data(flag='val')
            test_data, test_loader = exp._get_data(flag='test')

            # 截断训练集
            subset_size = int(len(train_data) * frac)
            train_data = Subset(train_data, range(subset_size))
            train_loader = DataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True
            )

            # 设置唯一 setting 名称
            setting = f"{args.model_id}_volume{int(frac*100)}"
            exp.model = exp._build_model().to(exp.device)  # 重新构建模型
            model = exp.train(setting)

            # 测试结果
            print(">>> Testing ...")
            preds, trues, masks = [], [], []
            exp.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(exp.device)
                    batch_x_mark = batch_x_mark.float().to(exp.device)
                    B, T, N = batch_x.shape
                    mask = torch.rand((B, T, N)).to(exp.device)
                    mask[mask <= args.mask_rate] = 0
                    mask[mask > args.mask_rate] = 1
                    inp = batch_x.masked_fill(mask == 0, 0)
                    outputs = exp.model(inp, batch_x_mark, None, None, mask)
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]
                    mask = mask[:, :, f_dim:]
                    preds.append(outputs.detach().cpu().numpy())
                    trues.append(batch_x.detach().cpu().numpy())
                    masks.append(mask.detach().cpu().numpy())

            preds = np.concatenate(preds, 0)
            trues = np.concatenate(trues, 0)
            masks = np.concatenate(masks, 0)

            from utils.metrics import metric
            mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
            volume_results[frac] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'mspe': mspe
            }

            print(f"📊 Volume {frac:.2f} -> MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.6f}, MSPE: {mspe:.6f}")

    # 保存 CSV
    result_path = os.path.join(args.checkpoints, f"{args.model_id}_volume_robustness.csv")
    pd.DataFrame(volume_results).T.to_csv(result_path)
    print(f"\n✅ Volume robustness results saved to: {result_path}")