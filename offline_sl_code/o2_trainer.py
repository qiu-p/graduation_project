import os
import argparse
import torch
import numpy as np
import datetime
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
# gp sr
from gplearn.genetic import SymbolicRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

import sys

sys.path.insert(0, os.getcwd())

from utils.mlp import MLP
from o1_dataloader import ScoreDataLoader, AreaDelayDataLoader
from utils.resnet import DeepQPolicy, BasicBlock, MBRLPPAModel
from utils.attention import AttentionValueNet


from ipdb import set_trace

class Trainer():
    def __init__(
        self,
        data_loader,
        policy,
        tb_logger,
        train_type="reward_regression",
        evaluate_freq=2,
        epochs=2,
        lr=1e-3,
        optim_class='Adam',
        device='cuda:0',
        save_pkl_path="./saved_models",
        bit_width="16_bits",
        # lr decay
        lr_decay=False,
        lr_decay_step=5,
        lr_decay_rate=0.96,
        # sr kwargs
        y_type="normalize_area",
        sr_epochs=1000,
        ml_model_type="symbolic_regression", # [symbolic_regression, lightgbm]
        split_ratio=0,
        random_state=0,
        function_set="v1", # [v1 +-*/, v2 +-*/log sqrt]
        const_bound=1
    ):
        self.data_loader = data_loader
        self.policy = policy
        self.tb_logger = tb_logger

        self.evaluate_freq = evaluate_freq
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.train_type = train_type
        self.save_pkl_path = save_pkl_path
        self.bit_width = bit_width
        self.y_type = y_type
        self.sr_epochs = sr_epochs
        self.ml_model_type = ml_model_type
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.function_set = function_set
        self.const_bound = const_bound

        if policy is not None:
            # optimizer
            self.optim_class = optim_class
            if isinstance(optim_class, str):
                optim_class = eval('optim.'+optim_class)
                self.optim_class = optim_class

            self.policy_optimizer = optim_class(
                self.policy.parameters(),
                lr=self.lr
            )
            # lr scheduler
            self.lr_decay = lr_decay
            self.lr_decay_step = lr_decay_step
            self.lr_decay_rate = lr_decay_rate
            if self.lr_decay:
                self.policy_lr_scheduler = lr_scheduler.StepLR(
                    self.policy_optimizer,
                    self.lr_decay_step,
                    gamma=self.lr_decay_rate
                )

            # loss fn
            self.loss = torch.nn.MSELoss()
        else:
            if self.ml_model_type == "symbolic_regression":
                if self.function_set == "v1":
                    FunctionSet = ('add', 'sub', 'mul', 'div')
                elif self.function_set == "v2":
                    FunctionSet = ('add', 'sub', 'mul', 'div','log','sqrt')
                ConstRange = (-1*self.const_bound, self.const_bound)

                self.est_gp = SymbolicRegressor(population_size=5000,
                                generations=self.sr_epochs, stopping_criteria=0.01,
                                p_crossover=0.7, p_subtree_mutation=0.1,
                                p_hoist_mutation=0.05, p_point_mutation=0.1,
                                max_samples=0.9, verbose=1, function_set=FunctionSet,
                                parsimony_coefficient=0.01, random_state=self.random_state, const_range=ConstRange)
            elif self.ml_model_type == "lightgbm":
                self.est_gp = LGBMRegressor(random_state=self.random_state)
            elif self.ml_model_type == "random_forest":
                self.est_gp = RandomForestRegressor(random_state=self.random_state)

        # total step
        self.step = 0

    def train_epoch(self, epoch):
        loss_all = 0
        cnt = 0
        for sample_x, sample_y, sample_action in self.data_loader.train_data_generator:
            sample_x = torch.tensor(
                sample_x,
                dtype=torch.float,
                device=self.device
            )
            sample_y = torch.tensor(
                sample_y,
                dtype=torch.float,
                device=self.device
            )
            reward_predictor = torch.zeros_like(
                sample_y
            )
            predict_y = self.policy(sample_x) # (batch_size, num_classes)
            
            for i in range(predict_y.shape[0]):
                reward_predictor[i] = predict_y[i, sample_action[i]]
            batch_loss = self.loss(reward_predictor, sample_y)
            # set_trace()
            self.policy_optimizer.zero_grad()
            batch_loss.backward()
            self.policy_optimizer.step()
            loss_all += batch_loss.item()
            cnt += 1
            self.step += 1
        if self.lr_decay:
            self.policy_lr_scheduler.step()
        return loss_all / cnt

    def train_epoch_ppa(self, epoch):
        loss_all = 0
        cnt = 0
        for sample_x, sample_y, sample_action in self.data_loader.train_data_generator:
            sample_x = torch.tensor(
                sample_x,
                dtype=torch.float,
                device=self.device
            )
            sample_y = torch.tensor(
                sample_y,
                dtype=torch.float,
                device=self.device
            )
            predict_y = self.policy(sample_x) # (batch_size, num_classes)
            batch_loss = self.loss(predict_y, sample_y)
            # set_trace()
            self.policy_optimizer.zero_grad()
            batch_loss.backward()
            self.policy_optimizer.step()
            loss_all += batch_loss.item()
            cnt += 1
            self.step += 1
        if self.lr_decay:
            self.policy_lr_scheduler.step()

        return loss_all / cnt

    def train_epoch_area_delay(self, epoch):
        loss_all = 0
        cnt = 0
        for sample_x, sample_y1, sample_y2 in self.data_loader.train_data_generator:
            sample_x = torch.tensor(
                sample_x,
                dtype=torch.float,
                device=self.device
            )
            sample_y1 = torch.tensor(
                sample_y1,
                dtype=torch.float,
                device=self.device
            )
            sample_y2 = torch.tensor(
                sample_y2,
                dtype=torch.float,
                device=self.device
            )
            
            predict_y1, predict_y2 = self.policy(sample_x) # (batch_size, num_classes)
            batch_loss1 = self.loss(predict_y1, sample_y1)
            batch_loss2 = self.loss(predict_y2, sample_y2)
            batch_loss = batch_loss1 + batch_loss2
            # set_trace()
            self.policy_optimizer.zero_grad()
            batch_loss.backward()
            self.policy_optimizer.step()
            loss_all += batch_loss.item()
            cnt += 1
            self.step += 1
        if self.lr_decay:
            self.policy_lr_scheduler.step()

        return loss_all / cnt

    def train_epoch_sr(self, epoch):
        loss_all = 0
        cnt = 0
        for train_x, train_y, _ in self.data_loader.train_data_generator:
            self.est_gp.fit(train_x, train_y)
            break

        return loss_all / (cnt+1)

    def evaluate_ppa(self, epoch, loss):
        test_loss_all = 0
        cnt = 0
        for sample_x, sample_y, sample_action in self.data_loader.test_data_generator:
            sample_x = torch.tensor(
                sample_x,
                dtype=torch.float,
                device=self.device
            )
            sample_y = torch.tensor(
                sample_y,
                dtype=torch.float,
                device=self.device
            )
            with torch.no_grad():
                predict_y = self.policy(sample_x)
                test_loss = torch.mean((predict_y - sample_y)**2)

            # 绘制reward predictor 和 sample y 的对比曲线
            fig1 = plt.figure()
            x = np.linspace(1, predict_y.shape[0], num=predict_y.shape[0])
            f1 = plt.scatter(x, (predict_y-sample_y).cpu().numpy())
            self.tb_logger.add_figure('prediction error', fig1, global_step=epoch)
            fig2 = plt.figure()
            x = np.linspace(1, predict_y.shape[0], num=predict_y.shape[0])
            f1 = plt.scatter(x, (predict_y).cpu().numpy())
            self.tb_logger.add_figure('predict reward', fig2, global_step=epoch)
            fig3 = plt.figure()
            x = np.linspace(1, predict_y.shape[0], num=predict_y.shape[0])
            f1 = plt.scatter(x, (sample_y).cpu().numpy())
            self.tb_logger.add_figure('true reward', fig3, global_step=epoch)
            test_loss_all += test_loss.item()
            cnt += 1
        test_loss = test_loss_all /  cnt
        self.tb_logger.add_scalar('test loss', test_loss, global_step=epoch)
        self.tb_logger.add_scalar('train loss', loss, global_step=epoch)
        print(f"*******epoch {epoch}, test loss {test_loss}*******")

    def evaluate_area_delay(self, epoch, loss):
        test_loss1_all = 0
        test_loss2_all = 0
        cnt = 0
        for sample_x, sample_y1, sample_y2 in self.data_loader.test_data_generator:
            sample_x = torch.tensor(
                sample_x,
                dtype=torch.float,
                device=self.device
            )
            sample_y1 = torch.tensor(
                sample_y1,
                dtype=torch.float,
                device=self.device
            )
            sample_y2 = torch.tensor(
                sample_y2,
                dtype=torch.float,
                device=self.device
            )
            with torch.no_grad():
                predict_y1, predict_y2 = self.policy(sample_x)
                test_loss1 = torch.mean((predict_y1 - sample_y1)**2)
                test_loss2 = torch.mean((predict_y2 - sample_y2)**2)
                
            # 绘制reward predictor 和 sample y 的对比曲线
            fig1 = plt.figure()
            x = np.linspace(1, predict_y1.shape[0], num=predict_y1.shape[0])
            f1 = plt.scatter(x, (predict_y1-sample_y1).cpu().numpy())
            self.tb_logger.add_figure('y1 prediction error', fig1, global_step=epoch)
            
            fig2 = plt.figure()
            x = np.linspace(1, predict_y2.shape[0], num=predict_y2.shape[0])
            f1 = plt.scatter(x, (predict_y2-sample_y2).cpu().numpy())
            self.tb_logger.add_figure('y2 prediction error', fig2, global_step=epoch)
            
            fig3 = plt.figure()
            x = np.linspace(1, predict_y1.shape[0], num=predict_y1.shape[0])
            f1 = plt.scatter(x, predict_y1.cpu().numpy())
            self.tb_logger.add_figure('y1 prediction', fig3, global_step=epoch)

            fig4 = plt.figure()
            x = np.linspace(1, predict_y2.shape[0], num=predict_y2.shape[0])
            f1 = plt.scatter(x, predict_y2.cpu().numpy())
            self.tb_logger.add_figure('y2 prediction', fig4, global_step=epoch)
            
            test_loss1_all += test_loss1.item()
            test_loss2_all += test_loss2.item()
            cnt += 1
        test_loss1 = test_loss1_all /  cnt
        test_loss2 = test_loss2_all /  cnt
        self.tb_logger.add_scalar('test loss y1', test_loss1, global_step=epoch)
        self.tb_logger.add_scalar('test loss y2', test_loss2, global_step=epoch)
        self.tb_logger.add_scalar('train loss', loss, global_step=epoch)

    def evaluate_sr(self, epoch, loss):
        test_loss1_all = 0
        test_loss2_all = 0
        cnt = 0
        for test_x, test_y, _ in self.data_loader.test_data_generator:
            predict_y = self.est_gp.predict(test_x)
            predict_y = np.expand_dims(predict_y, axis=1)
            test_y = test_y.cpu().numpy()

            test_loss = np.mean((predict_y - test_y)**2)

            # 绘制reward predictor 和 sample y 的对比曲线
            fig1 = plt.figure()
            x = np.linspace(1, predict_y.shape[0], num=predict_y.shape[0])
            f1 = plt.scatter(x, (predict_y-test_y))
            self.tb_logger.add_figure('y1 prediction error', fig1, global_step=epoch)
            
            test_loss1_all += test_loss.item()
            cnt += 1
        test_loss1 = test_loss1_all /  cnt
        self.tb_logger.add_scalar('test loss y1', test_loss1, global_step=epoch)
        self.tb_logger.add_scalar('train loss', loss, global_step=epoch)
        print(f"test_loss: {test_loss1}")
        print(f"train_loss: {loss}")

    def evaluate(self, epoch, loss):
        for sample_x, sample_y, sample_action in self.data_loader.test_data_generator:
            sample_x = torch.tensor(
                sample_x,
                dtype=torch.float,
                device=self.device
            )
            sample_y = torch.tensor(
                sample_y,
                dtype=torch.float,
                device=self.device
            )
            reward_predictor = torch.zeros_like(
                sample_y
            )

            with torch.no_grad():
                predict_y = self.policy(sample_x)
                for i in range(predict_y.shape[0]):
                    reward_predictor[i] = predict_y[i, sample_action[i]]

                test_loss = torch.mean((reward_predictor - sample_y)**2)

            # 绘制reward predictor 和 sample y 的对比曲线
            fig1 = plt.figure()
            x = np.linspace(1, reward_predictor.shape[0], num=reward_predictor.shape[0])
            f1 = plt.scatter(x, (reward_predictor-sample_y).cpu().numpy())
            self.tb_logger.add_figure('prediction error', fig1, global_step=epoch)
            fig2 = plt.figure()
            x = np.linspace(1, reward_predictor.shape[0], num=reward_predictor.shape[0])
            f1 = plt.scatter(x, (reward_predictor).cpu().numpy())
            self.tb_logger.add_figure('predict reward', fig2, global_step=epoch)
            fig3 = plt.figure()
            x = np.linspace(1, reward_predictor.shape[0], num=reward_predictor.shape[0])
            f1 = plt.scatter(x, (sample_y).cpu().numpy())
            self.tb_logger.add_figure('true reward', fig3, global_step=epoch)
            self.tb_logger.add_scalar('train loss', loss, global_step=epoch)
            self.tb_logger.add_scalar('test loss', test_loss.item(), global_step=epoch)
            print(f"*******epoch {epoch}, test loss {test_loss.item()}*******")

    def train(self):
        # mse loss
        for epoch in range(self.epochs):
            if self.train_type == "reward_regression":
                loss = self.train_epoch(epoch)
                print(f"*******epoch {epoch}, train loss {loss}*******")
                if epoch % self.evaluate_freq == 0:
                    self.evaluate(epoch, loss)
            elif self.train_type == "ppa_regression":
                loss = self.train_epoch_ppa(epoch)
                print(f"*******epoch {epoch}, train loss {loss}*******")
                if epoch % self.evaluate_freq == 0:
                    self.evaluate_ppa(epoch, loss)
            elif self.train_type == "area_delay_regression":
                loss = self.train_epoch_area_delay(epoch)
                print(f"*******epoch {epoch}, train loss {loss}*******")
                if epoch % self.evaluate_freq == 0:
                    self.evaluate_area_delay(epoch, loss)
            elif self.train_type == "reward_classify":
                pass
            elif self.train_type == "symbolic_regression":
                loss = self.train_epoch_sr(epoch)
                print(f"*******epoch {epoch}, train loss {loss}*******")
                if epoch % self.evaluate_freq == 0:
                    self.evaluate_sr(epoch, loss)
                break
            # save model
            if epoch % 250 == 0:
                self.save_checkpoint(epoch)
        self.save_checkpoint(epoch)
        
    def save_checkpoint(self, epoch):
        if self.policy is not None:
            save_pkl = os.path.join(self.save_pkl_path, f"ppa_model_{self.bit_width}_{epoch}.pkl")
            torch.save(self.policy.state_dict(), save_pkl)
        if self.train_type == "symbolic_regression":
            if self.ml_model_type == "symbolic_regression":
                save_pkl = os.path.join(self.save_pkl_path, f"sr_model_{self.bit_width}_{self.sr_epochs}_{self.y_type}_{self.split_ratio}.joblib")
                print(f"gpsr expression: {self.est_gp._program}")
            elif self.ml_model_type == "lightgbm":
                save_pkl = os.path.join(self.save_pkl_path, f"lightgbm_model_{self.bit_width}_{self.sr_epochs}_{self.y_type}_{self.split_ratio}.joblib")
            elif self.ml_model_type == "random_forest":
                save_pkl = os.path.join(self.save_pkl_path, f"randomforest_model_{self.bit_width}_{self.sr_epochs}_{self.y_type}_{self.split_ratio}.joblib")
            
            joblib.dump(self.est_gp, save_pkl)

if __name__ == "__main__":
    # 预测 reward 对比
        # mannual state
        # matrix state
        # seq state 
        # image state
    parser = argparse.ArgumentParser(description='Trainer Pipeline')
    parser.add_argument(
        '--train_type', type=str, default='image_state_reward') # image_state_reward
    parser.add_argument(
        '--y_type', type=str, default='normalize_area') # normalize_area
    parser.add_argument(
        '--y1_type', type=str, default='normalize_area') # normalize_area
    parser.add_argument(
        '--y2_type', type=str, default='normalize_delay') # normalize_area
    parser.add_argument(
        '--x_type', type=str, default='image_next_state') # image_next_state
    parser.add_argument(
        '--is_y_normalize', type=str, default='False') # False
    parser.add_argument(
        '--task_type', type=str, default='ppa_regression') # ppa_regression
    parser.add_argument(
        '--epochs', type=int, default=3000)
    parser.add_argument(
        '--sr_epochs', type=int, default=1000)
    parser.add_argument(
        '--bit_width', type=str, default="16_bits")
    parser.add_argument(
        '--train_batch_size', type=int, default=1024)
    parser.add_argument(
        '--ppa_scale', type=int, default=100)
    parser.add_argument(
        '--data_loader_type', type=str, default="area_delay") # ["area_delay", ["score"]]
    parser.add_argument(
        '--ml_model_type', type=str, default="symbolic_regression") # ["symbolic_regression", ["lightgbm"]]
    parser.add_argument(
        '--random_state', type=int, default=0)
    parser.add_argument(
        '--function_set', type=str, default="v1") # [v1, v2]
    parser.add_argument(
        '--const_bound', type=float, default=1)
    
    args = parser.parse_args()
    print(args.train_type)
    device = "cuda:0"
    
    learning_rate = 1e-3
    epochs = args.epochs
    if args.bit_width == "16_bits":
        if args.task_type == "reward_regression":
            num_classes = 128
        elif args.task_type == "ppa_regression":
            num_classes = 1
    elif args.bit_width == "32_bits":
        if args.task_type == "reward_regression":
            num_classes = 252
        elif args.task_type == "ppa_regression":
            num_classes = 1

    if args.train_type == "image_state_reward":
        policy = DeepQPolicy(
            BasicBlock,
            num_classes=num_classes
        ).to(device)
    elif args.train_type == "image_state_reward_v2":
        policy = DeepQPolicy(
            BasicBlock,
            input_channels=4,
            num_classes=num_classes
        ).to(device)
    elif args.train_type == "mannual_state":
        feature_dim = 4
        policy = MLP(
            feature_dim, output_dim=num_classes,
            hidden_sizes=[256,256], device=device
        ).to(device)
    elif args.train_type == "matrix_state":
        if args.bit_width == "16_bits":  
            feature_dim = 64
        elif args.bit_width == "32_bits":
            feature_dim = 126
        policy = MLP(
            feature_dim, output_dim=num_classes,
            hidden_sizes=[256,256], device=device
        ).to(device)
    elif args.train_type == "seq_state":
        policy = AttentionValueNet(
            8,128,8,8,num_classes
        ).to(device)
    elif args.train_type == "seq_state_relu2":
        policy = AttentionValueNet(
            8,128,8,8,num_classes, non_linear='relu2'
        ).to(device)
    elif args.train_type == "area_delay_model":
        policy = MBRLPPAModel(
            BasicBlock
        ).to(device)
    elif args.train_type == "symbolic_regression":
        policy = None

    YKey = args.y_type
    XKey = args.x_type
    Y1Key = args.y1_type
    Y2Key = args.y2_type
    # npy_data_path = "outputs/2023-11-11/09-48-44/logger_log/dqn_16bits/dqn16bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool/dqn16bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool_2023_11_11_09_48_50_0000--s-1/itr_3750.npy"
    # factor action sample 16 bits
    if args.bit_width == "16_bits":
        # easymac verilog
        # npy_data_path = "../outputs/2023-11-14/00-50-19/logger_log/dqn_16bits_factor_action/dqn16bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool_factoraction/dqn16bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool_factoraction_2023_11_14_00_50_24_0000--s-1/itr_3750.npy" # for v1 logic synthesis flow
        # npy_data_path = "../outputs/2023-11-19/10-19-06/logger_log/dqn_16bits_factor_action/dqn16bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool_factoraction_v2_synthesis/dqn16bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool_factoraction_v2_synthesis_2023_11_19_10_19_11_0000--s-1/itr_2500.npy" # for v2 logic synthesis flow
        
        # v1 脚本 加速Verilog
        npy_data_path = "../outputs/2024-03-12/14-08-16/logger_log/dqn_16bits_factor_action/dqn16bits_reset_rnd_factor_speedup_v1syn/dqn16bits_reset_rnd_factor_speedup_v1syn_2024_03_12_14_08_22_0000--s-1/itr_10000.npy"
        synthesis_type = "v1"
        # v2 脚本
        # npy_data_path = "../outputs/2024-03-12/14-09-47/logger_log/dqn_16bits_factor_action/dqn16bits_reset_rnd_factor_speedup_v2syn/dqn16bits_reset_rnd_factor_speedup_v2syn_2024_03_12_14_09_53_0000--s-1/itr_10000.npy"
        # synthesis_type = "v2"
    elif args.bit_width == "32_bits":
        # easymac verilog
        # factor action sample 32 bits
        # npy_data_path = "../outputs/2023-11-14/00-50-25/logger_log/dqn_rnd_reset_factor_action/dqn32bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool_v2state_max_stage12_factoraction/dqn32bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool_v2state_max_stage12_factoraction_2023_11_14_00_50_32_0000--s-1/itr_7500.npy"

        # v1 脚本 加速verilog
        npy_data_path = "../outputs/2024-04-08/10-14-43/logger_log/dqn_rnd_reset_factor_action/dqn32bits_reset_rnd_factor_speedup_v1syn/dqn32bits_reset_rnd_factor_speedup_v1syn_2024_04_08_10_14_49_0000--s-1/itr_10000.npy"
        synthesis_type = "v1"
    if args.is_y_normalize == 'True':
        is_y_normalize = True
    else:
        is_y_normalize = False
    
    if args.bit_width == "16_bits":
        data_loader_kwargs = {
            "bit_width": "16_bits_and",
            "MAX_STAGE_NUM": 6,
            "int_bit_width": 15.5,
            "wallace_area": 2064.5,
            "wallace_delay": 1.33
        }
    elif args.bit_width == "32_bits":
        data_loader_kwargs = {
            "bit_width": "32_bits_and",
            "MAX_STAGE_NUM": 12,
            "int_bit_width": 31.5,
            "wallace_area": 8221.5,
            "wallace_delay": 2.37
        }

    # SplitRatio = [0.8, 0.2]
    
    SplitRatios = [
        # [0.0001, 0.9999], # only one data
        # [0.001, 0.999],
        [0.01, 0.99],
        [0.05, 0.95],
        [0.1, 0.9],
        [0.02, 0.98],
        [0.03, 0.97],
        [0.04, 0.96],
        [0.06, 0.94],
        [0.07, 0.93],
        [0.08, 0.92],
        [0.09, 0.91]
        # [0.2, 0.8],
        # # [0.3, 0.7]
        # # [0.4, 0.6]
        # [0.5, 0.5],
        # [0.8, 0.2]
        # [0.99,0.01]
    ]
    for SplitRatio in SplitRatios:
        if args.data_loader_type == "score":
            data_loader = ScoreDataLoader(
                npy_data_path, x_key=XKey, y_key=YKey, is_y_normalize=is_y_normalize,
                train_batch_size=args.train_batch_size, split_ratio=SplitRatio, mode="train",
                random_shuffle=False, ppa_scale=args.ppa_scale,
                **data_loader_kwargs
            )
        elif args.data_loader_type == "area_delay":
            data_loader = AreaDelayDataLoader(
                npy_data_path, x_key=XKey, y1_key=Y1Key, y2_key=Y2Key, is_y_normalize=is_y_normalize,
                train_batch_size=args.train_batch_size, split_ratio=SplitRatio, mode="train",
                random_shuffle=False, ppa_scale=args.ppa_scale,
                **data_loader_kwargs
            )
        time_stamp = datetime.datetime.now().timestamp()
        base_log_dir = f"./offline_sl/outputs/{args.train_type}_{args.bit_width}_{args.y_type}_{args.y1_type}_{args.y2_type}_{args.is_y_normalize}_{args.task_type}_{args.epochs}_{SplitRatio}_{learning_rate}_{synthesis_type}_{args.ppa_scale}_{args.ml_model_type}_{args.random_state}_{args.function_set}_{args.const_bound}_random_shuffle_false_lr_decay_{time_stamp}"

        tb_logger = SummaryWriter(base_log_dir)

        trainer = Trainer(
            data_loader,
            policy,
            tb_logger,
            train_type=args.task_type,
            lr=learning_rate,
            epochs=epochs,
            device=device,
            bit_width=args.bit_width,
            lr_decay=True,
            lr_decay_step=10, # 10 
            lr_decay_rate=0.96, # 0.96
            y_type=YKey,
            sr_epochs=args.sr_epochs,
            ml_model_type=args.ml_model_type,
            split_ratio=SplitRatio,
            random_state=args.random_state,
            function_set=args.function_set,
            const_bound=args.const_bound

        )

        trainer.train()