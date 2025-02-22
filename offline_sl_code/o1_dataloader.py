import os
import sys
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
# 从npy load data
# 处理npy 数据得到各种各样的可能的(feature,y)数据集
# 生成各种可能得训练数据
from ipdb import set_trace

ColumnFeatureNum = 8

class ScoreDataset(Dataset):
    def __init__(self, x, y, action):
        self.x = x
        self.y = y
        self.action = action

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.action[index]

class AreaDelayDataset(Dataset):
    def __init__(self, x, y1, y2):
        self.x = x
        self.y1 = y1
        self.y2 = y2

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y1[index], self.y2[index]

"""
    x: image state for cnn; features for sr and tree model; proxy model;
    y: normalize area/delay;
   要做哪些实验
   1. 代理模型 vs 真实模型，估计误差分布；
        x 提取node num 估计or stage num估计, y提取normalize area/normalize delay --> 对应next state
        done stage number 确实都是5, 没有问题;
        误差分布这些数据放附录好嘞,正文还是放一些准确率,MSE的指标好嘞

   2. 不同模型学习对比
        1. 训练收敛时间对比; 训练时间vs准确率;
        2. 训练样本量vs准确率
        3. cnn 模型,树模型,SR模型
        4. xin
    
   3. sr模型的估计误差分布,和低精度模型,高精度模型构成多精度环境
"""

class ScoreDataLoader():
    def __init__(
        self,
        npy_data_path,
        bit_width="16_bits",
        MAX_STAGE_NUM=6,
        int_bit_width=16,
        # dataset hyperparameter
        x_key="matrix_state",
        y_key="delta_ppa",
        is_y_normalize=True,
        split_ratio=[0.8,0.2],
        train_batch_size=1024,
        seed=1,
        mode="analyze", # train
        wallace_area=2064.5,
        wallace_delay=1.33,
        wallace_node_num=719,
        wallace_stage_num=5,
        ppa_scale=100,
        random_shuffle=True
    ):
        self.npy_data_path = npy_data_path
        # hyperparameter
        self.MAX_STAGE_NUM = MAX_STAGE_NUM
        self.int_bit_width = int_bit_width
        self.bit_width = bit_width
        self.x_key = x_key
        self.y_key = y_key
        self.split_ratio = split_ratio
        self.train_batch_size = train_batch_size
        self.seed = seed
        self.is_y_normalize = is_y_normalize
        self.mode = mode
        self.wallace_area = wallace_area
        self.wallace_delay = wallace_delay
        self.wallace_node_num = wallace_node_num
        self.wallace_stage_num = wallace_stage_num
        self.ppa_scale = ppa_scale
        self.random_shuffle = random_shuffle

        np.random.seed(seed)
        self.dict_dataset = None
        self.training_dict_dataset = {}

        if self.mode == "train":
            train_x, train_y, test_x, test_y, train_action, test_action = self.load_data()
            self.training_dataset = ScoreDataset(train_x, train_y, train_action)
            self.testing_dataset = ScoreDataset(test_x, test_y, test_action)
            
            self.test_samples = test_x.shape[0]
            print(self.test_samples)

            self.train_data_generator = DataLoader(
                    self.training_dataset,
                    batch_size=self.train_batch_size, shuffle=True)
            self.test_data_generator = DataLoader(
                    self.testing_dataset,
                    batch_size=self.test_samples, shuffle=False)
        elif self.mode == "analyze":
            self.load_and_analyze_data()

    def load_and_analyze_data(self):
        raw_data = np.load(self.npy_data_path, allow_pickle=True).item()['replay_buffer'] # deque, list of dict
        dict_dataset = self.process_data(raw_data)

        # proxy estimation 
        area_estimation = np.stack(dict_dataset["normalize_node_num_v2"], axis=0)
        delay_estimation = np.stack(dict_dataset["normalize_stage_num"], axis=0)
        # true feedback
        area = np.stack(dict_dataset["normalize_area"], axis=0)
        delay = np.stack(dict_dataset["normalize_delay"], axis=0)

        # plot curves
        # 图1：estimation error 分布
        # 图2：x y 曲线
        area_estimation_error = area_estimation - area
        delay_estimation_error = delay_estimation - delay
        
        # print(area_estimation_error)
        # print(delay_estimation_error)
        # plt.hist(area_estimation_error)
        # plt.savefig("./results/proxy_hist_area.png")

        # plt.hist(delay_estimation_error)
        # plt.savefig("./results/proxy_hist_delay.png")
        
        # x = np.arange(area.shape[0])
        # plt.plot(x, area_estimation, label="proxy")
        # plt.plot(x, area, label="true")
        # plt.savefig("./results/proxy_trend_area.png")

        x = np.arange(delay.shape[0])
        plt.plot(x, delay_estimation, label="proxy")
        # plt.plot(x, delay, label="true")
        # print(delay)
        print(delay_estimation)
        plt.savefig("./results/proxy_trend_delay.png")
        
    def load_data(self):
        # load data
        raw_data = np.load(self.npy_data_path, allow_pickle=True).item()['replay_buffer'] # deque, list of dict
        ##### input features
        # 1. x = matrix state 展平，应该很垃圾
        # 2. x = image 
        # 3. x = seq state
        # 4. x = image 加上fanin fanout 信息
        ##### labels 
        # 1. ppa 绝对值
        # 2. scale ppa delta reward model
        # 3. 分类任务

        dict_dataset = self.process_data(raw_data)
        self.dict_dataset = dict_dataset
        train_x, train_y, test_x, test_y, train_action, test_action = self.get_split_dataset(dict_dataset)

        self.training_dict_dataset["train_x"] = train_x
        self.training_dict_dataset["train_y"] = train_y
        self.training_dict_dataset["test_x"] = test_x
        self.training_dict_dataset["test_y"] = test_y
        
        return train_x, train_y, test_x, test_y, train_action, test_action

    def _std_normalize(self, y):
        y_mean = np.mean(y)
        y_std = np.std(y)

        y = (y - y_mean) / (y_std + 1e-4)

        return y

    def get_split_dataset(self, dict_dataset):
        x = np.stack(dict_dataset[self.x_key], axis=0)
        y = np.stack(dict_dataset[self.y_key], axis=0)
        action = np.stack(dict_dataset["action"], axis=0)
        if self.is_y_normalize:
            y = self._std_normalize(y)

        indexes = np.arange(x.shape[0])
        if self.random_shuffle:
            np.random.shuffle(indexes)
        num_train = int(self.split_ratio[0] * x.shape[0]) 
        print(f"num_train: {num_train}")
        train_indexes = indexes[:num_train]
        test_indexes = indexes[num_train:]

        train_x = x[train_indexes]
        test_x = x[test_indexes]

        train_y = y[train_indexes]
        test_y = y[test_indexes]
        train_y = np.expand_dims(train_y, axis=1)
        test_y = np.expand_dims(test_y, axis=1)

        train_action = action[train_indexes]
        test_action = action[test_indexes]
        return train_x, train_y, test_x, test_y, train_action, test_action

    # def _get_sr_feature(self, ct32, ct22, stage_num):
    #     sr_feature = []
    #     # global feature
    #     sr_feature.append(stage_num)
    #     ct32_num = np.sum(ct32)
    #     ct22_num = np.sum(ct22)
    #     sr_feature.append(3*ct32_num+2*ct22_num)
    #     sr_feature.append(ct32_num)
    #     sr_feature.append(ct22_num)
        
    #     # get image state 
    #     image_state = self._get_image_state(ct32, ct22, stage_num)
    #     # stage feature for ct32 / ct22
    #     for i in range(image_state.shape[1]):
    #         stage_ct32 = np.sum(image_state[0,i])
    #         stage_ct22 = np.sum(image_state[1,i])
    #         sr_feature.append(stage_ct32)
    #         sr_feature.append(stage_ct22)

    #     # column feature for ct32 / ct22
    #     for i in range(image_state.shape[2]):
    #         column_ct32 = np.sum(image_state[0,:,i])
    #         column_ct22 = np.sum(image_state[1,:,i])
    #         sr_feature.append(column_ct32)
    #         sr_feature.append(column_ct22)
        
    #     return sr_feature

    def _get_sr_feature(self, ct32, ct22, stage_num):
        sr_feature = []
        # global feature
        sr_feature.append(stage_num)
        ct32_num = np.sum(ct32)
        ct22_num = np.sum(ct22)
        sr_feature.append(3*ct32_num+2*ct22_num)
        sr_feature.append(ct32_num)
        sr_feature.append(ct22_num)
        
        # get image state 
        image_state = self._get_image_state(ct32, ct22, stage_num)
        # stage feature for ct32 / ct22
        # for i in range(image_state.shape[1]):
        #     stage_ct32 = np.sum(image_state[0,i])
        #     stage_ct22 = np.sum(image_state[1,i])
        #     if stage_ct32 == 0:
        #         stage_ct32 = -1
        #     if stage_ct22 == 0:
        #         stage_ct22 = -1
        #     sr_feature.append(stage_ct32)
        #     sr_feature.append(stage_ct22)

        # column feature for ct32 / ct22
        # if self.is_sr_include_column:
        for i in range(1, image_state.shape[2]):
            column_ct32_ct22 = np.sum(image_state[0,:,i]) + np.sum(image_state[1,:,i])
            if column_ct32_ct22 == 0:
                column_ct32_ct22 = -1
            sr_feature.append(column_ct32_ct22)

        return sr_feature
    
    def _get_image_state(self, ct32, ct22, stage_num):
        ct32 = np.expand_dims(ct32, axis=0)
        ct22 = np.expand_dims(ct22, axis=0)
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = np.zeros((1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2)))
            ct32 = np.concatenate((ct32, zeros), axis=1)
            ct22 = np.concatenate((ct22, zeros), axis=1)
        image_state = np.concatenate((ct32, ct22), axis=0) # (2, max_stage-1, num_column)        
        return image_state

    def _get_ct_delay(self, ct32, ct22):
        ct_delay = np.zeros_like(ct32) # (stage_num, num_column)
        for i in range(1, ct32.shape[0]):
            # iterate over stage
            for j in range(ct32.shape[1]):
                # iterate over column
                if j == 0:
                    if ct32[i-1,j] > 0:
                        # 32 sum two delay
                        ct_delay[i,j] = ct_delay[i-1,j] + 2
                    elif ct22[i-1,j] > 0:
                        # 22 sum one delay
                        ct_delay[i,j] = ct_delay[i-1,j] + 1
                    else:
                        # no added delay
                        ct_delay[i,j] = ct_delay[i-1,j] + 0
                else:
                    if ct32[i-1,j] > 0:
                        current_sum = 2
                    elif ct22[i-1,j] > 0:
                        current_sum = 1
                    else:
                        current_sum = 0
                    if ct32[i-1,j-1] > 0 or ct22[i-1,j-1] > 0:
                        last_carry = 1
                    else:
                        last_carry = 0
                    if last_carry == 0:
                        ct_delay[i,j] = ct_delay[i-1,j] + current_sum
                    else:
                        ct_delay[i,j] = max(
                            ct_delay[i-1,j] + current_sum,
                            ct_delay[i-1,j-1] + last_carry
                        )
        return ct_delay
    
    def _get_image_state_v2(self, ct32, ct22, stage_num):
        ct_delay = self._get_ct_delay(ct32, ct22)
        ct32 = np.expand_dims(ct32, axis=0)
        ct22 = np.expand_dims(ct22, axis=0)
        ct_delay = np.expand_dims(ct_delay, axis=0)
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = np.zeros((1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2)))
            ct32 = np.concatenate((ct32, zeros), axis=1)
            ct22 = np.concatenate((ct22, zeros), axis=1)
            ct_delay = np.concatenate((ct_delay, zeros), axis=1)
        ct_area = 3 * ct32 + 2 * ct22
        image_state_v2 = np.concatenate((ct32, ct22, ct_area, ct_delay), axis=0) # (2, max_stage-1, num_column)        
        return image_state_v2

    def _get_estimated_delay(self, ct32_i, ct22_i):
        try:
            nonzero32 = list(np.nonzero(ct32_i))[-1]
            min_delay_32 = nonzero32[-1]
        except:
            # print(f"warning!!! ct32_i zero: {ct32_i}")
            min_delay_32 = 0
        
        try:
            nonzero31 = list(np.nonzero(ct22_i))[-1]
            min_delay_22 = nonzero31[-1]
        except:
            # print(f"warning!!! ct22_i zero: {ct22_i}")
            min_delay_22 = 0
        
        return max(min_delay_32, min_delay_22)
    
    def _get_seq_state(self, state, state_mask, ct32, ct22):
        # input matrix state and state mask
        # output sequence state
        """
            state feature vector definition
            [
                pp(1), position(1), mask(4), 
                cur_column: num32, num22, estimated area, estimated delay,
                last_column: (4)
                next_column: (4)
            ]
            state feature vector definition
            [
                mask(4), 
                cur_column: num32, num22, estimated area, estimated delay,
                ct32 each stage
                ct22 each stage
            ]
        """
        num_column = state.shape[1]
        state_features = np.zeros(
            (num_column, ColumnFeatureNum)
        )
        for i in range(num_column):
            # # pp
            # state_features[i,0] = initial_partial_product[i]
            # # position
            # state_features[i,1] = i
            # mask
            cur_column_mask = state_mask[4*i:4*(i+1)]
            state_features[i,0:4] = np.array(cur_column_mask, dtype=np.float)
            
            # column features
            state_features[i,4] = state[0,i]
            state_features[i,5] = state[1,i]
            state_features[i,6] = 3*state[0,i] + 2*state[1,i]
            # i-th column 32 delay
            estimated_delay = self._get_estimated_delay(ct32[:,i], ct22[:,i])
            state_features[i,7] = estimated_delay

        return state_features
    
    def process_data(self, raw_data):
        dict_dataset = {
            "matrix_state": [],
            "image_state": [],
            "image_state_v2": [],
            "seq_state": [],
            "mannual_state": [],
            "matrix_next_state": [],
            "image_next_state": [],
            "image_next_state_v2": [],
            "seq_next_state": [],
            "mannual_next_state": [],
            "action": [],
            "ppa": [],
            "area": [],
            "delay": [],
            "delta_ppa": [],
            "delta_ppa_class": [],
            "next_state_stage_num": [],
            "next_state_node_num_v1": [],
            "next_state_node_num_v2": [],
            "next_state_max_delay": [],
            "next_state_avg_delay": [],
            "normalize_area": [],
            "normalize_delay": [],
            "normalize_node_num_v2": [],
            "normalize_stage_num": [],
            "sr_features": []     
        }

        for transition_data in raw_data:
            # 1. matrix state
            dict_dataset["matrix_state"].append(
                transition_data.state.flatten().numpy()
            )
            dict_dataset["matrix_next_state"].append(
                transition_data.next_state.flatten().numpy()
            )
            # 2. image state
            ct32_state = transition_data.state_ct32
            ct22_state = transition_data.state_ct22
            stage_num = ct32_state.shape[0] - 1
            # print(f"stage_num: {stage_num}")
            image_state = self._get_image_state(
                ct32_state, ct22_state, stage_num
            )
            dict_dataset["image_state"].append(
                image_state
            )
            ct32_next_state = transition_data.next_state_ct32
            ct22_next_state = transition_data.next_state_ct22
            stage_num_next_state = ct32_next_state.shape[0] - 1
            # print(f"stage_num_next_state: {stage_num_next_state}")
            image_next_state = self._get_image_state(
                ct32_next_state, ct22_next_state, stage_num_next_state
            )
            dict_dataset["image_next_state"].append(
                image_next_state
            )

            sr_feature = self._get_sr_feature(ct32_next_state, ct22_next_state, stage_num_next_state)
            dict_dataset["sr_features"].append(
                sr_feature
            )

            # 22. image state with area delay information
            image_state_v2 = self._get_image_state_v2(
                ct32_state, ct22_state, stage_num
            )
            dict_dataset["image_state_v2"].append(
                image_state_v2
            )
            image_next_state_v2 = self._get_image_state_v2(
                ct32_next_state, ct22_next_state, stage_num_next_state
            )
            dict_dataset["image_next_state_v2"].append(
                image_next_state_v2
            )
            # next state stage num, node num, max delay
            dict_dataset["next_state_stage_num"].append(
                stage_num_next_state
            )
            dict_dataset["normalize_stage_num"].append(
                stage_num_next_state * self.ppa_scale / self.wallace_stage_num
            )
            
            node_num_v1 = ct32_next_state.sum() + ct22_next_state.sum()
            node_num_v2 = ct32_next_state.sum() * 3 + ct22_next_state.sum() * 2
            max_delay = np.max(image_next_state_v2[3])
            mean_delay = np.mean(image_next_state_v2[3])
            dict_dataset["next_state_node_num_v1"].append(
                node_num_v1
            )
            dict_dataset["next_state_node_num_v2"].append(
                node_num_v2
            )
            dict_dataset["normalize_node_num_v2"].append(
                node_num_v2 * self.ppa_scale / self.wallace_node_num
            )
            
            dict_dataset["next_state_max_delay"].append(
                max_delay
            )
            dict_dataset["next_state_avg_delay"].append(
                mean_delay
            )
            # 3. seq state
            state = transition_data.state.squeeze().numpy()
            state_mask = transition_data.mask.squeeze().numpy()
            seq_state = self._get_seq_state(
                state, state_mask, ct32_state, ct22_state
            )
            dict_dataset["seq_state"].append(
                seq_state
            )
            # 4. feature vector
            # [stage_num, node_num, number of 32, number of 22]
            mannual_state = np.array(
                [
                    stage_num,
                    3*ct32_state.sum() + 2*ct22_state.sum(),
                    ct32_state.sum(),
                    ct22_state.sum()
                ]
            )
            dict_dataset["mannual_state"].append(
                mannual_state
            )
            # 5. reward
            dict_dataset["delta_ppa"].append(
                transition_data.reward.item() * 10
            )
            # 6. reward class
            if transition_data.reward.item() >= 0:
                dict_dataset["delta_ppa_class"].append(
                1.
            )    
            else:
                dict_dataset["delta_ppa_class"].append(
                0.
            )    
            # 7. action type
            action = int(transition_data.action)
            dict_dataset["action"].append(action)
            # 8. ppa, area, delay
            ppa = transition_data.rewards_dict["avg_ppa"]
            area = np.mean(transition_data.rewards_dict["area"])
            delay = np.mean(transition_data.rewards_dict["delay"])
            dict_dataset["ppa"].append(ppa)
            dict_dataset["area"].append(area)
            dict_dataset["normalize_area"].append(area * self.ppa_scale / self.wallace_area)
            dict_dataset["delay"].append(delay)
            dict_dataset["normalize_delay"].append(delay * self.ppa_scale / self.wallace_delay)
            
        return dict_dataset

class AreaDelayDataLoader():
    def __init__(
        self,
        npy_data_path,
        bit_width="16_bits",
        MAX_STAGE_NUM=6,
        int_bit_width=16,
        # dataset hyperparameter
        x_key="matrix_state",
        y1_key="delta_ppa",
        y2_key="delta_ppa",
        is_y_normalize=True,
        split_ratio=[0.8,0.2],
        train_batch_size=1024,
        seed=1,
        mode="analyze", # train
        wallace_area=2064.5,
        wallace_delay=1.33,
        wallace_node_num=719,
        wallace_stage_num=5,
        ppa_scale=100,
        random_shuffle=False
    ):
        self.npy_data_path = npy_data_path
        # hyperparameter
        self.MAX_STAGE_NUM = MAX_STAGE_NUM
        self.int_bit_width = int_bit_width
        self.bit_width = bit_width
        self.x_key = x_key
        self.y1_key = y1_key
        self.y2_key = y2_key
        self.split_ratio = split_ratio
        self.train_batch_size = train_batch_size
        self.seed = seed
        self.is_y_normalize = is_y_normalize
        self.mode = mode
        self.wallace_area = wallace_area
        self.wallace_delay = wallace_delay
        self.wallace_node_num = wallace_node_num
        self.wallace_stage_num = wallace_stage_num
        self.ppa_scale = ppa_scale
        self.random_shuffle = random_shuffle

        np.random.seed(seed)
        self.dict_dataset = None
        if self.mode == "train":
            # TO MODIFY
            train_x, test_x, train_y1, test_y1, train_y2, test_y2 = self.load_data()

            self.training_dataset = ScoreDataset(train_x, train_y1, train_y2)
            self.testing_dataset = ScoreDataset(test_x, test_y1, test_y2)
            
            self.test_samples = test_x.shape[0]
            print(self.test_samples)

            self.train_data_generator = DataLoader(
                    self.training_dataset,
                    batch_size=self.train_batch_size, shuffle=True)
            # self.test_data_generator = DataLoader(
            #         self.testing_dataset,
            #         batch_size=self.test_samples, shuffle=False)
            self.test_data_generator = DataLoader(
                    self.testing_dataset,
                    batch_size=self.train_batch_size, shuffle=False)
        elif self.mode == "analyze":
            self.load_and_analyze_data()

    def load_and_analyze_data(self):
        raw_data = np.load(self.npy_data_path, allow_pickle=True).item()['replay_buffer'] # deque, list of dict
        dict_dataset = self.process_data(raw_data)

        # proxy estimation 
        area_estimation = np.stack(dict_dataset["normalize_node_num_v2"], axis=0)
        delay_estimation = np.stack(dict_dataset["normalize_stage_num"], axis=0)
        # true feedback
        area = np.stack(dict_dataset["normalize_area"], axis=0)
        delay = np.stack(dict_dataset["normalize_delay"], axis=0)

        # plot curves
        # 图1：estimation error 分布
        # 图2：x y 曲线
        area_estimation_error = area_estimation - area
        delay_estimation_error = delay_estimation - delay
        
        # print(area_estimation_error)
        # print(delay_estimation_error)
        # plt.hist(area_estimation_error)
        # plt.savefig("./results/proxy_hist_area.png")

        # plt.hist(delay_estimation_error)
        # plt.savefig("./results/proxy_hist_delay.png")
        
        # x = np.arange(area.shape[0])
        # plt.plot(x, area_estimation, label="proxy")
        # plt.plot(x, area, label="true")
        # plt.savefig("./results/proxy_trend_area.png")

        x = np.arange(delay.shape[0])
        plt.plot(x, delay_estimation, label="proxy")
        # plt.plot(x, delay, label="true")
        # print(delay)
        print(delay_estimation)
        plt.savefig("./results/proxy_trend_delay.png")
        
    def load_data(self):
        # load data
        raw_data = np.load(self.npy_data_path, allow_pickle=True).item()['replay_buffer'] # deque, list of dict
        ##### input features
        # 1. x = matrix state 展平，应该很垃圾
        # 2. x = image 
        # 3. x = seq state
        # 4. x = image 加上fanin fanout 信息
        ##### labels 
        # 1. ppa 绝对值
        # 2. scale ppa delta reward model
        # 3. 分类任务

        dict_dataset = self.process_data(raw_data)
        self.dict_dataset = dict_dataset

        # TO MODIFY
        train_x, test_x, train_y1, test_y1, train_y2, test_y2 = self.get_split_dataset(dict_dataset)

        return train_x, test_x, train_y1, test_y1, train_y2, test_y2

    def _std_normalize(self, y):
        y_mean = np.mean(y)
        y_std = np.std(y)

        y = (y - y_mean) / (y_std + 1e-4)

        return y

    def get_split_dataset(self, dict_dataset):
        x = np.stack(dict_dataset[self.x_key], axis=0)
        y1 = np.stack(dict_dataset[self.y1_key], axis=0)
        y2 = np.stack(dict_dataset[self.y2_key], axis=0)

        indexes = np.arange(x.shape[0])
        if self.random_shuffle:
            np.random.shuffle(indexes)
        num_train = int(self.split_ratio[0] * x.shape[0]) 
        print(f"num_train: {num_train}")
        train_indexes = indexes[:num_train]
        test_indexes = indexes[num_train:]

        train_x = x[train_indexes]
        test_x = x[test_indexes]

        train_y1 = y1[train_indexes]
        test_y1 = y1[test_indexes]
        train_y1 = np.expand_dims(train_y1, axis=1)
        test_y1 = np.expand_dims(test_y1, axis=1)

        train_y2 = y2[train_indexes]
        test_y2 = y2[test_indexes]
        train_y2 = np.expand_dims(train_y2, axis=1)
        test_y2 = np.expand_dims(test_y2, axis=1)

        return train_x, test_x, train_y1, test_y1, train_y2, test_y2

    def _get_image_state(self, ct32, ct22, stage_num):
        ct32 = np.expand_dims(ct32, axis=0)
        ct22 = np.expand_dims(ct22, axis=0)
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = np.zeros((1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2)))
            ct32 = np.concatenate((ct32, zeros), axis=1)
            ct22 = np.concatenate((ct22, zeros), axis=1)
        image_state = np.concatenate((ct32, ct22), axis=0) # (2, max_stage-1, num_column)        
        return image_state

    def _get_ct_delay(self, ct32, ct22):
        ct_delay = np.zeros_like(ct32) # (stage_num, num_column)
        for i in range(1, ct32.shape[0]):
            # iterate over stage
            for j in range(ct32.shape[1]):
                # iterate over column
                if j == 0:
                    if ct32[i-1,j] > 0:
                        # 32 sum two delay
                        ct_delay[i,j] = ct_delay[i-1,j] + 2
                    elif ct22[i-1,j] > 0:
                        # 22 sum one delay
                        ct_delay[i,j] = ct_delay[i-1,j] + 1
                    else:
                        # no added delay
                        ct_delay[i,j] = ct_delay[i-1,j] + 0
                else:
                    if ct32[i-1,j] > 0:
                        current_sum = 2
                    elif ct22[i-1,j] > 0:
                        current_sum = 1
                    else:
                        current_sum = 0
                    if ct32[i-1,j-1] > 0 or ct22[i-1,j-1] > 0:
                        last_carry = 1
                    else:
                        last_carry = 0
                    if last_carry == 0:
                        ct_delay[i,j] = ct_delay[i-1,j] + current_sum
                    else:
                        ct_delay[i,j] = max(
                            ct_delay[i-1,j] + current_sum,
                            ct_delay[i-1,j-1] + last_carry
                        )
        return ct_delay
    
    def _get_image_state_v2(self, ct32, ct22, stage_num):
        ct_delay = self._get_ct_delay(ct32, ct22)
        ct32 = np.expand_dims(ct32, axis=0)
        ct22 = np.expand_dims(ct22, axis=0)
        ct_delay = np.expand_dims(ct_delay, axis=0)
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = np.zeros((1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2)))
            ct32 = np.concatenate((ct32, zeros), axis=1)
            ct22 = np.concatenate((ct22, zeros), axis=1)
            ct_delay = np.concatenate((ct_delay, zeros), axis=1)
        ct_area = 3 * ct32 + 2 * ct22
        image_state_v2 = np.concatenate((ct32, ct22, ct_area, ct_delay), axis=0) # (2, max_stage-1, num_column)        
        return image_state_v2

    def _get_estimated_delay(self, ct32_i, ct22_i):
        try:
            nonzero32 = list(np.nonzero(ct32_i))[-1]
            min_delay_32 = nonzero32[-1]
        except:
            # print(f"warning!!! ct32_i zero: {ct32_i}")
            min_delay_32 = 0
        
        try:
            nonzero31 = list(np.nonzero(ct22_i))[-1]
            min_delay_22 = nonzero31[-1]
        except:
            # print(f"warning!!! ct22_i zero: {ct22_i}")
            min_delay_22 = 0
        
        return max(min_delay_32, min_delay_22)
    
    def _get_seq_state(self, state, state_mask, ct32, ct22):
        # input matrix state and state mask
        # output sequence state
        """
            state feature vector definition
            [
                pp(1), position(1), mask(4), 
                cur_column: num32, num22, estimated area, estimated delay,
                last_column: (4)
                next_column: (4)
            ]
            state feature vector definition
            [
                mask(4), 
                cur_column: num32, num22, estimated area, estimated delay,
                ct32 each stage
                ct22 each stage
            ]
        """
        num_column = state.shape[1]
        state_features = np.zeros(
            (num_column, ColumnFeatureNum)
        )
        for i in range(num_column):
            # # pp
            # state_features[i,0] = initial_partial_product[i]
            # # position
            # state_features[i,1] = i
            # mask
            cur_column_mask = state_mask[4*i:4*(i+1)]
            state_features[i,0:4] = np.array(cur_column_mask, dtype=np.float)
            
            # column features
            state_features[i,4] = state[0,i]
            state_features[i,5] = state[1,i]
            state_features[i,6] = 3*state[0,i] + 2*state[1,i]
            # i-th column 32 delay
            estimated_delay = self._get_estimated_delay(ct32[:,i], ct22[:,i])
            state_features[i,7] = estimated_delay

        return state_features
    
    def process_data(self, raw_data):
        dict_dataset = {
            "matrix_state": [],
            "image_state": [],
            "image_state_v2": [],
            "seq_state": [],
            "mannual_state": [],
            "matrix_next_state": [],
            "image_next_state": [],
            "image_next_state_v2": [],
            "seq_next_state": [],
            "mannual_next_state": [],
            "action": [],
            "ppa": [],
            "area": [],
            "delay": [],
            "delta_ppa": [],
            "delta_ppa_class": [],
            "next_state_stage_num": [],
            "next_state_node_num_v1": [],
            "next_state_node_num_v2": [],
            "next_state_max_delay": [],
            "next_state_avg_delay": [],
            "normalize_area": [],
            "normalize_delay": [],
            "normalize_node_num_v2": [],
            "normalize_stage_num": []      
        }

        for transition_data in raw_data:
            # 1. matrix state
            dict_dataset["matrix_state"].append(
                transition_data.state.flatten().numpy()
            )
            dict_dataset["matrix_next_state"].append(
                transition_data.next_state.flatten().numpy()
            )
            # 2. image state
            ct32_state = transition_data.state_ct32
            ct22_state = transition_data.state_ct22
            stage_num = ct32_state.shape[0] - 1
            # print(f"stage_num: {stage_num}")
            image_state = self._get_image_state(
                ct32_state, ct22_state, stage_num
            )
            dict_dataset["image_state"].append(
                image_state
            )
            ct32_next_state = transition_data.next_state_ct32
            ct22_next_state = transition_data.next_state_ct22
            stage_num_next_state = ct32_next_state.shape[0] - 1
            # print(f"stage_num_next_state: {stage_num_next_state}")
            image_next_state = self._get_image_state(
                ct32_next_state, ct22_next_state, stage_num_next_state
            )
            dict_dataset["image_next_state"].append(
                image_next_state
            )
            # 22. image state with area delay information
            image_state_v2 = self._get_image_state_v2(
                ct32_state, ct22_state, stage_num
            )
            dict_dataset["image_state_v2"].append(
                image_state_v2
            )
            image_next_state_v2 = self._get_image_state_v2(
                ct32_next_state, ct22_next_state, stage_num_next_state
            )
            dict_dataset["image_next_state_v2"].append(
                image_next_state_v2
            )
            # next state stage num, node num, max delay
            dict_dataset["next_state_stage_num"].append(
                stage_num_next_state
            )
            dict_dataset["normalize_stage_num"].append(
                stage_num_next_state * self.ppa_scale / self.wallace_stage_num
            )
            
            node_num_v1 = ct32_next_state.sum() + ct22_next_state.sum()
            node_num_v2 = ct32_next_state.sum() * 3 + ct22_next_state.sum() * 2
            max_delay = np.max(image_next_state_v2[3])
            mean_delay = np.mean(image_next_state_v2[3])
            dict_dataset["next_state_node_num_v1"].append(
                node_num_v1
            )
            dict_dataset["next_state_node_num_v2"].append(
                node_num_v2
            )
            dict_dataset["normalize_node_num_v2"].append(
                node_num_v2 * self.ppa_scale / self.wallace_node_num
            )
            
            dict_dataset["next_state_max_delay"].append(
                max_delay
            )
            dict_dataset["next_state_avg_delay"].append(
                mean_delay
            )
            # 3. seq state
            state = transition_data.state.squeeze().numpy()
            state_mask = transition_data.mask.squeeze().numpy()
            seq_state = self._get_seq_state(
                state, state_mask, ct32_state, ct22_state
            )
            dict_dataset["seq_state"].append(
                seq_state
            )
            # 4. feature vector
            # [stage_num, node_num, number of 32, number of 22]
            mannual_state = np.array(
                [
                    stage_num,
                    3*ct32_state.sum() + 2*ct22_state.sum(),
                    ct32_state.sum(),
                    ct22_state.sum()
                ]
            )
            dict_dataset["mannual_state"].append(
                mannual_state
            )
            # 5. reward
            dict_dataset["delta_ppa"].append(
                transition_data.reward.item() * 10
            )
            # 6. reward class
            if transition_data.reward.item() >= 0:
                dict_dataset["delta_ppa_class"].append(
                1.
            )    
            else:
                dict_dataset["delta_ppa_class"].append(
                0.
            )    
            # 7. action type
            action = int(transition_data.action)
            dict_dataset["action"].append(action)
            # 8. ppa, area, delay
            ppa = transition_data.rewards_dict["avg_ppa"]
            area = np.mean(transition_data.rewards_dict["area"])
            delay = np.mean(transition_data.rewards_dict["delay"])
            dict_dataset["ppa"].append(ppa)
            dict_dataset["area"].append(area)
            dict_dataset["normalize_area"].append(area * self.ppa_scale / self.wallace_area)
            dict_dataset["delay"].append(delay)
            dict_dataset["normalize_delay"].append(delay * self.ppa_scale / self.wallace_delay)
            
        return dict_dataset

if __name__ == "__main__":
    
    # npy_data_path = "outputs/2023-11-11/00-43-49/logger_log/dqn_16bits/dqn16bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool/dqn16bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool_2023_11_11_00_43_56_0000--s-1/itr_2500.npy"

    # # npy_data_path = "outputs/2023-11-11/09-48-44/logger_log/dqn_16bits/dqn16bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool/dqn16bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool_2023_11_11_09_48_50_0000--s-1/itr_3750.npy"

    # factor action sample 
    # v1 synthesis 
    npy_data_path = "../outputs/2024-03-12/14-08-16/logger_log/dqn_16bits_factor_action/dqn16bits_reset_rnd_factor_speedup_v1syn/dqn16bits_reset_rnd_factor_speedup_v1syn_2024_03_12_14_08_22_0000--s-1/itr_10000.npy"
    data_loader = ScoreDataLoader(
        npy_data_path,
        wallace_area=2064.5,
        wallace_delay=1.33,
        wallace_node_num=719,
        wallace_stage_num=5
    )
    
    # v2 synthesis
    # npy_data_path = "../outputs/2024-03-12/14-09-47/logger_log/dqn_16bits_factor_action/dqn16bits_reset_rnd_factor_speedup_v2syn/dqn16bits_reset_rnd_factor_speedup_v2syn_2024_03_12_14_09_53_0000--s-1/itr_10000.npy"
    # data_loader = ScoreDataLoader(
    #     npy_data_path,
    #     wallace_area=2576.5,
    #     wallace_delay=0.90345,
    #     wallace_node_num=719,
    #     wallace_stage_num=5
    # )