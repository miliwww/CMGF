import json
import os
import sys

import numpy as np
import random
import math

import numpy.random
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.io import savemat
import spectral
import torch
import torch.utils.data as dataf
import torch.nn as nn
#import matplotlib.pyplot as plt
from scipy import io
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
import gc
import time
import torch.nn.functional as F
from pymodel import pyCNN
from data_prepare import data_load, nor_pca, border_inter, con_data, getIndex, con_data_all, con_data2, con_data_even, \
    con_data2_even, con_data_all, random_occlusion_and_blur_pair_advanced, random_occlusion_and_blur_pair


# setting parameters
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

batchsize = 64
EPOCH = 200
LR = 0.0005
#LR = 0.00025
# dataset_name = "Houston"
dataset_name = "Trento"
# dataset_name = "MUUFL"
# dataset_name = "Augsburg"

for run in range(1):
    print("+"*30,"run:",run,"+"*30)

    gc.collect()
    torch.cuda.empty_cache()
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # load data
    (Data,Data2,TrLabel,TsLabel,additive_data,deadlines_data,
     kernal_data,poisson_data,salt_pepper_data,stripes_data,zmguass_data)= data_load(name=dataset_name)
    # TrLabel = small_sample(TrLabel, radito=0.2)
    img_row = len(Data2)
    img_col = len(Data2[0])
    print(f"Data 原始形状: {Data.shape}")
    print(f"Data 总元素数: {Data.size}")

    # normalization method 1: map to [0, 1]
    [m, n, l] = Data.shape
    PC,Data2,NC = nor_pca(Data,Data2,ispca=True)

    # boundary interpolation
    x, x2 = border_inter(PC,Data2,NC)
    # construct the training and testing set of HSI
    TrainPatch,TestPatch,TrainPatch2,TestPatch2,TrainLabel,TestLabel,TrainLabel2,TestLabel2 = con_data(x,x2,TrLabel,TsLabel,NC)
   # 数据增强
   #  TrainPatch, TrainLabel = \
   #      random_occlusion_and_blur_pair_advanced(
   #          TrainPatch,
   #          TrainLabel,
   #          occlusion_prob=0.8,  # 应用遮挡概率  Trento 0.8
   #          occlusion_size_range=(0.15, 0.25),  # 遮挡区域占原图的10%-25%
   #          blur_prob=0.5,  # 50%概率应用高斯模糊
   #          sigma_range=(0.5, 1.5)
   #      )

    input_data = [additive_data, deadlines_data, kernal_data, stripes_data,poisson_data, salt_pepper_data,  zmguass_data]
    noise_name = ['additive_data', 'deadlines_data', 'kernal_data','stripes_data', 'poisson_data', 'salt_pepper_data',  'zmguass_data']

    # step3: change data to the input type of PyTorch (tensor)
    TrainPatch1 = torch.from_numpy(TrainPatch)
    TrainLabel1 = torch.from_numpy(TrainLabel)-1
    TrainLabel1 = TrainLabel1.long()

    TestPatch1 = torch.from_numpy(TestPatch)
    TestLabel1 = torch.from_numpy(TestLabel)-1
    TestLabel1 = TestLabel1.long()
    Classes = len(np.unique(TrainLabel))

    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainLabel2 = torch.from_numpy(TrainLabel2)-1
    TrainLabel2 = TrainLabel2.long()

    dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel2)
    train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)
    TestPatch2 = torch.from_numpy(TestPatch2)
    TestLabel2 = torch.from_numpy(TestLabel2)-1
    TestLabel2 = TestLabel2.long()

    FM = 64
    # 定义可学习权重（单个参数）
    weight_raw = nn.Parameter(torch.tensor([0.5, 0.5], requires_grad=True))
    class infonce_loss(nn.Module):
        def __init__(self,temperature=0.07):
            super().__init__()
            self.temperature = temperature

        def NT_XentLoss(self,z1, z2, temperature=0.07):
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            N, Z = z1.shape
            #device = z1.device
            representations = torch.cat([z1, z2], dim=0)
            similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
            l_pos = torch.diag(similarity_matrix, N)
            r_pos = torch.diag(similarity_matrix, -N)
            positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
            diag = torch.eye(2 * N, dtype=torch.bool)
            diag[N:, :N] = diag[:N, N:] = diag[:N, :N]

            negatives = similarity_matrix[~diag].view(2 * N, -1)

            logits = torch.cat([positives, negatives], dim=1).to('cuda')
            logits /= temperature

            labels = torch.zeros(2 * N, dtype=torch.int64).to('cuda')

            loss = F.cross_entropy(logits, labels, reduction='sum')
            return loss / (2 * N)

        def forward(self, z1,z2):

            # 计算损失
            loss = self.NT_XentLoss(z1, z2)
            #print(f"InfoNCE Loss: {loss.item()}")
            return loss


    cnn = pyCNN(FM=FM,NC=NC,Classes=Classes)
    # move model to GPU
    cnn.cuda()

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    total_params, trainable_params = count_parameters(cnn)

    print("=" * 60)
    print("Model Architecture Summary")
    print(cnn)
    print("-" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 60)

    print("=" * 60)
    print("Parameter count by module")
    for name, module in cnn.named_children():
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:20s}: {module_params:,}")
    print("=" * 60)

    print("=" * 60)
    print("Architecture-level Hyperparameters")
    print(f"Dataset: {dataset_name}")
    print(f"Input patch size (HSI): {TrainPatch1.shape[2]} x {TrainPatch1.shape[3]}")
    print(f"Input patch size (LiDAR): {TrainPatch2.shape[2]} x {TrainPatch2.shape[3]}")
    print(f"Number of PCA components (NC): {NC}")
    print(f"Feature maps (FM): {FM}")
    print(f"Number of classes: {Classes}")
    print(f"Batch size: {batchsize}")
    print(f"Epochs: {EPOCH}")
    print(f"Learning rate: {LR}")
    print("=" * 60)

    optimizer = torch.optim.Adam([    {'params': cnn.parameters()},    {'params': [weight_raw]} ], lr=LR)
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    infonce_loss = infonce_loss()
    BestAcc = 0
    pred_img = TsLabel
    torch.cuda.synchronize()
    start = time.time()

    # train and test the designed model
    for epoch in range(EPOCH):

        for step, (b_x1, b_x2, b_y) in enumerate(train_loader):

            # move train data to GPU
            b_x1 = b_x1.cuda()
            b_x2 = b_x2.cuda()
            b_y = b_y.cuda()
            out1, out2, out3,input_CMFPG, output_CMFPG, in_1, in_2, in_3, out_1, out_2, out_3= cnn(b_x1, b_x2)
            del input_CMFPG, output_CMFPG, in_1, in_2, in_3, out_1, out_2, out_3

            loss1 = infonce_loss(out1, out2)
            loss3 = loss_func(out3, b_y)

            weight_celoss, weight_infonce = torch.softmax(weight_raw, dim=0)
            loss = weight_celoss * loss3 + weight_infonce * loss1

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            #print(f"weight_celoss: {weight_celoss.item(), loss3.item()},weight_infonce: {weight_infonce.item(), loss1.item()}")
            del out1, out2, out3

            if step % 50 == 0:
                cnn.eval()
                del b_x1, b_x2, b_y
                # temp1 = TrainPatch1
                # temp1 = temp1.cuda()
                # temp2 = TrainPatch2
                # temp2 = temp2.cuda()
                # temp3, temp4, temp5 = cnn(temp1, temp2)
                Classes = np.unique(TrainLabel1)
                pred_y = np.empty((len(TestLabel)), dtype='float32')
                number = len(TestLabel) // 1000
                for i in range(number):
                    temp = TestPatch1[i * 1000:(i + 1) * 1000, :, :, :]
                    temp = temp.cuda()
                    temp1 = TestPatch2[i * 1000:(i + 1) * 1000, :, :, :]
                    temp1 = temp1.cuda()
                    #temp2 = cnn(temp, temp1)[2] + cnn(temp, temp1)[1] + cnn(temp, temp1)[0]
                    temp2 = cnn(temp,temp1)[2]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[i * 1000:(i + 1) * 1000] = temp3.cpu().detach().numpy().astype('float32')
                    del temp, temp1, temp2, temp3

                if (i + 1) * 1000 < len(TestLabel):
                    temp = TestPatch1[(i + 1) * 1000:len(TestLabel), :, :, :]
                    temp = temp.cuda()
                    temp1 = TestPatch2[(i + 1) * 1000:len(TestLabel), :, :, :]
                    temp1 = temp1.cuda()
                    #temp2 = cnn(temp, temp1)[2] + cnn(temp, temp1)[1] + cnn(temp, temp1)[0]
                    temp2 = cnn(temp, temp1)[2]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[(i + 1) * 1000:len(TestLabel)] = temp3.cpu().detach().numpy().astype('float32')
                    del temp, temp1, temp2, temp3

                pred_y = torch.from_numpy(pred_y).long()
                print(np.unique(pred_y))
                print(np.unique(TestLabel1))
                accuracy = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.6f' % accuracy,'| ')

                # save the parameters in network
                if accuracy > BestAcc:
                    torch.save(cnn.state_dict(), 'BestAcc.pkl')
                    BestAcc = accuracy

                del pred_y, accuracy
                cnn.train()


    print('Best test acc:',BestAcc)
    torch.cuda.synchronize()
    end = time.time()
    Train_time = end - start

    # load the saved parameters
    cnn.load_state_dict(torch.load('BestAcc.pkl', weights_only=True))
    cnn.eval()
    torch.cuda.synchronize()

    from thop import profile, clever_format
    # 构造一个假的输入（batch=1 即可）
    dummy_hsi = torch.randn(1, TrainPatch1.shape[1],
                            TrainPatch1.shape[2],
                            TrainPatch1.shape[3]).cuda()

    dummy_lidar = torch.randn(1, TrainPatch2.shape[1],
                              TrainPatch2.shape[2],
                              TrainPatch2.shape[3]).cuda()
    # 计算 FLOPs 和参数量
    flops, params = profile(
        cnn,
        inputs=(dummy_hsi, dummy_lidar),
        verbose=False
    )
    flops, params = clever_format([flops, params], "%.3f")

    print("=" * 60)
    print("Computational Cost")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    print("=" * 60)


    print('-'*30,'Test','-'*30)
    start = time.time()

    pred_y = np.empty((len(TestLabel)), dtype='float32')
    number = len(TestLabel)//1000
    for i in range(number):
        temp = TestPatch1[i*1000:(i+1)*1000, :, :]
        temp = temp.cuda()
        temp1 = TestPatch2[i*1000:(i+1)*1000, :, :]
        temp1 = temp1.cuda()
        #temp2 =  1*cnn(temp, temp1)[2] +  0.01*cnn(temp, temp1)[1] +  0.01*cnn(temp, temp1)[0]
        temp2 = cnn(temp, temp1)[2]
        temp2_p = temp2.data
        # temp2 = cnn(temp, temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[i*1000:(i+1)*1000] = temp3.cpu().detach().numpy().astype('float32')
        del temp, temp2, temp3

    if (i+1)*1000 < len(TestLabel):
        temp = TestPatch1[(i+1)*1000:len(TestLabel), :, :]
        temp = temp.cuda()
        temp1 = TestPatch2[(i+1)*1000:len(TestLabel), :, :]
        temp1 = temp1.cuda()
        #temp2 = 1*cnn(temp, temp1)[2] + 0.01*cnn(temp, temp1)[1] + 0.01*cnn(temp, temp1)[0]
        temp2 = cnn(temp, temp1)[2]
        temp2_p = temp2.data
        # temp2 = cnn(temp, temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i+1)*1000:len(TestLabel)] = temp3.cpu().detach().numpy().astype('float32')
        del temp, temp2, temp3, temp2_p

    pred_y = torch.from_numpy(pred_y).long()
    print(np.unique(pred_y))
    print(np.unique(TestLabel1))
    OA = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)
    oa = OA.numpy()

    Classes = np.unique(TestLabel1)
    EachAcc = np.empty(len(Classes))
    pe = 0
    for i in range(len(Classes)):
        cla = Classes[i]
        right = 0
        sum = 0

        for j in range(len(TestLabel1)):
            if TestLabel1[j] == cla:
                sum += 1
            if TestLabel1[j] == cla and pred_y[j] == cla:
                right += 1
        pe += sum*right
        EachAcc[i] = right.__float__()/sum.__float__()

    AA = np.sum(EachAcc)/len(Classes)
    pe = pe / math.pow(TestLabel1.size(0), 2)
    kappa = (oa-pe)/(1-pe)
    print(dataset_name)
    print("batchsize: ",batchsize)
    print("LR: ",LR)
    print("OA:  ", OA)
    print("oa:  ", oa)
    print("EachAcc:  ", EachAcc)
    print("AA:  ", AA)
    print("kappa:  ", kappa)

    class_acc_list = EachAcc.tolist()
    class_acc_compact = "[" + ", ".join([f"{x:.6f}" for x in class_acc_list]) + "]"
    report1 = { "eval":{
                            "The Training time is: ":str(Train_time/60),
                            "name":dataset_name,
                            "batchsize":batchsize,
                            "epoch":EPOCH,
                            "LR":LR,
                            "seed":seed,
                            "Best test acc":BestAcc.item(),
                            "oa":oa.item(),
                            "aa":AA.item(),
                            "kappa":kappa.item(),
                            "classification":class_acc_compact,
                        }
                    }
    results = {
        "report1": report1,  # 假设report1已定义
        "report2_by_noise": {}  # 用于存储循环中的report2
    }

    torch.cuda.synchronize()
    end = time.time()
    print(end - start)
    Test_time = end - start
    print('The Training time is: ', Train_time)
    print('The Test time is: ', Test_time)

    # savemat("./png/predHouston.mat", {'pred':pred_y1})
    # savemat("./png/indexHouston.mat", {'index':index})
    print()
    print('Training size and testing size of HSI are:', TrainPatch.shape, 'and', TestPatch.shape)
    print('Training size and testing size of LiDAR are:', TrainPatch2.shape, 'and', TestPatch2.shape)
    # del TestPatch1,  TrainPatch1, TrainPatch2, TrainLabel1, TrainLabel2
    #print('Noise data size is:',noise_data[1].shape)

    AllLabel = TrLabel+TsLabel
    #全图成图
    AllPatch1, AllPatch2, allLabel, allLabel2 = con_data_all(x, x2, AllLabel, NC)
    AllPatch1 = torch.from_numpy(AllPatch1)
    allLabel = torch.from_numpy(allLabel)-1
    allLabel = allLabel.long()#torch.long 是 64 位有符号整数类型
    AllPatch2 = torch.from_numpy(AllPatch2)

    pred_y = np.empty((len(allLabel)), dtype='float32')
    number = len(allLabel) // 1000
    for i in range(number):
        temp = AllPatch1[i * 1000:(i + 1) * 1000, :, :]
        temp = temp.cuda()
        temp1 = AllPatch2[i * 1000:(i + 1) * 1000, :, :]
        temp1 = temp1.cuda()
        # temp2 =  1*cnn(temp, temp1)[2] +  0.01*cnn(temp, temp1)[1] +  0.01*cnn(temp, temp1)[0]
        temp2 = cnn(temp, temp1)[2]
        temp2_p = temp2.data
        # temp2 = cnn(temp, temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[i * 1000:(i + 1) * 1000] = temp3.cpu().detach().numpy().astype('float32')
        del temp, temp2, temp3

    if (i + 1) * 1000 < len(allLabel):
        temp = AllPatch1[(i + 1) * 1000:len(allLabel), :, :]
        temp = temp.cuda()
        temp1 = AllPatch2[(i + 1) * 1000:len(allLabel), :, :]
        temp1 = temp1.cuda()
        # temp2 = 1*cnn(temp, temp1)[2] + 0.01*cnn(temp, temp1)[1] + 0.01*cnn(temp, temp1)[0]
        temp2 = cnn(temp, temp1)[2]
        temp2_p = temp2.data
        # temp2 = cnn(temp, temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i + 1) * 1000:len(allLabel)] = temp3.cpu().detach().numpy().astype('float32')
        del temp, temp2, temp3

    pred_y = torch.from_numpy(pred_y).long()
    del AllPatch1, AllPatch2, allLabel, allLabel2


    def apply_prediction_to_mask(pred_y, pred_img, save_path="prediction_overlay.png", cmap="viridis"):
        """
        将pred_y的值赋给pred_img中大于零的位置，并将结果可视化

        参数:
        pred_y: 预测结果 (1D numpy array 或 torch tensor)
        pred_img: 原始掩码图像 (numpy array)
        save_path: 保存路径
        cmap: 颜色映射
        """
        # 确保pred_y是numpy数组
        if isinstance(pred_y, torch.Tensor):
            pred_y = pred_y.cpu().numpy()

        # 创建结果数组的副本，避免修改原始数据
        result = pred_img.copy()

        # 获取pred_img中大于零的位置的索引
        mask_indices = np.where(pred_img > 0)

        # 确保pred_y的长度与mask中的非零元素数量匹配
        if len(pred_y) != len(mask_indices[0]):
            raise ValueError(
                f"pred_y的长度({len(pred_y)})与mask中大于零的元素数量({len(mask_indices[0])})不匹配")

        # 将pred_y的值赋给mask中大于零的位置
        result[mask_indices] = pred_y + 1

        # 可视化原图
        plt.figure(figsize=(8, 6), dpi=600)
        # 创建颜色映射
        #muufl
        # colors = [
        #     'whitesmoke',  # Unclassified: 浅灰色
        #     'forestgreen',  # Trees: 深绿色，代表茂密的树木
        #     'yellowgreen',  # Grass: 浅绿色，代表草地
        #     'tan',  # Mixed ground surface: 棕褐色，代表混合地表
        #     'sandybrown',  # Sand: 沙褐色，代表沙地
        #     'lightsteelblue',  # Road: 深灰色，代表道路
        #     'dodgerblue',  # Water: 蓝色，代表水体
        #     'dimgray',  # Building shadow: 深灰色，代表建筑物阴影
        #     'orchid',  # Buildings: 砖红色，代表建筑物
        #     'lightgrey',  # Sidewalk: 浅灰色，代表人行道
        #     'gold',  # Yellow curb: 金色，代表黄色路缘
        #     'mediumpurple'  # Cloth panels: 中紫色，代表布面板
        # ]
        # class_labels = [
        #     'Unclassified', 'Trees', 'Grass', 'Mixed ground surface',
        #     'Sand', 'Road', 'Water', 'Building shadow', 'Buildings',
        #     'Sidewalk', 'Yellow curb', 'Cloth panels'
        # ]

        #Augsburg
        colors = [
            'whitesmoke',  # Unclassified:  oldlace
            'forestgreen',  # Forest:
            'orchid',  # Residential Area: 居住区域
            'lightsteelblue',  # Industrial Area: 现代工业感
            'yellowgreen',  # Low Plants: 植被
            'darkorange',  # Allotment: 农业用地
            'orangered',  # Commercial Area: 商业区域
            'dodgerblue'  # Water:
        ]
        class_labels = [
            'Unclassified', 'Trees', 'Grass', 'Mixed ground surface',
            'Sand', 'Road', 'Water', 'Building shadow'
        ]
        # Trento
        # colors = [
        #     'whitesmoke',  # Background: 背景
        #     'yellowgreen',  # Apple trees: 苹果树
        #     'orchid',  # Buildings: 建筑物
        #     'tan',  # Ground: 地面
        #     'forestgreen',  # Wood: 林地
        #     'mediumpurple',  # Vineyard: 葡萄园
        #     'lightsteelblue',  # Roads: 道路
        # ]
        #
        # class_labels = [
        #     'Background',
        #     'Apple trees',
        #     'Buildings',
        #     'Ground',
        #     'Wood',
        #     'Vineyard',
        #     'Roads',
        # ]
        cmap = ListedColormap(colors)
        plt.imshow(pred_img, cmap=cmap)
        print(np.unique(pred_img))

        # 移除刻度和标签
        plt.xticks([])
        plt.yticks([])
        plt.gca().axis('off')

        # 保存图像
        plt.savefig('./Result_Plot/gt_Ours', bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"结果已保存至: {'pred_img'}")

        # 可视化结果
        plt.figure(figsize=(8, 6), dpi=600)
        plt.imshow(result, cmap=cmap, vmin=0, vmax=7)#11,7
        print(np.unique(result))

        # 生成动态图例（仅显示数据中存在的类别）
        unique_values = np.unique(result)  # 此时result是整数数组，unique_values为整数
        handles, labels = [], []
        for val in unique_values:
            val = int(val)  # 将val转换为整数
            if 0 <= val < len(class_labels):  # 仅处理0~11的有效类别
                handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[val]))
                labels.append(class_labels[val])
            elif val == 8:  # 若存在超出范围的值（如pred_y=11时+1=12），可自定义处理
                handles.append(plt.Rectangle((0, 0), 1, 1, color='gray'))  # 无对应颜色时设为灰色
                labels.append('Unknown Class')  # 或抛出警告

        # 移除刻度和标签
        plt.xticks([])
        plt.yticks([])
        plt.gca().axis('off')

        # 保存图像
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"结果已保存至: {save_path}")
        del result, pred_y, pred_img


    pred_img = AllLabel  # 原始掩码图像
    save_path = "./Result_Plot/prediction_overlay.png"  # 保存路径
    result_img = apply_prediction_to_mask(pred_y, pred_img, save_path=save_path)

    #for data in input_data:
    for i, data in enumerate(input_data):
        PC, _ ,NC= nor_pca(data,Data2,ispca=True)
        data, _ = border_inter(PC,Data2,NC)

        TestPatch1 = con_data2(data,TsLabel,NC)
        name = noise_name[i]
        print('-' * 30, '%s %s data test'%(dataset_name,name), '-' * 30)
        start = time.time()
        TestPatch1 = torch.from_numpy(TestPatch1)

        pred_y = np.empty((len(TestLabel)), dtype='float32')
        number = len(TestLabel)//1000
        c = 0
        Input_CMFPG = None
        Output_CMFPG = None
        Input_1 = None
        Input_2 = None
        Input_3 = None
        Output_1 = None
        Output_2 = None
        Output_3 = None

        for i in range(number):
            temp = TestPatch1[i*1000:(i+1)*1000, :, :]
            temp = temp.cuda()
            temp1 = TestPatch2[i*1000:(i+1)*1000, :, :]
            temp1 = temp1.cuda()
            #temp2 =  1*cnn(temp, temp1)[2] +  0.01*cnn(temp, temp1)[1] +  0.01*cnn(temp, temp1)[0]
            temp2,input_CMFPG, output_CMFPG, in_1, in_2, in_3, out_1, out_2, out_3= cnn(temp, temp1)[2:]
            temp2_p = temp2.data
            # temp2 = cnn(temp, temp1)
            temp3 = torch.max(temp2, 1)[1].squeeze()
            pred_y[i*1000:(i+1)*1000] = temp3.cpu().detach().numpy().astype('float32')
            del temp, temp2, temp3
            # if c == 0:
            #     Input_CMFPG = input_CMFPG.cpu().detach()
            #     Output_CMFPG = output_CMFPG.cpu().detach()
            #     Input_1 = in_1.cpu().detach()
            #     Input_2 = in_2.cpu().detach()
            #     Input_3 = in_3.cpu().detach()
            #     Output_1 = out_1.cpu().detach()
            #     Output_2 = out_2.cpu().detach()
            #     Output_3 = out_3.cpu().detach()
            #
            #     c = 1
            # else:
            #     Input_CMFPG = torch.cat((Input_CMFPG, input_CMFPG.cpu().detach()), dim=0)
            #     Output_CMFPG = torch.cat((Output_CMFPG, output_CMFPG.cpu().detach()), dim=0)
            #     Input_1 = torch.cat((Input_1, in_1.cpu().detach()), dim=0)
            #     Input_2 = torch.cat((Input_2, in_2.cpu().detach()), dim=0)
            #     Input_3 = torch.cat((Input_3, in_3.cpu().detach()), dim=0)
            #     Output_1 = torch.cat((Output_1, out_1.cpu().detach()), dim=0)
            #     Output_2 = torch.cat((Output_2, out_2.cpu().detach()), dim=0)
            #     Output_3 = torch.cat((Output_3, out_3.cpu().detach()), dim=0)


        if (i+1)*1000 < len(TestLabel):
            temp = TestPatch1[(i+1)*1000:len(TestLabel), :, :]
            temp = temp.cuda()
            temp1 = TestPatch2[(i+1)*1000:len(TestLabel), :, :]
            temp1 = temp1.cuda()
            #temp2 = 1*cnn(temp, temp1)[2] + 0.01*cnn(temp, temp1)[1] + 0.01*cnn(temp, temp1)[0]
            temp2, input_CMFPG, output_CMFPG, in_1, in_2, in_3, out_1, out_2, out_3= cnn(temp, temp1)[2:]
            temp2_p = temp2.data
            # temp2 = cnn(temp, temp1)
            temp3 = torch.max(temp2, 1)[1].squeeze()
            pred_y[(i+1)*1000:len(TestLabel)] = temp3.cpu().detach().numpy().astype('float32')
            del temp, temp2, temp3
            # Input_CMFPG = torch.cat((Input_CMFPG, input_CMFPG.cpu().detach()), dim=0)
            # Output_CMFPG = torch.cat((Output_CMFPG, output_CMFPG.cpu().detach()), dim=0)
            # Input_1 = torch.cat((Input_1, in_1.cpu().detach()), dim=0)
            # Input_2 = torch.cat((Input_2, in_2.cpu().detach()), dim=0)
            # Input_3 = torch.cat((Input_3, in_3.cpu().detach()), dim=0)
            # Output_1 = torch.cat((Output_1, out_1.cpu().detach()), dim=0)
            # Output_2 = torch.cat((Output_2, out_2.cpu().detach()), dim=0)
            # Output_3 = torch.cat((Output_3, out_3.cpu().detach()), dim=0)


        pred_y = torch.from_numpy(pred_y).long()
        print(pred_y.shape)
        print(TestLabel.shape)
        # np.save(r'E:\Green\MS2CANet_pro2\Au__MS2CANet_2improve\feature_map/gt_'+name+'.npy', TestLabel)
        # np.save(r'E:\Green\MS2CANet_pro2\Au__MS2CANet_2improve\feature_map/Input_CMFPG_'+name+'.npy', Input_CMFPG.numpy())
        # np.save(r'E:\Green\MS2CANet_pro2\Au__MS2CANet_2improve\feature_map/Output_CMFPG_'+name+'.npy', Output_CMFPG.numpy())
        # np.save(r'E:\Green\MS2CANet_pro2\Au__MS2CANet_2improve\feature_map/Input_1_'+name+'.npy', Input_1.numpy())
        # np.save(r'E:\Green\MS2CANet_pro2\Au__MS2CANet_2improve\feature_map/Input_2_'+name+'.npy', Input_2.numpy())
        # np.save(r'E:\Green\MS2CANet_pro2\Au__MS2CANet_2improve\feature_map/Input_3_'+name+'.npy', Input_3.numpy())
        # np.save(r'E:\Green\MS2CANet_pro2\Au__MS2CANet_2improve\feature_map/Output_1_'+name+'.npy', Output_1.numpy())
        # np.save(r'E:\Green\MS2CANet_pro2\Au__MS2CANet_2improve\feature_map/Output_2_'+name+'.npy', Output_2.numpy())
        # np.save(r'E:\Green\MS2CANet_pro2\Au__MS2CANet_2improve\feature_map/Output_3_'+name+'.npy', Output_3.numpy())
        del Input_CMFPG, Output_CMFPG, Input_1, Input_2, Input_3, Output_1, Output_2, Output_3

        OA = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)
        oa = OA.numpy()

        Classes = np.unique(TestLabel1)
        EachAcc = np.empty(len(Classes))
        pe = 0
        for i in range(len(Classes)):
            cla = Classes[i]
            right = 0
            sum = 0

            for j in range(len(TestLabel1)):
                if TestLabel1[j] == cla:
                    sum += 1
                if TestLabel1[j] == cla and pred_y[j] == cla:
                    right += 1
            pe += sum*right
            EachAcc[i] = right.__float__()/sum.__float__()

        AA = np.sum(EachAcc)/len(Classes)
        pe = pe / math.pow(TestLabel1.size(0), 2)
        kappa = (oa-pe)/(1-pe)
        TestPatch1=0
        #print(dataset_name)
        print("OA:  ", OA)
        print("oa:  ", oa)
        print("EachAcc:  ", EachAcc)
        print("AA:  ", AA)
        print("kappa:  ", kappa)
        torch.cuda.synchronize()
        end = time.time()
        #print(end - start)
        Test_time = end - start
        print('The Test time is: ', Test_time)

        # savemat("./png/predHouston.mat", {'pred':pred_y1})
        # savemat("./png/indexHouston.mat", {'index':index})
        print()
        class_acc_list = EachAcc.tolist()
        class_acc_compact = "[" + ", ".join([f"{x:.6f}" for x in class_acc_list]) + "]"
        report2 = {"test": {
            "oa": oa.item(),
            "aa": AA.item(),
            "kappa": kappa.item(),
            "classification": class_acc_compact,
        }
        }

        results["report2_by_noise"][name] = report2

        # 全图成图
        AllPatch1, AllPatch2, allLabel, allLabel2 = con_data_all(data, x2, AllLabel, NC)
        AllPatch1 = torch.from_numpy(AllPatch1)
        allLabel = torch.from_numpy(allLabel) - 1
        allLabel = allLabel.long()  # torch.long 是 64 位有符号整数类型
        AllPatch2 = torch.from_numpy(AllPatch2)

        pred_y = np.empty((len(allLabel)), dtype='float32')
        number = len(allLabel) // 1000
        for i in range(number):
            temp = AllPatch1[i * 1000:(i + 1) * 1000, :, :]
            temp = temp.cuda()
            temp1 = AllPatch2[i * 1000:(i + 1) * 1000, :, :]
            temp1 = temp1.cuda()
            # temp2 =  1*cnn(temp, temp1)[2] +  0.01*cnn(temp, temp1)[1] +  0.01*cnn(temp, temp1)[0]
            temp2 = cnn(temp, temp1)[2]
            temp2_p = temp2.data
            # temp2 = cnn(temp, temp1)
            temp3 = torch.max(temp2, 1)[1].squeeze()
            pred_y[i * 1000:(i + 1) * 1000] = temp3.cpu().detach().numpy().astype('float32')
            del temp, temp2, temp3

        if (i + 1) * 1000 < len(allLabel):
            temp = AllPatch1[(i + 1) * 1000:len(allLabel), :, :]
            temp = temp.cuda()
            temp1 = AllPatch2[(i + 1) * 1000:len(allLabel), :, :]
            temp1 = temp1.cuda()
            # temp2 = 1*cnn(temp, temp1)[2] + 0.01*cnn(temp, temp1)[1] + 0.01*cnn(temp, temp1)[0]
            temp2 = cnn(temp, temp1)[2]
            temp2_p = temp2.data
            # temp2 = cnn(temp, temp1)
            temp3 = torch.max(temp2, 1)[1].squeeze()
            pred_y[(i + 1) * 1000:len(allLabel)] = temp3.cpu().detach().numpy().astype('float32')
            del temp, temp2, temp3

        pred_y = torch.from_numpy(pred_y).long()
        del AllPatch1, AllPatch2, allLabel, allLabel2
        # 可视化预测结果
        pred_img = AllLabel  # 原始掩码图像
        save_path = "./Result_Plot/prediction_ours_" + name + ".png"  # 保存路径
        result_img = apply_prediction_to_mask(pred_y, pred_img, save_path=save_path)

    path = './res/' + dataset_name + '/'
    if os.path.exists(path) == False:
        os.mkdir(path)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    save_path_json = f"{path}{dataset_name}_{timestamp_str}.json"
    print("save_path_json:", save_path_json)
    # save_path_json = path + save_path_json
    ss = json.dumps(results)

    with open(save_path_json, 'w') as f:
        json.dump(results, f, indent=4)

    import winsound
    winsound.Beep(1000, 2000)