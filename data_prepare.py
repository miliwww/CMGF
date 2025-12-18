from scipy import io
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

patchsize1 = 33
patchsize2 = 33
batchsize = 8
# EPOCH = 200
# LR = 0.001
pad_width = np.floor(patchsize1 / 2)
pad_width = np.int32(pad_width)
pad_width2 = np.floor(patchsize2 / 2)
pad_width2 = np.int32(pad_width2)



def data_load(name="Trento", split_percent=0.2):

    if name == "Trento":
        # DataPath1 = './dataset/Trento/HSI.mat'
        # DataPath2 = './dataset/Trento/LiDAR.mat'
        # TRPath = './dataset/Trento/TRLabel.mat'
        # TSPath = './dataset/Trento/TSLabel.mat'
        #
        # TrLabel = io.loadmat(TRPath)
        # TsLabel = io.loadmat(TSPath)
        # TrLabel = TrLabel['TRLabel']
        # TsLabel = TsLabel['TSLabel']
        #
        # Data = io.loadmat(DataPath1)
        # Data = Data['HSI']
        # Data = Data.astype(np.float32)
        #
        # Data2 = io.loadmat(DataPath2)
        # Data2 = Data2['LiDAR']
        # Data2 = Data2.astype(np.float32)

    # elif name == "Augsburg":
    #     DataPath1 = './dataset/Augsburg/data_DSM.mat'
    #     DataPath2 = './dataset/Augsburg/data_HS_LR.mat'
    #     TRPath = './dataset/Augsburg/TrainImage.mat'
    #     TSPath = './dataset/Augsburg/TestImage.mat'
    #     TrLabel = io.loadmat(TRPath)
    #     TsLabel = io.loadmat(TSPath)
    #     TrLabel = TrLabel['TrainImage']
    #     TsLabel = TsLabel['TestImage']
    #
    #     Data2 = io.loadmat(DataPath1)
    #     Data2 = Data2['data_DSM']
    #     Data2 = Data2.astype(np.float32)
    #
    #     Data = io.loadmat(DataPath2)
    #     Data = Data['data_HS_LR']
    #     Data = Data.astype(np.float32)

    # elif name == "Houston":
    #     DataPath1 = './dataset/Houston2013/Houston_HS_HR.mat'
    #     DataPath2 = './dataset/Houston2013/Houston_DSM_HR.mat'
    #     TRPath = './dataset/Houston2013/Houston_train.mat'
    #     TSPath = './dataset/Houston2013/Houston_test.mat'
    #     TrLabel = io.loadmat(TRPath)
    #     TsLabel = io.loadmat(TSPath)
    #     print("Keys in TrLabel:", TrLabel.keys())
    #     print("Keys in TrLabel:", TsLabel.keys())
    #     TrLabel = TrLabel['TrainImage']
    #     TsLabel = TsLabel['TestImage']
    #
    #     Data = io.loadmat(DataPath1)
    #     print("Keys in Data:", Data.keys())
    #     Data = Data['data_HS_HR']
    #     Data = Data.astype(np.float32)# 转换为浮点数类型
    #
    #     Data2 = io.loadmat(DataPath2)
    #     print("Keys in Data2:", Data2.keys())
    #     Data2 = Data2['DSM']
    #     Data2 = Data2.astype(np.float32)
        DataPath1 = "E:\Green\dataset\Trento\Trento_0.2_split.mat"
        DataPath2 = r"E:\Green\dataset\noise_Trento"
        data = io.loadmat(DataPath1)

        TrLabel = data['TR']
        TsLabel = data['TE']

        Data = data['HSI']
        Data = Data.astype(np.float32)  # 转换为浮点数类型

        Data2 = data['LiDAR']
        Data2 = Data2.astype(np.float32)
        # 加载噪声图像
        additive_data = io.loadmat(DataPath2 + "/additive.mat")['data'].astype(np.float32)
        deadlines_data = io.loadmat(DataPath2 + "/deadlines.mat")['data'].astype(np.float32)
        kernal_data = io.loadmat(DataPath2 + "/kernal.mat")['data'].astype(np.float32)
        poisson_data = io.loadmat(DataPath2 + "/poisson.mat")['data'].astype(np.float32)
        salt_pepper_data = io.loadmat(DataPath2 + "/salt_pepper.mat")['data'].astype(np.float32)
        stripes_data = io.loadmat(DataPath2 + "/stripes.mat")['data'].astype(np.float32)
        zmguass_data = io.loadmat(DataPath2 + "/zmguass.mat")['data'].astype(np.float32)

    elif name == "Houston":
        DataPath1 = "E:\Green\dataset\Houston\Houston_0.2_split.mat"
        DataPath2 = "E:\Green\dataset\\noise_Houston"
        data = io.loadmat(DataPath1)

        TrLabel = data['TR']
        TsLabel = data['TE']

        Data = data['HSI']
        Data = Data.astype(np.float32)# 转换为浮点数类型

        Data2 = data['LiDAR']
        Data2 = Data2.astype(np.float32)
        # 加载噪声图像
        additive_data = io.loadmat(DataPath2 + "/additive.mat")['data'].astype(np.float32)
        deadlines_data = io.loadmat(DataPath2 + "/deadlines.mat")['data'].astype(np.float32)
        kernal_data = io.loadmat(DataPath2 + "/kernal.mat")['data'].astype(np.float32)
        poisson_data = io.loadmat(DataPath2 + "/poisson.mat")['data'].astype(np.float32)
        salt_pepper_data = io.loadmat(DataPath2 + "/salt_pepper.mat")['data'].astype(np.float32)
        stripes_data = io.loadmat(DataPath2 + "/stripes.mat")['data'].astype(np.float32)
        zmguass_data = io.loadmat(DataPath2 + "/zmguass.mat")['data'].astype(np.float32)

    elif name == "MUUFL":
        DataPath1 = r"E:\Green\dataset\MUUFL\MUUFL_0.2_split.mat"
        DataPath2 = r"E:\Green\dataset\noise_MUUFL"
        data = io.loadmat(DataPath1)

        TrLabel = data['TR']
        TsLabel = data['TE']

        Data = data['HSI']
        Data = Data.astype(np.float32)# 转换为浮点数类型

        Data2 = data['LiDAR']
        Data2 = Data2.astype(np.float32)
        # 加载噪声图像
        additive_data = io.loadmat(DataPath2 + "/additive.mat")['data'].astype(np.float32)
        deadlines_data = io.loadmat(DataPath2 + "/deadlines.mat")['data'].astype(np.float32)
        kernal_data = io.loadmat(DataPath2 + "/kernal.mat")['data'].astype(np.float32)
        poisson_data = io.loadmat(DataPath2 + "/poisson.mat")['data'].astype(np.float32)
        salt_pepper_data = io.loadmat(DataPath2 + "/salt_pepper.mat")['data'].astype(np.float32)
        stripes_data = io.loadmat(DataPath2 + "/stripes.mat")['data'].astype(np.float32)
        zmguass_data = io.loadmat(DataPath2 + "/zmguass.mat")['data'].astype(np.float32)

    elif name == "Augsburg":
        DataPath1 = r"E:\Green\dataset\Augsburg\Augsburg_0.2_split.mat"
        DataPath2 = r"E:\Green\dataset\noise_Augsburg"
        data = io.loadmat(DataPath1)

        TrLabel = data['TR']
        TsLabel = data['TE']

        Data = data['HSI']
        Data = Data.astype(np.float32)# 转换为浮点数类型

        Data2 = data['LiDAR']
        Data2 = Data2.astype(np.float32)
        # 加载噪声图像
        additive_data = io.loadmat(DataPath2 + "/additive.mat")['data'].astype(np.float32)
        deadlines_data = io.loadmat(DataPath2 + "/deadlines.mat")['data'].astype(np.float32)
        kernal_data = io.loadmat(DataPath2 + "/kernal.mat")['data'].astype(np.float32)
        poisson_data = io.loadmat(DataPath2 + "/poisson.mat")['data'].astype(np.float32)
        salt_pepper_data = io.loadmat(DataPath2 + "/salt_pepper.mat")['data'].astype(np.float32)
        stripes_data = io.loadmat(DataPath2 + "/stripes.mat")['data'].astype(np.float32)
        zmguass_data = io.loadmat(DataPath2 + "/zmguass.mat")['data'].astype(np.float32)

    #spData, a, spTrLabel, b = train_test_split(Data, TrLabel, test_size=(1-split_percent), random_state=3, shuffle=False)# 划分训练集和测试集
    #spData2, a, spTrLabel2, b = train_test_split(Data2, TrLabel, test_size=(1-split_percent), random_state=3, shuffle=False)

    return (Data,Data2,TrLabel,TsLabel,additive_data,deadlines_data,
            kernal_data,poisson_data,salt_pepper_data,stripes_data,zmguass_data)


def nor_pca(Data,Data2,ispca=True):
    [m, n, l] = Data.shape
    for i in range(l):
        minimal = Data[:, :, i].min()
        maximal = Data[:, :, i].max()
        Data[:, :, i] = (Data[:, :, i] - minimal) / (maximal - minimal)

    minimal = Data2.min()
    maximal = Data2.max()
    Data2 = (Data2 - minimal) / (maximal - minimal)

    if ispca is True:
        NC = 20
        PC = np.reshape(Data, (m * n, l))
        pca = PCA(n_components=NC, copy=True, whiten=False)
        PC = pca.fit_transform(PC)
        PC = np.reshape(PC, (m, n, NC))
    else:
        NC = l
        PC = Data

    return PC,Data2,NC #349*1905*20,349*1905,20

#padding
def border_inter(PC,Data2,NC):
    temp = PC[:, :, 0]
    pad_width = np.floor(patchsize1 / 2)
    pad_width = np.int32(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')#359*1915
    [m2, n2] = temp2.shape
    x = np.empty((m2, n2, NC), dtype='float32')

    for i in range(NC):
        temp = PC[:, :, i]
        pad_width = np.floor(patchsize1 / 2)
        pad_width = np.int32(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x[:, :, i] = temp2

    x2 = Data2
    pad_width2 = np.floor(patchsize2 / 2)
    pad_width2 = np.int32(pad_width2)
    temp2 = np.pad(x2, pad_width2, 'symmetric')
    x2 = temp2
    return x, x2


def con_data(x,x2,TrLabel,TsLabel,NC):
    [ind1, ind2] = np.where(TrLabel != 0)# 返回非零元素的索引
    TrainNum = len(ind1)
    TrainPatch = np.empty((TrainNum, NC, patchsize1, patchsize1), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]#取patch 11*11*20
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))#121*20
        patch = np.transpose(patch)#20*121
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TrainPatch[i, :, :, :] = patch
        patchlabel = TrLabel[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel

    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TestPatch[i, :, :, :] = patch
        patchlabel = TsLabel[ind1[i], ind2[i]]
        TestLabel[i] = patchlabel

    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum = len(ind1)
    TrainPatch2 = np.empty((TrainNum, 1, patchsize2, patchsize2), dtype='float32')
    TrainLabel2 = np.empty(TrainNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        TrainPatch2[i, :, :, :] = patch
        patchlabel2 = TrLabel[ind1[i], ind2[i]]
        TrainLabel2[i] = patchlabel2

    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch2 = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
    TestLabel2 = np.empty(TestNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        TestPatch2[i, :, :, :] = patch
        patchlabel2 = TsLabel[ind1[i], ind2[i]]
        TestLabel2[i] = patchlabel2

    return TrainPatch,TestPatch,TrainPatch2,TestPatch2,TrainLabel,TestLabel,TrainLabel2,TestLabel2

# 偶数尺寸处理函数
def con_data_even(x, x2, TrLabel, TsLabel, NC):
    # ============================== 第一部分：处理x（多通道数据） ==============================
    # 对输入图像进行反射填充（处理边界）
    x_padded = np.pad(x, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')

    # 获取训练集非零标签的索引
    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum = len(ind1)
    TrainPatch = np.empty((TrainNum, NC, patchsize1, patchsize1), dtype='float32')
    TrainLabel = np.empty(TrainNum)

    # 提取训练集图像块（基于填充后的坐标）
    for i in range(TrainNum):
        # 在填充后的图像上直接截取块（无需调整索引）
        row_start = ind1[i]
        row_end = ind1[i] + patchsize1  # 偶数尺寸，直接+patchsize1
        col_start = ind2[i]
        col_end = ind2[i] + patchsize1

        patch = x_padded[row_start:row_end, col_start:col_end, :]  # 形状：(10,10,NC)
        patch = np.transpose(patch, (2, 0, 1))  # 直接转置为 (NC,10,10)
        TrainPatch[i] = patch
        TrainLabel[i] = TrLabel[ind1[i], ind2[i]]

    # 处理测试集（逻辑与训练集一致）
    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)
    for i in range(TestNum):
        patch = x_padded[ind1[i]:ind1[i] + patchsize1, ind2[i]:ind2[i] + patchsize1, :]
        TestPatch[i] = np.transpose(patch, (2, 0, 1))
        TestLabel[i] = TsLabel[ind1[i], ind2[i]]

    # ============================== 第二部分：处理x2（单通道数据） ==============================
    # 对输入图像进行反射填充
    x2_padded = np.pad(x2, ((pad_width2, pad_width2), (pad_width2, pad_width2)), mode='reflect')

    # 处理训练集
    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum = len(ind1)
    TrainPatch2 = np.empty((TrainNum, 1, patchsize2, patchsize2), dtype='float32')
    TrainLabel2 = np.empty(TrainNum)
    for i in range(TrainNum):
        patch = x2_padded[ind1[i]:ind1[i] + patchsize2, ind2[i]:ind2[i] + patchsize2]  # (6,6)
        TrainPatch2[i, 0] = patch  # 直接赋值，无需reshape
        TrainLabel2[i] = TrLabel[ind1[i], ind2[i]]

    # 处理测试集
    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch2 = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
    TestLabel2 = np.empty(TestNum)
    for i in range(TestNum):
        patch = x2_padded[ind1[i]:ind1[i] + patchsize2, ind2[i]:ind2[i] + patchsize2]
        TestPatch2[i, 0] = patch
        TestLabel2[i] = TsLabel[ind1[i], ind2[i]]

    return TrainPatch, TestPatch, TrainPatch2, TestPatch2, TrainLabel, TestLabel, TrainLabel2, TestLabel2


def con_data_all(x,x2,AllLabel,NC):

    [ind1, ind2] = np.where(AllLabel != 0)
    TestNum = len(ind1)
    Allpatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        Allpatch[i, :, :, :] = patch
        patchlabel = AllLabel[ind1[i], ind2[i]]
        TestLabel[i] = patchlabel


    [ind1, ind2] = np.where(AllLabel != 0)
    TestNum = len(ind1)
    Allpatch2 = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
    TestLabel2 = np.empty(TestNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        Allpatch2[i, :, :, :] = patch
        patchlabel2 = AllLabel[ind1[i], ind2[i]]
        TestLabel2[i] = patchlabel2

    return Allpatch,Allpatch2,TestLabel,TestLabel2

#噪声数据
def con_data2(x,TsLabel,NC):

    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TestPatch[i, :, :, :] = patch


    # 原有报错行
    return TestPatch

def con_data2_even(x, TsLabel, NC):

    # 对输入图像进行反射填充（处理边界）
    x_padded = np.pad(x, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')

    # 获取测试集非零标签的索引
    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')

    # 提取测试集图像块（基于填充后的坐标）
    for i in range(TestNum):
        # 在填充后的图像上直接截取块（无需调整索引）
        row_start = ind1[i]
        row_end = ind1[i] + patchsize1  # 偶数尺寸，直接+patchsize1
        col_start = ind2[i]
        col_end = ind2[i] + patchsize1

        # 提取块并调整维度
        patch = x_padded[row_start:row_end, col_start:col_end, :]  # 形状：(10,10,NC)
        patch = np.transpose(patch, (2, 0, 1))  # 直接转置为 (NC,10,10)
        TestPatch[i] = patch

    return TestPatch

def getIndex(TestLabel, temp):
    index = np.empty(shape=(2,temp), dtype=int)
    k = 0
    for i in range(len(TestLabel)):
        for j in range(len(TestLabel[0])):
            if TestLabel[i][j] != 0:
                index[0][k] = i+1
                index[1][k] = j+1
                k += 1

    return index


# 增强版本：更多遮挡类型
def random_occlusion_and_blur_pair_advanced(patch1,label1=None,
                                            occlusion_prob=0.5, occlusion_size_range=(0.15, 0.25),blur_prob=0.5,
                                            sigma_range=(0.5, 1.5)):
    """
    对数据进行随机遮挡和随机高斯模糊增强
    提供多种遮挡类型和参数随机化

     参数:
    - patch1: 第一组图像块 (HSI)，形状 (n_samples, n_bands1, height, width)
    - patch2: 第二组图像块 (雷达)，形状 (n_samples, n_bands2, height, width)
    - label1: 第一组标签
    - label2: 第二组标签
    - occlusion_prob: 应用遮挡的概率
    - occlusion_size_range: 遮挡区域相对于原图的比例范围 (min_ratio, max_ratio)
    - blur_prob: 应用高斯模糊的概率
    - sigma_range: 高斯模糊sigma的范围 (min_sigma, max_sigma)

     遮挡类型:
        zero遮挡：将遮挡区域设置为0（模拟信息完全缺失）
        噪声遮挡：将遮挡区域设置为随机噪声
        均值遮挡：将遮挡区域设置为该通道的均值
        多区域遮挡：在图像中遮挡多个小区域
        边缘遮挡：模拟传感器边缘问题

    返回:
    - augmented_patch1: 增强后的两组图像块（保持原始尺寸）
        - augmented_label1: 增强后的标签（如果提供了）
    """
    n_samples, n_bands1, height, width = patch1.shape

    # 初始化增强后的数据
    augmented_patch1 = np.zeros_like(patch1)


    for i in range(n_samples):
        occluded_patch1 = patch1[i].copy()

        # 1. 随机遮挡
        if np.random.random() < occlusion_prob:
            # 随机选择遮挡类型
            occlusion_type = np.random.choice(['zero', 'noise', 'mean', 'multi', 'border'])

            # 随机确定遮挡数量（1-3个）
            num_occlusions = np.random.randint(1, 4)

            for _ in range(num_occlusions):
                # 随机确定遮挡区域大小
                min_ratio, max_ratio = occlusion_size_range
                occlusion_ratio = np.random.uniform(min_ratio, max_ratio)

                occlusion_h = max(1, int(height * occlusion_ratio))
                occlusion_w = max(1, int(width * occlusion_ratio))

                # 随机确定遮挡位置
                max_h_start = height - occlusion_h
                max_w_start = width - occlusion_w

                if max_h_start >= 0 and max_w_start >= 0:
                    h_start = np.random.randint(0, max_h_start + 1)
                    w_start = np.random.randint(0, max_w_start + 1)

                    # 根据遮挡类型应用不同的遮挡
                    if occlusion_type == 'zero':
                        # 设置为0
                        occluded_patch1[:, h_start:h_start + occlusion_h, w_start:w_start + occlusion_w] = 0

                    elif occlusion_type == 'noise':
                        # 设置为随机噪声
                        noise_scale = np.random.uniform(0.05, 0.2)
                        random_noise1 = np.random.randn(n_bands1, occlusion_h, occlusion_w) * noise_scale
                        occluded_patch1[:, h_start:h_start + occlusion_h, w_start:w_start + occlusion_w] = random_noise1

                    elif occlusion_type == 'mean':
                        # 设置为通道均值
                        mean1 = np.mean(patch1[i], axis=(1, 2), keepdims=True)
                        occluded_patch1[:, h_start:h_start + occlusion_h, w_start:w_start + occlusion_w] = mean1

                    elif occlusion_type == 'multi':
                        # 多个小遮挡
                        sub_occlusion_h = max(1, occlusion_h // 2)
                        sub_occlusion_w = max(1, occlusion_w // 2)

                        for sub_h in range(h_start, h_start + occlusion_h, sub_occlusion_h):
                            for sub_w in range(w_start, w_start + occlusion_w, sub_occlusion_w):
                                sub_h_end = min(sub_h + sub_occlusion_h, h_start + occlusion_h)
                                sub_w_end = min(sub_w + sub_occlusion_w, w_start + occlusion_w)
                                occluded_patch1[:, sub_h:sub_h_end, sub_w:sub_w_end] = 0

                    elif occlusion_type == 'border':
                        # 边缘遮挡（模拟传感器边缘问题）
                        border_width = np.random.randint(1, 4)

                        # 随机选择遮挡哪条边
                        borders = np.random.choice(['top', 'bottom', 'left', 'right'],
                                                   size=np.random.randint(1, 3),
                                                   replace=False)

                        if 'top' in borders:
                            occluded_patch1[:, :border_width, :] = 0
                        if 'bottom' in borders:
                            occluded_patch1[:, height - border_width:, :] = 0
                        if 'left' in borders:
                            occluded_patch1[:, :, :border_width] = 0
                        if 'right' in borders:
                            occluded_patch1[:, :, width - border_width:] = 0

        # 2. 随机高斯模糊
        if np.random.random() < blur_prob:
            sigma = np.random.uniform(sigma_range[0], sigma_range[1])

            # 可以选择只模糊部分区域
            blur_whole = np.random.random() < 0.7  # 70%概率模糊整个图像

            if blur_whole:
                # 模糊整个图像
                for b in range(n_bands1):
                    occluded_patch1[b] = gaussian_filter(
                        occluded_patch1[b],
                        sigma=sigma,
                        mode='reflect'
                    )
            else:
                # 只模糊部分区域
                blur_ratio = np.random.uniform(0.3, 0.7)
                blur_h = int(height * blur_ratio)
                blur_w = int(width * blur_ratio)

                max_h_start = height - blur_h
                max_w_start = width - blur_w

                if max_h_start >= 0 and max_w_start >= 0:
                    h_start = np.random.randint(0, max_h_start + 1)
                    w_start = np.random.randint(0, max_w_start + 1)

                    for b in range(n_bands1):
                        # 提取区域并模糊
                        region = occluded_patch1[b, h_start:h_start + blur_h, w_start:w_start + blur_w]
                        blurred_region = gaussian_filter(region, sigma=sigma, mode='reflect')
                        occluded_patch1[b, h_start:h_start + blur_h, w_start:w_start + blur_w] = blurred_region

        augmented_patch1[i] = occluded_patch1

    if label1 is not None:
        return (augmented_patch1,
                label1.copy())
    else:
        return augmented_patch1


def random_occlusion_and_blur_pair(patch1, label1=None,
                                   occlusion_prob=0.5, occlusion_size_range=(0.1, 0.3),
                                   blur_prob=0.5, sigma_range=(0.5, 1.5)):
    """
    对HSI数据进行随机遮挡和随机高斯模糊增强

    参数:
    - patch1: 第一组图像块 (HSI)，形状 (n_samples, n_bands1, height, width)
    - label1: 第一组标签
    - occlusion_prob: 应用遮挡的概率
    - occlusion_size_range: 遮挡区域相对于原图的比例范围 (min_ratio, max_ratio)
    - blur_prob: 应用高斯模糊的概率
    - sigma_range: 高斯模糊sigma的范围

    返回:
    - augmented_patch1: 增强后的两组图像块（保持原始尺寸）
    - augmented_label1: 增强后的标签（如果提供了）
    """
    n_samples, n_bands1, height, width = patch1.shape

    # 初始化增强后的数据（保持原始尺寸）
    augmented_patch1 = np.zeros_like(patch1)

    for i in range(n_samples):
        # 复制原始数据
        occluded_patch1 = patch1[i].copy()

        # 1. 随机遮挡
        if np.random.random() < occlusion_prob:
            # 随机确定遮挡区域的大小（相对于原图的比例）
            occlusion_ratio = np.random.uniform(occlusion_size_range[0], occlusion_size_range[1])

            # 计算遮挡区域的绝对尺寸
            occlusion_h = int(height * occlusion_ratio)
            occlusion_w = int(width * occlusion_ratio)

            # 确保遮挡区域至少为1×1
            occlusion_h = max(1, occlusion_h)
            occlusion_w = max(1, occlusion_w)

            # 随机确定遮挡区域的起始位置
            max_h_start = height - occlusion_h
            max_w_start = width - occlusion_w

            if max_h_start >= 0 and max_w_start >= 0:
                h_start = np.random.randint(0, max_h_start + 1)
                w_start = np.random.randint(0, max_w_start + 1)

                # 在两组数据的相同位置应用遮挡
                # 方法1：设置为0（黑色遮挡）
                occluded_patch1[:, h_start:h_start + occlusion_h, w_start:w_start + occlusion_w] = 0

                # 方法2：设置为随机噪声（可选，取消注释使用）
                # random_noise1 = np.random.randn(n_bands1, occlusion_h, occlusion_w) * 0.1
                # random_noise2 = np.random.randn(n_bands2, occlusion_h, occlusion_w) * 0.1
                # occluded_patch1[:, h_start:h_start+occlusion_h, w_start:w_start+occlusion_w] = random_noise1
                # occluded_patch2[:, h_start:h_start+occlusion_h, w_start:w_start+occlusion_w] = random_noise2

                # 方法3：设置为通道均值（可选，取消注释使用）
                # mean1 = np.mean(patch1[i], axis=(1, 2), keepdims=True)
                # mean2 = np.mean(patch2[i], axis=(1, 2), keepdims=True)
                # occluded_patch1[:, h_start:h_start+occlusion_h, w_start:w_start+occlusion_w] = mean1
                # occluded_patch2[:, h_start:h_start+occlusion_h, w_start:w_start+occlusion_w] = mean2

        # 2. 随机高斯模糊（只对HSI数据）
        if np.random.random() < blur_prob:
            sigma = np.random.uniform(sigma_range[0], sigma_range[1])

            # 对HSI数据的每个波段分别应用高斯模糊
            for b in range(n_bands1):
                occluded_patch1[b] = gaussian_filter(
                    occluded_patch1[b],
                    sigma=sigma,
                    mode='reflect'
                )

        # 保存增强后的数据
        augmented_patch1[i] = occluded_patch1

    # 返回结果
    if label1 is not None :
        return (augmented_patch1,
                label1.copy(),)
    else:
        return augmented_patch1