# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 将Y转换为one-hot编码
# Y = onehotEncoder(Y, 10)

# one-hot编码函数
def onehotEncoder(Y, ny):
    return np.eye(ny)[Y]

class Conv3x3:# 使用3x3滤波器的卷积层

    def __init__(self, num_filters):
        self.num_filters = num_filters
        '''
        filters是一个三维数组，维度为(num_filters, 3, 3)
        我们除以9来减少初始值的方差
        '''
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        生成所有可能的3x3图像区域，使用有效填充。
        - image是一个二维numpy数组
        '''
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j
        '''
        将 im_region, i, j 以 tuple 形式存储到迭代器中
        以便后面遍历使用
        '''
    def forward(self, input):
        '''
        使用给定的输入执行卷积层的前向传播。
        返回一个三维numpy数组，维度为(h, w, num_filters)。
        - input是一个二维numpy数组
        '''
        self.last_input = input
        '''
        input 为 image，即输入数据
        output 为输出框架，默认都为 0，都为 1 也可以，反正后面会覆盖
        input: 20x20
        output: 18x18x8
        '''
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            '''卷积运算，点乘再相加，ouput[i, j] 为向量，8 层'''
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        ''' 最后将输出数据返回，便于下一层的输入使用'''
        return output

    def backprop(self, d_L_d_out, learn_rate):
        '''
        执行卷积层的反向传播。
        - d_L_d_out是这一层输出的损失梯度。
        - learn_rate是一个浮点数。
        '''
        
        '''# 初始化一组为 0 的 gradient，3x3x8'''
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                '''
                按 f 分层计算，一次算一层，然后累加起来
                d_L_d_filters[f]: 3x3 matrix
                d_L_d_out[i, j, f]: num
                im_region: 3x3 matrix in image
                '''
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # 更新滤波器
        self.filters -= learn_rate * d_L_d_filters

        return None

class MaxPool2:
    # 使用2x2池化大小的最大池化层

    def iterate_regions(self, image):
        '''
        生成不重叠的2x2图像区域进行池化。
        - image是一个三维numpy数组
        '''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        '''
        使用给定的输入执行最大池化层的前向传播。
        返回一个三维numpy数组，维度为(h / 2, w / 2, num_filters)。
        - input是一个三维numpy数组，维度为(h, w, num_filters)
        '''
        
        '''存储 池化层 的输入参数，18x18x8'''
        self.last_input = input
        '''input: 卷基层的输出，池化层的输入'''
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output

    def backprop(self, d_L_d_out):
        '''
        执行最大池化层的反向传播。
        返回这一层输入的损失梯度。
        - d_L_d_out是这一层输出的损失梯度。
        '''
        
        '''池化层输入数据，18x18x8，默认初始化为 0'''
        d_L_d_input = np.zeros(self.last_input.shape)
        '''
        每一个 im_region 都是一个 3x3x8 的8层小矩阵
        修改 max 的部分，首先查找 max
        '''
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            '''获取 im_region 里面最大值的索引向量，一叠的感觉'''
            amax = np.amax(im_region, axis=(0, 1))
            '''遍历整个 im_region，对于传递下去的像素点，修改 gradient 为 loss 对 output 的gradient'''
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        '''如果该像素是最大值，则将梯度复制到该像素。'''
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input

class Softmax:
    # 带有softmax激活的标准全连接层

    def __init__(self, input_len, nodes):
        '''
        我们除以input_len来减少初始值的方差
        input_len: 输入层的节点个数，池化层输出拉平之后的
        nodes: 输出层的节点个数，本例中为 10
        构建权重矩阵，初始化随机数，不能太大
        '''
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        使用给定的输入执行softmax层的前向传播。
        返回一个一维numpy数组，包含相应的概率值。
        - input可以是任何维度的数组。
        '''
        #9x9x8
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.weights.shape
        '''
        input: 9x9x8 = 648
        self.weights: (648, 10)
        以上叉乘之后为 向量，648个节点与对应的权重相乘再加上bias得到输出的节点
        '''
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        '''
        执行softmax层的反向传播。
        返回这一层输入的损失梯度。
        - d_L_d_out是这一层输出的损失梯度。
        - learn_rate是一个浮点数
        '''
        # 我们知道 d_L_d_out 中只有一个元素是非零的
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
            # e^totals
            t_exp = np.exp(self.last_totals)
            # 所有e^totals的总和
            S = np.sum(t_exp)
            # 计算out[i]相对于totals的梯度
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
            # 计算totals相对于权重/偏差/输入的梯度
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights
            # 计算损失相对于totals的梯度
            d_L_d_t = gradient * d_out_d_t
            # 计算损失相对于权重/偏差/输入的梯度
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t
            # 更新权重和偏差
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            # 将矩阵从 1d 转为 3d
            # 648 to 9x9x8
            return d_L_d_inputs.reshape(self.last_input_shape)

def cost(Y_hat, Y):
    n = Y.shape[0]
    correct_log_probs = -np.log(Y_hat[range(n), Y])
    c = np.sum(correct_log_probs) / n
    return c

def test(Y_hat, Y):
    predictions = np.argmax(Y_hat, axis=1)
    acc = np.mean(predictions == Y)
    return acc

# 加载数据
train_data = np.load("train_data.npy")
label = np.load("train_label.npy")
test_data = np.load("test_data.npy")

#X = train_data.reshape(train_data.shape[0], 28, 28).transpose(0,2,1)
X = train_data.reshape(train_data.shape[0], 28, 28)

X_test = test_data.reshape(test_data.shape[0], 28, 28)
Y = label.astype(np.int32)
(n, L, _) = X.shape

# 打乱数据
permutation = np.random.permutation(X.shape[0])
X_shuffled = X[permutation]
Y_shuffled = Y[permutation]

# 分割数据
# split_index = int(X.shape[0] * 0.9)  # 70%用于训练
split_index = int(X.shape[0])  # 100%用于训练
train_images = X_shuffled[:split_index]
#test_images = X_shuffled[split_index:]
test_images = X_test
train_labels = Y_shuffled[:split_index]
#test_labels = Y_shuffled[split_index:]

num_filter = 3 #卷积核数量
conv = Conv3x3(num_filter)  # 28x28x8 -> 26x26x8
pool = MaxPool2()  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * num_filter, 10)  # 9x9x8 -> 10

def forward(image, label):
    '''
    完成CNN的前向传播，并计算准确率和交叉熵损失。
    - image是一个二维numpy数组
    - label是一个数字
    '''
    out = conv.forward((image / 255) - 0.5)
    #out = conv.forward(image)
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

def train(im, label, lr=.01):#学习率参数设置
    '''
    对给定的图像和标签完成完整的训练步骤。
    返回交叉熵损失和准确率。
    - image是一个二维numpy数组
    - label是一个数字
    - lr是学习率
    '''
    out, loss, acc = forward(im, label)

    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc

print('CNN初始化完成!')

# 初始化存储损失的列表
losses = []
train_acc_history = []
test_acc_history = []

# 训练CNN N个周期
for epoch in range(5):#训练次数参数设置
    print('\n--- 第%d周期 ---' % (epoch + 1))

    # 打乱训练数据
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # 开始训练
    loss = 0
    num_correct = 0

    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(
                '[步骤 %d] 过去100步: 平均损失 %.3f | 准确率: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc

        # 收集损失值
        losses.append(l)
        
    # 在每个周期结束时计算训练和测试集的准确率
    Y_hat_train = np.array([forward(im, None)[0] for im in train_images])
    train_acc = test(Y_hat_train, train_labels)

    #Y_hat_test = np.array([forward(im, None)[0] for im in test_images])
    #test_acc = test(Y_hat_test, test_labels)

    train_acc_history.append(train_acc)
    #test_acc_history.append(test_acc)

    #print(f'\n周期 {epoch + 1} \n训练准确率: {train_acc} 测试准确率: {test_acc}')
    print(f'\n周期 {epoch + 1} \n训练准确率: {train_acc}')


print("\n")
print("----正在计算损失和准确率...请稍候... ---")

# 训练循环结束时，对训练集和测试集进行评估
Y_hat_train = []
for im in train_images:
    out, _, _ = forward(im, None)
    Y_hat_train.append(out)
Y_hat_train = np.array(Y_hat_train)

train_loss = cost(Y_hat_train, train_labels)
train_acc = test(Y_hat_train, train_labels)

Y_hat_test = []
for im in test_images:
    out, _, _ = forward(im, None)
    Y_hat_test.append(out)
Y_hat_test = np.array(Y_hat_test)

# test_loss = cost(Y_hat_test, test_labels)
# test_acc = test(Y_hat_test, test_labels)

# 首先获取每个测试样本的预测类别
predicted_labels = np.argmax(Y_hat_test, axis=1)

# 创建一个与预测结果数量相匹配的索引数组
indices = np.arange(0, len(predicted_labels))

# 将索引数组和预测标签数组合并为一个二维数组
predictions_with_indices = np.vstack((indices, predicted_labels)).T

# 保存为 CSV 文件
np.savetxt("predict.csv", predictions_with_indices, fmt='%d', delimiter=',', header='Index,PredictedLabel', comments='')


print("训练损失:", train_loss)
print("训练准确率:", train_acc)
# print("测试损失:", test_loss)
# print("测试准确率:", test_acc)  

# 绘制训练损失和准确率曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Training Accuracy')
# plt.plot(test_acc_history, label='Testing Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()  

    