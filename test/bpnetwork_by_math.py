# 引用math处理包
import math
# 引用随机数处理包
import random

# 每次生成同样的随机数
random.seed(0)


# 生成一个ab之间的随机数
def rand(a, b):
    return (b - a) * random.random() + a


# 生成一个m*n的矩阵，fill表示填充值
def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


# 自定义一个sigmoid激活非线性函数
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# 定义对于sigmoid线性函数的导数函数
def sigmoid_derivative(x):
    return x * (1 - x)


# 定义一个BP神经网络类
class BPNeuralNetwork:
    # 初始化这个网络
    def __init__(self):
        # 输入值层数
        self.input_n = 0
        # 隐藏层层数
        self.hidden_n = 0
        # 输出层层数
        self.output_n = 0
        # 输入单元
        self.input_cells = []
        # 隐藏单元
        self.hidden_cells = []
        # 输出单元
        self.output_cells = []
        # 输入系数权重矩阵
        self.input_weights = []
        # 输出系数权重矩阵
        self.output_weights = []
        # 输入偏移量矩阵
        self.input_correction = []
        # 输出偏移量矩阵
        self.output_correction = []

    # 网络设置函数
    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # 初始化层次单元，多少层就乘以等到多少个单位矩阵
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # 初始化权重矩阵
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # 随机数非线性激活网络
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                # 输入权重为输入层*隐藏层大小的矩阵
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                # 输入权重为隐藏层*输出层大小的矩阵
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # 初始化偏移量矩阵
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    # 预测函数，根据输入进行预测输出的值
    def predict(self, inputs):
        # 激活输入层
        for i in range(self.input_n - 1):
            # 将对应的输入赋值到对应的输入层
            self.input_cells[i] = inputs[i]
        # 激活隐藏层
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # 激活输出层
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    # 反向传播函数，即对模型参数进行更新，参数为源数据，目标数据，学习率，学习步长
    def back_propagate(self, case, label, learn, correct):
        # 正向传播，即正向传输数据到网络模型中
        self.predict(case)
        # 获取输出层的损失值，即计算计算值与目标的偏差
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # 获取隐藏层的损失值
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # 更新输出层的权重系数矩阵
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # 更新输入层的权重系数矩阵
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # 获取全局的损失值
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    # 训练方法，参数为输入情况，对应标签，训练最大次数，学习率，变动步长
    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        # 一共训练10000次
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

    def test(self):
        # 设置亦或的四个实例
        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        # 设置对应输出的标签
        labels = [[0], [1], [1], [0]]
        # 这是个2个输入层，5个隐藏层，1个输出层的网络
        # 即对应2个输入，1一个输出，5次对数据进行处理
        self.setup(2, 5, 1)
        # 对模型进行训练
        self.train(cases, labels, 10000, 0.05, 0.1)
        # 调用预测方法对源数据进行预测
        for i in range(len(cases)):
            predict = self.predict(cases[i])[0]
            print("实际值：{0}，预测值：{1}，损失值：{2}".format(labels[i][0], predict, labels[i][0] - predict))


if __name__ == '__main__':
    # 建立一个BP神经网络
    nn = BPNeuralNetwork()
    # 调用测试方法
    nn.test()
