import numpy as np
#from icecream import ic
from collections import  defaultdict


class Node:
    def __init__(self, inputs=[], name=None, trainable=False):#遍历赋予每个节点对应值
        #self捆绑，挂载，调用时直接Node.input........
        self.inputs = inputs  # array of Nodes
        self.outputs = []     # this == self   
        self.name = name      #用于附加名字

        for n in self.inputs:     #这里是input
            n.outputs.append(self)#这里是output，append进行连接,append() 是给原列表对象添加元素，而直接赋值则属于创建一个新对象
            '''
            self.l2 = l1 #这一行实际是给一个空筐取了个别名，下面的那一行用了别名去操作这个筐
            self.l2.append('qqq') #这行往空筐里放了qqq
            self.l2 = ['111'] #这行实际已经换了一个筐，这个筐已经装好了111
            '''
            # Node: w1，在这里时调用了__rept__
            # connect self node and its input nodes
            # 将input与output连接
            '''
            a = [1,2,3]
            b = []
            b.append(a)
            print(b)
            '''

        self.value = None #f1（运算节点）输入一个东西会求的得一个值
        self.loss_gradient = defaultdict(float)#loss对这个值的偏导      
        '''
        （1）defaultdict(int)：初始化为 0
        （2）defaultdict(float)：初始化为 0.0
        （3）defaultdict(str)：初始化为 ”
        '''
        self.trainable = trainable

    def __repr__(self):#定义调用该类时自动运行，放名字用于调试
        return 'Node: {}'.format(self.name)#该定义有被使用，虽然没有被调用

    def forward(self):#每一个节点会向前运算
        raise NotImplementedError

    def backward(self):#也会向后运算
        raise NotImplementedError


class Placeholder(Node):
    def __init__(self, name=None, trainable=False):
        Node.__init__(self, name=name, trainable=trainable)#跳Node类

    def forward(self):
        # print('我是placeholder，我的值通过人工初始化得到了：{}'.format(self.value))
        pass

    def backward(self):
        for out in self.outputs:
            gradient = out.loss_gradient[self]
            self.loss_gradient[self] += gradient
        # print('get gradient: {}'.format(self.loss_gradient[self]))


class Variable(Placeholder):#自写可训练函数
    def __init__(self, name=None):
        Placeholder.__init__(self, name=name, trainable=True)#可训练，跳Placeholder定义下的__init__
        #


class Linear(Node):
    def __init__(self, x, w, b, name=None):
        Node.__init__(self, inputs=[x, w, b], name=name)

    def forward(self):
        x, w, b = self.inputs
        self.value = w.value * x.value + b.value

        # print('依据self.inputs: {}, 计算目前该节点的值: {}'.format(self.inputs, self.value))

    def backward(self):
        x, w, b = self.inputs
        for out in self.outputs:
            self.loss_gradient[x] += out.loss_gradient[self] * w.value #self.outputs[0].loss_gradient[self] = self.loss_gradient[yhat]
            self.loss_gradient[w] += out.loss_gradient[self] * x.value
            self.loss_gradient[b] += out.loss_gradient[self] * 1

        # print('get gradient: {}'.format(self.loss_gradient))


class Sigmoid(Node):
    def __init__(self, x, name=None):
        Node.__init__(self, inputs=[x], name=name)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))#计算节点

    def forward(self):
        self.value = self._sigmoid(self.inputs[0].value)#得值

        # print('依据self.inputs: {}, 计算目前该节点的值: {}'.format(self.inputs, self.value))

    def backward(self):
        x = self.inputs[0]

        for out in self.outputs:
            self.loss_gradient[x] += out.loss_gradient[self] * self.value * (1 - self.value)#求解loss对每一个X偏导的时候，只需要求当前这个loss节点的值再乘以对于它输出节点的值
        # print('get gradient: {}'.format(self.loss_gradient))


class Loss(Node):
    def __init__(self, y, yhat, name=None):
        Node.__init__(self, inputs=[y, yhat], name=name)

    def forward(self):
        y, yhat = self.inputs

        self.value = np.mean( (y.value - yhat.value) ** 2 )
        # print('依据self.inputs: {}, 计算目前该节点的值: {}'.format(self.inputs, self.value))

    def backward(self):
        y, yhat = self.inputs
        self.loss_gradient[yhat] = -2 * np.mean(y.value - yhat.value)#在此计算偏导 载入yhat的loss_gradictent
        self.loss_gradient[y] = 2 * np.mean(y.value - yhat.value)

        # print('get gradient: {}'.format(self.loss_gradient))