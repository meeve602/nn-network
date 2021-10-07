from numpy.core.records import array
from nn import Placeholder, Linear, Sigmoid, Loss, Variable
import numpy as np
from utils import convert_feed_dict_to_graph, topological_sort, forward_and_backward, optimize
#from icecream import ic
from sklearn.datasets import load_boston#加载数据
import pandas as pd
import matplotlib.pyplot as plt
#主
#加载数据CSV文件
data = load_boston()
x_data = data['data']
y = data['target']
desc = data['DESCR']

# x, y ; x with 13 dimensions
# let computer could predict house price using some features automatically

# correlation analysis

dataframe = pd.DataFrame(x_data)
'''
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, np.nan],  # np.nan表示NA
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
DataFrame(data)  # 默认生成整数索引, 字典的键作列,值作行
JSON格式？
输出结果为：
    state	year	pop
0	Ohio	2000.0	1.5
1	Ohio	2001.0	1.7
2	Ohio	2002.0	3.6
3	Nevada	2001.0	2.4
4	Nevada	NaN	2.9
'''

dataframe.columns = data['feature_names']#将取首行featur Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'price'],dtype='object')
dataframe['price'] = y #设price为y轴
#对Y进行列表化
rm = dataframe['RM']
lstat = dataframe['LSTAT']
y_ = dataframe['price']

#构造网络
x = Placeholder(name='x')#x Node: x
y = Placeholder(name='y-true')# Node: y-true
b1 = Variable(name='b1')
w1 = Variable(name='w1')
b2 = Variable(name='b2')
w2 = Variable(name='w2')
b3 = Variable(name='b3')
w3 = Variable(name='w3')
b4 = Variable(name='b4')
f1 = Linear(x=x, w=w1, b=b1, name='f1')#f1.forward()会自动导向linear.forward()函数
f2 = Linear(x=x, w=w2, b=b2, name='f2')
f3 = Sigmoid(x=f1, name='f3-sigmoid')
f5 = Linear(x=f2, w=w3, b=b3, name='f5')
f4 = Linear(x=f3, w=f5, b=b4, name='yhat')#w=f5????
loss = Loss(y=y, yhat=f4, name='loss')
#初始化数值
feed_dict = {               #随机赋予一个初始值
    w1: np.random.normal(),
    b1: np.random.normal(),
    w2: np.random.normal(),
    b2: np.random.normal(),
    w3: np.random.normal(),
    b3: np.random.normal(),
    b4: np.random.normal(),
    x: rm,
    y: y_
}
#训练 
losse = []
a = np.arange(0,50)
graph = list(topological_sort(convert_feed_dict_to_graph(feed_dict)))# list() 方法用于将元组转换为列表     先连成图，再排成有序数组（拓扑排序）
for i in range(50):
    forward_and_backward(graph)
    optimize(graph, lr=1e-8)
    print('loss: {}'.format(loss.value))#小写
    losse.append(loss.value)

plt.plot(a ,losse,"ob")
plt.show()    
# print('w1{}'.format(w1.value))
# print('b1{}'.format(b1.value))
# print('w2{}'.format(w2.value))
# print('b2{}'.format(b2.value))
# print('w3{}'.format(w3.value))
# print('b3{}'.format(b3.value))
# print('b4{}'.format(b4.value))
# print('x{}'.format(x.value))
# print('y{}'.format(y.value))


