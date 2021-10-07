from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from icecream import ic
import random
from functools import reduce
from collections import defaultdict
from nn import Placeholder#导入


# sns.heatmap(dataframe.corr())

# x, y ; x with 13 dimensions
# sns.heatmap(dataframe.corr())

# plt.subplots(1, 2, figsize=(20, 20))

# plt.scatter(dataframe['RM'], dataframe['price'])
# plt.scatter(dataframe['LSTAT'], dataframe['price'])

# plt.show()

#介绍KNN
def k_nearest_neighbors(train_rm, train_lstat, train_y, query_rm, query_lstat, topn=3):
    """"
    KNN model
    --
    input is the rm and lstat value of a perspective house
    return: predicted house price
    """

    elements = [(r, ls, y) for r, ls, y in zip(train_rm, train_lstat, train_y)]

    def distance(e): return (e[0] - query_rm) ** 2 + (e[1] - query_lstat) ** 2

    neighbors = sorted(elements, key=distance, reverse=True)[:topn]

    return np.mean([y for r, ls, y in neighbors])

# => rm -> price


#有关计算数学公式
def random_linear(x):
    w, b = np.random.normal(scale=10, size=(1, 2))[0]
    return linear(x, w, b)


def linear(x, w, b):
    return w * x + b


def loss(yhat, y):
    return np.mean((yhat - y) ** 2)


def partial_w(y, yhat, x):
    return -2 * np.mean((y - yhat) * x)


def partial_b(y, yhat):
    return -2 * np.mean(y - yhat)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def complexity_function_fitting():
    sub_x = np.linspace(-5, 5)
    random_i = np.random.randint(0, len(sub_x))
    left, right = sub_x[:random_i], sub_x[random_i:]

    output = np.concatenate((
        random_linear(sigmoid(random_linear(left))),
        random_linear(sigmoid(random_linear(right)))
    ))

    plt.plot(sub_x, output)


def topological_sort(graph: dict):#拓扑排序 1. topological sorting 
    """
    :graph: {
        'node': [adjacent1, adjacent2, .. adjacentN],
    }
    :return: the topological sorting for this graph
    """

    while graph:#图不为空时循环执行
        all_inputs = reduce(lambda a, b: a + b, map(list,graph.values()))#list与graph进行相加，reduce() 函数会对参数序列中元素进行累积 此函数功能是将value单独分离
        # print(all_inputs)
        '''
        map() 会根据提供的函数对指定序列做映射。
        >>> map(square, [1,2,3,4,5])   # 计算列表各个元素的平方
        [1, 4, 9, 16, 25]
        >>> map(lambda x: x ** 2, [1, 2, 3, 4, 5])  # 使用 lambda 匿名函数
        [1, 4, 9, 16, 25]
        '''
        need_remove = set(graph.keys()) - set(all_inputs)#输入减去输出 = 有输出无输入值

        if need_remove:  # len(need_remove) > 0 #将节点进行遍历输出，并将其删除输出了的节点
            node = random.choice(list(need_remove))#随机选择节点
            # print(node)#如b3
            exit_node = graph[node][0]#随机选择对应计算节点
            # print(exit_node)#如f5
            graph.pop(node)
            # print(graph)#b3:f5被移除

            yield node# yield用于返回多个值，本式存到node数组中
            if not graph: 
                yield exit_node #解决最后一个单独节点问题
            '''
            return：在程序函数中返回某个值，返回之后函数不在继续执行，彻底结束。
            yield: 带有yield的函数是一个迭代器，函数返回某个值时，会停留在某个位置，返回函数值后，会在前面停留的位置继续执行，            直到程序结束
            '''
        else:
            raise TypeError('the graph contain a cycle, the computing graph need acyclic graph')#有环图错误


def convert_feed_dict_to_graph(feed_dict: dict):
    computing_graph = defaultdict(list)#defaultdict(list),会构建一个默认value为list的字典，
    """
    from collections import defaultdict
    result = defaultdict(list)
    data = [("p", 1), ("p", 2), ("p", 3),
            ("h", 1), ("h", 2), ("h", 3)]
    
    for (key, value) in data:
        result[key].append(value)
    print(result)#defaultdict(<class 'list'>, {'p': [1, 2, 3], 'h': [1, 2, 3]})
    """
    nodes = list(feed_dict.keys())
    
    print(feed_dict.keys())
    print(feed_dict.values())
    
    while nodes:        #循环把节点连接起来，形成图
                        #node里没有f1,f2...计算节点
        n = nodes.pop(0)#删除表中内容
        print(n)
        if n in computing_graph: continue  #代替方案，直接初始化f1,f2,f3,f4，不需要通过append引出，需要有序？？？

        if isinstance(n, Placeholder):#判断两个类型是否相同推荐使用 isinstance()
            n.value = feed_dict[n]

        for m in n.outputs:
            computing_graph[n].append(m)#列表末尾添加新的对象.append()     computing_graph[n]是defaultdict类型，写法result[key].append(value)直接载入，与传统数组不同
            # print(n.outputs)
            # print(computing_graph)
            nodes.append(m)#连接,计算节点从这里被append进去
            print(nodes)

    return computing_graph#所有包括计算节点连成的图会被返回


def forward_and_backward(graph):
    for node in graph:#正向排序输出
        node.forward()

    for node in graph[::-1]:#反向排序输出
        node.backward()


def optimize(nodes, lr):
    for node in nodes:
        if node.trainable:
            node.value = node.value - node.loss_gradient[node] * lr

# remains
"""
[done] 1. topological sorting 
2. using topological sorting implement auto-grade
3. create a neural network framework
4. convert single-dimension version to multiply version
5. distribute neural network framework to internet (pip)
"""

if __name__ == '__main__':
    data = load_boston()

    x_data = data['data']
    y = data['target']
    desc = data['DESCR']

    # x, y ; x with 13 dimensions
    # let computer could predict house price using some features automatically

    # correlation analysis

    dataframe = pd.DataFrame(x_data)
    dataframe.columns = data['feature_names']
    dataframe['price'] = y

    rm = dataframe['RM']
    lstat = dataframe['LSTAT']
    y = dataframe['price']

    complex_graph = {#键值对
        'x': ['f1', 'f2'],
        'b1': ['f1'],
        'w1': ['f1'],
        'f1': ['f3'],
        'f3': ['f4', 'f5'],
        'f2': ['f5'],
        'w2': ['f2'],
        'b2':['f2'],
        'f5': ['loss'],
        'f4': ['loss'],
        'y': ['loss']
    }

    ic(list(topological_sort(complex_graph)))

