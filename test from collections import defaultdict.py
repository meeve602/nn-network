from collections import defaultdict

computing_graph = defaultdict(list)#defaultdict(list),会构建一个默认value为list的字典，
"""
for (key, value) in data:
    result[key].append(value)
print(result)#defaultdict(<class 'list'>, {'p': [1, 2, 3], 'h': [1, 2, 3]})
"""
n = 'p'
m = [1,2,23]
computing_graph[n].append(m)
print(computing_graph)
