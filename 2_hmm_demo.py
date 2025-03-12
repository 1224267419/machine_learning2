import numpy as np
from hmmlearn import hmm
import math
def demo0():
    # 设定隐藏状态的集合
    states = ["box 1", "box 2", "box3"]
    n_states = len(states)

    # 设定观察状态的集合
    observations = ["red", "white"]
    n_observations = len(observations)

    # 设定初始状态分布
    start_probability = np.array([0.2, 0.4, 0.4])

    # 设定状态转移概率分布矩阵
    transition_probability = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])

    # 设定观测状态概率矩阵
    emission_probability = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ])
    # 设定模型参数
    model = hmm.CategoricalHMM(n_components=n_states)
    model.startprob_ = start_probability  # 初始状态分布
    model.transmat_ = transition_probability  # 状态转移概率分布矩阵
    model.emissionprob_ = emission_probability  # 观测状态概率矩阵
    seen = np.array([[0, 1, 0]]).reshape(-1,1)  # 设定观测序列
    model.n_trials=0
    box = model.predict(seen)

    #用map函数实现了一一对应
    print("球的观测顺序为：\n", ", ".join(map(lambda x: observations[x], seen.flatten())))
    # 注意：需要使⽤flatten⽅法，把seen从⼆维变成⼀维
    print("最可能的隐藏状态序列为:\n", ", ".join(map(lambda x: states[x], box)))

    print(model.score(seen))
    print("该隐藏状态序列的概率为:",math.exp(model.score(seen)))
if __name__== '__main__':
    demo0()