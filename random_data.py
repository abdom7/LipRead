import numpy as np

label_array = range(51)

def Data(max_seqlen=30, image_size=1600, data_num=1000):
    return np.random.rand(data_num, max_seqlen, image_size)

def Label(data_num=1000):
    return np.random.choice(label_array, size=data_num)


def Train(data_num=6000):
    return Data(data_num=data_num), Label(data_num=data_num)


def Test(data_num=1000):
    return Data(data_num=data_num), Label(data_num=data_num)


def Val(data_num=1000):
    return Data(data_num=data_num), Label(data_num=data_num)

