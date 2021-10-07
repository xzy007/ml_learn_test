import tensorflow as tf
# keras一般通过tensorflow.keras来使用，但是pycharm没有提示，原因是因为实际的keras路径放在tensorflow/python/keras
try:
    from tensorflow.python import keras
    from tensorflow.python.keras import layers
except:
    from tensorflow.keras import keras
base_path = "../data/fashion_mnist_data"

# 下载fastion数据
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28, 28) #784

    return images, labels
def load_fastion_mnist_data():
    # fashion_mnist = keras.datasets.fashion_mnist
    # (X_train_full,Y_train_full), (X_test, Y_test) = fashion_mnist.load_data()
    X_train_full, Y_train_full = load_mnist("%s/"%base_path, "train")
    print(X_train_full.shape)
    print(X_train_full.dtype)
    X_test, Y_test = load_mnist("%s/" % base_path, "t10k")
    print(X_test[:2], Y_test[:2])
    #训练数据，再次切为训练数据+验证集数据
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    Y_valid, Y_train = Y_train_full[:5000], Y_train_full[5000:]
    # label
    class_names = 'T-shirt/top,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,Ankle boot'.split(',')
    print(class_names[Y_test[0]])
    return X_train, Y_train, X_valid, Y_valid

# 构造模型 - 顺序式构造
def create_model():
    # X_train, Y_train, X_valid, Y_valid = load_fastion_mnist_data()
    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28])) #该层位适配的问题，其实就是reshape的操作，转为一维数组，[-1, 1]
    model.add(layers.Dense(300, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(1, activation='softmax'))
    print(model.summary())

    # 获取模型参数细节
    print(model.layers)
    print(model.layers[0].name)
    
    return model


create_model()

