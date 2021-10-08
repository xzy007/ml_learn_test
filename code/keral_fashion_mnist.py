import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.util import compat
import os
from tensorflow.python.keras import *
from tensorflow.python import keras
# keras一般通过tensorflow.keras来使用，但是pycharm没有提示，原因是因为实际的keras路径放在tensorflow/python/keras
# try:
#     from tensorflow.python import keras
#     from tensorflow.python.keras import layers
#     from tensorflow.python.keras import losses
#     from tensorflow.python.keras import optimizers
#     from tensorflow.python.keras import metrics
#     from tensorflow.python.keras import utils
#     from tensorflow.python.keras import backend as K
#     from tensorflow.python.keras import callbacks
# except:
#     from tensorflow.keras import keras
base_path = "../data/housing" #

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
    X_train_full, Y_train_full = load_mnist("%s/"%base_path, "train")
    print(X_train_full.shape)
    print(X_train_full.dtype)
    X_test, Y_test = load_mnist("%s/" % base_path, "t10k")
    # print(X_test[:2], Y_test[:2])
    #训练数据，再次切为训练数据+验证集数据
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    Y_valid, Y_train = Y_train_full[:5000], Y_train_full[5000:]
    # label
    # class_names = 'T-shirt/top,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,Ankle boot'.split(',')
    # print(class_names[Y_test[0]])
    return X_train, Y_train, X_valid, Y_valid

#导出模型pb
def export_savedmodel(model):
    '''
        传入keras model会自动保存为pb格式
    '''
    model_path = "model/" # 模型保存的路径
    model_version = 0 # 模型保存的版本
    # 从网络的输入输出创建预测的签名
    model_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input': model.input},
        outputs={'output': model.output})
    # 使用utf-8编码将 字节或Unicode 转换为字节
    export_path = os.path.join(compat.as_bytes(model_path), compat.as_bytes(str(model_version))) # 将保存路径和版本号join
    builder = tf.saved_model.builder.SavedModelBuilder(export_path) # 生成"savedmodel"协议缓冲区并保存变量和模型
    builder.add_meta_graph_and_variables( # 将当前元图添加到savedmodel并保存变量
        sess=K.get_session(), # 返回一个 session 默认返回tf的sess,否则返回keras的sess,两者都没有将创建一个全新的sess返回
        tags=[tf.saved_model.tag_constants.SERVING], # 导出模型tag为SERVING(其他可选TRAINING,EVAL,GPU,TPU)
        clear_devices=True, # 清除设备信息
        signature_def_map={ # 签名定义映射
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: # 默认服务签名定义密钥
            model_signature # 网络的输入输出策创建预测的签名
            })
    builder.save() # 将"savedmodel"协议缓冲区写入磁盘.
    print("save model pb success ...")
# 构造模型 - 顺序式构造 原始结构
def create_model():
    X_train_full, Y_train_full = load_mnist("%s/" % base_path, "train")
    print(X_train_full.shape)
    print(Y_train_full.dtype)
    X_test, Y_test = load_mnist("%s/" % base_path, "t10k")
    X_test = X_test / 255.0
    print(X_test.shape)
    print(X_test.dtype)
    #训练数据，再次切为训练数据+验证集数据
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0 #255
    Y_valid, Y_train = Y_train_full[:5000], Y_train_full[5000:]
    print(X_train.shape)
    print(X_valid.dtype)

    # class_names = 'T-shirt/top,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,Ankle boot'.split(',')
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid('off')
    #     plt.imshow(X_test[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[Y_test[i]])  #[]
    # plt.show()
    # exit(0)

    model = models.Sequential([
        layers.Flatten(input_shape=[28, 28]),
        layers.Dense(300, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(10, activation='softmax')
        ]
    )
    print(model.summary())

    # 获取模型参数细节
    # print(model.layers)
    # w, b = model.layers[1].get_weights()

    # 编译模型
    model.compile(loss='sparse_categorical_crossentropy', #losses.sparse_categorical_crossentropy 针对稀疏label设置的
                  optimizer='sgd', #optimizers.SGD()
                  metrics=['accuracy'])#metrics.accuracy()

    # 训练模型
    # print(X_train[:1, :, :])
    # print(Y_train[:1])
    print(X_train.shape)
    print(Y_train.shape)
    history = model.fit(X_train, Y_train, batch_size=32, epochs=1, validation_data=(X_valid, Y_valid))

    # 参数显示
    # print(history.history)
    # pd.DataFrame(history.history).plot(figsize=(8, 5))
    # plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    # plt.show()

    # test评估
    eval = model.evaluate(X_test, Y_test, batch_size=32)
    print(eval)

    # 单独预测
    # batch_size = 32
    # total_loss = 0.0
    # total_num = 0
    # for i in range(313):
    #     y_p = model.predict(X_test[i*batch_size:(i+1) * batch_size])
    #     loss = losses.sparse_categorical_crossentropy(Y_test[i*batch_size : (i+1) * batch_size], y_p)
    #     total_loss += np.sum(loss.numpy())
    #     total_num += len(loss.numpy())
    # print(total_num)
    # print(total_loss / float(total_num))

    #保存
    model.save("%s/my_keras_model.h5"%base_path)
    keras.models.save_model(model, "%s/my_keras_model_pb"%base_path)
    # 加载
    model2 = keras.models.load_model("%s/my_keras_model.h5"%base_path)
    eval = model2.evaluate(X_test, Y_test, batch_size=32)
    print(eval)

    model3 = keras.models.load_model("%s/my_keras_model_pb"%base_path)
    eval = model3.evaluate(X_test, Y_test, batch_size=32)
    print(eval)


    return model

# 构造模型的多样行
def create_model_v1():
    # 数据获取处理
    X_train_full, Y_train_full = load_mnist("%s/" % base_path, "train")
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    Y_valid, Y_train = Y_train_full[:5000], Y_train_full[5000:]
    X_test, Y_test = load_mnist("%s/" % base_path, "t10k")
    X_test = X_test / 255.0
    # 构建模型
    model = keras.models.Sequential([
        layers.Flatten(input_shape=[28, 28]),
        layers.Dense(300, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(10, activation='softmax')
        ]
    )
    print(model.summary())

    # 编译模型
    model.compile(loss='sparse_categorical_crossentropy', #losses.sparse_categorical_crossentropy 针对稀疏label设置的
                  optimizer='sgd', #optimizers.SGD()
                  metrics=['accuracy'])#metrics.accuracy()

    # 训练评估模型
    history = model.fit(X_train, Y_train, batch_size=32, epochs=30, validation_data=(X_valid, Y_valid))

    # test评估
    eval = model.evaluate(X_test, Y_test, batch_size=32)
    print(eval)

    # 保存加载验证
    model.save("%s/my_keras_model.h5" % base_path) #训练完成后，保存模型
    model2 = keras.models.load_model("%s/my_keras_model.h5" % base_path)
    eval = model2.evaluate(X_test, Y_test, batch_size=32)
    print(eval)

class WideAndDeepModel(Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = layers.Dense(units, activation=activation)
        self.hidden2 = layers.Dense(units, activation=activation)
        self.main_out = layers.Dense(1)
        self.aux_out = layers.Dense(1)
    def call(self, inputs):
        inputs_A, inputs_B = inputs
        hidden1 = self.hidden1(inputs_B)
        hidden2 = self.hidden2(hidden1)
        concat = layers.concatenate([inputs_A, hidden2])
        main_out = self.main_out(concat)
        aux_out = self.aux_out(hidden2)
        return main_out, aux_out

def housing_data_train():
    housing = fetch_california_housing()
    X_train_full,  X_test, Y_train_full, Y_test = train_test_split(housing.data, housing.target)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full)
    # 仅做伸缩变化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_valid = scaler.transform(X_valid)
    X_new = X_test[:3]
    print(X_train.shape)

    # 构建多路输入
    X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
    X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
    X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
    X_new_A, X_new_B = X_new[:, :5], X_new[:, 2:]
    # 建模
    # 1: 顺序建模
    # model = keras.models.Sequential([
    #     layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]), #
    #     layers.Dense(1)
    # ])

    # 2：拓扑结构1 函数式编程
    # input_ = layers.Input(shape=X_train.shape[1:])
    # dense1 = layers.Dense(30, activation='relu')(input_)
    # dense2 = layers.Dense(30, activation='relu')(dense1)
    # concat = layers.Concatenate()([input_, dense2])
    # output = layers.Dense(1)(concat)
    # model = keras.Model(inputs=[input_], outputs=[output])
    # print(model.summary())

    # 3：拓扑结构2 函数式编程 多路输入
    # inputA = layers.Input(shape=[5], name='wide_input')
    # inputB = layers.Input(shape=[6], name='deep_input')
    # dense1 = layers.Dense(30, activation='relu')(inputB)
    # dense2 = layers.Dense(30, activation='relu')(dense1)
    # concat = layers.Concatenate()([inputA, dense2])
    # output = layers.Dense(1)(concat)
    # model = keras.Model(inputs=[inputA, inputB], outputs=[output])
    # print(model.summary())

    # 4：拓扑结构3 函数式编程 多路输入，多路输出
    inputA = layers.Input(shape=[5], name='wide_input')
    inputB = layers.Input(shape=[6], name='deep_input')
    dense1 = layers.Dense(30, activation='relu')(inputB)
    dense2 = layers.Dense(30, activation='relu')(dense1)
    concat = layers.Concatenate()([inputA, dense2])
    output = layers.Dense(1)(concat)
    aux_output = layers.Dense(1)(dense2)
    model = keras.Model(inputs=[inputA, inputB], outputs=[output, aux_output])
    print(model.summary())

    # 5：拓扑结构4 命令式编程，子类化 多路输入，多路输出
    # inputA = layers.Input(shape=[5], name='wide_input')
    # inputB = layers.Input(shape=[6], name='deep_input')
    # output, aux_output = WideAndDeepModel()([inputA, inputB])
    # model = keras.Model(inputs=[inputA, inputB], outputs=[output, aux_output])
    # print(model.summary())

    # 编译
    model.compile(loss=['mean_squared_error', 'mean_squared_error'],
                  loss_weights=[0.9, 0.1],
                  optimizer='sgd')
    # 回调函数设置
    model_file = "%s/housing_wide_deep_model.h5"%base_path
    checkpoint_cb = callbacks.ModelCheckpoint(model_file, save_best_only=True)
    early_stop_cb = callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # 训练
    histroy = model.fit([X_train_A, X_train_B], [Y_train, Y_train],
                        epochs=20,
                        validation_data=((X_valid_A, X_valid_B), [Y_valid, Y_valid]),
                        callbacks=[checkpoint_cb, early_stop_cb])

    # 评估
    eval = model.evaluate((X_test_A, X_test_B), [Y_test, Y_test])
    print("eval:", eval)

    # 单独预测
    y_pred, y_pred_aux = model.predict((X_new_A, X_new_B))
    print("y_true:", Y_test[:3])
    print("y_pred:", y_pred)

    # # 保存模型
    # model.save("%s/housing_wide_deep_model.h5"%base_path)
    # # 加载模型
    # model_new = keras.models.load_model("%s/housing_wide_deep_model.h5"%base_path)
    # eval = model_new.evaluate((X_test_A, X_test_B), [Y_test, Y_test])
    # print("new eval:", eval)



housing_data_train()


