import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.util import compat
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import activations
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import utils
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
def create_sample_model():
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

    #构建
    model = keras.models.Sequential([
        layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]), #
        layers.Dense(1)
    ])
    print(model.summary())

    # 编译模型
    model.compile(loss=huber_fn, optimizer='nadam')


    # 训练
    histroy = model.fit(X_train, Y_train,
                        epochs=1,
                        validation_data=(X_valid, Y_valid))

    # 评估
    eval = model.evaluate(X_test, Y_test)
    print("eval:", eval)

    # 单独预测
    y_pred = model.predict(X_new)
    print("y_true:", Y_test[:3])
    print("y_pred:", y_pred)

base_path = "../data/housing" #
# 自定损失函数--函数式，比较不自由，简单模式
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    loss = tf.where(is_small_error, tf.square(error) / 2, tf.abs(error) - 0.5)
    return loss
# 损失函数类形式，复杂模式
class HuberLoss(losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs) #外界可以通过字典调整父类两个构造参数，name, reduction 的值
    def call(self, y_true, y_pred):
        threshold = self.threshold
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        loss = tf.where(is_small_error, tf.square(error) / 2, threshold * tf.abs(error) - threshold**2 / 2)
        return loss
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold":self.threshold}

# 自定义初始化/正则化/约束条件
def my_softplus(x):
    return tf.math.log(tf.exp(x) + 1.0) #低阶api
def my_glorot_initializer(shape, dytpe=tf.float32): #输入输出梯度的变化保持一致
    stddev = tf.sqrt(2.0 / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dytpe=dytpe)
def my_l1_regularizer(w):
    return tf.reduce_sum(tf.abs(0.01 * w))
def my_positive_weights(w):
    return tf.where(w < 0.0, tf.zeros_like(w), w)
# 类的模式
class MySoftPlus(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
    def call(self, inputs, *args, **kwargs):
        return  tf.math.log(tf.exp(inputs) + 1.0) #低阶api
class MyL1Regularizer(regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, w):
        return tf.reduce_sum(tf.abs(self.factor * w))
    def get_config(self): # 非父类定义的
        return {"factor":self.factor}
class MyGlorotInitializer(initializers.initializers_v2.Initializer):
    def __init__(self):
        pass
    def __call__(self, shape, dtype=tf.float32):
        stddev = tf.sqrt(2.0 / (shape[0] + shape[1]))
        return tf.random.normal(shape, stddev=stddev, dtype=dtype)
class MyPositiveWeights(constraints.Constraint):
    def __init__(self):
        pass
    def __call__(self, w):
        return tf.where(w < 0.0, tf.zeros_like(w), w)

def create_huber(threshold):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        loss = tf.where(is_small_error, tf.square(error) / 2, threshold * tf.abs(error) - threshold**2 / 2)
        return loss
    return huber_fn
# 自定义指标 流式指标
class HuberMetric(metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros") #初始化
        self.count = self.add_weight("count", initializer="zeros")
    def update_state(self, y_true, y_pred, *args, **kwargs):
        metric = self.huber_fn(y_true, y_pred) #定义如何更新指标
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), dtype=tf.float32))
    def result(self): #返回结果
        return self.total / self.count
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold":self.threshold}

# 自定义层
class MyDense(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(kwargs)
        self.units = units
        self.activation = activations.get(activation)

    def build(self, input_shape): #定义图的单元结构
        self.kernel = self.add_weight(name="kernel", shape=[input_shape[-1], self.units], initializer='glorot_normal')
        self.bias = self.add_weight(name='bias', shape=[self.units], initializer='zeros')
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs): #定义流式过程
        return self.activation(inputs @ self.kernel + self.bias)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape.as_list()[:-1] + [self.units])
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units":self.units, "activation":activations.serialize(self.activation)}
class MyMultiLayer(layers.Layer): #多路输入输出
    def __init__(self, **kwargs):
        super().__init__(kwargs)
    def call(self, inputs, *args, **kwargs):
        x1, x2 = inputs
        return [x1 + x2, x1 * x2, x1 / x2]
    def compute_output_shape(self, input_shape):
        b1, b2 = input_shape
        return [b1, b1, b1]
class MyGaussianNoise(layers.Layer): #同流程不同时机做不同操作
    def __init__(self, stddev, **kwargs):
        super().__init__(kwargs)
        self.stddev = stddev
    def call(self, inputs, is_train=True, *args, **kwargs):
        if is_train:
            noise = tf.random.normal(tf.shape(inputs), stddev=self.stddev)
            return inputs + noise
        else:
            return inputs
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "stddev":self.stddev}
class ResidualBlock(layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super(ResidualBlock, self).__init__(kwargs)
        self.hidden = [layers.Dense(n_neurons, activation='elu', kernel_initializer='he_normal') for _ in range(n_layers)]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x


# 自定义模型,本质是层的子类
class ResidualRegressor(keras.Model):
    def __init__(self, out_dim, **kwargs):
        super().__init__(kwargs)
        self.hidden1 = layers.Dense(30, activation='elu', kernel_initializer='he_normal')
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = layers.Dense(out_dim)

    def call(self, inputs, training=None, mask=None):
        x = self.hidden1(inputs)
        for _ in range(1 + 3):
            x = self.block1(x)
        x = self.block2(x)
        return self.out(x)
class ReconstructingRegressor(keras.Model):
    def __init__(self,out_dim,**kwargs):
        super().__init__(**kwargs)
        self.hidden = [layers.Dense(30, activation='selu', kernel_initializer='lecun_normal') for _ in range(5)]
        self.out = layers.Dense(out_dim)
        self.mean_metrics = metrics.Mean()
    def build(self, input_shape):
        input_n = input_shape[-1]
        self.reconstruct = layers.Dense(input_n)
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        reconstruct = self.reconstruct(x)
        recon_loss = tf.reduce_mean(tf.square(reconstruct - inputs))
        metrics_ = self.mean_metrics(recon_loss)
        self.add_loss(0.05 * recon_loss)
        self.add_metric(metrics_, name='recon_loss')
        return self.out(x)




def run_auto_define_class():
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

    #构建
    # model = keras.models.Sequential([
    #     layers.Dense(30,
    #                  activation=MySoftPlus(),
    #                  input_shape=X_train.shape[1:],
    #                  kernel_initializer=MyGlorotInitializer(),
    #                  kernel_constraint=MyPositiveWeights(),
    #                  kernel_regularizer=MyL1Regularizer(0.01)),
    #     MyDense(15, activation=MySoftPlus()),
    #     layers.Dense(1)
    # ])
    input = layers.Input(shape=[X_train.shape[-1]], name='input_x')
    # out = ResidualRegressor(1)(input)
    out = ReconstructingRegressor(1)(input)
    model = keras.Model(inputs=[input], outputs=[out])

    print(model.summary())

    # 编译模型
    model.compile(loss=HuberLoss(1.0), optimizer='nadam',
                  metrics=[HuberMetric(1.0)])


    # 训练
    histroy = model.fit(X_train, Y_train,
                        epochs=1,
                        validation_data=(X_valid, Y_valid))

    # 评估
    eval = model.evaluate(X_test, Y_test)
    print("eval:", eval)

    # 单独预测
    y_pred = model.predict(X_new)
    print("y_true:", Y_test[:3])
    print("y_pred:", y_pred)
    # 保存模型
    model.save_weights("%s/auto_define_model.h5.w"%base_path)
    # tf.saved_model.save(model, "%s/auto_define_pb_model"%base_path)

    # 加载模型
    new_model = keras.Model(inputs=[input], outputs=[out])
    new_model.load_weights("%s/auto_define_model.h5.w"%base_path)
    print(new_model.summary())
    y_pred = new_model.predict(X_new)
    print("y_true:", Y_test[:3])
    print("y_pred:", y_pred)
    # model = tf.saved_model.load("%s/auto_define_pb_model"%base_path)
    # pred_fun = model.signatures["serving_default"]
    # out = pred_fun(input_x=tf.constant(X_new.tolist()))  # 这里的输入要转化为列表
    # print(out)

def load_model():
    # 加载模型
    model = keras.models.load_model("%s/auto_define_model.h5"%base_path,
                                    custom_objects={"HuberLoss":HuberLoss, #会调用from_config 自动构造对象
                                                    "MySoftPlus":MySoftPlus,
                                                    "MyGlorotInitializer":MyGlorotInitializer,
                                                    "MyPositiveWeights": MyPositiveWeights,
                                                    "MyL1Regularizer": MyL1Regularizer,
                                                    "MyDense":MyDense,
                                                    "HuberMetric":HuberMetric
                                                    })


# load_model()
# run_auto_define_class()
b=np.array([1, 2])
a=np.array([[1, 2, 3], [4, 5, 6]])
np.savez('123.npz', a=a, b=b)
data = np.load('123.npz')
print(data)
print(data["a"])
print(data["b"])