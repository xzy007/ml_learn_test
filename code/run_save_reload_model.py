import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.util import compat
import os
from tensorflow.python import keras
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import utils
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
if tf.executing_eagerly():
   tf.compat.v1.disable_eager_execution()

base_path = "../data/housing" #
def train_model():
    housing = fetch_california_housing()
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(housing.data, housing.target)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full)
    # 仅做伸缩变化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_valid = scaler.transform(X_valid)
    X_new = X_test[:3]
    print(X_train.shape)
    # 建模
    # 1: 顺序建模
    input_ = layers.Input(shape=X_train.shape[1:], name='x')
    input_2 = layers.Input(shape=X_train.shape[1:], name='x2')
    dense1 = layers.Dense(30, activation='relu')(input_)
    dense2 = layers.Dense(30, activation='relu')(dense1)
    concat = layers.Concatenate()([input_, dense2])
    output = layers.Dense(1, name='score')(concat)
    model = keras.Model(inputs=[input_, input_2], outputs=[output])
    print(model.summary())

    # 编译
    model.compile(loss=['mean_squared_error'],
                  optimizer='sgd')

    # 回调函数设置
    model_file = "%s/housing_wide_deep_model.h5"%base_path
    # run_logdir = "%s/my_logs/%s"%(base_path, time.strftime("run_%Y_%m_%d_%H_%M_%S"))
    # tensorboard_cb = callbacks.TensorBoard(run_logdir, histogram_freq=1, write_grads=True)
    checkpoint_cb = callbacks.ModelCheckpoint(model_file, save_best_only=True)
    early_stop_cb = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    # 训练
    histroy = model.fit([X_train, X_train], Y_train,
                        epochs=10,
                        validation_data=([X_valid, X_valid], Y_valid),
                        callbacks=[checkpoint_cb, early_stop_cb],
                        verbose=1
                        )

    # 评估
    eval = model.evaluate([X_test,X_test], Y_test)
    print("eval:", eval)
    exit(0)
    # 单独预测
    y_pred = model.predict([X_new, X_new])
    print("y_true:", Y_test[:3])
    print("y_pred:", y_pred)

    # # 保存模型
    model.save("%s/housing_wide_deep_model.h5"%base_path)
    # 加载模型
    model_new = keras.models.load_model("%s/housing_wide_deep_model.h5"%base_path)
    eval = model_new.evaluate([X_test,X_test], Y_test)
    print("new eval:", eval)

    # 保存为pb结构
    # tf.keras.models.save_model(model_new, "%s/housing_wide_deep_model_pb"%base_path)
    keras.models.save_model(model_new, "%s/housing_wide_deep_model_pb"%base_path)
    # export_savedmodel(model_new, "F:\\tmp/model_pb", "housing_wide_deep_model_pb")
    sess = K.get_session()
    meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], "%s/housing_wide_deep_model_pb"%base_path)
    signature = meta_graph_def.signature_def
    print(signature)
def export_savedmodel(model, output_dir, model_name):
    # 从网络的输入输出创建预测的签名
    print(model.input)
    inputs = {
        "x": tf.compat.v1.saved_model.build_tensor_info(model.input[0])
    }
    outputs = {
        "score":tf.compat.v1.saved_model.build_tensor_info(model.output)
    }
    print(outputs)
    # for key, tensor in inputs.items():
    #     print(type(tensor))
    #     print(tensor.get_shape().as_proto())
    # exit(0)
    # model_signature = tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(inputs=inputs, outputs=outputs)
    model_signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    sess = tf.compat.v1.Session() #K.get_session()
    sess.run(tf.compat.v1.initialize_all_variables())
    signature_def_map = {'%s_signa' % model_name: model_signature}

    model_path = "%s/%s"%(output_dir, model_name)
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(model_path) # 生成"savedmodel"协议缓冲区并保存变量和模型
    builder.add_meta_graph_and_variables( # 将当前元图添加到savedmodel并保存变量
        sess=sess, # 返回一个 session 默认返回tf的sess,否则返回keras的sess,两者都没有将创建一个全新的sess返回
        tags=[tf.compat.v1.saved_model.tag_constants.SERVING], # 导出模型tag为SERVING(其他可选TRAINING,EVAL,GPU,TPU)
        clear_devices=True, # 清除设备信息
        signature_def_map=signature_def_map,
        main_op=tf.compat.v1.tables_initializer(),
        strip_default_attrs=True
    )
    builder.save() # 将"savedmodel"协议缓冲区写入磁盘.
    print("save model pb success ...")



def reload_pb_model():
    housing = fetch_california_housing()
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(housing.data, housing.target)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full)
    # 仅做伸缩变化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_valid = scaler.transform(X_valid)
    X_new = X_test[:3]
    print(X_train.shape)


    # new eval: 0.3423307538032532
    sess = K.get_session()
    meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], "%s/housing_wide_deep_model_pb"%base_path)
    signature = meta_graph_def.signature_def
    outputs_key = ["score"]
    input_keys = ["x"]
    signature_key = 'serving_default'
    outputs_tensor = {val: sess.graph.get_tensor_by_name(signature[signature_key].outputs[val].name) for val in outputs_key}
    print(outputs_tensor)
    inputs_tensor = {val: sess.graph.get_tensor_by_name(signature[signature_key].inputs[val].name) for val in input_keys}
    print(inputs_tensor)

    x = {"x": [[0.0 for i in range(8)]]}
    s = {inputs_tensor[key]: val for key, val in x.items()}
    q_eval = sess.run([outputs_tensor[n] for n in outputs_key], feed_dict=s)
    print("============================ score ============================")
    print(q_eval)
    print("============================ score ============================")

train_model()