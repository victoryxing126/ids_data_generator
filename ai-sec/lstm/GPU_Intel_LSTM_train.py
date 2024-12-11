#coding=utf-8
import os
import json
import base64
import hashlib
import datetime
import sys
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from configures.app_conf import *
import warnings
import pymysql
import pandas

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

def DictException(e):
    """整理异常对象为dict"""
    try:
        return {
            'summary': str(e),
            'err_file': e.__traceback__.tb_frame.f_globals["__file__"],
            'err_line': e.__traceback__.tb_lineno
        }
    except:
        return False

def Md5Strs(strings):
    """返回字符串的MD5签名"""
    return hashlib.md5(str(strings).encode('utf-8')).hexdigest()

def Md5File(path):
    try:
        with open(path, 'rb') as f:
            md5_obj = hashlib.md5()
            md5_obj.update(f.read())
            return dict(code=200, data=md5_obj.hexdigest(), msg='成功')
    except Exception as e:
        return dict(code=500, data=DictException(e), msg='失败')

def DB_Conn():
    """创建并返回数据库连接"""
    return pymysql.connect(
        host=DB_CONN_DICT.get("host"),
        port=DB_CONN_DICT.get("port"),
        user=DB_CONN_DICT.get("user"),
        password=DB_CONN_DICT.get("password"),
        database=DB_CONN_DICT.get("database")
    )

def DB_ChangeSql(dbcon, sql_statment, isCommit=True, isClose=False):
    """执行SQL语句并处理异常"""
    try:
        if str(sql_statment).lower().startswith(("create", "drop", "truncate", "alter")):
            return {'code': 500, 'data': '', 'msg': '禁止执行敏感SQL'}
        
        if str(sql_statment).lower().startswith("delete") and "where" not in str(sql_statment).lower():
            return {'code': 500, 'data': '', 'msg': '禁止执行无WHERE限制的DELETE语句！'}
        
        dbcon.cursor().execute(sql_statment)
        if isCommit:
            dbcon.commit()
        if isClose:
            dbcon.close()
        return RESP_200
    except Exception as e:
        _err = DictException(e)
        dbcon.rollback()
        return {'code': 500, 'data': _err, 'msg': 'SQL错误'}

def DirTravelToFileList(original_trainData_dir="./0-Data-TrainTest"):
    """遍历目录返回子文件列表"""
    return [os.path.join(original_trainData_dir, f) for f in os.listdir(original_trainData_dir) if os.path.isfile(os.path.join(original_trainData_dir, f))]

def Base64Encode(strings):
    """将字符串转成Base64码返回"""
    try:
        return base64.b64encode(strings.encode("utf8")).decode("utf8")
    except Exception as e:
        print(DictException(e))
        return False

def Base64Decode(strings):
    """将Base64码还原成字符串返回"""
    try:
        return base64.b64decode(strings.encode("utf8")).decode("utf8")
    except Exception as e:
        print(DictException(e))
        return False

def CalcMaxSampleLength():
    """返回最大样本长度"""
    con = DB_Conn()
    _max = pandas.read_sql(""" SELECT GREATEST(
        (SELECT MAX(test.length_sample) FROM sec_ai.data_test_list test),
        (SELECT MAX(train.length_sample) FROM sec_ai.data_train_list train),
        (SELECT MAX(val.length_sample) FROM sec_ai.data_val_list val)
    ) AS max """, con).to_dict(orient="records")[0].get("max")
    con.commit()
    con.close()
    return int(_max * 1.1)

# 设置TensorFlow使用Intel GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

START_TIME = int(time.time())
model_name = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
conn = DB_Conn()

print("开始读取训练数据集...")
x_train_data = pandas.read_sql(SQLQueyData.format(table_name="data_train_list"), conn).to_dict(orient='list')
x_train = x_train_data.get("sample")
x_train_tag = x_train_data.get("is_abnormal")   

# 读取验证数据集
x_validate_data = pandas.read_sql(SQLQueyData.format(table_name="data_val_list"), conn).to_dict(orient='list')
x_val = x_validate_data.get("sample")
y_val_tag = x_validate_data.get("is_abnormal")   

# 实例化 Tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x_train + x_val)

# 文本转换为整数序列
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_val_seq = tokenizer.texts_to_sequences(x_val)

# 计算 pad_sequences 的 maxlen 参数
_maxlen = max([max(x_train_data.get("length_sample")), max(x_validate_data.get("length_sample"))])
x_train_padded = pad_sequences(x_train_seq, maxlen=_maxlen)
x_val_padded = pad_sequences(x_val_seq, maxlen=_maxlen)

# 将标签列表转换为 numpy 数组
x_train_tag = np.array(x_train_tag)
y_val_tag = np.array(y_val_tag)

# 定义模型
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=_maxlen),
    tf.keras.layers.LSTM(units=128),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 使用生成器按需加载数据
def data_generator(features, labels, batch_size):
    while True:
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            yield features[start:end], labels[start:end]

batch_size = 32
train_gen = data_generator(x_train_padded, x_train_tag, batch_size)
val_gen = data_generator(x_val_padded, y_val_tag, batch_size)

# 训练模型
history = model.fit(train_gen, steps_per_epoch=len(x_train_padded) // batch_size, epochs=int(input("输入训练迭代轮次（epochs, 正整数）：")),
                    validation_data=val_gen, validation_steps=len(x_val_padded) // batch_size)

# 保存模型
_path_h5 = f"./SaveModel/{model_name}_model.h5"
model.save(_path_h5)
_path_SavedModel = f"./SaveModel/{model_name}_SavedModel.keras"
model.save(_path_SavedModel)

# 在DB记录模型摘要信息
from tensorflow.keras.layers import Embedding, LSTM, Dense
for layer in model.layers:
    if isinstance(layer, LSTM):
        LSTM_CONF_INFO = json.dumps(layer.get_config())
        LSTM_UNITS = layer.get_config()['units']
    elif isinstance(layer, Embedding):
        Embedding_CONF_INFO = json.dumps(layer.get_config())
    elif isinstance(layer, Dense):
        Dense_CONF_INFO = json.dumps(layer.get_config())

optimizer = str(model.optimizer.name)
loss_function = str(model.loss)
learning_rate = round(float(model.optimizer.learning_rate.numpy()), 5)
pad_sequences_maxlen = _maxlen
epochs = model.history.params['epochs']
training_end_date = str(datetime.datetime.now())
duration = int(time.time() - START_TIME)

right_train_sample_num = x_train_data.get("is_abnormal").count(0)
wrong_train_sample_num = x_train_data.get("is_abnormal").count(1)
right_val_sample_num = x_validate_data.get("is_abnormal").count(0)
wrong_val_sample_num = x_validate_data.get("is_abnormal").count(1)
train_sample_ids = ",".join([str(d) for d in x_train_data.get("id")])
val_sample_ids = ",".join([str(d) for d in x_validate_data.get("id")])
final_train_accuracy = history.history['accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]

DB_ChangeSql(DB_Conn(), f""" INSERT INTO sec_ai.models_trained_lstm 
(final_val_loss, final_train_loss, final_train_accuracy, final_val_accuracy, tf_keras_layers_LSTM_conf, tf_keras_layers_Embedding_conf, tf_keras_layers_Dense_conf, LSTM_UNITS, right_train_sample_num,
wrong_train_sample_num, right_val_sample_num, wrong_val_sample_num, train_sample_ids, val_sample_ids, 
model_name,  optimizer, loss_function, learning_rate, batch_size, pad_sequences_maxlen, epochs,
model_file_path_h5, model_file_path_SavedModel, training_end_date, train_duration_seconds) 
VALUES
( '{final_val_loss}','{final_train_loss}','{final_train_accuracy}','{final_val_accuracy}', '{LSTM_CONF_INFO}', '{Embedding_CONF_INFO}', '{Dense_CONF_INFO}', '{LSTM_UNITS}', '{right_train_sample_num}',
'{wrong_train_sample_num}', '{right_val_sample_num}', '{wrong_val_sample_num}', '{train_sample_ids}', '{val_sample_ids}',
'{model_name}', '{optimizer}', '{loss_function}', '{learning_rate}', '{batch_size}', '{pad_sequences_maxlen}', {epochs},
'{_path_h5}', '{_path_SavedModel}', '{training_end_date}', '{duration}'); """)

try:
    conn.close()
except:
    pass
print(f"训练耗时：{duration} 秒")
print("""====END====""")
