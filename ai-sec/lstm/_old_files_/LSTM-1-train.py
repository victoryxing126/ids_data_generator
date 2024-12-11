#coding=utf-8
from common_func import * 

START_TIME = int(time.time())
model_name = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
# 创建数据库连接
conn = DB_Conn()

print("开始读取训练数据集...")
x_train_data = pandas.read_sql(SQLQueyData.format(table_name="data_train_list"), conn).to_dict(orient='list')
x_train = x_train_data.get("sample")
#针对 x_train 的打标结果
x_train_tag = x_train_data.get("is_abnormal")   


#二、读取验证数据集 (x_val):
x_validate_data = pandas.read_sql(SQLQueyData.format(table_name="data_val_list"), conn).to_dict(orient='list')
#验证数据集
x_val = x_validate_data.get("sample")
#针对 x_val 的打标结果
y_val_tag = x_validate_data.get("is_abnormal")   


#三、实例化 Tokenizer
tokenizer = Tokenizer(num_words=10000)  # 使用最频繁的 10000 个词
tokenizer.fit_on_texts(x_train + x_val)  # 训练和验证数据一起拟合

# 将文本转换为整数序列
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_val_seq = tokenizer.texts_to_sequences(x_val)

#根据训练集长度、计算 pad_sequences 的 maxlen 参数
##直接取三个表中最大的length_sample
#_maxlen = CalcMaxSampleLength()
_maxlen = max([max(x_train_data.get("length_sample")), max(x_validate_data.get("length_sample"))])
# 序列填充，确保所有序列长度相同
x_train_padded = pad_sequences(x_train_seq, maxlen=_maxlen)
x_val_padded = pad_sequences(x_val_seq, maxlen=_maxlen)

# 将标签列表转换为 numpy 数组
x_train_tag = np.array(x_train_tag)
y_val_tag = np.array(y_val_tag)

# 定义模型
# units：根据硬件资源（内存）调整，值越大、启动时占用内存越大。
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=_maxlen),
    tf.keras.layers.LSTM(units=128),
    #tf.keras.layers.LSTM(units=64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 使用生成器按需加载数据，避免一次性加载所有数据到内存中
def data_generator(features, labels, batch_size):
    while True:
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            yield features[start:end], labels[start:end]

batch_size = 32
train_gen = data_generator(x_train_padded, x_train_tag, batch_size)
val_gen = data_generator(x_val_padded, y_val_tag, batch_size)
# 使用生成器训练模型
history = model.fit(train_gen, steps_per_epoch=len(x_train_padded) // batch_size, epochs=int(input("输入训练迭代轮次（epochs, 正整数）：")),
                    validation_data=val_gen, validation_steps=len(x_val_padded) // batch_size)

# 保存模型为HDF5格式
_path_h5 = f"./SaveModel/{model_name}_model.h5"
model.save(_path_h5)
# 保存模型为 SavedModel 格式
_path_SavedModel = f"./SaveModel/{model_name}_SavedModel.keras"
model.save(_path_SavedModel)

#在DB记录模型摘要信息
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
#batch_size = str(model._feed_input_shapes[0][0])
#pad_sequences_maxlen = model._feed_input_shapes[0][1]
pad_sequences_maxlen = _maxlen
epochs = model.history.params['epochs']
training_end_date = str(datetime.datetime.now())
duration = int(time.time()-START_TIME)
#样本信息
right_train_sample_num = x_train_data.get("is_abnormal").count(0)	#训练数据集中、正向样本数
wrong_train_sample_num = x_train_data.get("is_abnormal").count(1)	#训练数据集中、负向样本数

right_val_sample_num = x_validate_data.get("is_abnormal").count(0)  #验证数据集中、正向样本数
wrong_val_sample_num	= x_validate_data.get("is_abnormal").count(1)   #验证数据集中、负向样本数

train_sample_ids = ",".join([str(d) for d in x_train_data.get("id")])	    #用到的训练集主键ID
val_sample_ids = ",".join([str(d) for d in x_validate_data.get("id")])	#用到的验证集主键ID
# 获取训练结束后的最后一个 epoch 的 accuracy 和 loss
final_train_accuracy = history.history['accuracy'][-1]
final_train_loss = history.history['loss'][-1]

# 获取验证集的 accuracy 和 loss
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