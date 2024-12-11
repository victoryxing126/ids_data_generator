#coding=utf-8
from common_func import *  
from tensorflow.keras.models import load_model

#QuerySQL(引用时通过.format替换表名)
#SQLQueyData = """ SELECT id,md5_sample,convert(FROM_BASE64(sample_b64) using UTF8MB4) AS sample,is_abnormal,attack_type,attack_type_zh,length_sample  FROM sec_ai.{table_name} LIMIT 1000; """
#is_abnormal：0-正常，1-攻击

sample_num_normal = int(input("输入学习训练使用的正常数据样本量（攻击样本量为此数值的1/2）："))
sample_num_abnormal = int(0.5 * sample_num_normal)

SQLQueyData = """(SELECT id,md5_sample,convert(FROM_BASE64(sample_b64) using UTF8MB4) AS sample,is_abnormal,attack_type,attack_type_zh,length_sample FROM sec_ai.{table_name} 
WHERE is_abnormal=0 ORDER BY RAND() LIMIT """+str(sample_num_normal) + """)
UNION
(SELECT id,md5_sample,convert(FROM_BASE64(sample_b64) using UTF8MB4) AS sample,is_abnormal,attack_type,attack_type_zh,length_sample FROM sec_ai.{table_name} WHERE is_abnormal=1 ORDER BY RAND() LIMIT """+str(sample_num_abnormal)+""") ; """


def preprocess_data(x_train, x_val, x_test, maxlen):
    tokenizer = Tokenizer(num_words=10000)  # 使用最频繁的 10000 个词
    tokenizer.fit_on_texts(x_train + x_val)  # 训练和验证数据一起拟合
    
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_val_seq = tokenizer.texts_to_sequences(x_val)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    
    x_train_padded = pad_sequences(x_train_seq, maxlen=maxlen)
    x_val_padded = pad_sequences(x_val_seq, maxlen=maxlen)
    x_test_padded = pad_sequences(x_test_seq, maxlen=maxlen)
    
    return x_train_padded, x_val_padded, x_test_padded, tokenizer

def create_model(maxlen, lstm_units=128):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=maxlen),
        tf.keras.layers.LSTM(units=lstm_units),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def data_generator(features, labels, batch_size):
    while True:
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            yield features[start:end], labels[start:end]

def train_model(conn, epochs):
    START_TIME = int(time.time())
    model_name = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")

    print("开始读取训练数据集...")
    x_train_data = pandas.read_sql(SQLQueyData.format(table_name="data_train_list"), conn).to_dict(orient='list')
    x_train = x_train_data.get("sample")
    x_train_tag = np.array(x_train_data.get("is_abnormal"))

    x_validate_data = pandas.read_sql(SQLQueyData.format(table_name="data_val_list"), conn).to_dict(orient='list')
    x_val = x_validate_data.get("sample")
    y_val_tag = np.array(x_validate_data.get("is_abnormal"))

    _maxlen = max([max(x_train_data.get("length_sample")), max(x_validate_data.get("length_sample"))])
    
    x_train_padded, x_val_padded, _, tokenizer = preprocess_data(x_train, x_val, [], _maxlen)

    model = create_model(_maxlen)

    batch_size = 32
    train_gen = data_generator(x_train_padded, x_train_tag, batch_size)
    val_gen = data_generator(x_val_padded, y_val_tag, batch_size)

    history = model.fit(train_gen, steps_per_epoch=len(x_train_padded) // batch_size, epochs=epochs,
                        validation_data=val_gen, validation_steps=len(x_val_padded) // batch_size)

    _path_h5 = f"./SaveModel/{model_name}_model.h5"
    model.save(_path_h5)
    _path_SavedModel = f"./SaveModel/{model_name}_SavedModel.keras"
    model.save(_path_SavedModel)

    # 记录模型信息到数据库
    model_info = {
        "LSTM_CONF_INFO": json.dumps(model.get_layer(index=1).get_config()),
        "LSTM_UNITS": model.get_layer(index=1).get_config()['units'],
        "Embedding_CONF_INFO": json.dumps(model.get_layer(index=0).get_config()),
        "Dense_CONF_INFO": json.dumps(model.get_layer(index=2).get_config()),
        "optimizer": str(model.optimizer.name),
        "loss_function": str(model.loss),
        "learning_rate": round(float(model.optimizer.learning_rate.numpy()), 5),
        "pad_sequences_maxlen": _maxlen,
        "epochs": epochs,
        "training_end_date": str(datetime.datetime.now()),
        "duration": int(time.time()-START_TIME),
        "right_train_sample_num": x_train_data.get("is_abnormal").count(0),
        "wrong_train_sample_num": x_train_data.get("is_abnormal").count(1),
        "right_val_sample_num": x_validate_data.get("is_abnormal").count(0),
        "wrong_val_sample_num": x_validate_data.get("is_abnormal").count(1),
        "train_sample_ids": ",".join([str(d) for d in x_train_data.get("id")]),
        "val_sample_ids": ",".join([str(d) for d in x_validate_data.get("id")]),
        "final_train_accuracy": history.history['accuracy'][-1],
        "final_train_loss": history.history['loss'][-1],
        "final_val_accuracy": history.history['val_accuracy'][-1],
        "final_val_loss": history.history['val_loss'][-1],
    }

    DB_ChangeSql(conn, f"""
    INSERT INTO sec_ai.models_trained_lstm 
    (final_val_loss, final_train_loss, final_train_accuracy, final_val_accuracy, tf_keras_layers_LSTM_conf, 
    tf_keras_layers_Embedding_conf, tf_keras_layers_Dense_conf, LSTM_UNITS, right_train_sample_num,
    wrong_train_sample_num, right_val_sample_num, wrong_val_sample_num, train_sample_ids, val_sample_ids, 
    model_name,  optimizer, loss_function, learning_rate, batch_size, pad_sequences_maxlen, epochs,
    model_file_path_h5, model_file_path_SavedModel, training_end_date, train_duration_seconds) 
    VALUES
    ('{model_info['final_val_loss']}', '{model_info['final_train_loss']}', '{model_info['final_train_accuracy']}', 
    '{model_info['final_val_accuracy']}', '{model_info['LSTM_CONF_INFO']}', '{model_info['Embedding_CONF_INFO']}', 
    '{model_info['Dense_CONF_INFO']}', '{model_info['LSTM_UNITS']}', '{model_info['right_train_sample_num']}',
    '{model_info['wrong_train_sample_num']}', '{model_info['right_val_sample_num']}', '{model_info['wrong_val_sample_num']}', 
    '{model_info['train_sample_ids']}', '{model_info['val_sample_ids']}',
    '{model_name}', '{model_info['optimizer']}', '{model_info['loss_function']}', '{model_info['learning_rate']}', 
    '{batch_size}', '{model_info['pad_sequences_maxlen']}', {model_info['epochs']},
    '{_path_h5}', '{_path_SavedModel}', '{model_info['training_end_date']}', '{model_info['duration']}');
    """)
    with open(f"./SaveModel/{model_name}_tokenizer.json", 'w') as f:
        f.write(json.dumps(tokenizer.to_json()))
    
    print(f"训练耗时：{model_info['duration']} 秒")
    return model_name

def test_model(conn, model_id):
    START_TIME = int(time.time())
    START_TIME_datetime = str(datetime.datetime.now())

    m_details = pandas.read_sql(f"SELECT * FROM sec_ai.models_trained_lstm WHERE model_id='{model_id}';", conn).to_dict(orient='records')
    if len(m_details) != 1:
        print("输入的模型ID有误、终止")
        sys.exit()
    m_details = m_details[0]

    x_train_data = pandas.read_sql(SQLQueyData.format(table_name="data_train_list"), conn).to_dict(orient='list')
    x_val_data = pandas.read_sql(SQLQueyData.format(table_name="data_val_list"), conn).to_dict(orient='list')
    x_test_data = pandas.read_sql(SQLQueyData.format(table_name="data_test_list"), conn).to_dict(orient='list')

    x_train = x_train_data.get("sample")
    x_val = x_val_data.get("sample")
    x_test = x_test_data.get("sample")
    y_test = np.array(x_test_data.get("is_abnormal"))

    _, _, x_test_padded, _ = preprocess_data(x_train, x_val, x_test, m_details.get("pad_sequences_maxlen"))

    loaded_model = load_model(m_details.get("model_file_path_SavedModel"))
    test_loss, test_acc = loaded_model.evaluate(x_test_padded, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

    duration = int(time.time() - START_TIME)
    test_info = {
        "right_train_sample_num": x_train_data.get("is_abnormal").count(0),
        "wrong_train_sample_num": x_train_data.get("is_abnormal").count(1),
        "right_val_sample_num": x_val_data.get("is_abnormal").count(0),
        "wrong_val_sample_num": x_val_data.get("is_abnormal").count(1),
        "right_test_sample_num": x_test_data.get("is_abnormal").count(0),
        "wrong_test_sample_num": x_test_data.get("is_abnormal").count(1),
        "train_sample_ids": ",".join([str(d) for d in x_train_data.get("id")]),
        "val_sample_ids": ",".join([str(d) for d in x_val_data.get("id")]),
        "test_sample_ids": ",".join([str(d) for d in x_test_data.get("id")]),
    }

    DB_ChangeSql(conn, f"""
    INSERT INTO sec_ai.models_test_result_lstm
    (model_id, test_start_date, test_end_date, train_duration_seconds, right_train_sample_num, 
    wrong_train_sample_num, right_val_sample_num, wrong_val_sample_num, right_test_sample_num, 
    wrong_test_sample_num, train_sample_ids, val_sample_ids, test_sample_ids, test_loss, test_accuracy)
    VALUES
    ('{model_id}', '{START_TIME_datetime}', CURRENT_TIMESTAMP, '{duration}', 
    '{test_info['right_train_sample_num']}', '{test_info['wrong_train_sample_num']}', 
    '{test_info['right_val_sample_num']}', '{test_info['wrong_val_sample_num']}', 
    '{test_info['right_test_sample_num']}', '{test_info['wrong_test_sample_num']}', 
    '{test_info['train_sample_ids']}', '{test_info['val_sample_ids']}', '{test_info['test_sample_ids']}', 
    '{test_loss}', '{test_acc}');
    """)
    
    print(f"测试耗时：{duration} 秒")

def main():

    conn = DB_Conn()
    choice = input("1. 训练模型 \n2. 测试模型 \n3. 训练并测试\t\n选择操作：")
    
    if choice == '1':
        epochs = int(input("输入训练迭代轮次（epochs, 正整数）："))
        model_name = train_model(conn, epochs)
        print(f"模型训练完成，保存为：{model_name}")
    elif choice == '2':
        model_id = input("输入要测试的模型ID：")
        test_model(conn, model_id)
    elif choice == '3':
        epochs = int(input("输入训练迭代轮次（epochs, 正整数）："))
        model_name = train_model(conn, epochs)
        print(f"模型训练完成，保存为：{model_name}")
        model_id = pandas.read_sql(f""" SELECT `model_id` FROM sec_ai.models_trained_lstm WHERE model_name = '{model_name}' ; """, conn).to_dict(orient="records")[0].get("model_id")
        test_model(conn, model_id)
    else:
        print("无效的选择")

    conn.close()

if __name__ == "__main__":
    main()
