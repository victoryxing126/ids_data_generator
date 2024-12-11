#coding=utf-8
from common_func import *

START_TIME = int(time.time())
START_TIME_datetime = str(datetime.datetime.now())
m_id = input("开始测试, 输入要测试的模型ID：")
# 创建数据库连接
conn = DB_Conn()
#查询模型摘要信息
m_details = pandas.read_sql(f"""SELECT * FROM sec_ai.models_trained_lstm WHERE model_id='{m_id}' ;""", conn).to_dict(orient='records')
if len(m_details) != 1:
    print("输入的模型ID有误、终止")
    sys.exit()
m_details = m_details[0]

#一、读取训练数据
x_train_data = pandas.read_sql(SQLQueyData.format(table_name="data_train_list"), conn).to_dict(orient='list')
x_train = x_train_data.get("sample")

#二、读取验证数据集 (x_val):
x_validate_data = pandas.read_sql(SQLQueyData.format(table_name="data_val_list"), conn).to_dict(orient='list')
x_val = x_validate_data.get("sample")

#准备和预处理测试集 x_test 和 y_test
x_test_data = pandas.read_sql(SQLQueyData.format(table_name="data_test_list"), conn).to_dict(orient='list')
x_test = x_test_data.get("sample")
y_test = x_test_data.get("is_abnormal")


# 将文本转换为整数序列
tokenizer = Tokenizer(num_words=10000)  # 使用最频繁的 10000 个词
tokenizer.fit_on_texts(x_train + x_val)  # 训练和验证数据一起拟合

# 将文本转换为整数序列
x_test_seq = tokenizer.texts_to_sequences(x_test)

# 序列填充，确保所有序列长度相同
x_test_padded = pad_sequences(x_test_seq, maxlen=m_details.get("pad_sequences_maxlen"))

# 将标签列表转换为 numpy 数组
y_test = np.array(y_test)

# 加载保存的模型
from tensorflow.keras.models import load_model

## 加载保存的HDF5格式模型
#loaded_model_h5 = load_model(m_details.get("model_file_path_h5"))
# 加载保存的SavedModel格式模型
loaded_model_savedmodel = load_model(m_details.get("model_file_path_SavedModel"))

##使用预处理后的测试数据集评估模型性能
#test_loss, test_acc = loaded_model_h5.evaluate(x_test_padded, y_test)
#print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
test_loss, test_acc = loaded_model_savedmodel.evaluate(x_test_padded, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

#记录本次测试结果
end_ts = time.time()
duration = end_ts - START_TIME
right_train_sample_num = x_train_data.get("is_abnormal").count(0)
wrong_train_sample_num = x_train_data.get("is_abnormal").count(1)
right_val_sample_num = x_validate_data.get("is_abnormal").count(0)
wrong_val_sample_num = x_validate_data.get("is_abnormal").count(1)
right_test_sample_num = x_test_data.get("is_abnormal").count(0)
wrong_test_sample_num = x_test_data.get("is_abnormal").count(1)
train_sample_ids = ",".join([str(d) for d in x_train_data.get("id")])
val_sample_ids = ",".join([str(d) for d in x_validate_data.get("id")])
test_sample_ids = ",".join([str(d) for d in x_test_data.get("id")])
test_loss = test_loss
test_accuracy = test_acc

DB_ChangeSql(conn, f""" INSERT INTO sec_ai.models_test_result_lstm
(model_id, test_start_date, test_end_date, train_duration_seconds, right_train_sample_num, wrong_train_sample_num, right_val_sample_num, wrong_val_sample_num, right_test_sample_num, wrong_test_sample_num, train_sample_ids, val_sample_ids, test_sample_ids, test_loss, test_accuracy)
VALUES( '{m_id}', '{START_TIME_datetime}', CURRENT_TIMESTAMP, '{duration}', '{right_train_sample_num}', '{wrong_train_sample_num}', '{right_val_sample_num}', '{wrong_val_sample_num}', '{right_test_sample_num}', '{wrong_test_sample_num}', '{train_sample_ids}', '{val_sample_ids}', '{test_sample_ids}', '{test_loss}', '{test_acc}'); """)

try:
    conn.close()
except:
    pass
print("测试耗时：" + str(int(time.time()-START_TIME))+" 秒")
print("""====END====""")