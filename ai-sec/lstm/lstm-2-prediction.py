#coding=utf-8
from common_func import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from tensorflow.keras.preprocessing.text import tokenizer_from_json

def predict_data(conn, model_id, data):
    # 获取模型信息
    m_details = pandas.read_sql(f"SELECT * FROM sec_ai.models_trained_lstm WHERE model_id='{model_id}';", conn).to_dict(orient='records')
    if len(m_details) != 1:
        print("输入的模型ID有误、终止")
        return None
    m_details = m_details[0]
    # 加载模型
    loaded_model = load_model(m_details.get("model_file_path_SavedModel"))
    # 加载tokenizer
    with open(f"./SaveModel/{m_details['model_name']}_tokenizer.json") as f:
        tokenizer = tokenizer_from_json(json.load(f))
    # 预处理数据
    x_pred_seq = tokenizer.texts_to_sequences([data])
    x_pred_padded = pad_sequences(x_pred_seq, maxlen=m_details.get("pad_sequences_maxlen"))
    # 进行预测
    prediction = loaded_model.predict(x_pred_padded)
    # 返回预测结果
    return prediction[0][0]


def RemarkMistake(conn, deviation, model_id, data, table, result):
    """为预测偏差过大的数据打标; 当预测结果(result)与实际打标值(0或1)的差值超过 deviation、将修改该数据的 special 为 yes ，并在 remark 列追加原因"""
    if abs(float(data.get('is_abnormal')) - float(result)) >= float(deviation):
        print(f"""偏差超过{str(deviation)}、将在数据集表remark记录""")
        DB_ChangeSql(conn, f""" UPDATE sec_ai.{table} SET special='yes',
                remark = CONCAT(remark, '{datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")} {model_id}号模型预测结果为{result}、偏差超过{str(deviation)}\n') 
                WHERE md5_sample='{data.get("md5_sample")}' ; """, isCommit=True)        
    return conn.affected_rows()

def main():
    conn = DB_Conn()
    model_id_list = pandas.read_sql(""" SELECT A.model_id, CONCAT(A.final_val_accuracy*100,'%') AS Train_val_accuracy,CONCAT(B.test_accuracy*100,'%') AS Test_accuracy,
    CONCAT(A.final_val_loss*100,'%') AS Train_val_loss, CONCAT(B.test_loss*100,'%') AS Test_loss ,A.remark FROM sec_ai.models_trained_lstm A 
    LEFT JOIN sec_ai.models_test_result_lstm B ON B.model_id=A.model_id
    ORDER BY A.model_id ASC; """, conn).to_dict(orient='records')
    print("当前可用的模型ID及备注信息：")
    print(json.dumps(model_id_list, ensure_ascii=False, indent='\t'))
    model_id = input("输入要使用的模型ID：")
    _table = random.choice(['data_val_list', 'data_train_list', 'data_test_list'])
    #查询模型训练时使用的数据集ID、抽取预测数据时排除上述记录
    
    train_val_sample_ids = pandas.read_sql(f""" SELECT CONCAT(train_sample_ids,val_sample_ids) AS to_be_exclude_ids FROM sec_ai.models_trained_lstm where model_id='{model_id}' ;""", conn).to_dict(orient='records')
    if len(train_val_sample_ids) == 0:
        print("输入的模型ID有误")
        sys.exit()
    train_val_sample_ids = train_val_sample_ids[0].get("to_be_exclude_ids")
    #_table = data_test_list #只用test数据集进行测试
    sample_num = int(input("要随机抽取多少条数据进行预测（正负样本量各占1/2）："))
    
    SQLQueyData = f"""(SELECT id,md5_sample,convert(FROM_BASE64(sample_b64) using UTF8MB4) AS sample,is_abnormal,attack_type,attack_type_zh,length_sample FROM sec_ai.{_table} 
    WHERE is_abnormal=0 AND id NOT IN ({train_val_sample_ids}) ORDER BY RAND() LIMIT {str(sample_num)})
    UNION
    (SELECT id,md5_sample,convert(FROM_BASE64(sample_b64) using UTF8MB4) AS sample,is_abnormal,attack_type,attack_type_zh,length_sample FROM sec_ai.{_table}
    WHERE is_abnormal=1 AND id NOT IN ({train_val_sample_ids}) ORDER BY RAND() LIMIT {str(sample_num)}) ; """
    
    data_list = pandas.read_sql(SQLQueyData, conn).to_dict(orient='records')

    print(f"使用 {model_id} 号模型h5文件对下列样本的预测结果：")    
    for each in data_list:
        result = predict_data(conn, model_id, each.get("sample"))
        print(f"模型预测异常概率：{result * 100:.2f}%")
        print(f"实际打标异常概率：{str(int(each.get('is_abnormal'))*100)}%")
        print(f"""该样本【md5: {each.get("md5_sample")} 】打标的攻击类型为："""+each.get("attack_type_zh"))
        RemarkMistake(conn, 0.2, model_id, each, _table, result)
            
    conn.close()


if __name__ == "__main__":
    main()