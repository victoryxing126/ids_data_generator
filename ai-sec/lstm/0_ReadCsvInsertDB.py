#coding=utf-8
import warnings, pymysql, pandas
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")
from common_func import * 


def ReadCsvToSql(csv_file_path, table=''):
    """将CSV文件整理成SQL语句"""
    ori_data = pandas.read_csv(csv_file_path)
    ori_data_list = ori_data.to_dict(orient="records")
    db_conn = DB_Conn()
    for i in ori_data_list:
        if  i.get("request_line") == None:
            continue
        http_method = str(i.get("request_line")).split(" ")[0]
        length_req_line = len(str(i.get("request_line")))
        req_line_b64 = Base64Encode(str(i.get("request_line")))
        
        _headers = json.loads(i.get("headers"))
        _headers_strings_list = []
        for hk, hv in _headers.items():
            _headers_strings_list.append(str(hk) + ":" + str(hv))
        _req_line = i.get("request_line") 
        _headers_strings = "\r\n".join(_headers_strings_list)
        length_http_header_text = len(_headers_strings)
        
        if str(i.get("payload_location")).lower() == 'body':
            _body = i.get("payload") if i.get("payload") != None else ''
        else:
            _body = ''
        _sample_ori = _req_line+"\r\n"+_headers_strings+"\r\n"*2+_body
        _sample = Base64Encode(_sample_ori)
        md5_sample = Md5Strs(_sample_ori)
        length_sample = len(_sample_ori)
        
        _is_abnormal = '1' if str(i.get("action")).lower() == 'block' else '0'
        attack_type = i.get("attack")
        attack_type_zh = vulnType.get(str(attack_type).lower()) if vulnType.get(str(attack_type).lower()) != None else ''
        _this_sql = f""" REPLACE INTO `{table}` 
        (remark, length_req_body, http_method, md5_sample, sample_b64, is_abnormal, attack_type, attack_type_zh, length_sample, req_line_b64, length_req_line,
        http_header_json_b64, length_http_header_text, req_body_b64, payload_location, eval_payload_b64, length_eval_payload)
        VALUES
        ('', '{len(_body)}', '{http_method}', '{md5_sample}', '{_sample}', '{_is_abnormal}', '{attack_type}',  '{attack_type_zh}', '{length_sample}', '{req_line_b64}', '{length_req_line}', 
        '{Base64Encode(str(i.get("headers")))}', '{length_http_header_text}', '{Base64Encode(_body)}', '{i.get("payload_location")}', '{Base64Encode(str(i.get("payload")))}', '{len(str(i.get("payload")))}'); """
        _result = DB_ChangeSql(db_conn, _this_sql)
        if _result.get("code") != 200 and "Duplicate entry" not in _result.get('data').get('summary'):
            print(_this_sql)
            print("执行上述SQL报错："+str(_result))
    db_conn.commit()
    db_conn.close()
    return True
        


if "__main__" == __name__:
    for _each_file in DirTravelToFileList():
        if os.sep + "train-" in str(_each_file):
            ReadCsvToSql(_each_file, 'data_train_list')
            try:
                os.remove(_each_file)
            except:
                pass
        elif os.sep + "val-" in str(_each_file):
            ReadCsvToSql(_each_file, 'data_val_list')
            try:
                os.remove(_each_file)
            except:
                pass
        elif os.sep + "test-" in str(_each_file):
            ReadCsvToSql(_each_file, 'data_test_list')
            try:
                os.remove(_each_file)
            except:
                pass
        else:
            print("跳过: "+str(_each_file))

print("Done~")