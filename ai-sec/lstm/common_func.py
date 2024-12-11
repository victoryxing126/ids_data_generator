#coding=utf-8
import os, json, base64, hashlib, datetime, sys, time, random
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from configures.app_conf import *
import warnings, pymysql, pandas
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")


def DictException(e):
    """入参e必须是 Exception 实例; 整理异常对象为dict:\n{'summary':str(e), 'err_file':'', 'err_line':''}"""
    try:
        return {'summary': str(e), 'err_file': e.__traceback__.tb_frame.f_globals["__file__"],
                'err_line': e.__traceback__.tb_lineno}
    except:
        return False


def Md5Strs(strings):
    """返回字符串的MD5签名"""
    return hashlib.md5(str(strings).encode('utf-8')).hexdigest()


def Md5File(path):
    try:
        f = open(path, 'rb')
        md5_obj = hashlib.md5()
        md5_obj.update(f.read())
        hash_code = md5_obj.hexdigest()
        f.close()
        return dict(code=200, data=hash_code, msg='成功')
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
    """将指定 Insert/Update/Delete 类型的 sql_statment 写入 dbcon 对应的数据库表中,然后关闭DB连接"""
    try:
        if str(sql_statment).lower().startswith("create") or \
           str(sql_statment).lower().startswith("drop") or \
           str(sql_statment).lower().startswith("truncate") or \
           str(sql_statment).lower().startswith("alter"):
            return {'code': 500, 'data': '', 'msg': '禁止drop、truncate等敏感SQL'}
        #2023-1-20 禁止执行无WHERE限制的DELETE语句
        if str(sql_statment).lower().startswith("delete") and "where" not in str(sql_statment).lower():
            return {'code': 500, 'data': '', 'msg': '禁止执行无WHERE限制的DELETE语句！'}
        #2023-1-20 禁止执行无WHERE限制的DELETE语句====END
        #2023-1-5 对于连接超时的场景，尝试补录到REDIS供后续修复数据
        try:
            dbcon.cursor().execute(sql_statment)
            if isCommit == True:
                dbcon.commit()
            if isClose == True:
                dbcon.close()
            return RESP_200
        except Exception as e:
            _err = DictException(e)
            return {'code': 500, 'data': _err, 'msg': 'SQL错误'}
    except Exception as e:
        _err = DictException(e)
        try:    #防止锁表
            dbcon.commit()
        except:
            pass
        try:
            dbcon.close()
        except:
            pass
        print(_err)
        return {'code': 500, 'data': _err, 'msg': 'SQL错误'}


def DirTravelToFileList(original_trainData_dir="./0-Data-TrainTest"):
    """遍历目录返回子文件列表"""
    _file_list = os.listdir(original_trainData_dir)
    _new_file_list = []
    for i in _file_list:
        if os.path.isfile(original_trainData_dir+os.sep+i):
            _new_file_list.append(original_trainData_dir+os.sep+i)
    return _new_file_list


def Base64Encode(strings):
    """将传入的字符串转成str类型的Base64码返回(过于复杂的请求体、直接存入mysql时可能会报错；可以使用这个方法)"""
    try:
        return base64.b64encode(strings.encode("utf8")).decode("utf8")
    except Exception as e:
        print(DictException(e))
        return False


def Base64Decode(strings):
    """将传入的str类型Base64码还原成str类型返回(和Base64Encode配对使用)"""
    try:
        return base64.b64decode(strings.encode("utf8")).decode("utf8")
    except Exception as e:
        print(DictException(e))
        return False


def CalcMaxSampleLength():
    """返回所有训练、验证、测试数据集 的最大长度*1.1，作为 pad_sequences ~ maxlen 参数"""
    con = DB_Conn()
    _max = pandas.read_sql(""" SELECT GREATEST(
    (SELECT MAX(test.length_sample) FROM sec_ai.data_test_list test),
    (SELECT MAX(train.length_sample) FROM sec_ai.data_train_list train),
    (SELECT MAX(val.length_sample) FROM sec_ai.data_val_list val)
    ) AS max """, con).to_dict(orient="records")[0].get("max")
    con.commit()
    con.close()
    return int(_max * 1.1)