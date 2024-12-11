# coding: utf-8

import sqlite3, os, uuid
from datetime import datetime


def Table_Create(create_sql, db_con=None, sql_file="datasets_info.db"):
    """创建新表"""
    if db_con != None:
        conn = db_con
    else:
        # 创建或连接到 SQLite 数据库
        conn = sqlite3.connect(sql_file)
    cursor = conn.cursor()
    # 创建表格（如果不存在）
    cursor.execute(create_sql)
    # 提交更改并关闭连接
    conn.commit()
    conn.close()
    return True


def insert_dataset_info(d_name, alias_name, path_strings, db_con=None, sql_file="datasets_info.db"):
    """向 datasets_info 表插入新数据"""
    if db_con is not None:
        conn = db_con
    else:
        conn = sqlite3.connect(sql_file)
    
    cursor = conn.cursor()
    
    # 检查是否已经存在相同的 d_name
    cursor.execute(f"SELECT save_path FROM dataset_info WHERE dataset_name = '{d_name}' ")
    result = cursor.fetchone()
    
    if result:
        existing_path = result[0]
        # 检查路径是否存在于磁盘上
        if os.path.exists(existing_path):
            print(f"数据集 '{d_name}' 已存在于路径 '{existing_path}' 上，数据已下载。")
            conn.close()
            return False
        else:
            # 删除旧记录
            cursor.execute("DELETE FROM dataset_info WHERE dataset_name = ?", (d_name,))
    
    # 插入新的数据集信息
    cursor.execute(f'''
        REPLACE INTO dataset_info (id, dataset_name, alias_name, download_date, save_path)
        VALUES ('{str(uuid.uuid4())}', '{d_name}', '{alias_name}', '{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', '{path_strings}')
        ''')
    
    conn.commit()
    conn.close()
    
    print(f"'{d_name}' 下载、保存成功：{path_strings}")
    return True


#已下载数据集基本信息表
Table_Create('''
CREATE TABLE IF NOT EXISTS dataset_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alias_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    download_date TEXT NOT NULL,
    save_path TEXT NOT NULL
)''', sql_file="datasets_info.db")