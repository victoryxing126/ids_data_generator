# coding: utf-8

from _lib import * 

#已下载数据集基本信息表
Table_Create('''
CREATE TABLE IF NOT EXISTS dataset_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alias_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    download_date TEXT NOT NULL,
    save_path TEXT NOT NULL
)''', sql_file="datasets_info.db")

