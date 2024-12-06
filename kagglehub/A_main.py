# coding: utf-8

from _lib import * 
import kagglehub


# https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks/data
alias_name = 'cyber-security-attacks'
dataset_name = 'teamincribo/cyber-security-attacks'
path = kagglehub.dataset_download(dataset_name)
insert_dataset_info(d_name=dataset_name, alias_name=alias_name, path_strings=path)

#https://www.kaggle.com/datasets/hassan06/nslkdd
alias_name = 'NSL-KDD'
dataset_name = 'hassan06/nslkdd'
path = kagglehub.dataset_download(dataset_name)
insert_dataset_info(d_name=dataset_name, alias_name=alias_name, path_strings=path)

#https://www.kaggle.com/datasets/hassan06/nslkdd
alias_name = 'KDD Cup 99'
dataset_name = 'galaxyh/kdd-cup-1999-data'
path = kagglehub.dataset_download(dataset_name)
insert_dataset_info(d_name=dataset_name, alias_name=alias_name, path_strings=path)

print('end')