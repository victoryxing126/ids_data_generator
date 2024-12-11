SELECT id, md5_sample, special, remark, CONVERT(from_base64(sample_b64) using utf8) as sample, is_abnormal, attack_type_zh FROM sec_ai.data_test_list 
#SELECT * FROM sec_ai.data_test_list 
#where md5_sample='9fa3b2349f84a50190814417953ecc5d'
where special IS NOT NULL
ORDER BY update_time DESC;
SELECT id, md5_sample, special, remark, CONVERT(from_base64(sample_b64) using utf8) as sample, is_abnormal, attack_type_zh FROM sec_ai.data_train_list 
#where md5_sample='9fa3b2349f84a50190814417953ecc5d'
where special IS NOT NULL
ORDER BY update_time DESC;
SELECT id, md5_sample, special, remark, CONVERT(from_base64(sample_b64) using utf8) as sample, is_abnormal, attack_type_zh FROM sec_ai.data_val_list 
#where md5_sample='9fa3b2349f84a50190814417953ecc5d'
where special IS NOT NULL
ORDER BY update_time DESC;


SELECT A.* FROM sec_ai.data_train_list A where md5_sample='9fa3b2349f84a50190814417953ecc5d';
SELECT A.* FROM sec_ai.data_val_list A where md5_sample='9fa3b2349f84a50190814417953ecc5d';
SELECT A.* FROM sec_ai.data_test_list A where md5_sample='9fa3b2349f84a50190814417953ecc5d';

SELECT COUNT(*) AS TEST_NUM, is_abnormal FROM sec_ai.data_test_list GROUP BY is_abnormal;
SELECT COUNT(*) AS TRAIN_NUM, is_abnormal FROM sec_ai.data_train_list GROUP BY is_abnormal;
SELECT COUNT(*) AS VAL_NUM, is_abnormal FROM sec_ai.data_val_list GROUP BY is_abnormal;

#训练模型列表
SELECT A.model_id, CONCAT(A.final_val_accuracy*100,'%') AS Train_val_accuracy,CONCAT(B.test_accuracy*100,'%') AS Test_accuracy,
CONCAT(A.final_val_loss*100,'%') AS Train_val_loss, CONCAT(B.test_loss*100,'%') AS Test_loss ,A.remark FROM sec_ai.models_trained_lstm A 
LEFT JOIN sec_ai.models_test_result_lstm B ON B.model_id=A.model_id
ORDER BY A.model_id ASC;


SELECT CONCAT(train_sample_ids,val_sample_ids) FROM sec_ai.models_trained_lstm;