CREATE DATABASE `sec_ai` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;

CREATE TABLE `data_train_list` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `md5_sample` varchar(45) NOT NULL COMMENT '主键，sample的md5由MYSQL,由MYSQL触发器自动生成，无需手填',
  `sample` mediumtext NOT NULL COMMENT '样本',
  `is_abnormal` int NOT NULL COMMENT '正常样本还是攻击样本；0-正常，1-攻击',
  `attack_type` varchar(128) DEFAULT NULL COMMENT '攻击类型-简写\n',
  `attack_type_zh` mediumtext COMMENT '攻击类型-汉语',
  `length_sample` int DEFAULT NULL COMMENT 'sample的长度，协助模型知道训练数据长度；由MYSQL触发器自动生成，无需手填',
  PRIMARY KEY (`md5_sample`),
  UNIQUE KEY `id_UNIQUE` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='ltsm训练数据集（注：不可以和验证数据集混用!）';

CREATE DEFINER=`root`@`localhost` TRIGGER `data_train_list_calc_sample_md5_for_pk_before_insert` BEFORE INSERT ON `data_train_list` FOR EACH ROW BEGIN
    SET NEW.md5_sample = MD5(NEW.sample);
    SET NEW.length_sample = LENGTH(NEW.sample);
END

CREATE DEFINER=`root`@`localhost` TRIGGER `data_train_list_calc_sample_md5_for_pk_before_update` BEFORE UPDATE ON `data_train_list` FOR EACH ROW BEGIN
    SET NEW.md5_sample = MD5(NEW.sample);
    SET NEW.length_sample = LENGTH(NEW.sample);
END

CREATE TABLE `data_val_list` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `md5_sample` varchar(45) NOT NULL COMMENT '主键，sample的md5由MYSQL,由MYSQL触发器自动生成，无需手填',
  `sample` mediumtext NOT NULL COMMENT '样本',
  `is_abnormal` int NOT NULL COMMENT '正常样本还是攻击样本；0-正常，1-攻击',
  `attack_type` varchar(128) DEFAULT NULL COMMENT '攻击类型-简写\n',
  `attack_type_zh` mediumtext COMMENT '攻击类型-汉语',
  `length_sample` int DEFAULT NULL COMMENT 'sample的长度，协助模型知道训练数据长度；由MYSQL触发器自动生成，无需手填',
  PRIMARY KEY (`md5_sample`),
  UNIQUE KEY `id_UNIQUE` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='ltsm验证数据集（注：不可以和训练数据集混用!）';

CREATE DEFINER=`root`@`localhost` TRIGGER `dataval_list_calc_sample_md5_for_pk_before_insert` BEFORE INSERT ON `data_val_list` FOR EACH ROW BEGIN
    SET NEW.md5_sample = MD5(NEW.sample);
    SET NEW.length_sample = LENGTH(NEW.sample);
END

CREATE DEFINER=`root`@`localhost` TRIGGER `data_val_list_calc_sample_md5_for_pk_before_update` BEFORE UPDATE ON `data_val_list` FOR EACH ROW BEGIN
    SET NEW.md5_sample = MD5(NEW.sample);
    SET NEW.length_sample = LENGTH(NEW.sample);
END

