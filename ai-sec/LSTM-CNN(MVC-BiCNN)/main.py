import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from tensorflow.keras.models import Model

def create_mvc_bicnn(max_length, vocab_size, embedding_dim, num_classes):
    # 输入层
    input_layer = Input(shape=(max_length,))
    
    # 嵌入层
    embedding = Embedding(vocab_size, embedding_dim, input_length=max_length)(input_layer)
    
    # 双向LSTM层
    bilstm = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    
    # CNN层
    conv1 = Conv1D(64, 3, activation='relu')(bilstm)
    conv2 = Conv1D(64, 4, activation='relu')(bilstm)
    conv3 = Conv1D(64, 5, activation='relu')(bilstm)
    
    # 全局最大池化
    pooled1 = GlobalMaxPooling1D()(conv1)
    pooled2 = GlobalMaxPooling1D()(conv2)
    pooled3 = GlobalMaxPooling1D()(conv3)
    
    # 连接层
    concatenated = Concatenate()([pooled1, pooled2, pooled3])
    
    # 全连接层
    dense = Dense(64, activation='relu')(concatenated)
    
    # 输出层
    output = Dense(num_classes, activation='softmax')(dense)
    
    # 创建模型
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# 创建模型实例
model = create_mvc_bicnn(max_length=100, vocab_size=10000, embedding_dim=100, num_classes=2)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型概要
model.summary()

# 假设数据预处理和多视图生成已完成
X_train, y_train = preprocess_and_generate_views(train_data)
X_test, y_test = preprocess_and_generate_views(test_data)

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")

# 使用LIME解释模型预测
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=['Normal', 'SQLi'])
exp = explainer.explain_instance(X_test[0], model.predict, num_features=10)
exp.show_in_notebook()
