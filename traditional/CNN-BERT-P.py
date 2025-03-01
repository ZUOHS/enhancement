import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Concatenate, Embedding, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm

# -------------------------------
# 全局参数设置
# -------------------------------
MAX_LEN = 512  # BERT最大输入长度
EMBEDDING_DIM = 768  # BERT基础模型的隐藏层维度
CREATOR_EMBEDDING_DIM = 300
EPOCHS = 20
BATCH_SIZE = 8

# -------------------------------
# 加载BERT模型和tokenizer
# -------------------------------
bert_model_name = '/root/autodl-tmp/code/enhancement/source/models/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = TFBertModel.from_pretrained(bert_model_name)


# -------------------------------
# 修改后的数据读取函数
# -------------------------------
def read_data(fold_id, FLAG, csv_file='/root/autodl-tmp/code/enhancement/source/summary6_niu2.csv'):
    """
    从 CSV 文件中读取数据，并根据 FLAG 进行划分：
    FLAG == 0：随机平均划分十折，使用字段 folds_id 进行划分；
    FLAG == 1：十个项目划分十折，使用字段 product 进行划分。
    """
    # 初始化各列表
    # summary_train, summary_val, summary_test = [], [], []
    text_train, text_val, text_test = [], [], []
    y_train, y_val, y_test = [], [], []
    creator_train, creator_val, creator_test = [], [], []
    role_train, role_val, role_test = [], [], []
    freq_train, freq_val, freq_test = [], [], []
    id_train, id_val, id_test = [], [], []

    # 定义标签映射字典
    dict_resolution = {"FIXED": 1, "INVALID": 0, "DUPLICATE": 0, "WONTFIX": 0,
                       "INCOMPLETE": 0, "WORKSFORME": 0, "EXPIRED": 0, "MOVED": 0, "INACTIVE": 0}
    dict_senti = {"negative": 0, "positive": 1, "neutral": 2}

    # 读取 CSV 数据
    df = pd.read_csv(csv_file)

    if FLAG == 0:
        # 根据 folds_id 划分：测试集为 folds_id == fold_id，其余为训练集
        test_df = df[df['fold'] == fold_id]
        val_df = df[(df['fold'].notnull()) & (df['fold'] == 2)]
        train_df = df[(df['fold'].notnull()) & (df['fold'] == 0)]
    elif FLAG == 1:
        product_list = ["Bugzilla", "SeaMonkey", "Core Graveyard", "Core", "MailNews Core",
                        "Toolkit", "Firefox", "Thunderbird", "Calendar", "Camino Graveyard"]
        # 根据 product 划分：测试集为 product == product_list[fold_id]，其他为训练集
        test_df = df[(df['product'] == product_list[fold_id])]
        train_df = df[(df['product'] != product_list[fold_id])]
    else:
        raise ValueError("FLAG 必须为 0 或 1")

    # 读取测试集数据
    for index, element in test_df.iterrows():
        id_test.append(element['id'])
        # summary_test.append(str(element['preprocessed12']).replace("\n", " "))
        text_test.append(str(element['preprocessed12345']).replace("\n", " "))
        y_test.append(dict_resolution[element['resolution']])

        # 若 creator_developer 字段为空则赋值为 0
        if pd.isnull(element['role']):
            role_test.append(0)
        else:
            role_test.append(element['role'])
        creator_test.append(element['creator'])
        freq_test.append(int(element['creator_freq']))

    # 读取训练集数据
    for index, element in train_df.iterrows():
        id_train.append(element['id'])
        # summary_train.append(str(element['preprocessed12']).replace("\n", " "))
        text_train.append(str(element['preprocessed12345']).replace("\n", " "))
        y_train.append(dict_resolution[element['resolution']])

        if pd.isnull(element['role']):
            role_train.append(0)
        else:
            role_train.append(element['role'])
        creator_train.append(element['creator'])
        freq_train.append(int(element['creator_freq']))

    for index, element in val_df.iterrows():
        id_val.append(element['id'])
        # summary_val.append(str(element['preprocessed12']).replace("\n", " "))
        text_val.append(str(element['preprocessed12345']).replace("\n", " "))
        y_val.append(dict_resolution[element['resolution']])

        if pd.isnull(element['role']):
            role_val.append(0)
        else:
            role_val.append(element['role'])
        creator_val.append(element['creator'])
        freq_val.append(int(element['creator_freq']))

    return (text_train, text_val, text_test,
            y_train, y_val, y_test,
            creator_train, creator_val, creator_test,
            role_train, role_val, role_test,
            freq_train, freq_val, freq_test,
            id_train, id_val, id_test)


# -------------------------------
# BERT文本编码函数
# -------------------------------
def bert_encode(texts, tokenizer, max_len=MAX_LEN):
    input_ids = []
    attention_masks = []

    print("🔄 正在对文本进行 BERT 编码...")
    for text in tqdm(texts, desc="BERT 编码进度", ncols=80):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return np.array(input_ids), np.array(attention_masks)


# -------------------------------
# 构建CNN-BERT模型
# -------------------------------
def build_model():
    # 文本输入
    input_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')

    # BERT嵌入
    bert_output = bert_model([input_ids, attention_mask])[0]  # 取最后一层隐藏状态

    # 多尺寸卷积层
    conv_blocks = []
    for kernel_size in [2, 3, 4]:
        conv = Conv1D(
            filters=128,
            kernel_size=kernel_size,
            padding='valid',
            activation='relu',
            strides=1)(bert_output)
        pool = MaxPooling1D(pool_size=MAX_LEN - kernel_size + 1)(conv)
        flatten = Flatten()(pool)
        conv_blocks.append(flatten)

    text_features = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    # 创建者特征
    creator_input = Input(shape=(1,), name='creator_input')
    creator_embedding = Embedding(input_dim=10000, output_dim=CREATOR_EMBEDDING_DIM)(creator_input)
    creator_flatten = Flatten()(creator_embedding)

    # 角色特征（one-hot）
    role_input = Input(shape=(3,), name='role_input')  # 假设已经one-hot编码

    # 频率特征
    freq_input = Input(shape=(1,), name='freq_input')

    # 合并所有特征
    merged = Concatenate()([text_features, creator_flatten, role_input, freq_input])

    # 全连接层
    dropout = Dropout(0.5)(merged)
    dense = Dense(256, activation='relu')(dropout)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(
        inputs=[input_ids, attention_mask, creator_input, role_input, freq_input],
        outputs=output
    )

    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


# -------------------------------
# 主程序流程
# -------------------------------
def main():
    print("🚀 开始加载数据...")

    # 读取数据
    (text_train, text_val, text_test,
            y_train, y_val, y_test,
            creator_train, creator_val, creator_test,
            role_train, role_val, role_test,
            freq_train, freq_val, freq_test,
            id_train, id_val, id_test) = read_data(fold_id=1, FLAG=0)

    print(f"✅ 数据加载完成！训练集: {len(text_train)}，测试集: {len(text_test)}")

    # 组合 summary 和 description
    # text_train = [s + " " + d for s, d in zip(summary_train, descrip_train)]
    # text_val = [s + " " + d for s, d in zip(summary_val, descrip_val)]
    # text_test = [s + " " + d for s, d in zip(summary_test, descrip_test)]
    
    print(f"📌 组合 summary 和 description 完成，示例: {text_train[0][:100]}...")

    # BERT编码
    print("🔄 正在对文本进行 BERT 编码...")
    train_input_ids, train_attention_masks = bert_encode(text_train, tokenizer)
    val_input_ids, val_attention_masks = bert_encode(text_val, tokenizer)
    test_input_ids, test_attention_masks = bert_encode(text_test, tokenizer)
    print("✅ BERT 编码完成！")

    # 编码创建者ID
    print("🔄 对创建者 ID 进行编码...")
    creator_encoder = LabelEncoder()
    creator_encoder.fit(np.concatenate([creator_train, creator_val, creator_test]))
    creator_train_encoded = creator_encoder.transform(creator_train)
    creator_val_encoded = creator_encoder.transform(creator_val)
    creator_test_encoded = creator_encoder.transform(creator_test)
    print(f"✅ 创建者 ID 编码完成！唯一创建者数: {len(creator_encoder.classes_)}")

    # 编码角色（one-hot）
    print("🔄 对角色进行 One-Hot 编码...")
    role_encoder = OneHotEncoder(sparse=False)
    role_encoder.fit(np.concatenate([role_train, role_val, role_test]).reshape(-1, 1))
    role_train_encoded = role_encoder.transform(np.array(role_train).reshape(-1, 1))
    role_val_encoded = role_encoder.transform(np.array(role_val).reshape(-1, 1))
    role_test_encoded = role_encoder.transform(np.array(role_test).reshape(-1, 1))
    print(f"✅ 角色 One-Hot 编码完成！类别数: {role_train_encoded.shape[1]}")

    # 构建模型
    print("🔧 正在构建 CNN-BERT 模型...")
    model = build_model()
    print("✅ 模型构建完成！")

    # 训练配置
    # early_stopping = EarlyStopping(
    #     monitor='val_accuracy',
    #     patience=3,
    #     restore_best_weights=True
    # )

    # reduce_lr = ReduceLROnPlateau(
    #     monitor='val_accuracy',
    #     factor=0.2,
    #     patience=2,
    #     min_lr=1e-6
    # )
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, mode='auto',
                                  min_delta=0.001, cooldown=0, min_lr=0)

    # 模型训练
    print("🚀 开始训练模型...")
    # 确保所有输入是 NumPy 数组
    freq_train = np.array(freq_train, dtype=np.float32)
    freq_val = np.array(freq_val, dtype=np.float32)
    freq_test = np.array(freq_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    creator_train_encoded = np.array(creator_train_encoded, dtype=np.int32)
    creator_val_encoded = np.array(creator_val_encoded, dtype=np.int32)
    creator_test_encoded = np.array(creator_test_encoded, dtype=np.int32)

    if not isinstance(role_train_encoded, np.ndarray):
        role_train_encoded = np.array(role_train_encoded, dtype=np.float32)
    if not isinstance(role_val_encoded, np.ndarray):
        role_val_encoded = np.array(role_val_encoded, dtype=np.float32)
    if not isinstance(role_test_encoded, np.ndarray):
        role_test_encoded = np.array(role_test_encoded, dtype=np.float32)

    # 训练模型
    history = model.fit(
        x=[train_input_ids, train_attention_masks, creator_train_encoded, role_train_encoded, freq_train],
        y=y_train,
        validation_data=(
            [val_input_ids, val_attention_masks, creator_val_encoded, role_val_encoded, freq_val], y_val
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    print("✅ 训练完成！")

    # 模型评估
    print("📊 正在评估模型...")
    results = model.evaluate(
        [test_input_ids, test_attention_masks, creator_test_encoded, role_test_encoded, freq_test],
        y_test,
        verbose=1  # 显示评估进度
    )

    # 计算预测结果
    print("🔄 计算预测结果...")
    y_pred_probs = model.predict([test_input_ids, test_attention_masks, creator_test_encoded, role_test_encoded, freq_test])
    y_pred = (y_pred_probs > 0.5).astype(int)  # 设定 0.5 作为阈值

    # 计算 ACC 和 AUC
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)

    # 计算 Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    print("\n📊 评估结果:")
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ AUC: {auc:.4f}")
    print(f"✅ 正例 (1) - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1-score: {f1[1]:.4f}")
    print(f"✅ 反例 (0) - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1-score: {f1[0]:.4f}")

    print("\n🎯 评估完成！")

    test_results_df = pd.DataFrame({
        "id": id_test,
        "true_label": y_test,
        "predicted_label": y_pred.flatten(),  # 确保维度匹配
        "predicted_prob": y_pred_probs.flatten()
    })

    results_filename = "/root/autodl-tmp/code/enhancement/0131/prediction_results4-6.csv"
    test_results_df.to_csv(results_filename, index=False, encoding="utf-8-sig")

    # 保存结果到 CSV
    df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision (1)", "Recall (1)", "F1-score (1)", "Precision (0)", "Recall (0)", "F1-score (0)"],
        "Value": [acc, auc, precision[1], recall[1], f1[1], precision[0], recall[0], f1[0]]
    })

    csv_filename = "/root/autodl-tmp/code/enhancement/0131/evaluation_results4-6.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")  # 适用于中文字符

    print(f"📁 结果已保存至 {csv_filename}")


if __name__ == "__main__":
    main()
