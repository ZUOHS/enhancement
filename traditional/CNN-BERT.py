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


MAX_LEN = 512  
EMBEDDING_DIM = 768  
CREATOR_EMBEDDING_DIM = 300
EPOCHS = 20
BATCH_SIZE = 8


bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = TFBertModel.from_pretrained(bert_model_name)



def read_data(fold_id, FLAG, csv_file='../data/enhancement.csv'):

    text_train, text_val, text_test = [], [], []
    y_train, y_val, y_test = [], [], []
    creator_train, creator_val, creator_test = [], [], []
    role_train, role_val, role_test = [], [], []
    freq_train, freq_val, freq_test = [], [], []

    dict_resolution = {"FIXED": 1, "INVALID": 0, "DUPLICATE": 0, "WONTFIX": 0,
                       "INCOMPLETE": 0, "WORKSFORME": 0, "EXPIRED": 0, "MOVED": 0, "INACTIVE": 0}


    df = pd.read_csv(csv_file)

    if FLAG == 0:
        test_df = df[df['fold'] == fold_id]
        val_df = df[(df['fold'].notnull()) & (df['fold'] == 2)]
        train_df = df[(df['fold'].notnull()) & (df['fold'] == 0)]
    elif FLAG == 1:
        product_list = ["Bugzilla", "SeaMonkey", "Core Graveyard", "Core", "MailNews Core",
                        "Toolkit", "Firefox", "Thunderbird", "Calendar", "Camino Graveyard"]
        test_df = df[(df['product'] == product_list[fold_id])]
        train_df = df[(df['product'] != product_list[fold_id])]
    else:
        raise ValueError("")


    for index, element in test_df.iterrows():
        text_test.append(str(element['preprocessed12345']).replace("\n", " "))
        y_test.append(dict_resolution[element['resolution']])

        if pd.isnull(element['role']):
            role_test.append(0)
        else:
            role_test.append(element['role'])
        creator_test.append(element['creator'])
        freq_test.append(int(element['creator_freq']))


    for index, element in train_df.iterrows():
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
            freq_train, freq_val, freq_test)


def bert_encode(texts, tokenizer, max_len=MAX_LEN):
    input_ids = []
    attention_masks = []

    for text in tqdm(texts, desc="BERT", ncols=80):
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



def build_model():
    input_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')


    bert_output = bert_model([input_ids, attention_mask])[0]


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

    dropout = Dropout(0.5)(merged)
    dense = Dense(256, activation='relu')(dropout)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(
        inputs=[input_ids, attention_mask],
        outputs=output
    )

    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model



def main():

    (text_train, text_val, text_test,
            y_train, y_val, y_test,
            creator_train, creator_val, creator_test,
            role_train, role_val, role_test,
            freq_train, freq_val, freq_test) = read_data(fold_id=1, FLAG=0)



    train_input_ids, train_attention_masks = bert_encode(text_train, tokenizer)
    val_input_ids, val_attention_masks = bert_encode(text_val, tokenizer)
    test_input_ids, test_attention_masks = bert_encode(text_test, tokenizer)

    creator_encoder = LabelEncoder()
    creator_encoder.fit(np.concatenate([creator_train, creator_val, creator_test]))
    creator_train_encoded = creator_encoder.transform(creator_train)
    creator_val_encoded = creator_encoder.transform(creator_val)
    creator_test_encoded = creator_encoder.transform(creator_test)
  
    role_encoder = OneHotEncoder(sparse=False)
    role_encoder.fit(np.concatenate([role_train, role_val, role_test]).reshape(-1, 1))
    role_train_encoded = role_encoder.transform(np.array(role_train).reshape(-1, 1))
    role_val_encoded = role_encoder.transform(np.array(role_val).reshape(-1, 1))
    role_test_encoded = role_encoder.transform(np.array(role_test).reshape(-1, 1))

    model = build_model()


    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, mode='auto',
                                  min_delta=0.001, cooldown=0, min_lr=0)

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


    history = model.fit(
        x=[train_input_ids, train_attention_masks],
        y=y_train,
        validation_data=(
            [val_input_ids, val_attention_masks], y_val
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

  
    results = model.evaluate(
        [test_input_ids, test_attention_masks, ],
        y_test,
        verbose=1  
    )


    y_pred_probs = model.predict([test_input_ids, test_attention_masks])
    y_pred = (y_pred_probs > 0.5).astype(int)  


    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)


    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

    print(f" Accuracy: {acc:.4f}")
    print(f" AUC: {auc:.4f}")
    print(f"  (1) - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1-score: {f1[1]:.4f}")
    print(f"  (0) - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1-score: {f1[0]:.4f}")



    df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision (1)", "Recall (1)", "F1-score (1)", "Precision (0)", "Recall (0)", "F1-score (0)"],
        "Value": [acc, auc, precision[1], recall[1], f1[1], precision[0], recall[0], f1[0]]
    })

    csv_filename = "metric.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig") 



if __name__ == "__main__":
    main()
