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
from keras.layers import LSTM


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
    id_train, id_val, id_test  = [], [], []

    dict_resolution = {"FIXED": 1, "INVALID": 0, "DUPLICATE": 0, "WONTFIX": 0,
                       "INCOMPLETE": 0, "WORKSFORME": 0, "EXPIRED": 0, "MOVED": 0, "INACTIVE": 0}
    dict_senti = {"negative": 0, "positive": 1, "neutral": 2}

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
        id_test.append(element['id'])
        text_test.append(str(element['preprocessed12345']).replace("\n", " "))
        y_test.append(dict_resolution[element['resolution']])


        if pd.isnull(element['role']):
            role_test.append(0)
        else:
            role_test.append(element['role'])
        creator_test.append(element['creator'])
        freq_test.append(int(element['creator_freq']))

    for index, element in train_df.iterrows():
        id_train.append(element['id'])

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

    lstm_output = LSTM(128, return_sequences=True)(bert_output)
    lstm_output = LSTM(64)(lstm_output)


    creator_input = Input(shape=(1,), name='creator_input')
    creator_embedding = Embedding(input_dim=10000, output_dim=CREATOR_EMBEDDING_DIM)(creator_input)
    creator_flatten = Flatten()(creator_embedding)


    role_input = Input(shape=(3,), name='role_input') 


    freq_input = Input(shape=(1,), name='freq_input')


    merged = Concatenate()([lstm_output, creator_flatten, role_input, freq_input])

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




def main():



    (text_train, text_val, text_test,
            y_train, y_val, y_test,
            creator_train, creator_val, creator_test,
            role_train, role_val, role_test,
            freq_train, freq_val, freq_test,
            id_train, id_val, id_test) = read_data(fold_id=1, FLAG=0)



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

    results = model.evaluate(
        [test_input_ids, test_attention_masks, creator_test_encoded, role_test_encoded, freq_test],
        y_test,
        verbose=1
    )


    y_pred_probs = model.predict([test_input_ids, test_attention_masks, creator_test_encoded, role_test_encoded, freq_test])
    y_pred = (y_pred_probs > 0.5).astype(int) 


    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)


    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    

    print(f" Accuracy: {acc:.4f}")
    print(f" AUC: {auc:.4f}")
    print(f" (1) - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1-score: {f1[1]:.4f}")
    print(f" (0) - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1-score: {f1[0]:.4f}")


    test_results_df = pd.DataFrame({
        "id": id_test,
        "true_label": y_test,
        "predicted_label": y_pred.flatten(),  
        "predicted_prob": y_pred_probs.flatten()
    })

    results_filename = "result.csv"
    test_results_df.to_csv(results_filename, index=False, encoding="utf-8-sig")


    df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision (1)", "Recall (1)", "F1-score (1)", "Precision (0)", "Recall (0)", "F1-score (0)"],
        "Value": [acc, auc, precision[1], recall[1], f1[1], precision[0], recall[0], f1[0]]
    })

    csv_filename = "metric.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig") 



if __name__ == "__main__":
    main()
