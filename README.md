# Automatic Classification of Software Enhancement Requests

This repository contains experimental code for automatic classification of software enhancement requests using different models, divided into three main parts: decoder-only models, encoder-only models, and reproduction of traditional machine learning methods.

## Project Structure

```
.
├── data/                # Data files
│   ├── train.csv       # Training data
│   ├── val.csv         # Validation data
│   ├── test.csv        # Test data
│   └── enhancement.csv # Original enhancement request data
├── decoder/            # Decoder-only model code
│   ├── binary_text.py              # Directb Binary classification task
│   ├── binary_text_profile.py      # Directb Binary classification with creator profile
│   ├── binary_text_profile_10shots.py  # 10-shot Directb Binary classification
│   ├── multi_text.py               # Multi to binary classification task
│   ├── multi_text_profile.py       # Multi to binary with creator profile
│   └── multi_text_profile_10shots.py   # 10-shot Multi to binary classification
├── encoder/            # Encoder-only model code
│   ├── bert.py                 # BERT base model
│   ├── bert-p.py               # BERT model with creator profile
│   ├── bert-large.py           # BERT-large model
│   ├── bert-large-p.py         # BERT-large model with creator profile
│   ├── roberta.py              # RoBERTa model
│   ├── roberta-p.py            # RoBERTa model with creator profile
│   ├── roberta-large.py        # RoBERTa-large model
│   ├── roberta-large-p.py      # RoBERTa-large model with creator profile
│   ├── deberta.py              # DeBERTa model
│   ├── deberta-p.py            # DeBERTa model with creator profile
│   ├── deberta-large.py        # DeBERTa-large model
│   ├── deberta-large-p.py      # DeBERTa-large model with creator profile
│   ├── electra.py              # ELECTRA model
│   ├── electra-p.py            # ELECTRA model with creator profile
│   ├── electra-large.py        # ELECTRA-large model
│   ├── electra-large-p.py      # ELECTRA-large model with creator profile
│   ├── xlnet.py                # XLNet model
│   ├── xlnet-p.py              # XLNet model with creator profile
│   ├── xlnet-large.py          # XLNet-large model
│   └── xlnet-large-p.py        # XLNet-large model with creator profile
└── traditional/        # Traditional machine learning methods
    ├── CNN-BERT.py             # CNN model with BERT embeddings
    ├── CNN-BERT-P.py           # CNN-BERT model with creator profile
    ├── CNN-GLOVE.py            # CNN model with GloVe embeddings
    ├── CNN-GLOVE-P.py          # CNN-GloVe model with creator profile
    ├── LSTM-BERT.py            # LSTM model with BERT embeddings
    ├── LSTM-BERT-P.py          # LSTM-BERT model with creator profile
    ├── LSTM-GLOVE.py           # LSTM model with GloVe embeddings
    └── LSTM-GLOVE-P.py         # LSTM-GloVe model with creator profile
```

## Environment Requirements

### Common Environment
- Python 3.8+
- pandas
- numpy
- scikit-learn

### Decoder-only Models
- OpenAI API or compatible interface
- tiktoken
- concurrent.futures

### Encoder-only Models
- Transformers
- PyTorch
- datasets
- tensorflow

### Traditional Machine Learning Methods
- Keras
- TensorFlow
- gensim
- imblearn (imbalanced-learn)
- keras_preprocessing

## Data Preparation

1. Place the required data files in the `data/` directory:
   - `train.csv` - Training dataset
   - `val.csv` - Validation dataset
   - `test.csv` - Test dataset
   - `enhancement.csv` - Original enhancement request data

2. For models using GloVe embeddings, download the pre-trained GloVe word vectors and place them in the data directory:
   ```
   cd data/
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   ```

## Running Instructions

### Decoder-only Models

1. Configure API key: Before running, set the API key and base URL in the respective script:
   ```python
   client = OpenAI(
       api_key="YOUR_API_KEY",
       base_url="YOUR_BASE_URL"
   )
   ```

2. Running example:
   ```bash
   cd decoder/
   python binary_text.py
   ```

### Encoder-only Models

1. Running example:
   ```bash
   cd encoder/
   python bert.py
   ```

2. View results: Results will be saved in the current directory as `results.csv` and `metrics.csv` files.

### Traditional Machine Learning Methods

 Running example:
   ```bash
   cd traditional/
   python LSTM-GLOVE.py
   ```

## Experiment Description

This project compares the performance of different types of models on software enhancement request classification tasks:

1. **Decoder-only Models**: Using GPT-like models for text classification tasks, including both binary and multi-class settings.
   - Files with the `profile` suffix include creator profile information
   - Files with the `10shots` suffix use few-shot learning methods

2. **Encoder-only Models**: Using different pre-trained language models (BERT, RoBERTa, DeBERTa, ELECTRA, XLNet, etc.) for text classification.
   - Files with the `-p` suffix include creator profile information
   - Files with the `-large` suffix use the corresponding large model version

3. **Traditional Machine Learning Methods**: Using classic deep learning architectures such as CNN and LSTM, combined with BERT and GloVe embedding methods.
   - Files with the `-P` suffix include creator profile information

## Notes

- Running decoder code requires a valid API key and access permissions
- Most models require significant memory and GPU resources, especially large models
- Some scripts may need appropriate modifications based on your environment