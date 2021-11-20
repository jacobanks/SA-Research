import pandas as pd
import numpy as np
import codecs
import os
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.tag import pos_tag
import contractions
from nltk.tokenize import word_tokenize

def load_embeddings(embedding_path):
  #Loads embedings, returns weight matrix and dict from words to indices.
  print('loading word embeddings from %s' % embedding_path)
  weight_vectors = []
  word_idx = {}
  with codecs.open(embedding_path, encoding='utf-8') as f:
    for line in f:
        word, vec = line.split(u' ', 1)
        word_idx[word] = len(weight_vectors)
        weight_vectors.append(np.array(vec.split(), dtype=np.float32))
  # '(' and ')' are replaced by '-LRB-' and '-RRB-'
  word_idx[u'-LRB-'] = word_idx.pop(u'(')
  word_idx[u'-RRB-'] = word_idx.pop(u')')
  # Random embedding vector for unknown words.
  weight_vectors.append(np.random.uniform(-0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
  return np.stack(weight_vectors), word_idx


def read_data():
    # read dictionary into df
    print("Reading Training Data...")
    df_data_sentence = pd.read_table('Data/dictionary.txt')
    df_data_sentence_processed = df_data_sentence['Phrase|Index'].str.split('|', expand=True)
    df_data_sentence_processed = df_data_sentence_processed.rename(columns={0: 'Phrase', 1: 'phrase_ids'})

    # read sentiment labels into df
    df_data_sentiment = pd.read_table('Data/sentiment_labels.txt')
    df_data_sentiment_processed = df_data_sentiment['phrase ids|sentiment values'].str.split('|', expand=True)
    df_data_sentiment_processed = df_data_sentiment_processed.rename(columns={0: 'phrase_ids', 1: 'sentiment_values'})

    #combine data frames containing sentence and sentiment
    df_processed_all = df_data_sentence_processed.merge(df_data_sentiment_processed, how='inner', on='phrase_ids')

    return df_processed_all

def training_data_split(all_data, splitPercent):
    print("Splitting Training Data...")
    msk = np.random.rand(len(all_data)) < splitPercent
    train_only = all_data[msk]
    test_and_dev = all_data[~msk]

    msk_test = np.random.rand(len(test_and_dev)) < 0.5
    test_only = test_and_dev[msk_test]
    dev_only = test_and_dev[~msk_test]

    dev_only.to_csv(os.path.join('Data/TrainingData/dev.csv'))
    test_only.to_csv(os.path.join('Data/TrainingData/test.csv'))
    train_only.to_csv(os.path.join('Data/TrainingData/train.csv'))

    return train_only, test_only, dev_only

def embed_words(data, word_idx, max_seq_len):
    print("Processing data for tensorflow...")
    no_rows = len(data)
    ids = np.zeros((no_rows, max_seq_len), dtype='int32')
    # convert keys in dict to lower case
    word_idx_lwr =  {k.lower(): v for k, v in word_idx.items()}
    idx = 0

    lengths = []
    for index, row in data.iterrows():
        sentence = (row['Phrase'])
        sentence_words = word_tokenize(contractions.fix(sentence))
        lengths.append(len(sentence_words))

        i = 0
        for word in sentence_words:
            word_lwr = word.lower()
            try:
                ids[idx][i] =  word_idx_lwr[word_lwr]
            except Exception as e:
                if str(e) == word:
                    ids[idx][i] = 0
                continue
            i = i + 1
        idx = idx + 1

    print(max(lengths))
    return ids

def labels_matrix(data):
    labels = data['sentiment_values']
    lables_float = labels.astype(float)

    cats = ['0','1','2','3','4']
    labels_mult = (lables_float * 5).astype(int)
    dummies = pd.get_dummies(labels_mult, prefix='', prefix_sep='')
    dummies = dummies.T.reindex(cats).T.fillna(0)
    labels_matrix = dummies.values

    return labels_matrix