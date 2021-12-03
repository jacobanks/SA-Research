import pandas as pd
import numpy as np
import codecs
import os
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
import re, string, emoji, contractions, json
from nltk.tokenize import word_tokenize
import pytreebank
import sys

def load_embeddings(embedding_path):
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
    # print("Loading Glove Model")
    # glove_model = {}
    # weight_vectors = []
    # with open(embedding_path,'r', encoding='utf-8') as f:
    #     for line in f:
    #         split_line = line.split()
    #         word = split_line[0]
    #         embedding = np.array(split_line[1:], dtype=np.float32)
    #         glove_model[word] = len(weight_vectors)
    #         if len(embedding) != 100:
    #             print(len(embedding))
    #             print(word + str(embedding))
    #         weight_vectors.append(embedding)

    # glove_model[u'-LRB-'] = glove_model.pop(u'(')
    # glove_model[u'-RRB-'] = glove_model.pop(u')')
    # weight_vectors.append(np.random.uniform(-0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
    # print(f"{len(glove_model)} words loaded!")
    # return np.stack(weight_vectors), glove_model

def read_data():
    # read dictionary into df
    print("Reading Training Data...")
    train_data = pd.DataFrame(data=json.load(open("Data/TrainingData/sst_train.json",)))
    test_data = pd.DataFrame(data=json.load(open("Data/TrainingData/sst_test.json",)))
    val_data = pd.DataFrame(data=json.load(open("Data/TrainingData/sst_val.json",)))
    return train_data, test_data, val_data

def embed_words(data, word_idx, max_seq_len):
    no_rows = len(data)
    ids = np.zeros((no_rows, max_seq_len), dtype='int32')
    word_idx_lwr =  {k.lower(): v for k, v in word_idx.items()}
    idx = 0

    for index, row in data.iterrows():
        sentence = row['phrase']
        sentence_words = process_input_text(sentence.lower())
        i = 0
        for word in sentence_words:
            word_lwr = word
            try:
                ids[idx][i] =  word_idx_lwr[word_lwr]
            except Exception as e:
                if str(e) == word:
                    ids[idx][i] = 0
                continue
            i += 1
        idx += 1
    return ids

def labels_matrix(data):
    labels = data['label']
    lables_float = labels.astype(int)

    cats = ['0','1','2','3','4']
    labels_mult = ((lables_float * 10) / 2).astype(int)
    dummies = pd.get_dummies(labels_mult, prefix='', prefix_sep='')
    dummies = dummies.T.reindex(cats).T.fillna(0)
    labels_matrix = dummies.values

    return labels_matrix

def process_input_text(input_text):
    input_text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', input_text)
    input_text = re.sub("(@[A-Za-z0-9_]+)","", input_text)
    unprocessed_tokens = word_tokenize(contractions.fix(input_text))
    processed_tokens = []

    for token, tag in pos_tag(unprocessed_tokens):
        if token not in string.punctuation:
            token = emoji.get_emoji_regexp().sub(u'', token)
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)
            if len(token) > 0:
                processed_tokens.append(token)
                
    return processed_tokens

def create_training_files():
    out_path = os.path.join(sys.path[0], 'sst_{}.json')
    dataset = pytreebank.load_sst('./raw_data')

    # Store train, dev and test in separate files
    for category in ['train', 'test', 'val']:
        with open(out_path.format(category), 'w') as outfile:
            data = []
            for item in dataset[category]:
                data.append({"phrase": item.to_labeled_lines()[0][1], "label": item.to_labeled_lines()[0][0] + 1})
            outfile.write(json.dumps(data, indent=4))