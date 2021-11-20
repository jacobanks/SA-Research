import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.layers import Dropout
import utility_functions as uf
from keras.models import load_model
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist
# from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
# from nltk.tag import pos_tag
import json
import emoji
import contractions
import re, string

def load_training_data(gloveFile):
    # Load embeddings for the filtered glove list
    weight_matrix, word_idx = uf.load_embeddings(gloveFile)

    # create test, validation and trainng data
    all_data = uf.read_data()
    train_data, test_data, dev_data = uf.training_data_split(all_data, 0.8)

    train_data = train_data.reset_index()
    dev_data = dev_data.reset_index()
    test_data = test_data.reset_index()

    maxSeqLength = 280

    # load Training data matrix
    train_x = uf.embed_words(train_data, word_idx, maxSeqLength)
    test_x = uf.embed_words(test_data, word_idx, maxSeqLength)
    val_x = uf.embed_words(dev_data, word_idx, maxSeqLength)

    # load labels data matrix
    train_y = uf.labels_matrix(train_data)
    val_y = uf.labels_matrix(dev_data)
    test_y = uf.labels_matrix(test_data)

    return train_x, train_y, test_x, test_y, val_x, val_y, weight_matrix, word_idx

def build_model(weight_matrix, max_words, EMBEDDING_DIM):
    # create the model
    model = Sequential()
    model.add(Embedding(len(weight_matrix), EMBEDDING_DIM, weights=[weight_matrix], input_length=max_words, trainable=False))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, activation='tanh')))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(5, activation='softmax'))
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model

def train_model(model, num_epochs, train_x, train_y, test_x, test_y, val_x, val_y, batch_size) :
    # save the best model and early stopping
    saveBestModel = keras.callbacks.ModelCheckpoint('model/best_model_100_2.hdf5', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

    # Fit the model
    model.fit(train_x, train_y, batch_size=batch_size, epochs=num_epochs, validation_data=(val_x, val_y), callbacks=[saveBestModel, earlyStopping])
    # Final evaluation of the model
    score, acc = model.evaluate(test_x, test_y, batch_size=batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)

    return model

def predict_sentiments(trained_model, data, word_idx, max_words):
    live_list = []
    live_list_np = np.zeros((max_words,1))
    # split the sentence into its words and remove any punctuations.
    unprocessed_tokens = word_tokenize(contractions.fix(data))
    processed_tokens = []
    # stop_words = stopwords.words('english')

    for token in unprocessed_tokens:
        if token not in string.punctuation:
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)
            token = re.sub('\'.*?\s',' ', token)
            token = re.sub(r'http\S+',' ', token)
            token = emoji.get_emoji_regexp().sub(u'', token)
            token = token.lower()
            # if tag.startswith("NN"):
            #     pos = 'n'
            # elif tag.startswith('VB'):
            #     pos = 'v'
            # else:
            #     pos = 'a'

            # lemmatizer = WordNetLemmatizer()
            # token = lemmatizer.lemmatize(token, pos)
            processed_tokens.append(token)

    print(processed_tokens)

    if len(processed_tokens) > max_words:
        chunks = [processed_tokens[i * max_words:(i + 1) * max_words] for i in range((len(processed_tokens) + max_words - 1) // max_words )] 

        scores = []
        for chunk in chunks:
            data_index = np.array([word_idx[word] if word in word_idx else 0 for word in chunk])

            # padded with zeros of length 280
            padded_array = np.zeros(max_words)
            padded_array[:data_index.shape[0]] = data_index
            data_index_np_pad = padded_array.astype(int)
            live_list.append(data_index_np_pad)
            live_list_np = np.asarray(live_list)
            type(live_list_np)

            # get score from the model
            score = trained_model.predict(live_list_np, batch_size=1, verbose=0)

            # single_score = np.round(np.argmax(score)/5, decimals=2) # maximum of the array i.e single band

            # weighted score of top 3 bands
            top_3_index = np.argsort(score)[0][-3:]
            top_3_scores = score[0][top_3_index]
            top_3_weights = top_3_scores/np.sum(top_3_scores)
            single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/5, decimals = 2)
            scores.append(single_score_dot)
        final_score = np.mean(scores)
        print(scores)
        print(final_score)
    else:
        data_index = np.array([word_idx[word] if word in word_idx else 0 for word in processed_tokens])

        # padded with zeros of length 280
        padded_array = np.zeros(max_words)
        padded_array[:data_index.shape[0]] = data_index
        data_index_np_pad = padded_array.astype(int)
        live_list.append(data_index_np_pad)
        live_list_np = np.asarray(live_list)
        type(live_list_np)

        # get score from the model
        score = trained_model.predict(live_list_np, batch_size=1, verbose=0)

        # single_score = np.round(np.argmax(score)/5, decimals=2) # maximum of the array i.e single band

        # weighted score of top 3 bands
        top_3_index = np.argsort(score)[0][-3:]
        top_3_scores = score[0][top_3_index]
        top_3_weights = top_3_scores/np.sum(top_3_scores)
        single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/5, decimals = 2)
        final_score = single_score_dot

    return final_score

if __name__ == "__main__":
    max_words = 56 # max no of words in training data
    batch_size = 750 # batch size for training
    EMBEDDING_DIM = 200 # size of the word embeddings
    train_flag = False # set True if in training mode else False if in prediction mode
    epochs = 100
    gloveFile = 'Data/glove/glove.twitter.27B/glove.twitter.27B.200d.txt'

    if train_flag:
        # Load Training Data
        train_x, train_y, test_x, test_y, val_x, val_y, weight_matrix, word_idx = load_training_data(gloveFile)
        # create lstm model
        model = build_model(weight_matrix, max_words, EMBEDDING_DIM)
        # train the model
        trained_model = train_model(model, epochs, train_x, train_y, test_x, test_y, val_x, val_y, batch_size)

        # serialize weights to HDF5
        model.save("model/best_model_final_100_2.hdf5")
        print("Saved model to disk")
    else:
        print("Predicting...")
        weight_matrix, word_idx = uf.load_embeddings(gloveFile)
        weight_path = ['model/best_model_100_2.hdf5']

        # nlp = spacy.load("en_core_web_sm")

        for model in weight_path:
            loaded_model = load_model(model)
            loaded_model.summary()

            print("Loading data...")
            # biden_file = open('../../SA-Research/experiment-data/biden_comments.json',)
            # biden_submissions = json.load(biden_file)

            biden_submissions = ["Biden isn't the best president ever.", "Biden is the best president ever.", "Biden is awesome", "I don't know about Biden.", "Biden is terrible!", "I don't like Biden."]

            print("Predicting Sentiment...")
            scores = []
            polarity_sum = []
            for data_sample in biden_submissions:
                input_text = data_sample
                #if input_text contains the words 'trump' and 'biden'
                # if 'trump' in input_text.lower() and 'biden' in input_text.lower():
                #     sentences = sent_tokenize(input_text)
                #     if len(sentences) > 1:                              # if the input text contains more than one sentence
                #         scores.append("mixed sentences")
                #         for sentence in sentences:
                #             if 'trump' in sentence.lower(): 
                #                 if 'biden' in sentence.lower():         # if the sentence contains both trump and biden
                #                     scores.append("mixed sentence")
                #                 else:                                   # if the sentence contains only trump
                #                     scores.append("trump sentence")
                #             elif 'biden' in sentence.lower():
                #                 if 'trump' in sentence.lower():         # if the sentence contains both biden and trump
                #                     scores.append("mixed sentence")
                #                 else:                                   # if the sentence contains only biden
                #                     scores.append("biden sentence")
                #     else:                                               # if the input text contains only one sentence
                #         scores.append("mixed single comment")
                # elif 'trump' in input_text.lower():                     # if the input text contains only 'trump'
                #     scores.append("trump")
                # elif 'biden' in input_text.lower():                     # if the input text contains only 'biden'
                #     scores.append("biden")
                # scores.append("short comment")

                if len(word_tokenize(input_text)) < 280:
                    result = predict_sentiments(loaded_model, input_text, word_idx, max_words)
                #     break
            
                    if result < 0.2:
                        print(input_text + " -------- Very Negative Score: " + str(result))
                        polarity_sum.append("Very Negative")
                    elif result >= 0.2 and result < 0.4:
                        print(input_text + " -------- Negative Score: " + str(result))
                        polarity_sum.append("Negative")
                    elif result >= 0.4 and result < 0.6:
                        print(input_text + " -------- Neutral Score: " + str(result))
                        polarity_sum.append("Neutral")
                    elif result >= 0.6 and result < 0.8:
                        print(input_text + " -------- Positive Score: " + str(result))
                        polarity_sum.append("Positive")
                    elif result >= 0.8:
                        print(input_text + " -------- Very Positive Score: " + str(result))
                        polarity_sum.append("Very Positive")
            
            freq_dist_pos = FreqDist(scores)
            print("\nMost Common Scores: ")
            print(freq_dist_pos.most_common(20))
            #
            # freq_dist_pos = FreqDist(polarity_sum)
            # print("\nMost Common Sentiments: ")
            # print(freq_dist_pos.most_common(5))