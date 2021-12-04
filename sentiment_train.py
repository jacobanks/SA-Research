import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.models import load_model
from nltk.tokenize import sent_tokenize
import json
import helpers as helper

def load_training_data(gloveFile, max_words):
    # Load embeddings for the filtered glove list
    weight_matrix, word_idx = helper.load_embeddings(gloveFile)

    # create test, validation and trainng data
    all_data = helper.read_data()
    train_data, test_data, dev_data = helper.training_data_split(all_data, 0.8)

    train_data = train_data.reset_index()
    dev_data = dev_data.reset_index()
    test_data = test_data.reset_index()

    # load Training data matrix
    print("Processing words...")
    train_x = helper.embed_words(train_data, word_idx, max_words)
    test_x = helper.embed_words(test_data, word_idx, max_words)
    val_x = helper.embed_words(dev_data, word_idx, max_words)

    # load labels data matrix
    print("Building labels matrices...")
    train_y = helper.labels_matrix(train_data)
    print(train_y)
    val_y = helper.labels_matrix(dev_data)
    test_y = helper.labels_matrix(test_data)

    return train_x, train_y, test_x, test_y, val_x, val_y, weight_matrix, word_idx
        
def build_model(weight_matrix, max_words, EMBEDDING_DIM):
    # create the model
    model = Sequential()
    model.add(Embedding(len(weight_matrix), EMBEDDING_DIM, weights=[weight_matrix], input_length=max_words, trainable=False))
    model.add(Bidirectional(LSTM(512, dropout=0.3, activation='tanh')))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.50))
    model.add(Dense(5, activation='softmax'))
    # try using different optimizers and different optimizer configs
    # tf.keras.Optimizer.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def train_model(model, num_epochs, train_x, train_y, test_x, test_y, val_x, val_y, batch_size) :
    # save the best model and early stopping
    saveBestModel = keras.callbacks.ModelCheckpoint('model/optimized_model_binary.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')

    # Fit the model
    model.fit(train_x, train_y, batch_size=batch_size, epochs=num_epochs, validation_data=(val_x, val_y), callbacks=[saveBestModel, earlyStopping], shuffle=True)
    # Final evaluation of the model
    score, acc = model.evaluate(test_x, test_y, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return model

def predict_sentiments(trained_model, input_text, word_idx, max_words):
    processed_tokens = helper.process_input_text(input_text)
    # Break text into chunks of max_words length
    chunks = [processed_tokens[i * max_words:(i + 1) * max_words] for i in range((len(processed_tokens) + max_words - 1) // max_words )] 
    scores = []
    final_score = 0

    # Analyze each chunk
    for chunk in chunks:
        live_list = []
        live_list_np = np.zeros((max_words,1))
        data_index = np.array([word_idx[word] if word in word_idx else 0 for word in chunk])
        # padded with zeros of length 280
        padded_array = np.zeros(max_words)
        padded_array[:data_index.shape[0]] = data_index
        data_index_np_pad = padded_array.astype(int)
        live_list.append(data_index_np_pad)
        live_list_np = np.asarray(live_list)

        # get score from the model
        score = trained_model.predict(live_list_np, batch_size=1, verbose=0)
        single_score = np.argmax(score)/5# maximum of the array i.e single band

        # weighted score of top 3 bands
        # top_3_index = np.argsort(score)[0][-3:]
        # top_3_scores = score[0][top_3_index]
        # top_3_weights = top_3_scores/np.sum(top_3_scores)
        # top_3_score = np.round(np.dot(top_3_index, top_3_weights)/5, decimals = 2)
        scores.append(single_score)

    if len(scores) > 1:
        final_score = np.mean(scores)     # Get the average score of all the chunks
    elif len(scores) == 1:
        final_score = scores[0]

    return final_score

if __name__ == "__main__":
    max_words = 56 # max no of words in training data
    batch_size = 2048 # batch size for training
    EMBEDDING_DIM = 100 # size of the word embeddings
    train_flag = True # set True if in training mode else False if in prediction mode
    epochs = 100
    gloveFile = 'Data/glove/glove.twitter.27B/glove.twitter.27B.100d.txt'

    if train_flag:
        # Load Training Data
        train_x, train_y, test_x, test_y, val_x, val_y, weight_matrix, word_idx = load_training_data(gloveFile, max_words)
        # create lstm model
        model = build_model(weight_matrix, max_words, EMBEDDING_DIM)
        # train the model
        trained_model = train_model(model, epochs, train_x, train_y, test_x, test_y, val_x, val_y, batch_size)
        # serialize weights to HDF5
        # trained_model.save("model/optimized_model_final_1.hdf5")
        # print("Saved model to disk")
    else:
        weight_matrix, word_idx = helper.load_embeddings(gloveFile)
        model = 'model/optimized_model.hdf5'
        sites = ['Twitter']
        data = ['biden']

        print("Loading Model from " + model)
        loaded_model = load_model(model)
        loaded_model.summary()
        polarity_sum = {"Very Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "Very Positive": 0}

        for site in sites:
            for file in data:
                print("Loading data from " + file + "_comments.json")
                posts_file = 'Data/' + site + '/' + file
                if site == 'Reddit':
                    posts_file = posts_file + '_comments.json'
                else:
                    posts_file = posts_file + '_tweets.json'

                submissions = json.load(open(posts_file,))
                # submissions_test = ["Biden isn't the best president ever.", "Biden is the best president ever. https://github.com/jacobanks", "Biden is the worst president ever.", "Biden is awesome", "I don't know about Biden.", "Biden is terrible!", "I don't like Biden."]
                analyzed_text = []

                count = 0
                for data_sample in submissions[:100000]:
                    input_text = str(data_sample["body"])

                    print("\rPredicting sentiment polarity... analyzed {} posts.".format(count), end="")
                    if len(input_text) > 5:    
                        input_text = input_text.lower()                                 
                        mentions_both_list = []
                        # Analyze parts that mention different candidates
                        if 'trump' in input_text and 'biden' in input_text:
                            sentences = sent_tokenize(input_text)
                            if len(sentences) > 1:                              # if the input text contains more than one sentence
                                for sentence in sentences:
                                    if 'trump' in sentence and 'biden' not in sentence:                                 # if the sentence contains only trump
                                        score = predict_sentiments(loaded_model, sentence, word_idx, max_words)
                                        mentions_both_list.append({"sentence": sentence, "score": score, "topic": "trump"})
                                    elif 'biden' in sentence and 'trump' not in sentence:                                 # if the sentence contains only biden
                                        score = predict_sentiments(loaded_model, sentence, word_idx, max_words)
                                        mentions_both_list.append({"sentence": sentence, "score": score, "topic": "biden"})
                                    elif 'biden' in sentence and 'trump' in sentence:                                 # if the sentence contains both biden and trump
                                        score = predict_sentiments(loaded_model, sentence, word_idx, max_words)
                                        mentions_both_list.append({"sentence": sentence, "score": score, "topic": "both"})
                                    else:
                                        score = predict_sentiments(loaded_model, sentence, word_idx, max_words)
                                        mentions_both_list.append({"sentence": sentence, "score": score, "topic": "none"})
                            else:  
                                mentions_both_list.append({"topic": "both"})
                        elif 'trump' in input_text:
                            mentions_both_list.append({"topic": "trump"})
                        elif 'biden' in input_text:
                            mentions_both_list.append({"topic": "biden"})
                        else:
                            mentions_both_list.append({"topic": "none"})

                        # Analyze whole post
                        overall_result = predict_sentiments(loaded_model, input_text, word_idx, max_words)
                        analyzed_text.append({"id": data_sample["id"], "score": overall_result, "mentions": mentions_both_list})

                        count += 1
                        if count % 1000 == 0:
                            with open('Data/' + site + '/scores/' + file + '_scores_final.json', 'w') as outfile:
                                outfile.write(json.dumps(analyzed_text, indent=4))
                
                print("\nSaving analyzed text to " + file + "_scores.json")
                with open('Data/' + site + '/scores/' + file + '_scores_final.json', 'w') as outfile:
                    outfile.write(json.dumps(analyzed_text, indent=4))