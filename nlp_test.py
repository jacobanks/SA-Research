from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import movie_reviews, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk import bigrams
import datetime as dt
import re, string, random
import csv
import emoji
import json
import contractions
import pytreebank

def process_data(tweet_tokens, stop_words = ()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        token = re.sub('\'.*?\s',' ', token)
        token = re.sub(r'http\S+',' ', token)
        token = emoji.get_emoji_regexp().sub(u'', token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words and token not in ["..."]:
            cleaned_tokens.append(token.lower())

    # bigram_tokens = []
    # for item in bigrams(cleaned_tokens):
    #     bigram_tokens.append(' '.join(item))

    return cleaned_tokens

def format_for_model(cleaned_tokens_list):
    dataset = []
    for tweet_tokens in cleaned_tokens_list:
        dataset.append((dict([token, True] for token in tweet_tokens[0]), tweet_tokens[1]))
    return dataset

def classify_comment_list(comments):
    total_positive = 0
    total_negative = 0
    total_neutral = 0
    all_tokens = []
    subreddits = []
    stop_words = stopwords.words('english')

    length = []
    
    for comment in comments:
        length.append(len(word_tokenize(comment["body"])))
        # process comment
        input_text = contractions.fix(comment["body"])
        custom_tokens = process_data(word_tokenize(input_text), stop_words)
        # add tokens and subreddits for freq dist
        all_tokens.extend(custom_tokens)
         subreddits.append(comment["subreddit"])
        
         Classify tokens
        sa_value = classifier.prob_classify(dict([token, True] for token in custom_tokens))
        positive = sa_value.prob("Positive")
        negative = sa_value.prob("Negative")
        print(negative)
        print(positive)
        if not negative > 0.5:
            if positive - negative > 0:
                total_positive += 1
                print(comment + " --------- Positive")
            else:
                total_neutral += 1
                print(comment + " --------- Neutral")
        elif not positive > 0.5:
            if positive - negative <= 0:
                total_negative += 1
                print(comment + " --------- Negative")
            else:
                total_neutral += 1
                print(comment + " --------- Neutral")
        else:
            total_neutral += 1
            print(comment + " --------- Neutral")

    freq_dist_pos = FreqDist(length)
    print("\nMost Common Tokens: ")
    print(freq_dist_pos.most_common(10))
    print(max(length))
    print("Positive: " + str(total_positive) + " -------- " + "Negative: " + str(total_negative) + " -------- " + "Neutral: " + str(total_neutral))
    freq_dist_pos = FreqDist(all_tokens)
    print("\nMost Common Tokens: ")
    print(freq_dist_pos.most_common(10))
    freq_dist_pos = FreqDist(subreddits)
    print("\nMost Common Subs: ")
    print(freq_dist_pos.most_common(20))
    print("-----------------------------------------")

if __name__ == "__main__":
    stop_words = stopwords.words('english')

    print("Loading Training Data...")
    training_data_file = open('trainingandtestdata/training.1600000.processed.noemoticon.csv', encoding='latin1')
    unprocesses_data = list(csv.reader(training_data_file))
    pos = unprocesses_data[:100000]
    neg = unprocesses_data[-100000:]
    unprocesses_data = pos + neg

    # Split into positive and negative and tokenize/expand contractions
    print("Preprocessing Training Data...")
    training_data = []
    split = []
    for row in unprocesses_data:
        if row[0] == "0":
            split.append("Negative")
            tokens = word_tokenize(contractions.fix(row[5]))
            training_data.append([process_data(tokens, stop_words), "Negative"])
        elif row[0] == "4":
            split.append("Positive")
            tokens = word_tokenize(contractions.fix(row[5]))
            training_data.append([process_data(tokens, stop_words), "Positive"])

    freq_dist_pos = FreqDist(split)
    print("Positive / Negative Split: ")
    print(freq_dist_pos.most_common(10))

    with open('processed_training.txt', 'w') as outfile:
        outfile.write(str(training_data))

    print("Loading and Preprocessing Testing Data...")
    dataset = pytreebank.load_sst("trees/")
    testing_data = []
    for sentence in dataset["test"]:
        label = sentence.to_labeled_lines()[0][0]
        text = sentence.to_labeled_lines()[0][1]
        if label <= 1:
            tokens = word_tokenize(contractions.fix(text))
            testing_data.append([process_data(tokens, stop_words), "Negative"])
        elif label >= 3:
            tokens = word_tokenize(contractions.fix(text))
            testing_data.append([process_data(tokens, stop_words), "Positive"])

    # Format for Model
    print("Formatting for Model...")
    training_data = format_for_model(training_data)
    testing_data = format_for_model(testing_data)
    random.shuffle(training_data)
    random.shuffle(testing_data)

    # Train Model
    print("Training Model...\n")
    classifier = NaiveBayesClassifier.train(training_data)

    print("Accuracy is:", str(classify.accuracy(classifier, testing_data) * 100) + "%")
    print(classifier.show_most_informative_features(10))

    # Run experiments
    print("\nClassifying Biden Comments...")
    print("-----------------------------------------")
    biden_file = open('experiment-data/biden_comments.json',)
    biden_submissions = json.load(biden_file)
#    biden_submissions = ["I really like Biden.", "Biden is awesome", "I don't know about Biden.", "Biden is terrible!", "I don't like Biden."]
    classify_comment_list(biden_submissions)

#    print("\nClassifying Trump Comments...")
#    print("-----------------------------------------")
    trump_file = open('experiment-data/trump_comments.json',)
    trump_submissions = json.load(trump_file)
    classify_comment_list(trump_submissions)
