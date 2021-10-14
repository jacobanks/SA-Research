from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
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
    
#    bigram_tokens = []
#    for item in bigrams(cleaned_tokens):
#            bigram_tokens.append(' '.join(item))
      
    return cleaned_tokens

def get_tweets_for_model(cleaned_tokens_list):
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

    for comment in comments:
        # process comment
        input_text = contractions.fix(comment["body"])
        custom_tokens = process_data(word_tokenize(input_text), stop_words)
        # add tokens and subreddits for freq dist
        all_tokens.extend(custom_tokens)
        subreddits.append(comment["subreddit"])
        
        # Classify tokens
        sa_value = classifier.prob_classify(dict([token, True] for token in custom_tokens))
        positive = sa_value.prob("Positive")
        negative = sa_value.prob("Negative")
        if not negative > 0.05:
            if positive - negative > 0:
                total_positive += 1
            else:
                total_neutral += 1
        elif not positive > 0.05:
            if positive - negative <= 0:
                total_negative += 1
            else:
                total_neutral += 1
        else:
            total_neutral += 1
        
    print("Positive: " + str(total_positive) + " -------- " + "Negative: " + str(total_negative) + " -------- " + "Neutral: " + str(total_neutral))
    freq_dist_pos = FreqDist(all_tokens)
    print("\nMost Common Tokens: ")
    print(freq_dist_pos.most_common(10))
    freq_dist_pos = FreqDist(subreddits)
    print("\nMost Common Subs: ")
    print(freq_dist_pos.most_common(20))
    print("-----------------------------------------")

if __name__ == "__main__":
    print("Loading Training Data...")
    training_data_file = open('trainingandtestdata/training.1600000.processed.noemoticon.csv', encoding='latin1')
    unprocesses_data = list(csv.reader(training_data_file))
#    pos = unprocesses_data[:10000]
#    neg = unprocesses_data[-10000:]
#    unprocesses_data = pos + neg
    
    # Split into positive and negative and tokenize/expand contractions
    training_data = []
    for row in unprocesses_data:
        if row[0] == "0":
            training_data.append([word_tokenize(contractions.fix(row[5])), "Negative"])
        elif row[0] == "4":
            training_data.append([word_tokenize(contractions.fix(row[5])), "Positive"])

    stop_words = stopwords.words('english')
    cleaned_tokens_list = []

    # Process data by removing stop words, lemmatizing
    print("Preprocessing Training Data...")
    for row in training_data:
        tokens = row[0]
        cleaned_tokens_list.append([process_data(tokens, stop_words), row[1]])

    # Format for Model
    dataset = get_tweets_for_model(cleaned_tokens_list)
    random.shuffle(dataset)
    
    # Split training and test data
    train_data = dataset[:1280000]
    test_data = dataset[-320000:]

    # Train Model
    print("Training Model...\n")
    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", str(classify.accuracy(classifier, test_data) * 100) + "%")
    print(classifier.show_most_informative_features(10))

    # Run experiments
    print("\nClassifying Biden Comments...")
    print("-----------------------------------------")
    biden_file = open('experiment-data/biden_comments.json',)
    biden_submissions = json.load(biden_file)
    classify_comment_list(biden_submissions)
    
    print("\nClassifying Trump Comments...")
    print("-----------------------------------------")
    trump_file = open('experiment-data/trump_comments.json',)
    trump_submissions = json.load(trump_file)
    classify_comment_list(trump_submissions)
