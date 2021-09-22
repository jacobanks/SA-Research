import praw
import pandas as pd
from praw.models import MoreComments
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import emoji
import re
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
import pprint

reddit = praw.Reddit(client_id="1qZUyoMeJ07WAw",
                     client_secret="L6BmNqvsVNQWh-_k0Z4_6Uza4CqSww",
                     user_agent="Sentiment")

#subreddit = reddit.subreddit('politics')
#for post in subreddit.hot(limit=5):
##    print(post.data)
#    pprint.pprint(vars(post))

#    print(post.id, '\n')

post = reddit.submission(id='mtcclr')
all_comments = []
post.comments.replace_more(limit=None)
for comments in post.comments.list():
    all_comments.append(comments.body)

#print(all_comments, '\n')
#print('Total: ', len(all_comments))

list = all_comments
list = [str(i) for i in list]
string_uncleaned = ' , '.join(list)
emojiless = emoji.get_emoji_regexp().sub(u'', string_uncleaned)
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
tokenized_string = tokenizer.tokenize(emojiless)
lowered_string = [word.lower() for word in tokenized_string]

nlp = spacy.load('en_core_web_sm')
all_stopwords = nlp.Defaults.stop_words
text = lowered_string
tokens_without_sw = [word for word in text if not word in all_stopwords]
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = ([lemmatizer.lemmatize(w) for w in tokens_without_sw])
stemmer = PorterStemmer()
stem_tokens = ([stemmer.stem(s) for s in tokens_without_sw])

sia = SIA()
results = []

for sentences in lemmatized_tokens:
    polarity = sia.polarity_scores(sentences)
    polarity['words'] = sentences
    results.append(polarity)

pd.set_option('display.max_columns', None, 'max_colwidth', None)
df = pd.DataFrame.from_records(results)
df['label'] = 0
df.loc[df['compound'] > 0.05, 'label'] = 1
df.loc[df['compound'] < -0.05, 'label'] = -1
#print(df.label.value_counts())
df_no_neutral = df.loc[df['label']]

fig, ax = plt.subplots(figsize=(8, 8))
counts = df_no_neutral.label.value_counts(normalize=True) * 100
sns.barplot(x=counts.index, y=counts, ax=ax)
ax.set_xticklabels(['Negative', 'Positive'])
ax.set_ylabel("Percentage")
plt.show()

