# from datetime import datetime
import json, csv, codecs
from nltk import FreqDist
import matplotlib.pyplot as plt
import numpy as np

# Reddit OVERALL Distribution of sentiment: {'V. Negative': 44411, 'Negative': 207189, 'Neutral': 140932, 'Positive': 7081, 'V. Positive': 0}
# Trump sents: {'V. Negative': 27665, 'Negative': 141155, 'Neutral': 95600, 'Positive': 4705, 'V. Positive': 0}
# Biden sents: {'V. Negative': 14093, 'Negative': 104364, 'Neutral': 77905, 'Positive': 4516, 'V. Positive': 0}
# Both sents: {'V. Negative': 7755, 'Negative': 46819, 'Neutral': 36514, 'Positive': 1933, 'V. Positive': 0}
# mentions: {'Trump': 157048, 'Biden': 113876, 'Both': 128688}

# Twitter OVERALL Distribution of sentiment: {'V. Negative': 29452, 'Negative': 197847, 'Neutral': 162813, 'Positive': 9806, 'V. Positive': 0}
# Trump sents: {'V. Negative': 14872, 'Negative': 97676, 'Neutral': 72859, 'Positive': 3736, 'V. Positive': 0}
# Biden sents: {'V. Negative': 6507, 'Negative': 82936, 'Neutral': 63007, 'Positive': 4559, 'V. Positive': 0} 157,009
# Both sents: {'V. Negative': 3084, 'Negative': 75656, 'Neutral': 50984, 'Positive': 1236, 'V. Positive': 0} 130,960
# mentions: {'Trump': 136185, 'Biden': 109219, 'Both': 154420}

def scale_sentiment(result, polarity_sum):
    if result < 0.2:
        # print(input_text + " -------- V. Negative Score: " + str(result))
        polarity_sum = {"V. Negative": polarity_sum["V. Negative"] + 1, "Negative": polarity_sum["Negative"], 
        "Neutral": polarity_sum["Neutral"], "Positive": polarity_sum["Positive"], "V. Positive": polarity_sum["V. Positive"]}
    elif result >= 0.2 and result < 0.4:
        # print(input_text + " -------- Negative Score: " + str(result))
        polarity_sum = {"V. Negative": polarity_sum["V. Negative"], "Negative": polarity_sum["Negative"] + 1, 
        "Neutral": polarity_sum["Neutral"], "Positive": polarity_sum["Positive"], "V. Positive": polarity_sum["V. Positive"]}
    elif result >= 0.4 and result < 0.6:
        # print(input_text + " -------- Neutral Score: " + str(result))
        polarity_sum = {"V. Negative": polarity_sum["V. Negative"], "Negative": polarity_sum["Negative"], 
        "Neutral": polarity_sum["Neutral"] + 1, "Positive": polarity_sum["Positive"], "V. Positive": polarity_sum["V. Positive"]}
    elif result >= 0.6 and result < 0.8:
        # print(input_text + " -------- Positive Score: " + str(result))
        polarity_sum = {"V. Negative": polarity_sum["V. Negative"], "Negative": polarity_sum["Negative"], 
        "Neutral": polarity_sum["Neutral"], "Positive": polarity_sum["Positive"] + 1, "V. Positive": polarity_sum["V. Positive"]}
    elif result >= 0.8:
        # print(input_text + " -------- V. Positive Score: " + str(result))
        polarity_sum = {"V. Negative": polarity_sum["V. Negative"], "Negative": polarity_sum["Negative"], 
        "Neutral": polarity_sum["Neutral"], "Positive": polarity_sum["Positive"], "V. Positive": polarity_sum["V. Positive"] + 1}

    return polarity_sum

times = []
files = ['Data/Twitter/scores/biden_scores_final.json', 'Data/Twitter/scores/trump_scores_final.json', 'Data/Reddit/scores/biden_scores_final.json', 'Data/Reddit/scores/trump_scores_final.json']
# polarity_sum_trump = {"V. Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "V. Positive": 0}
polarity_sum_biden_twitter = {"V. Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "V. Positive": 0}
polarity_sum_biden_reddit = {"V. Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "V. Positive": 0}
# polarity_sum_both = {"V. Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "V. Positive": 0}
candidates = ["trump", "biden"]
sites = ["Twitter", "Reddit"]
# for file in files:
for candidate in candidates:
    for site in sites:
        polarity_scores = json.load(open('Data/' + site + '/scores/' + candidate + '_scores_final.json',))
        for data_sample in polarity_scores:
            for data in data_sample["mentions"]:
                if "score" in data:
                    result = data["score"]
                else:
                    result = data_sample["score"]
        
                if data["topic"] == "trump":
                    if site == "Twitter":
                        polarity_sum_biden_twitter = scale_sentiment(result, polarity_sum_biden_twitter)
                    else:
                        polarity_sum_biden_reddit = scale_sentiment(result, polarity_sum_biden_reddit)

        # elif data["topic"] == "trump":
        #     polarity_sum_trump = scale_sentiment(result, polarity_sum_trump)
        # elif data["topic"] == "both":
        #     polarity_sum_both = scale_sentiment(result, polarity_sum_both)    

        # if data_sample["mentions"][0]["topic"] == "biden":
        #     polarity_sum_biden = scale_sentiment(result, polarity_sum_biden)
        # elif data_sample["mentions"][0]["topic"] == "trump":
        #     polarity_sum_trump = scale_sentiment(result, polarity_sum_trump)
        # elif data_sample["mentions"][0]["topic"] == "both":
        #     polarity_sum_both = scale_sentiment(result, polarity_sum_both)       

polarity_sum_biden_twitter_percent = {"V. Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "V. Positive": 0}
polarity_sum_biden_reddit_percent = {"V. Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "V. Positive": 0}
trump_total = sum(polarity_sum_biden_reddit.values())
for key, value in polarity_sum_biden_reddit.items():
    if key == "V. Negative":
        print(key + ": " + str(value / trump_total * 100) + "%")
        polarity_sum_biden_reddit_percent = {"V. Negative": value / trump_total * 100, "Negative": polarity_sum_biden_reddit_percent["Negative"], "Neutral": polarity_sum_biden_reddit_percent["Neutral"], "Positive": polarity_sum_biden_reddit_percent["Positive"], "V. Positive": polarity_sum_biden_reddit_percent["V. Positive"]}
    elif key == "Negative":
        print(key + ": " + str(value / trump_total * 100) + "%")
        polarity_sum_biden_reddit_percent = {"V. Negative": polarity_sum_biden_reddit_percent["V. Negative"], "Negative": value / trump_total * 100, "Neutral": polarity_sum_biden_reddit_percent["Neutral"], "Positive": polarity_sum_biden_reddit_percent["Positive"], "V. Positive": polarity_sum_biden_reddit_percent["V. Positive"]}
    elif key == "Neutral":
        print(key + ": " + str(value / trump_total * 100) + "%")
        polarity_sum_biden_reddit_percent = {"V. Negative": polarity_sum_biden_reddit_percent["V. Negative"], "Negative": polarity_sum_biden_reddit_percent["Negative"], "Neutral": value / trump_total * 100, "Positive": polarity_sum_biden_reddit_percent["Positive"], "V. Positive": polarity_sum_biden_reddit_percent["V. Positive"]}
    elif key == "Positive":
        print(key + ": " + str(value / trump_total * 100) + "%")
        polarity_sum_biden_reddit_percent = {"V. Negative": polarity_sum_biden_reddit_percent["V. Negative"], "Negative": polarity_sum_biden_reddit_percent["Negative"], "Neutral": polarity_sum_biden_reddit_percent["Neutral"], "Positive": value / trump_total * 100, "V. Positive": polarity_sum_biden_reddit_percent["V. Positive"]}
    elif key == "V. Positive":
        print(key + ": " + str(value / trump_total * 100) + "%")
        polarity_sum_biden_reddit_percent = {"V. Negative": polarity_sum_biden_reddit_percent["V. Negative"], "Negative": polarity_sum_biden_reddit_percent["Negative"], "Neutral": polarity_sum_biden_reddit_percent["Neutral"], "Positive": polarity_sum_biden_reddit_percent["Positive"], "V. Positive": value / trump_total * 100}

print("Reddit sents: {}".format(polarity_sum_biden_reddit_percent))
print("Reddit Sum: {}\n".format(trump_total))

biden_total = sum(polarity_sum_biden_twitter.values())
for key, value in polarity_sum_biden_twitter.items():
    if key == "V. Negative":
        print(key + ": " + str(value / biden_total * 100) + "%")
        polarity_sum_biden_twitter_percent = {"V. Negative": value / biden_total * 100, "Negative": polarity_sum_biden_twitter_percent["Negative"], "Neutral": polarity_sum_biden_twitter_percent["Neutral"], "Positive": polarity_sum_biden_twitter_percent["Positive"], "V. Positive": polarity_sum_biden_twitter_percent["V. Positive"]}
    elif key == "Negative":
        print(key + ": " + str(value / biden_total * 100) + "%")
        polarity_sum_biden_twitter_percent = {"V. Negative": polarity_sum_biden_twitter_percent["V. Negative"], "Negative": value / biden_total * 100, "Neutral": polarity_sum_biden_twitter_percent["Neutral"], "Positive": polarity_sum_biden_twitter_percent["Positive"], "V. Positive": polarity_sum_biden_twitter_percent["V. Positive"]}
    elif key == "Neutral":
        print(key + ": " + str(value / biden_total * 100) + "%")
        polarity_sum_biden_twitter_percent = {"V. Negative": polarity_sum_biden_twitter_percent["V. Negative"], "Negative": polarity_sum_biden_twitter_percent["Negative"], "Neutral": value / biden_total * 100, "Positive": polarity_sum_biden_twitter_percent["Positive"], "V. Positive": polarity_sum_biden_twitter_percent["V. Positive"]}
    elif key == "Positive":
        print(key + ": " + str(value / biden_total * 100) + "%")
        polarity_sum_biden_twitter_percent = {"V. Negative": polarity_sum_biden_twitter_percent["V. Negative"], "Negative": polarity_sum_biden_twitter_percent["Negative"], "Neutral": polarity_sum_biden_twitter_percent["Neutral"], "Positive": value / biden_total * 100, "V. Positive": polarity_sum_biden_twitter_percent["V. Positive"]}
    elif key == "V. Positive":
        print(key + ": " + str(value / biden_total * 100) + "%")
        polarity_sum_biden_twitter_percent = {"V. Negative": polarity_sum_biden_twitter_percent["V. Negative"], "Negative": polarity_sum_biden_twitter_percent["Negative"], "Neutral": polarity_sum_biden_twitter_percent["Neutral"], "Positive": polarity_sum_biden_twitter_percent["Positive"], "V. Positive": value / biden_total * 100}

print("Twitter sents: {}".format(polarity_sum_biden_twitter_percent))
print("Twitter Sum: {}\n".format(biden_total))

# both_total = sum(polarity_sum_both.values())
# for key, value in polarity_sum_both.items():
#     if key == "V. Negative":
#         print(key + ": " + str(value / both_total * 100) + "%")
#     elif key == "Negative":
#         print(key + ": " + str(value / both_total * 100) + "%")
#     elif key == "Neutral":
#         print(key + ": " + str(value / both_total * 100) + "%")
#     elif key == "Positive":
#         print(key + ": " + str(value / both_total * 100) + "%")
#     elif key == "V. Positive":
#         print(key + ": " + str(value / both_total * 100) + "%")
# print("Both sents: {}\n".format(polarity_sum_both))
# print("Both Sum: {}\n".format(both_total))

names = list(polarity_sum_biden_twitter_percent.keys())
values_trump = list(polarity_sum_biden_reddit_percent.values())
values_biden = list(polarity_sum_biden_twitter_percent.values())
# values_both = list(polarity_sum_both.values())

fig, axs = plt.subplots(1, 1, figsize=(5, 4), sharey=True)
axs.bar(names, values_biden, color=(0.2, 0.4, 0.7, 0.6), label="Twitter")
axs.bar(names, values_trump, color=(0.6, 0.1, 0.1, 0.6), bottom=values_biden, label="Reddit")
# values_trump = np.array(values_trump)
# values_biden = np.array(values_biden)
# axs.bar(names, values_both, color=(0.4, 0.4, 0.4, 0.6), bottom=values_trump+values_biden, label="Both")
axs.legend()

plt.title('Sentiment Distribution Percentage Trump')

plt.show()