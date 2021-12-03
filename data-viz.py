# from datetime import datetime
import json
from nltk import FreqDist
import matplotlib.pyplot as plt
import numpy as np

# Reddit OVERALL Distribution of sentiment: {'Very Negative': 44411, 'Negative': 207189, 'Neutral': 140932, 'Positive': 7081, 'Very Positive': 0}
# Trump sents: {'Very Negative': 27665, 'Negative': 141155, 'Neutral': 95600, 'Positive': 4705, 'Very Positive': 0}
# Biden sents: {'Very Negative': 14093, 'Negative': 104364, 'Neutral': 77905, 'Positive': 4516, 'Very Positive': 0}
# Both sents: {'Very Negative': 7755, 'Negative': 46819, 'Neutral': 36514, 'Positive': 1933, 'Very Positive': 0}
# mentions: {'Trump': 157048, 'Biden': 113876, 'Both': 128688}

# Twitter OVERALL Distribution of sentiment: {'Very Negative': 29452, 'Negative': 197847, 'Neutral': 162813, 'Positive': 9806, 'Very Positive': 0}
# Trump sents: {'Very Negative': 14872, 'Negative': 97676, 'Neutral': 72859, 'Positive': 3736, 'Very Positive': 0}
# Biden sents: {'Very Negative': 6507, 'Negative': 82936, 'Neutral': 63007, 'Positive': 4559, 'Very Positive': 0} 157,009
# Both sents: {'Very Negative': 3084, 'Negative': 75656, 'Neutral': 50984, 'Positive': 1236, 'Very Positive': 0} 130,960
# mentions: {'Trump': 136185, 'Biden': 109219, 'Both': 154420}

def scale_sentiment(result, polarity_sum):
    if result < 0.2:
        # print(input_text + " -------- Very Negative Score: " + str(result))
        polarity_sum = {"Very Negative": polarity_sum["Very Negative"] + 1, "Negative": polarity_sum["Negative"], 
        "Neutral": polarity_sum["Neutral"], "Positive": polarity_sum["Positive"], "Very Positive": polarity_sum["Very Positive"]}
    elif result >= 0.2 and result < 0.4:
        # print(input_text + " -------- Negative Score: " + str(result))
        polarity_sum = {"Very Negative": polarity_sum["Very Negative"], "Negative": polarity_sum["Negative"] + 1, 
        "Neutral": polarity_sum["Neutral"], "Positive": polarity_sum["Positive"], "Very Positive": polarity_sum["Very Positive"]}
    elif result >= 0.4 and result < 0.6:
        # print(input_text + " -------- Neutral Score: " + str(result))
        polarity_sum = {"Very Negative": polarity_sum["Very Negative"], "Negative": polarity_sum["Negative"], 
        "Neutral": polarity_sum["Neutral"] + 1, "Positive": polarity_sum["Positive"], "Very Positive": polarity_sum["Very Positive"]}
    elif result >= 0.6 and result < 0.8:
        # print(input_text + " -------- Positive Score: " + str(result))
        polarity_sum = {"Very Negative": polarity_sum["Very Negative"], "Negative": polarity_sum["Negative"], 
        "Neutral": polarity_sum["Neutral"], "Positive": polarity_sum["Positive"] + 1, "Very Positive": polarity_sum["Very Positive"]}
    elif result >= 0.8:
        # print(input_text + " -------- Very Positive Score: " + str(result))
        polarity_sum = {"Very Negative": polarity_sum["Very Negative"], "Negative": polarity_sum["Negative"], 
        "Neutral": polarity_sum["Neutral"], "Positive": polarity_sum["Positive"], "Very Positive": polarity_sum["Very Positive"] + 1}

    return polarity_sum

times = []
files = ['Data/Twitter/scores/trump_scores.json', 'Data/Twitter/scores/biden_scores.json']
polarity_sum_trump = {"Very Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0}
polarity_sum_biden = {"Very Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0}
polarity_sum_both = {"Very Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0}

for file in files:
    polarity_scores = json.load(open(file,))
    for data_sample in polarity_scores:
        for data in data_sample["mentions"]:
            if "score" in data:
                result = data["score"]
            else:
                result = data_sample["score"]

            if data["topic"] == "biden":
                polarity_sum_biden = scale_sentiment(result, polarity_sum_biden)
            elif data["topic"] == "trump":
                polarity_sum_trump = scale_sentiment(result, polarity_sum_trump)
            elif data['topic'] == "both":
                polarity_sum_both = scale_sentiment(result, polarity_sum_both)      

trump_total = sum(polarity_sum_trump.values())
for key, value in polarity_sum_trump.items():
    if key == "Very Negative":
        print(key + ": " + str(value / trump_total * 100) + "%")
    elif key == "Negative":
        print(key + ": " + str(value / trump_total * 100) + "%")
    elif key == "Neutral":
        print(key + ": " + str(value / trump_total * 100) + "%")
    elif key == "Positive":
        print(key + ": " + str(value / trump_total * 100) + "%")
    elif key == "Very Positive":
        print(key + ": " + str(value / trump_total * 100) + "%")
print("Trump sents: {}\n".format(polarity_sum_trump))

biden_total = sum(polarity_sum_biden.values())
for key, value in polarity_sum_biden.items():
    if key == "Very Negative":
        print(key + ": " + str(value / biden_total * 100) + "%")
    elif key == "Negative":
        print(key + ": " + str(value / biden_total * 100) + "%")
    elif key == "Neutral":
        print(key + ": " + str(value / biden_total * 100) + "%")
    elif key == "Positive":
        print(key + ": " + str(value / biden_total * 100) + "%")
    elif key == "Very Positive":
        print(key + ": " + str(value / biden_total * 100) + "%")
print("Biden sents: {}\n".format(polarity_sum_biden))

both_total = sum(polarity_sum_both.values())
for key, value in polarity_sum_both.items():
    if key == "Very Negative":
        print(key + ": " + str(value / both_total * 100) + "%")
    elif key == "Negative":
        print(key + ": " + str(value / both_total * 100) + "%")
    elif key == "Neutral":
        print(key + ": " + str(value / both_total * 100) + "%")
    elif key == "Positive":
        print(key + ": " + str(value / both_total * 100) + "%")
    elif key == "Very Positive":
        print(key + ": " + str(value / both_total * 100) + "%")
print("Both sents: {}\n".format(polarity_sum_both))

names = list(polarity_sum_trump.keys())
values_trump = list(polarity_sum_trump.values())
values_biden = list(polarity_sum_biden.values())
values_both = list(polarity_sum_both.values())

fig, axs = plt.subplots(1, 1, figsize=(5.5, 4), sharey=True)
axs.bar(names, values_biden, color=(0.2, 0.4, 0.7, 0.6), label="Biden")
axs.bar(names, values_trump, color=(0.6, 0.1, 0.1, 0.6), bottom=values_biden, label="Trump")
values_trump = np.array(values_trump)
values_biden = np.array(values_biden)
axs.bar(names, values_both, color=(0.4, 0.4, 0.4, 0.6), bottom=values_trump+values_biden, label="Both")
axs.legend()
fig.suptitle('Candidate Sentiment Distribution Twitter')

plt.show()
