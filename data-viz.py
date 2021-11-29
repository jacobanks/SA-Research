from datetime import datetime
import json, csv
from nltk import FreqDist
from nltk.tokenize import sent_tokenize
import helpers as helper

with open('Data/Twitter/biden_tweets.json', 'r') as outfile:
    submissions = json.load(outfile)
    times = []
    polarity_scores = json.load(open('Data/Reddit/analyzed/biden_scores.json',))

    polarity_sum = []

    count = 0
    for data_sample in submissions:
        input_text = data_sample['body']
        polarity_object = list(filter(lambda score: score['id'] == data_sample['id'], polarity_scores))
        if len(polarity_object) > 0:
            result = polarity_object[0]['score'] 
            
            if len(input_text) > 5:
                if result < 0.2:
                    # print(input_text + " -------- Very Negative Score: " + str(result))
                    polarity_sum.append("Very Negative")
                elif result >= 0.2 and result < 0.4:
                    # print(input_text + " -------- Negative Score: " + str(result))
                    polarity_sum.append("Negative")
                elif result >= 0.4 and result < 0.6:
                    # print(input_text + " -------- Neutral Score: " + str(result))
                    polarity_sum.append("Neutral")
                elif result >= 0.6 and result < 0.8:
                    # print(input_text + " -------- Positive Score: " + str(result))
                    polarity_sum.append("Positive")
                elif result >= 0.8:
                    # print(input_text + " -------- Very Positive Score: " + str(result))
                    polarity_sum.append("Very Positive")

                freq_dist_pos = FreqDist(polarity_sum)
                print("\rMost Common Sentiments: {}".format(freq_dist_pos.most_common(5)), end="")
                
                dt_object = datetime.fromtimestamp(data_sample['created_utc'])
                times.append(str(dt_object.month) + "/" + str(dt_object.day))

    freq_dist_times = FreqDist(times)
    print("\nMost Common Times: {}".format(freq_dist_times.most_common(20)))