from datetime import datetime
import json
from nltk import FreqDist
from nltk.tokenize import sent_tokenize

# def check_for_mixed(input_text):
#     # if input_text contains the words 'trump' and 'biden'
#     scores = []
#     if 'trump' in input_text.lower() and 'biden' in input_text.lower():
#         sentences = sent_tokenize(input_text)
#         if len(sentences) > 1:                              # if the input text contains more than one sentence
#             scores.append("mixed sentences")
#             for sentence in sentences:
#                 if 'trump' in sentence.lower(): 
#                     if 'biden' in sentence.lower():         # if the sentence contains both trump and biden
#                         scores.append("mixed sentence")
#                     else:                                   # if the sentence contains only trump
#                         scores.append("trump sentence")
#                 elif 'biden' in sentence.lower():
#                     if 'trump' in sentence.lower():         # if the sentence contains both biden and trump
#                         scores.append("mixed sentence")
#                     else:                                   # if the sentence contains only biden
#                         scores.append("biden sentence")
#         else:                                               # if the input text contains only one sentence
#             scores.append("mixed single comment")
#     elif 'trump' in input_text.lower():                     # if the input text contains only 'trump'
#         scores.append("trump")
#     elif 'biden' in input_text.lower():                     # if the input text contains only 'biden'
#         scores.append("biden")
    
#     return scores

# with open('biden_comments.json', 'r') as outfile:
#     submissions = json.load(outfile)
#     times = []
#     polarity_scores = json.load(open('biden_comments_analyzed.json',))

#     scores = []
#     polarity_sum = []

#     count = 0
#     for data_sample in submissions:
#         input_text = data_sample['body']
#         polarity_object = list(filter(lambda score: score['id'] == data_sample['id'], polarity_scores))
#         if len(polarity_object) > 0:
#             result = polarity_object[0]['score'] 
            
#             if len(input_text) > 5:
#                 if result < 0.2:
#                     # print(input_text + " -------- Very Negative Score: " + str(result))
#                     polarity_sum.append("Very Negative")
#                     scores = scores + check_for_mixed(input_text)
#                 elif result >= 0.2 and result < 0.4:
#                     # print(input_text + " -------- Negative Score: " + str(result))
#                     polarity_sum.append("Negative")
#                     scores = scores + check_for_mixed(input_text)
#                 elif result >= 0.4 and result < 0.6:
#                     # print(input_text + " -------- Neutral Score: " + str(result))
#                     polarity_sum.append("Neutral")
#                     scores = scores + check_for_mixed(input_text)
#                 elif result >= 0.6 and result < 0.8:
#                     # print(input_text + " -------- Positive Score: " + str(result))
#                     polarity_sum.append("Positive")
#                     scores = scores + check_for_mixed(input_text)
#                 elif result >= 0.8:
#                     # print(input_text + " -------- Very Positive Score: " + str(result))
#                     polarity_sum.append("Very Positive")
#                     scores = scores + check_for_mixed(input_text)

#                 freq_dist_pos = FreqDist(polarity_sum)
#                 print("\rMost Common Sentiments: {}".format(freq_dist_pos.most_common(5)), end="")
                
#                 dt_object = datetime.fromtimestamp(data_sample['created_utc'])
#                 times.append(str(dt_object.month) + "/" + str(dt_object.day))

#     freq_dist_times = FreqDist(times)
#     print("\nMost Common Times: {}".format(freq_dist_times.most_common(20)))
        
#     freq_dist_pos = FreqDist(scores)
#     print("\nMost Common Scores: {}".format(freq_dist_pos.most_common(20)))