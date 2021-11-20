from psaw import PushshiftAPI
import datetime as dt
from datetime import date, datetime
import json
from nltk import FreqDist
from langdetect import detect
import re

# Get Dates frequency distribution
with open('biden_comments.json', 'r') as outfile:
    data = json.load(outfile)
    times = []
    for item in data:
        stringliteral = re.sub(r'http\S+', '', item['body'])
        if stringliteral:
            try:
                language = detect(stringliteral)
            except:
                language = "error"
                print("This row throws an error:", stringliteral)
            times.append(language)
        # dt_object = datetime.fromtimestamp(item['created_utc'])
        # if 'trump' in item['body'].lower():
        #     times.append("mixed")            
        # elif 'biden' in item['body'].lower():
        #     times.append("biden")
        # else:
        #     times.append("none")
        #     print(item)
        # times.append(str(dt_object.month) + "/" + str(dt_object.day))
        # print("dt_object =", dt_object)

    freq_dist_pos = FreqDist(times)
    print("\nMost Common times: ")
    print(freq_dist_pos.most_common(20))

# Gathering data from Reddit
# api = PushshiftAPI()

# start_epoch=int(dt.datetime(2020, 10, 20).timestamp())
# end_epoch=int(dt.datetime(2020, 11, 3).timestamp())

# for search in ["biden", "trump"]:
#     print(api.metadata_.get('shards'))

#     if search == "trump":
#         search_term = "trump|donald trump"
#     else:
#         search_term = "biden|joe biden"

#     # Search for comments with the above search terms
#     submissions = api.search_comments(q=search_term, after=start_epoch, before=end_epoch, filter=['id','permalink', 'author', 'body', 'subreddit', 'score'])

#     comments = []

#     # Get all comments
#     print("\rFetching PSAW {} posts... fetched 0 posts.".format(search), end="")
#     for comment in submissions:
#         if comment.subreddit != "csci040temp":
#             # Append new comment to the list
#             comments.append(comment[-1])
#             count = len(comments)
#             # Every 5000 comments, save the list to a file in json format
#             if count % 5000 == 0:
#                 print("\nSaved to " + search + "_comments.json at count: " + str(count))
#                 with open(search + '_comments.json', 'w') as outfile:
#                     outfile.write(json.dumps(comments, indent=4))
            
#             if count >= 500000:
#                 break
#             print("\rFetching PSAW {} posts... fetched {} posts.".format(search, count), end="")
        
#     print(len(comments))
