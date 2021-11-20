from psaw import PushshiftAPI
import datetime as dt
from datetime import date, datetime
import json
from nltk import FreqDist

# # Get Dates frequency distribution
# with open('trump_comments.json', 'r') as outfile:
#     data = json.load(outfile)
#     times = []
#     for item in data:
#         dt_object = datetime.fromtimestamp(item['created_utc'])
#         times.append(str(dt_object.month) + "/" + str(dt_object.day))
#         # print("dt_object =", dt_object)

#     freq_dist_pos = FreqDist(times)
#     print("\nMost Common times: ")
#     print(freq_dist_pos.most_common(20))

# Gathering data from Reddit
api = PushshiftAPI()

start_epoch=int(dt.datetime(2020, 10, 20).timestamp())
end_epoch=int(dt.datetime(2020, 11, 3).timestamp())

for search in ["trump", "biden"]:
    if search == "trump":
        search_term = "trump|donald trump"
    else:
        search_term = "biden|joe biden"
    submissions = api.search_comments(q=search_term, after=start_epoch, before=end_epoch, filter=['permalink', 'author', 'body', 'subreddit', 'score'])

    print(api.metadata_.get('shards'))

    comments = []
    count = 0
    print("\rFetching PSAW {} posts... fetched 0 posts.".format(search), end="")
    for comment in submissions:
        if comment.subreddit != "csci040temp":
            comments.append(comment[-1])
            if count % 5000 == 0:
                print("\nSaved to " + search + "_comments.json at count: " + str(count))
                with open(search + '_comments.json', 'w') as outfile:
                    outfile.write(json.dumps(comments, indent=4))
            count += 1
            print("\rFetching PSAW {} posts... fetched {} posts.".format(search, count), end="")
        
    print(len(comments))
