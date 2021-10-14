import praw
from psaw import PushshiftAPI
import datetime as dt
import json

api = PushshiftAPI()

start_epoch=int(dt.datetime(2020, 9, 1).timestamp())
end_epoch=int(dt.datetime(2020, 11, 1).timestamp())

for search in ["trump"]:
    submissions = list(api.search_comments(q=search, after=start_epoch, before=end_epoch, filter=['permalink', 'author', 'body', 'subreddit'], limit=50000))

    print(api.metadata_.get('shards'))

    comments = []
    for comment in submissions:
        if comment.subreddit != "csci040temp":
            comments.append(comment[-1])
        
    print(len(comments))

    with open('experiment-data/' + search + '_comments.json', 'w') as outfile:
        outfile.write(json.dumps(comments, indent=4))
