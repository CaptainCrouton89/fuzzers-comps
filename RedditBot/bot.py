# %%
import requests
import json
import praw

credentials = json.load(open("credentials.json", "r"))

reddit = praw.Reddit(
    client_id=credentials["client_id"],
    client_secret=credentials["client_secret"],
    user_agent=credentials["user_agent"],
    redirect_uri=credentials["redirect_uri"],
    refresh_token=credentials["refresh_token"]
)

subr = 'pythonsandlot' # Choose your subreddit
 
subreddit = reddit.subreddit(subr) # Initialize the subreddit to a variable
 
title = 'Just Made My first Post on Reddit Using Python.'
 
selftext = '''
I am learning how to use the Reddit API with Python using the PRAW wrapper.
By following the tutorial on https://www.jcchouinard.com/post-on-reddit-api-with-python-praw/
This post was uploaded from my Python Script
'''

subreddit.submit(title,selftext=selftext)


# json.dump(data, open("pythonSubredditData.json", "w+"))
# %%


# print(data["data"].keys())  # let's see what we get
