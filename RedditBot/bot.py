# %%
import requests
import json

personal_id_dict = json.load(open("reddit-key-silas.json", "r"))
bot_id_dict = json.load(open("reddit-key-bot-agent.json", "r"))

# note that CLIENT_ID refers to 'personal use script' and SECRET_TOKEN to 'token'
auth = requests.auth.HTTPBasicAuth(bot_id_dict["U"], bot_id_dict["P"])

# here we pass our login method (password), username, and password
data = {'grant_type': 'password',
        'username': personal_id_dict["U"],
        'password': personal_id_dict["P"]}

# setup our header info, which gives reddit a brief description of our app
headers = {'User-Agent': 'MyBot/0.0.1'}

# send our request for an OAuth token
res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers)

# convert response to JSON and pull access_token value
TOKEN = res.json()['access_token']

# add authorization to our headers dictionary
headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}


res = requests.get("https://oauth.reddit.com/r/python/hot",
                   headers=headers)

data = res.json()
json.dump(data, open("pythonSubredditData.json", "w+"))
# %%


print(data["data"].keys())  # let's see what we get
