import time
import os
import bot
import argparse
import json

subreddits = [
    "insanepeoplefacebook",
    "football",
    "pics",
    "trashy"
]

parser = argparse.ArgumentParser(
    description='Enables testing of neural network.')
parser.add_argument("-c", "--config",
                    help="config file for running model. Should correspond to model.",
                    default="configs/reddit.json")
parser.add_argument("-l", "--loglevel",
                    help="Level at which to log events.",
                    default="INFO")
args = parser.parse_args()


with open(str(args.config)) as f:
    config = json.load(f)

i = 0
while True:
    sub = subreddits[i % len(subreddits)]
    try:
        print("posting to", sub)
        bot.main(config, sub)
    except:
        print("failed to execute")
    time.sleep(600)
    i += 1
