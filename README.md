# fuzzers-comps
https://docs.google.com/document/d/11Jetb9c4ESyEJDJimQIl_9iA-YA4926-yMGC6GdH9j4/edit?usp=sharing

## Setup Config
1. Make a copy of template.json
2. Fill out the copy, and rename as wanted

## Training the Model
1. Enter the `network` directory
2. Run `python3 pipeline.py -c <your_config_file_path>`

## Generate Mass Results
1. Enter the `network` directory
2. Run `python3 masstest.py -c <your_config_file_path>`

## Testing the Results
1. Enter the `network` directory
2. Run `python3 testing.py -c <your_config_file_path>`
3. Respond to the prompts in the terminal window
4. ???
5. Profit

## If you got sent this repo because you're being helpful
1. Run `python3`
    a. Type `import torch`
    b. Type `torch.cuda.is_available()`
    c. If this prints `False`, tell the person who sent you this and stop.
2. Enter the `network` directory
3. Run `python3 pipeline.py -c configs/GridSearchToo/big_reddit_1000.json`
    a. Optionally, create a screen or go through some other method that
       allows you to keep python programs running in the background.
4. Run `python3 masstest.py -c configs/GridSearchToo/big_reddit_1000.json`
5. When it's done, contact the person who sent you this with to figure out
   how to transfer the relevant files.
