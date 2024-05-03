import json
import random

with open("dataset/valid.json") as f:
    valid = json.load(f)
    
random.shuffle(valid)

json.dump(valid[:1000], open("dataset/mytrain.json", "w"), indent = 4)
json.dump(valid[1000:], open("dataset/myvalid.json", "w"), indent = 4)