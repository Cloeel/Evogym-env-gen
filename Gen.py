from bardapi import Bard
import os
import api
import json

os.environ['_BARD_API_KEY'] = api.api_key

with open('./json-list-data/CaveCrawler-v0.json', 'r') as f:
    data = json.load(f)

with open('./json-list-data/Balancer-v0.json', 'r') as f:
    data1 = json.load(f)

respon = Bard().get_answer("二次元のマップである{},{}のようなマップを作成して。".format(data,data1))['content']
print(respon)