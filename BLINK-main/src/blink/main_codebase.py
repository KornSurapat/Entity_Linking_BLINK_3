# import necessary files & library
import blink.main_dense as main_dense
import argparse
import json


# Set up model 
models_path = "models/" # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": False, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)


def transform(test_data, index):
  data_to_link = []
  for i in range(len(test_data[index]["gold_spans"])):
    record = {}
    record["id"] = i
    record["label"] = "unknown"
    record["label_id"] = -1
    record["context_left"] = test_data[index]["text"][
        : test_data[index]["gold_spans"][i]["start"]].lower()
    record["mention"] = test_data[index]["text"][
        test_data[index]["gold_spans"][i]["start"] :  test_data[index]["gold_spans"][i]["start"] +  test_data[index]["gold_spans"][i]["length"]].lower()
    record["context_right"] = test_data[index]["text"][
        test_data[index]["gold_spans"][i]["start"] +  test_data[index]["gold_spans"][i]["length"] : -1].lower()
    data_to_link.append(record)
  return data_to_link

# 1. ace
test1_ace = []
with open("blink/ace2004.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test1_ace.append(json_object)

for i in range(len(test1_ace)):
  data_to_link = transform(test1_ace, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
  # Just test
  for mention in data_to_link:
    print("mention: "+ str(mention["mention"]))
    print("predictions: ")
    for i in range(len(ids)):
      print("  " + str(i + 1) + ". id: " + str(ids[i]) + ", title: " + str(predictions[i]) + ", score: " + str(scores[i]))
  print("____________________________________________________________________________________________________")

# 2. aida
test2_aida = []
with open("blink/aida.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test2_aida.append(json_object)

# 3. aquaint
test3_aquaint = []
with open("blink/aquaint.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test3_aquaint.append(json_object)

# 4. cweb
test4_cweb = []
with open("blink/cweb.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test4_cweb.append(json_object)

# 5. graphq
test5_graphq = []
with open("blink/graphq.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test5_graphq.append(json_object)

# 6. mintaka
test6_mintaka = []
with open("blink/mintaka.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test6_mintaka.append(json_object)

# 7. msnbc
test7_msnbc = []
with open("blink/msnbc.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test7_msnbc.append(json_object)

# 8. reddit_comments
test8_reddit_comments = []
with open("blink/reddit_comments.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test8_reddit_comments.append(json_object)

# 9. reddit_posts
test9_reddit_posts = []
with open("blink/reddit_posts.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test9_reddit_posts.append(json_object)

# 10. shadow
test10_shadow = []
with open("blink/shadow.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test10_shadow.append(json_object)

# 11. tail
test11_tail = []
with open("blink/tail.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test11_tail.append(json_object)

# 12. top
test12_top = []
with open("blink/top.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test12_top.append(json_object)

# 13. tweeki
test13_tweeki = []
with open("blink/tweeki.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test13_tweeki.append(json_object)

# 14. webqsp
test14_webqsp = []
with open("blink/webqsp.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test14_webqsp.append(json_object)

# 15. wiki
test15_wiki = []
with open("blink/wiki.jsonl", 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test15_wiki.append(json_object)

# data_to_link = [ {
#                     "id": 0,
#                     "label": "unknown",
#                     "label_id": -1,
#                     "context_left": "".lower(),
#                     "mention": "Shakespeare".lower(),
#                     "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
#                 },
#                 {
#                     "id": 1,
#                     "label": "unknown",
#                     "label_id": -1,
#                     "context_left": "Shakespeare's account of the Roman general".lower(),
#                     "mention": "Julius Caesar".lower(),
#                     "context_right": "'s murder by his friend Brutus is a meditation on duty.".lower(),
#                 }
#                 ]
#
# _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
