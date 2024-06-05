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

def reform_result(data_to_link, ids, predictions, scores):
  result_data = []
  for mention in data_to_link:
    record = {}
    record["mention"] = mention["mention"]
    record["predictions"] = []
    for i in range(len(ids)):
      inner_record = {}
      inner_record["id"] = ids[i]
      inner_record["title"] = predictions[i]
      inner_record["score"] = scores[i]
      record["predictions"].append(inner_record)
    result_data.append(record)
  return result_data

def write_jsonl(file_path, outer_result_data):
  with open(file_path, 'w', encoding='utf-8') as f:
    for item in outer_result_data:
      f.write(json.dumps(item) + '\n')

# 1. ace
test1_ace = []
with open("src/blink/ace2004.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test1_ace.append(json_object)

outer_result_data = []

for i in range(len(test1_ace)):
  # Prepare data
  outer_record = {}
  outer_record["text"] = test1_ace[i]["text"]
  # Run model
  data_to_link = transform(test1_ace, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
  # Just show
  for mention in data_to_link:
    print("mention: "+ str(mention["mention"]))
    print("predictions: ")
    for i in range(len(ids)):
      print("  " + str(i + 1) + ". id: " + str(ids[i]) + ", title: " + str(predictions[i]) + ", score: " + str(scores[i]))
  print("____________________________________________________________________________________________________")
  # Reform result data
  outer_record["result"] = reform_result(data_to_link, ids, predictions, scores)
  # store data
  outer_result_data.append(outer_record)

# write_jsonl("src/blink/ace2004_pred.jsonl", outer_result_data)

# 2. aida
test2_aida = []
with open("src/blink/aida.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test2_aida.append(json_object)

for i in range(len(test2_aida)):
  data_to_link = transform(test2_aida, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)


# 3. aquaint
test3_aquaint = []
with open("src/blink/aquaint.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test3_aquaint.append(json_object)

for i in range(len(test3_aquaint)):
  data_to_link = transform(test3_aquaint, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

# 4. cweb
test4_cweb = []
with open("src/blink/cweb.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test4_cweb.append(json_object)

for i in range(len(test4_cweb)):
  data_to_link = transform(test4_cweb, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

# 5. graphq
test5_graphq = []
with open("src/blink/graphq.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test5_graphq.append(json_object)

for i in range(len(test5_graphq)):
  data_to_link = transform(test5_graphq, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

# 6. mintaka
test6_mintaka = []
with open("src/blink/mintaka.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test6_mintaka.append(json_object)

for i in range(len(test6_mintaka)):
  data_to_link = transform(test6_mintaka, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

# 7. msnbc
test7_msnbc = []
with open("src/blink/msnbc.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test7_msnbc.append(json_object)

for i in range(len(test7_msnbc)):
  data_to_link = transform(test7_msnbc, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

# 8. reddit_comments
test8_reddit_comments = []
with open("src/blink/reddit_comments.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test8_reddit_comments.append(json_object)

for i in range(len(test8_reddit_comments)):
  data_to_link = transform(test8_reddit_comments, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

# 9. reddit_posts
test9_reddit_posts = []
with open("src/blink/reddit_posts.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test9_reddit_posts.append(json_object)

for i in range(len(test9_reddit_posts)):
  data_to_link = transform(test9_reddit_posts, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

# 10. shadow
test10_shadow = []
with open("src/blink/shadow.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test10_shadow.append(json_object)

for i in range(len(test10_shadow)):
  data_to_link = transform(test10_shadow, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

# 11. tail
test11_tail = []
with open("src/blink/tail.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test11_tail.append(json_object)

for i in range(len(test11_tail)):
  data_to_link = transform(test11_tail, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

# 12. top
test12_top = []
with open("src/blink/top.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test12_top.append(json_object)

for i in range(len(test12_top)):
  data_to_link = transform(test12_top, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

# 13. tweeki
test13_tweeki = []
with open("src/blink/tweeki.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test13_tweeki.append(json_object)

for i in range(len(test13_tweeki)):
  data_to_link = transform(test13_tweeki, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

# 14. webqsp
test14_webqsp = []
with open("src/blink/webqsp.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test14_webqsp.append(json_object)

for i in range(len(test14_webqsp)):
  data_to_link = transform(test14_webqsp, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

# 15. wiki
test15_wiki = []
with open("src/blink/wiki.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    test15_wiki.append(json_object)

for i in range(len(test15_wiki)):
  data_to_link = transform(test15_wiki, i)
  _, _, _, _, _, ids, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

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
