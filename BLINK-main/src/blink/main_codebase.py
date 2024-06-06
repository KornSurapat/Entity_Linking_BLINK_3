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


def read_jsonl(file_path):
  test_data = []
  with open(file_path, 'r', encoding='utf-8') as f:
    for l in f:
      json_object = json.loads(l.strip())
      test_data.append(json_object)
  return test_data

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

def reform_result(added_data, index, predictions, scores):
  for mention in range(len(predictions)):
    record = []
    for candidate in range(len(predictions[mention])):
      record.append([predictions[mention][candidate], float(scores[mention][candidate])])
    added_data[index]["gold_spans"][mention]["predictions"] = record
  return added_data

def write_jsonl(file_path, added_data):
  with open(file_path, 'w', encoding='utf-8') as f:
    for item in added_data:
      f.write(json.dumps(item) + '\n')

def process(file_path_read, file_path_write):
  # Read input data
  test_data = read_jsonl(file_path_read)
  # Prepare container
  added_data = test_data
  for i in range(len(test_data)):
    # Run model
    data_to_link = transform(test_data, i)
    _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
    # Reform result data
    added_data = reform_result(added_data, i, predictions, scores)
  # store data
  write_jsonl(file_path_write, added_data)


# 1. ace
# # Read input data
# test1_ace = []
# with open("src/blink/ace2004.jsonl", 'r', encoding='utf-8') as f:
#   for l in f:
#     json_object = json.loads(l.strip())
#     test1_ace.append(json_object)
# # Prepare container
# added_data = test1_ace
# for i in range(len(test1_ace)):
#   # Run model
#   data_to_link = transform(test1_ace, i)
#   _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
#   # Just show
#   # j = 0
#   # for mention in data_to_link:
#   #   print("mention: "+ str(mention["mention"]))
#   #   print("predictions: ")
#   #   for k in range(len(predictions[j])):
#   #     print("  " + str(k + 1) + ". title: " + str(predictions[j][k]) + ", score: " + str(scores[j][k]))
#   #   j += 1
#   # print("____________________________________________________________________________________________________")
#   # Just show
#   # Reform result data
#   added_data = reform_result(added_data, i, predictions, scores)
# # store data
# write_jsonl("src/blink/ace2004_pred.jsonl", added_data)

process("src/blink/ace2004.jsonl", "src/blink/ace2004_pred.jsonl")
# Just test
print("Test____________________________________________________________________________________________________")
with open("src/blink/ace2004_pred.jsonl", 'r', encoding='utf-8') as f:
  for l in f:
    json_object = json.loads(l.strip())
    print(json_object)
print("Test____________________________________________________________________________________________________")
# Just test

# 2. aida
# process("src/blink/aida.jsonl", "src/blink/aida_pred.jsonl")

# 3. aquaint
# process("src/blink/aquaint.jsonl", "src/blink/aquaint_pred.jsonl")

# 4. cweb
# process("src/blink/cweb.jsonl", "src/blink/cweb_pred.jsonl")

# 5. graphq
# process("src/blink/graphq.jsonl", "src/blink/graphq_pred.jsonl")

# 6. mintaka
# process("src/blink/mintaka.jsonl", "src/blink/mintaka_pred.jsonl")

# 7. msnbc
# process("src/blink/msnbc.jsonl", "src/blink/msnbc_pred.jsonl")

# 8. reddit_comments
# process("src/blink/reddit_comments.jsonl", "src/blink/reddit_comments_pred.jsonl")

# 9. reddit_posts
# process("src/blink/reddit_posts.jsonl", "src/blink/reddit_posts_pred.jsonl")

# 10. shadow
# process("src/blink/shadow.jsonl", "src/blink/shadow_pred.jsonl")

# 11. tail
# process("src/blink/tail.jsonl", "src/blink/tail_pred.jsonl")

# 12. top
# process("src/blink/top.jsonl", "src/blink/top_pred.jsonl")

# 13. tweeki
# process("src/blink/tweeki.jsonl", "src/blink/tweeki_pred.jsonl")

# 14. webqsp
# process("src/blink/webqsp.jsonl", "src/blink/webqsp_pred.jsonl")

# 15. wiki
# process("src/blink/wiki.jsonl", "src/blink/wiki_pred.jsonl")

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
