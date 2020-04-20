import time
import h5py
import torch
import numpy as np

from transformers import BertModel, BertTokenizer

text = "we want to know how bert embeddings can be used"
words = text.split()
word_to_token_map = []

bert_name = "bert-base-uncased"
model = BertModel.from_pretrained(bert_name)
model.eval()
tokenizer = BertTokenizer.from_pretrained(bert_name)

# input_ids = torch.tensor([tokenizer.encode(text)])

# tokens = tokenizer.tokenize(text)


# for i, token in enumerate(tokens):
#   if token.startswith("##"):
#     continue
#   word_to_token_map.append(i+1)

# tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
start = time.time()

tokens = []
for w in words:
  word_to_token_map.append(len(tokens)+1)
  tokens += tokenizer.tokenize(w)
word_to_token_map.append(len(tokens)+1)
print(tokens)
print(word_to_token_map)
# word_to_token_map = np.array(word_to_token_map) + 1

input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)
# input_ids = tokenizer.prepare_for_model(input_ids, max_length=16, pad_to_max_length=True)['input_ids']
# print(input_ids)

with torch.no_grad():
# model.detach()
  features, _ = model(torch.tensor([[tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]]))
# print(features)

# word_features = features[0][word_to_token_map[:-1]]

word_features = np.zeros((len(words), features.size()[-1]), dtype=np.float32)
for i in range(len(words)):
  word_features[i] = torch.mean(features[0][word_to_token_map[i]:word_to_token_map[i+1]], dim=0)#.numpy()

print(word_features.dtype)
print(time.time() - start, "seconds")

# with h5py.File('test.h5', 'w') as f:
#   dset = f.create_dataset('emb', shape=word_features.shape, 
#                             maxshape=(None,) + word_features.shape[1:],
#                             dtype=word_features.dtype)

#   dset[:] = word_features
#   row_count = word_features.shape[0]
#   for i in range(1000):
#     dset.resize(row_count + word_features.shape[0], axis=0)
#     dset[row_count:] = word_features
#     row_count += word_features.shape[0]