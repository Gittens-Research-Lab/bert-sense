import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from utils import Params
#from train import Net, BertDataset, combine_samples
from w2v_bert import W2VNet, BertCtxDataset, combine_samples

def load_vocab(path):
  word2id = {}
  id2word = []
  with open(path) as f:
    for i, line in enumerate(f):
      word = line.split()[0]
      word2id[word] = i
      id2word.append(word)
  
  return word2id, id2word

SCRATCH = "/gpfs/u/home/BERT/BERTnksh/scratch/"
data_path = os.path.join(SCRATCH, "data/bert-sense/small_750k_bert_ctx_d5.h5")
vocab_path = os.path.join(SCRATCH, 'data/bert-sense/vocab10_small_750k.txt')
vocab, id2word = load_vocab(vocab_path)
#model_dir = os.path.join(SCRATCH, 'output/bert-sense/small-250k-noshuf-bert-d5-s3-adam-lr1e3')
model_dir = os.path.join(SCRATCH, 'output/bert-sense/small-750k-noshuf-dissg-bert-ctx-d5-s3-d300-adam-lr1e3-sense-temp4-alpha2')
print(model_dir)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = Params(os.path.join(model_dir, 'params.json'))
params.device = device 
model = W2VNet(params).to(device)
checkpoint = torch.load(os.path.join(model_dir, 'model-4.tar'))
model.load_state_dict(checkpoint['model_state_dict'])

small_dataset = BertCtxDataset(data_path, vocab_path, params, load_all=False)
bert_loader = DataLoader(small_dataset, batch_size=50000, shuffle=False, 
                        num_workers=20, collate_fn=combine_samples)
data_len = len(small_dataset)
all_alpha = torch.zeros((params.vocab_size, params.num_sense)).to(device)
alpha_count = torch.zeros((params.vocab_size, params.num_sense)).to(device)
c = 0
total = 0

for b in bert_loader:
  with torch.no_grad():
    alpha = model.get_alpha(b[0].to(device), b[1].to(device)).squeeze(-1)

  #print(alpha.shape)
  #print(all_alpha[b[0], :].shape)
  all_alpha[b[0], :] += alpha
  alpha_count[b[0], :] += 1
  c+=1
  total += b[0].shape[0]
  if c % 2 == 0:
    print("Processed {:,d}/{:,d} words ({:.2%})".format(total, data_len, total/data_len), end='\r')
    #break

all_alpha = (all_alpha / alpha_count).cpu().numpy()

with open('stats/alpha.txt', 'w') as f:
  for i, alpha in enumerate(all_alpha):
    f.write("{}\t{}\n".format(id2word[i], "\t ".join(map(str, alpha))))

#words = ["apple", "rock", "plant", "star", "fox"]
# words = ["fox"]
# word_ids = [vocab[word] for word in words]
# alphas = defaultdict(list)
# c = 0
# 
# for b in bert_loader:
#   word_id = b[0].item()
#   #print(word_id, end='\r')
#   if word_id in word_ids:
#     with torch.no_grad():
#       #alpha = model(b[0], b[2], return_alpha=True)
#       alpha = model.get_alpha(b[0], b[1])
#     print(alpha)
#     word = id2word[word_id]
#     alphas[word].append(alpha)
#     c += 1
#     if c % 5 == 0:
#       print([(k,len(v)) for k,v in alphas.items()], end='\r')
#     if len(alphas[word]) == 100:
#       word_ids.remove(word_id)
#   if len(word_ids) == 0:
#     break
# 
# for word in alphas:
#   print("Average alpha out of 100 contexts for", word)
#   print(sum(alphas[word]) / len(alphas[word]))
