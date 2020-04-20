import os

import torch
import numpy as np

from train import Net
from utils import Params

def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)

def load_vocab(path):
  word2id = {}
  id2word = []
  with open(path) as f:
    for i, line in enumerate(f):
      word = line.split()[0]
      word2id[word] = i
      id2word.append(word)
  
  return word2id, id2word


class Embedder():
  def __init__(self, model, vocab, params):
    if params.use_M:
      self.M = model.M.detach().numpy()
    else:
      self.M = None
    self.emb = model.emb.weight.detach().numpy()
    self.emb_n = self.emb / np.linalg.norm(self.emb, axis=1, keepdims=True)
    self.global_emb = np.mean(np.reshape(self.emb, [params.vocab_size, params.num_sense, model.emb_dim]),axis=1)
    self.global_emb_n = self.global_emb / np.linalg.norm(self.global_emb, axis=1, keepdims=True)
    self.word2id, self.id2word = vocab
    self.num_sense = params.num_sense
    self.word_dim = model.emb_dim
    if params.disamb:
      self.sense_disamb = np.reshape(model.disamb.weight.detach().numpy(), [params.vocab_size, params.num_sense, model.emb_dim])
    else:
      self.sense_disamb = None
  
  def get_sense_emb(self, word):
    assert word in self.word2id, "Word is not in the vocabulary"
    word_id = self.word2id[word]
    return self.emb[word_id * self.num_sense: word_id * self.num_sense + self.num_sense]
  
  def nearby(self, word, from_emb="sense", to_emb="sense", context="", num_nns=10):
    word_id = self.word2id[word]
    if from_emb == "sense":
      word_emb = self.emb_n[word_id * self.num_sense: word_id * self.num_sense + self.num_sense]
    elif from_emb == "global":
      word_emb = self.global_emb_n[word_id:word_id+1]

    if to_emb == "sense":
      dist = np.dot(self.emb_n, word_emb.T)
    elif to_emb == "global":
      dist = np.dot(self.global_emb_n, word_emb.T) 
    # print("Top 10 highest similarity for %s" %word)
    
    # if len(context) > 0:
    #   sense_list =  [self.disambiguate(word,context)]
    # else:
    sense_list = range(dist.shape[1])

    for i in sense_list:
      if from_emb == "sense":
        print("sense %d:" %i, end=" ")
      
      highsim_idxs = dist[:,i].argsort()[::-1]
      # select top num_nns (linear) indices with the highest cosine similarity
      highsim_idxs = highsim_idxs[1:num_nns+1]
      if to_emb == "sense":
        words = ["{}:{}".format(self.id2word[int(idx/self.num_sense)], idx%self.num_sense) for idx in highsim_idxs]
      elif to_emb == "global":
        words = [self.id2word[j] for j in highsim_idxs]
      
      print(" ".join(words))

if __name__=="__main__":
  SCRATCH = "/gpfs/u/home/BERT/BERTnksh/scratch/"
  vocab = load_vocab(os.path.join(SCRATCH, 'data/bert-sense/vocab5_small_250k.txt'))
  model_dir = os.path.join(SCRATCH, 'output/bert-sense/small-250k-noshuf-bert-d5-s3-adam-lr1e3')
  
  params = Params(os.path.join(model_dir, 'params.json'))
  model = Net(params)
  checkpoint = torch.load(os.path.join(model_dir, 'model-3.tar'))#, map_location=torch.device('cpu'))
  model.load_state_dict(checkpoint['model_state_dict'])
  embed = Embedder(model, vocab, params)
  #print(model.M)
  # _start_shell(locals())
  
  while True:
    try:
      word = input("\nEnter word:\t")
      # if ":" in text:
      #   word, context = text.split(':')
      # else:
      #   word, context = text, ""
      embed.nearby(word, "sense", "sense")
    except KeyboardInterrupt:
      print()
      break
    except AssertionError as e:
      print(e)
      
