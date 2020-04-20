import os

import torch
import numpy as np

from dissg import Net
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

def np_softmax(x, axis=-1):
  exp_x = np.exp(x)
  return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class Embedder():
  def __init__(self, model, vocab, params):
    self.emb = model.emb.weight.detach().numpy()
    self.emb_n = self.emb / np.linalg.norm(self.emb, axis=1, keepdims=True)
    self.global_emb = model.global_emb.weight.detach().numpy()
    self.global_emb_n = self.global_emb / np.linalg.norm(self.global_emb, axis=1, keepdims=True)
    self.sense_emb = np.reshape(self.emb, [params.vocab_size, params.num_sense, params.emb_dim])
    self.sense_disamb = np.reshape(model.disamb.weight.detach().numpy(), [params.vocab_size, params.num_sense, params.emb_dim])
    
    self.word2id, self.id2word = vocab
    self.num_sense = params.num_sense
    self.word_dim = params.emb_dim
  
  def get_sense_emb(self, word):
    assert word in self.word2id, "Word is not in the vocabulary"
    word_id = self.word2id[word]
    return self.emb[word_id * self.num_sense: word_id * self.num_sense + self.num_sense]
  
  def get_ctx_emb(self, context_ids):
    if len(context_ids) == 0:
      #print("No words in context")
      return np.zeros((self.word_dim,), np.float32)
    
    contexts = []
    ctx_sense, ctx_disamb = [], []
    for context_word_id in context_ids:
      contexts.append(self.global_emb[context_word_id])
      ctx_sense.append(self.sense_emb[context_word_id])
      ctx_disamb.append(self.sense_disamb[context_word_id])
    
    context_mean = np.mean(np.stack(contexts, axis=1), axis=1, keepdims=True) # (d,1)
    ctx_sense = np.stack(ctx_sense, axis=0) # (c,k,d)
    ctx_disamb = np.stack(ctx_disamb, axis=0) # (c,k,d)
    ctx_alpha = np_softmax(np.matmul(ctx_disamb, np.tile(context_mean, (len(contexts),1,1))), axis=1) # (c,k,1)
    context_mean = np.mean(np.sum(ctx_sense * ctx_alpha, axis=1), axis=0) # (d,)
    
    return context_mean

  def nearby(self, word, from_emb="sense", to_emb="sense", context="", num_nns=10):
    word_id = self.word2id[word]
    if from_emb == "sense":
      word_emb = self.emb[word_id * self.num_sense: word_id * self.num_sense + self.num_sense]
    elif from_emb == "global":
      word_emb = self.global_emb[word_id:word_id+1]

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
  SCRATCH = "/gpfs/u/home/BERT/BERTnksh/scratch"
  vocab = load_vocab(os.path.join(SCRATCH, 'data/bert-sense/vocab5_small_250k.txt'))
  #vocab = load_vocab(os.path.join(SCRATCH, 'data/bert-sense/vocab100_westwiki-para2'))
  model_dir = os.path.join(SCRATCH, 'output/bert-sense/small-250k-noshuf-dissg-bert-ctx-d5-s3-d100-adam-lr1e3-sense-temp4-alpha2')
  #model_dir = os.path.join(SCRATCH, 'output/bert-sense/westwiki-100m-d5-s3-adam-lr1e2-ss3')

  params = Params(os.path.join(model_dir, 'params.json'))
  params.device = torch.device('cuda:0')
  model = Net(params)
  checkpoint = torch.load(os.path.join(model_dir, 'model-10.tar'))
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
      
