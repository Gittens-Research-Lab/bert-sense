import os, re
import argparse
import time
import sys

import torch
from nltk import word_tokenize
import numpy as np
from scipy.stats import spearmanr
np.random.seed(1)

from utils import Params
from dissg import Net

def load_vocab(path):
  word2id = {}
  id2word = []
  with open(path) as f:
    for i, line in enumerate(f):
      word = line.split()[0]
      word2id[word] = i
      id2word.append(word)
  
  return word2id, id2word

def cosine_sim(a,b):
  dot_product = np.dot(a, b)
  norm_a = np.linalg.norm(a)
  norm_b = np.linalg.norm(b)
  return dot_product / (norm_a * norm_b)

def np_softmax(x, axis=-1):
  exp_x = np.exp(x)
  return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class Embedder():
  def __init__(self, model, vocab, params, tokens_to_keep):
    self.word2id, self.id2word = vocab
    self.num_sense = params.num_sense
    self.word_dim = params.emb_dim
    
    self.sense_emb = np.reshape(model.emb.weight.detach().numpy(), [params.vocab_size, params.num_sense, params.emb_dim])
    self.sense_emb_n = self.sense_emb / np.linalg.norm(self.sense_emb, axis=-1, keepdims=True)
    self.global_emb = model.global_emb.weight.detach().numpy()
    self.global_emb_n = self.global_emb / np.linalg.norm(self.global_emb, axis=1, keepdims=True)
    self.sense_disamb = np.reshape(model.disamb.weight.detach().numpy(), [params.vocab_size, params.num_sense, params.emb_dim])/params.emb_dim
    
    tokens_found = 0
    for token in tokens_to_keep:
      if token in self.word2id:
        tokens_found += 1
    
    print('\nModelfile contains {} pretrained word embeddings with {} dimensions'.format(params.vocab_size, params.emb_dim))
    print('Found %d out of %d words in the dataset' %(tokens_found, len(tokens_to_keep)))

  def global_embedding(self, word):
    return self.global_emb[self.word2id.get(word,0)]
    
  def get_cluster_centers(self, word):
    if self.sense_disamb is not None:
      return self.sense_disamb[self.word2id.get(word,0)]
    else:
      return self.sense_emb[self.word2id.get(word,0)]
  
  def context_probability(self, word, context):
    contexts = []
    alpha2 = 1 
    if alpha2:
      ctx_sense, ctx_disamb = [], []
    for context_word in context:
      if context_word in self.word2id:
        contexts.append(self.global_emb[self.word2id[context_word]])
        if alpha2:
          ctx_sense.append(self.sense_emb[self.word2id[context_word]])
          ctx_disamb.append(self.sense_disamb[self.word2id[context_word]])
    
    context_mean = np.mean(np.stack(contexts, axis=1), axis=1, keepdims=True) # (d,1)
    if alpha2:
      ctx_sense = np.stack(ctx_sense, axis=0) # (c,k,d)
      ctx_disamb = np.stack(ctx_disamb, axis=0) # (c,k,d)
      ctx_alpha = np_softmax(np.matmul(ctx_disamb, np.tile(context_mean, (len(contexts),1,1))), axis=1) # (c,k,1)
      context_mean = np.mean(np.sum(ctx_sense * ctx_alpha, axis=1), axis=0).reshape((-1,1)) # (d,1)
      ctx_alpha = np_softmax(np.matmul(ctx_disamb, np.tile(context_mean, (len(contexts),1,1))), axis=1) # (c,k,1)
      context_mean = np.mean(np.sum(ctx_sense * ctx_alpha, axis=1), axis=0) # (d,)

    prob = np_softmax(np.dot(self.get_cluster_centers(word), context_mean), axis=0) # (k,)
     
    return prob

  def pairwise_sim(self, word1, word2, criterion='avg', sim='cosine'):
    ncl1 = ncl2 = self.num_sense

    scores = np.zeros((ncl1,ncl2))
    for i in range(ncl1):
      for j in range(ncl2):
        scores[i,j] = cosine_sim(self.sense_emb[self.word2id.get(word1,0)][i], 
                                 self.sense_emb[self.word2id.get(word2,0)][j])
        
    if criterion == 'avg':
      return np.mean(scores)
    elif criterion == 'max':
      return np.max(scores)
  
  def average_simC(self, word1, context1, word2, context2):
    ncl1 = ncl2 = self.num_sense
    
    p1 = self.context_probability(word1, context1).reshape([ncl1,1])
    p2 = self.context_probability(word2, context2).reshape([1,ncl2])
    probs = np.dot(p1, p2)
    scores = np.zeros((ncl1,ncl2))
    for i in range(ncl1):
      for j in range(ncl2):
        scores[i,j] = cosine_sim(self.sense_emb[self.word2id.get(word1,0)][i], 
                                 self.sense_emb[self.word2id.get(word2,0)][j])
    
    return np.sum(scores * probs)

  def best_sense(self, word, context):
    num_senses = self.num_sense
    if num_senses < 2:
      return self.sense_emb[self.word2id.get(word,0)][0]
    
    scores = self.context_probability(word, context)    
    cl_max = np.argmax(scores[:])
    return self.sense_emb[self.word2id.get(word,0)][cl_max]

def tokenize(text):
  return [t for t in word_tokenize(text) if sum([c.isalnum() for c in t])]

def get_context(tokens, position, window_size, freqs, total_count):
  """
  Args
    position: select tokens from the beginning or the end of the tokens list
  """
  if position == 'pre':
    if freqs is None:
      return tokens[:window_size]
    token_list = tokens
  elif position == 'post':
    if freqs is None:
      return tokens[-window_size:]
    token_list = tokens[::-1]
  
  context = []
  for word in token_list:
    if word in freqs:
      freq = freqs[word]
      # subsampling
      if np.random.uniform() <= np.sqrt( 0.001 * total_count / freq ):
        context.append(word)    
        if len(context) == window_size:
          break    

  if position == 'pre':
    return context
  elif position == 'post':
    return context[::-1]

def process_huang(filename='SCWS/ratings.txt',
                context_window=5, freqs=None):
  dirname = "/gpfs/u/home/BERT/BERTnksh/scratch/data/bert-sense"
  filepath = os.path.join(dirname, filename)
  f = open(filepath, 'r')
  result_list = []
  if freqs is None: total_freq = 0
  else: total_freq = sum(freqs.values())    
  for line in f:
    ob = re.search(r'(.*)<b>(.*)</b>(.*)<b>(.*)</b>(.*?)\t(.+)', line.lower())
    pre1 = tokenize(ob.group(1))
    word1 = ob.group(2).strip()
    middle = tokenize(ob.group(3))
    word2 = ob.group(4).strip()
    post2 = tokenize(ob.group(5))
    scores = ob.group(6).split()
    
    pre1 = get_context(pre1, 'post', context_window, freqs, total_freq)
    post1 = get_context(middle, 'pre', context_window, freqs, total_freq)
    pre2 = get_context(middle, 'post', context_window, freqs, total_freq)
    post2 = get_context(post2, 'pre', context_window, freqs, total_freq)
        
    scores = [float(score) for score in scores]
    ave_score = np.mean(np.array(scores))
        
    result = (word1, pre1+post1, word2, pre2+post2, ave_score)
    result_list.append(result)
  print("Read %d word, context pairs with %d window size" 
  %(len(result_list), context_window))
  return result_list

def calc_correlaton(args):
  # freqs = dict()
  # for line in open(os.path.join(args.model, 'vocab.txt')):
  #   word, freq = line.strip().split()
  #   freqs[word] = int(freq)
  freqs = None
  data = process_huang(context_window=5, freqs=freqs)
  words1, contexts1, words2, contexts2, targets = zip(*data)
  # for i in range(100,110):
  #   print("Word1:", words1[i], contexts1[i], "\tWord2:", words2[i], contexts2[i])
  # return
  words = set(words1+words2)
  if args.mode > 0:
    for context in contexts1+contexts2:
      for w in context:
        words.add(w)
  
  SCRATCH = "/gpfs/u/home/BERT/BERTnksh/scratch/"
  vocab = load_vocab(os.path.join(SCRATCH, 'data/bert-sense/vocab10_small_750k.txt'))
  #vocab = load_vocab(os.path.join(SCRATCH, 'data/bert-sense/vocab100_westwiki-para2'))
  #model_dir = os.path.join(SCRATCH, 'output/bert-sense/small-750k-noshuf-dissg-bert-ctx-d5-s3-d300-b2048-ns10-adam-lr1e3-sense-temp4-alpha3')
  model_dir = os.path.join(SCRATCH, 'output/bert-sense/westwiki-200m-vocab750k-d5-s3-d300-adam-lr1e3-ss0-b2048-ns10-sense-alpha2')
  epoch = 1 
  print("model:", model_dir, "epoch-%d"%epoch)

  params = Params(os.path.join(model_dir, 'params.json'))
  params.device = torch.device('cuda:0')
  model = Net(params)
  checkpoint = torch.load(os.path.join(model_dir, f'model-{epoch}.tar'))
  model.load_state_dict(checkpoint['model_state_dict'])  

  embed = Embedder(model, vocab, params, words)
  
  def print_scores(model_scores, targets):
    print("Spearman score: %0.2f" %(spearmanr(targets, model_scores).correlation*100))

  # c1, c2, sim = 'max', 'mean', 'energy'
  if args.mode == 0: # Average
    model_scores = []
    print("\nUsing global method")
    for i in range(len(words1)):
      w1v = embed.global_embedding(words1[i])
      w2v = embed.global_embedding(words2[i])
      model_scores.append(cosine_sim(w1v, w2v))

    print_scores(model_scores, targets)

  elif args.mode >= 1: # AveSim
    model_scores = []
    print("\nUsing avgSim method")
    for i in range(len(words1)):
      model_scores.append(embed.pairwise_sim(words1[i], words2[i], 'avg', 'cosine'))
    
    print_scores(model_scores, targets)
    
    # AveSimC
    model_scores = []
    print("\nUsing avgSimC method")
    for i in range(len(words1)):
      model_scores.append(embed.average_simC(words1[i], contexts1[i], words2[i], contexts2[i]))
    
    print_scores(model_scores, targets)
  
    # MaxSim
    model_scores = []
    print("\nUsing maxSim method")
    for i in range(len(words1)):
      model_scores.append(embed.pairwise_sim(words1[i], words2[i], 'max', 'cosine'))
    
    print_scores(model_scores, targets)
  
    # Best Local Cluster
    model_scores = []
    print("\nUsing localSim method")
    for i in range(len(words1)):
      w1v = embed.best_sense(words1[i], contexts1[i])
      w2v = embed.best_sense(words2[i], contexts2[i])
      model_scores.append(cosine_sim(w1v, w2v))
      # w1g = embed.global_embedding(words1[i])
      # w2g = embed.global_embedding(words2[i])
      # model_scores.append((cosine_sim(w1v, w2g) + cosine_sim(w1g,w2v))/2)
    
    print_scores(model_scores, targets)


def main(args):
  calc_correlaton(args)
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, help="Path to the embedding file")
  parser.add_argument('--epoch', type=int, default=-1, help="Epoch of model")  
  parser.add_argument('--mode', type=int, default=0, help="Whether \
  select best cluster or average")
  args = parser.parse_args()
  main(args)
