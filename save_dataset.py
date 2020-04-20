import os
import time
import datetime
from collections import Counter

import h5py
import numpy as np
import torch

from transformers import BertModel, BertTokenizer

def load_vocab(path):
  word2id = {}
  with open(path) as f:
    for i, line in enumerate(f):
      word2id[line.split()[0]] = i
    
  return word2id

def create_vocab(datapath, name, delta=5, min_freq=5):
  counts = Counter()
  with open(datapath) as f:
    for line in f:
      words = line.split()
      if len(words) < 2*delta+1:
        continue
      counts.update(words)
  
  word2id = {"UNK":0}
  word2freq = {"UNK":0}
  c = 0
  for i, (word, freq) in enumerate(counts.most_common()):
    c += freq
    if freq >= min_freq:
      word2id[word] = i+1
      word2freq[word] = freq
  
  print("Total words:", c)
  print("Vocab size:", len(word2id))
  with open('vocab_' + name + '.txt', 'w') as f:
    for w,freq in word2freq.items():
      f.write(f"{w}\t{freq}\n")

  return word2id

def process_data(file_path, bert_vocab, bert_weight, vocab, delta=5):
  window_size = 2 * delta + 1
  bert_tokenizer = BertTokenizer.from_pretrained(bert_vocab)
  bert_model = BertModel.from_pretrained(bert_weight)
  bert_model.eval()
  
  with open(file_path) as f:
    for line in f:
      words = line.split()
      word_to_token_map = []
      tokens = []
      for w in words:
        word_to_token_map.append(len(tokens))
        tokens += bert_tokenizer.tokenize(w)
      
      word_to_token_map.append(len(tokens))
      word_to_token_map = np.array(word_to_token_map)
      
      word_ids = []
      context_tokens = []
      context_token_maps = []
      for i in range(len(words) - window_size + 1):
        word_ids.append(vocab.get(words[i+delta], 0))
        context_token_maps.append(word_to_token_map[i:i+window_size+1] - word_to_token_map[i] + 1)
        context_tokens.append(tokens[word_to_token_map[i]:word_to_token_map[i+window_size]])
      
      max_len = max([len(c) for c in context_tokens])
      context_ids = []
      for ctokens in context_tokens:
        context_ids.append([bert_tokenizer.cls_token_id] \
                          + bert_tokenizer.convert_tokens_to_ids(ctokens) \
                          + [bert_tokenizer.sep_token_id] \
                          + [bert_tokenizer.pad_token_id] * (max_len - len(ctokens)))
      
      with torch.no_grad():
        output_states, _ = bert_model(torch.tensor(context_ids))
      
      context_states = np.zeros((len(word_ids), window_size, output_states.shape[-1]), dtype=np.float32)
      for i, output_state in enumerate(output_states):
        for j in range(window_size):
          context_states[i,j] = torch.mean(output_state[context_token_maps[i][j]:context_token_maps[i][j+1]], dim=0)
      # print(output_states.shape, context_states.shape)
      
      bert_embeds = context_states[:,delta,:]
      bert_context_embeds = np.mean(np.concatenate([context_states[:,:delta,:], context_states[:,delta+1:,:]], axis=1), axis=1)
      
      # print(context_tokens)
      # print(context_token_maps)

      yield np.array(word_ids, dtype=np.int32), bert_embeds, bert_context_embeds


if __name__ == "__main__":
  SCRATCH = "/gpfs/u/home/BERT/BERTnksh/scratch/"
  data_name = "small_250k"
  data_path = os.path.join(SCRATCH, "data/bert-sense/{}.txt".format(data_name))
  bert_path = os.path.join(SCRATCH, "data/bert-sense/{}_bert_d5.h5".format(data_name)) 
  bert_vocab = "bert-base-uncased-vocab.txt"
  bert_weight = "bert_base_weights"

  # vocab = load_vocab('vocab')
  vocab = create_vocab(data_path, data_name)
  delta = 5
  data_gen = process_data(data_path, bert_vocab, bert_weight, vocab, delta)
  
  with h5py.File(bert_path, 'w') as f:
    start = time.time()
    batch = next(data_gen)
    f.attrs['delta'] = delta
    f.attrs['bert_dim'] = batch[1].shape[-1]
    f.attrs['vocab_size'] = len(vocab)
    word_dset = f.create_dataset('word', shape=batch[0].shape, 
                              maxshape=(None,) + batch[0].shape[1:],
                              dtype=batch[0].dtype)
    bert_dset = f.create_dataset('bert', shape=batch[1].shape, 
                              maxshape=(None,) + batch[1].shape[1:],
                              dtype=batch[1].dtype)
    ctx_dset = f.create_dataset('ctx', shape=batch[2].shape, 
                              maxshape=(None,) + batch[2].shape[1:],
                              dtype=batch[2].dtype)

    word_dset[:] = batch[0]
    bert_dset[:] = batch[1]
    ctx_dset[:] = batch[2]
    row_count = batch[0].shape[0]

    for batch in data_gen:
      # batch = next(data_gen)
      word_dset.resize(row_count + batch[0].shape[0], axis=0)
      word_dset[row_count:] = batch[0]
      bert_dset.resize(row_count + batch[1].shape[0], axis=0)
      bert_dset[row_count:] = batch[1]
      ctx_dset.resize(row_count + batch[2].shape[0], axis=0)
      ctx_dset[row_count:] = batch[2]

      row_count += batch[0].shape[0]

      t = time.time() - start
      print(f"{row_count:,} words in {datetime.timedelta(seconds=int(t))} at {row_count/t:,.2f} words/sec", end='\r')
    
    t = time.time() - start
    print(f"{row_count:,} words in {datetime.timedelta(seconds=int(t))} at {row_count/t:,.2f} words/sec")
