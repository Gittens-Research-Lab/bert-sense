from collections import Counter
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer

class BERTDataset(Dataset):
  def __init__(self, file_path, bert_name, delta=5):
    """
    Load the whole data file into memory
    for easier access during batching
    """
    self.delta = delta
    self.window_size = 2 * delta + 1
    self.create_vocab(file_path)
    # self.save_vocab("vocab")
    
    self.bert_tokenizer = BertTokenizer.from_pretrained(bert_name)
    self.bert_model = BertModel.from_pretrained(bert_name)
    self.bert_model.eval()
    

  def create_vocab(self, file_path):
    counts = Counter()
    self.data = []
    with open(file_path) as f:
      for line in f:
        words = line.split()
        if len(words) < self.window_size:
          continue
        self.data.append(words)
        counts.update(words)
    
    self.word2id = {"UNK":0}
    self.id2word = ["UNK"]
    self.word2freq = {"UNK":0}
    for i, (word, freq) in enumerate(counts.most_common()):
      if freq >= 5:
        self.word2id[word] = i+1
        self.id2word.append(word)
        self.word2freq[word] = freq
    
    print("Vocab size:", len(self.id2word))
  
  def save_vocab(self, file_path):
    with open(file_path, 'w') as f:
      for w in self.id2word:
        f.write(f"{w} {self.word2freq[w]}\n")
  
  def process(self, line):
    """
    Returns words and contexts from line

    Args
      line: list of words
    """
    # bert_model = BertModel.from_pretrained(self.bert_name)
    # bert_tokenizer = BertTokenizer.from_pretrained(self.bert_name)
    # bert_model.eval()
    word_to_token_map = []
    tokens = []
    for w in line:
      word_to_token_map.append(len(tokens))
      tokens += self.bert_tokenizer.tokenize(w)
    
    word_to_token_map.append(len(tokens))
    word_to_token_map = np.array(word_to_token_map)
    
    words = []
    context_tokens = []
    context_token_maps = []
    for i in range(len(line) - self.window_size + 1):
      words.append(self.word2id.get(line[i+self.delta], 0))
      context_token_maps.append(word_to_token_map[i:i+self.window_size] - word_to_token_map[i] + 1)
      context_tokens.append(tokens[word_to_token_map[i]:word_to_token_map[i+self.window_size]])
    
    max_len = max([len(c) for c in context_tokens])
    context_ids = []
    for ctokens in context_tokens:
      context_ids.append([self.bert_tokenizer.cls_token_id] \
                        + self.bert_tokenizer.convert_tokens_to_ids(ctokens) \
                        + [self.bert_tokenizer.sep_token_id] \
                        + [self.bert_tokenizer.pad_token_id] * (max_len - len(ctokens)))
    
    context_token_maps = context_token_maps

    with torch.no_grad():
      output_states, _ = self.bert_model(torch.tensor(context_ids))
    context_states = []
    for i, output_state in enumerate(output_states):
      context_states.append(output_state[context_token_maps[i]])
    context_states = torch.stack(context_states,dim=0)
    # print(output_states.size(), context_states.size())
    
    bert_embeds = context_states[:,self.delta,:]
    bert_context_embeds = torch.cat([context_states[:,:self.delta,:], context_states[:,self.delta+1:,:]], dim=1).view((len(context_tokens),-1))
    
    # print(context_tokens)
    # print(max_len)
    # print(bert_embeds[0].size())
    # print(bert_context_embeds[0].size())

    return words, bert_embeds, bert_context_embeds#, torch.stack(bert_embeds, dim=0), torch.stack(bert_context_embeds, dim=0)
  
  def __len__(self):
    return len(self.data)
  

  def __getitem__(self, idx):
    """
    Access a specific sample from the dataset
    """
    words, berts, contexts = self.process(self.data[idx])
    return {'word':words, 'bert':berts, 'context':contexts}


def combine_samples(batch):
  """
  Combines a list of samples returned from 
  '__getitem__' of BERTDataset

  Args
    batch: list of samples in a batch

  Returns:
    a batch of contexts and embeddings for use in training
  """
  words = torch.tensor([w for sample in batch for w in sample['word']])
  berts = torch.cat([sample['bert'] for sample in batch], dim=0)
  contexts = torch.cat([sample['context'] for sample in batch], dim=0)
  return words, berts, contexts

if __name__ == "__main__":

  small_dataset = BERTDataset("/home/anik/Documents/f18/embeddings/data/west-wiki/small.txt", "bert-base-uncased")

  bert_loader = DataLoader(small_dataset, batch_size=2, shuffle=False, 
                                        num_workers=0, collate_fn=combine_samples)

  start = time.time()
  c = 0
  for i, b in enumerate(bert_loader):
    # print(b[0].size(),'\n', b[1].size(), '\n', b[2].size())
    c+= len(b[0])
    if i % 2 == 0:
      t = time.time() - start
      print(f"{c:,} words in {t:0.2f} seconds at {c/t:,.2f} words/sec", end='\r')
    # break
  t = time.time() - start
  print(f"{c:,} words in {t:0.2f} seconds at {c/t:,.2f} words/sec")