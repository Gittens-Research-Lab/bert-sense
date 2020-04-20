import time, datetime
import sys
import os
import pickle
import logging

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import Params, set_logger

class Net(nn.Module):

  def __init__(self, params):
    super(Net, self).__init__()
    self.num_sense = params.num_sense
    self.emb_dim = params.emb_dim
    if params.use_M:
      self.M = nn.Parameter(torch.randn(params.emb_dim, params.bert_dim))
    else:
      self.M = None
    self.emb = nn.Embedding(params.vocab_size * params.num_sense, self.emb_dim)
    self.emb.weight.data.uniform_(-1/self.emb_dim, 1/self.emb_dim)
    self.disamb = nn.Embedding(params.vocab_size * params.num_sense, self.emb_dim)
    self.disamb.weight.data.uniform_(-1/self.emb_dim, 1/self.emb_dim)
    
    # if params.global_emb:
    #   self.global_emb = nn.Embedding(params.vocab_size, self.emb_dim)
    #   self.global_emb.weight.data.uniform_(-1/self.emb_dim, 1/self.emb_dim)

  def forward(self, word_ids, ctx, return_alpha=False, temp=1):
    sense_ids = []
    for i in range(self.num_sense):
      sense_ids.append(word_ids * self.num_sense + i)
    sense_ids = torch.stack(sense_ids, dim=1)

    sense_embs = self.emb(sense_ids)
    disamb_embs = self.disamb(sense_ids)

    if self.M is not None:
      sense_embs = torch.matmul(sense_embs, self.M)
      disamb_embs = torch.matmul(disamb_embs, self.M)
    alpha = torch.matmul(disamb_embs, ctx.unsqueeze(-1)) #/ torch.sqrt(self.emb_dim)
    
    if return_alpha:
      return (alpha/temp).softmax(dim=1).squeeze(dim=-1)
    else:
      alpha = F.softmax(alpha, dim=1)
      sum_emb = torch.sum(torch.mul(alpha, sense_embs), dim=1)
      return sum_emb

  def get_parameters(self):
    trainable_params = list(self.emb.parameters()) + list(self.disamb.parameters())
    if self.M is not None:
      trainable_params += [self.M]

    return trainable_params  
  
  def get_embedding(self):
    return self.emb.weight

def logloss(a_emb, b_emb):
  dot_prod = torch.sum(torch.mul(a_emb, b_emb), dim=1)
  return - torch.mean(F.logsigmoid(dot_prod))


class BertDataset(Dataset):
  def __init__(self, data_path, load_all=True):
    with h5py.File(data_path, 'r', libver='latest', swmr=True) as data:
      self.data_len = len(data['word'])
    
    self.data = None
    self.data_path = data_path
    self.load_all = load_all
    
  def __len__(self):
    return self.data_len

  def __getitem__(self, idx):
    if self.data is None:
      self.data = h5py.File(self.data_path, 'r', libver='latest', swmr=True)
      self.word = self.data['word'][:] if self.load_all else self.data['word']
      self.bert = self.data['bert'][:] if self.load_all else self.data['bert']
      self.ctx = self.data['bert_ctx'][:] if self.load_all else self.data['bert_ctx']
    sample = {}
    sample['word'] = self.word[idx]
    sample['bert'] = self.bert[idx]
    sample['ctx'] = self.ctx[idx]
    return sample

def combine_samples(batch):
  """
  Combine a list of samples
  """
  words = torch.LongTensor([sample['word'] for sample in batch])#, dim=0)
  berts = torch.tensor([sample['bert'] for sample in batch])#, dim=0)
  ctxs = torch.tensor([sample['ctx'] for sample in batch])#, dim=0)

  return words, berts, ctxs

if __name__ == "__main__":
  SCRATCH = "/gpfs/u/home/BERT/BERTnksh/scratch"
  data_path = os.path.join(SCRATCH, "data/bert-sense/small_750k_bert_ctx_d5.h5")
  exp_dir = os.path.join(SCRATCH, "output/bert-sense", "small-750k-noshuf-bert-d5-s3-adam-lr1e3")
  if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

  restore_file = os.path.join(exp_dir, 'model-10.tar')
  #restore_file = None
 
  params = Params('params.json')
  with h5py.File(data_path, 'r', libver='latest', swmr=True) as data_file:
    d = {}
    for k,v in data_file.attrs.items():
      if isinstance(v, np.int64):
        d[k] = int(v)
      else:
        d[k] = v		
    params.update_dict(d)

  # Set the logger
  set_logger(os.path.join(exp_dir, "train.log")) 
  
  small_dataset = BertDataset(data_path, load_all=False)
  logging.info("Dataset size: {}".format(len(small_dataset)))
  bert_loader = DataLoader(small_dataset, batch_size=params.batch_size, shuffle=False, 
                          num_workers=params.num_workers, collate_fn=combine_samples)
  
  logging.info(params)
  params.save(os.path.join(exp_dir, 'params.json'))
  # sys.exit(0)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logging.info(device)
  model = Net(params).to(device)
  optimizer = torch.optim.Adam(model.get_parameters(), lr=params.learning_rate)

  loss_fn = torch.nn.MSELoss()
   
  start = time.time()
  start_epoch = 0
  c = 0
  loss_vals = []
  temp_vals = []
  if restore_file is not None:
    logging.info("Restoring model parameters from {}".format(restore_file))
    checkpoint = torch.load(restore_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    start_epoch = checkpoint['epoch']
    #with open(os.path.join(exp_dir, "loss.pkl"), "rb") as f:
    #  loss_vals = pickle.load(f)


  for e in range(start_epoch, params.num_epochs):
    for step, b in enumerate(bert_loader):
      # print(b[0].size(),'\n', b[1].size(), '\n', b[2].size())
      word_id, bert_emb, bert_ctx = b
      if params.normalize:
        bert_emb = F.normalize(bert_emb, dim=1)
      c+= len(word_id)

      sense_emb = model(word_id.to(device), bert_ctx.to(device))
      loss = loss_fn(bert_emb.to(device), sense_emb) 
      temp_vals.append(loss.item())
      
      # Zero gradients, perform a backward pass, and update the weights.
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (step+1) % 50 == 0:
        loss_vals.append(np.mean(temp_vals))
        temp_vals.clear()
        t = time.time() - start
        logging.info(f"{datetime.timedelta(seconds=int(t))}"
                f" E{e+1}:s{step+1:<5,d}"
                f" Loss: {loss:<,.6f}"
                f" words/sec: {c/t:<6,.2f}")
      #if (step+1) == 200: 
      #  break
    
    save_path = os.path.join(exp_dir,"model-{}.tar".format(e+1))
    torch.save({'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'epoch': e+1},
                save_path)
 
  t = time.time() - start
  logging.info(f"\n{c:<6,d} words in {datetime.timedelta(seconds=int(t))} at {c/t:<,.2f} words/sec")

  plt.plot(loss_vals, 'o', markersize=2)
  plt.title(f"Loss ({params.optimizer} optimizer, lr = {params.learning_rate})")
  plt.savefig(os.path.join(exp_dir, "plot.png"), dpi=200)
  #plt.show()

  with open(os.path.join(exp_dir, "loss.pkl"), "wb") as f:
    pickle.dump(loss_vals, f)

  
