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
from train import Net as BertNet

class W2VNet(nn.Module):
  def __init__(self, params):
    super(W2VNet, self).__init__()
    self.num_sense = params.num_sense
    self.emb_dim = params.emb_dim
    self.num_samples = params.num_samples
    self.tiny_float = torch.tensor(torch.finfo(torch.float).tiny).to(params.device)
    self.device = params.device

    self.emb = nn.Embedding(params.vocab_size * params.num_sense, params.emb_dim)
    self.emb.weight.data.uniform_(-1/params.emb_dim, 1/params.emb_dim)
    self.global_emb = nn.Embedding(params.vocab_size, params.emb_dim)
    self.global_emb.weight.data.uniform_(-1/params.emb_dim, 1/params.emb_dim)    
    self.disamb = nn.Embedding(params.vocab_size * params.num_sense, params.emb_dim)
    self.disamb.weight.data.uniform_(-1/params.emb_dim, 1/params.emb_dim)

  def forward(self, word_ids, context_ids, neg_ids, return_alpha=False, temp=1):
    sense_ids = []
    for i in range(self.num_sense):
      sense_ids.append(word_ids * self.num_sense + i)
    sense_ids = torch.stack(sense_ids, dim=1)
    
    # Shape written to the right
    # b = batch_size, k = num_sense, d = emb_dim, c = num_context, n = num_neg
    sense_embs = self.emb(sense_ids) # (b,k,d)
    disamb_embs = self.disamb(sense_ids) # (b,k,d)
    context_embs = self.global_emb(context_ids) # (b,c,d)
    sample_embs = self.global_emb(neg_ids) # (b,c*n,d)
    # word_embs = self.global_emb(word_ids).unsqueeze(-1) # (b,d,1)
    # context_alpha = torch.matmul(context_embs, word_embs).softmax(dim=1) # (b,c,1)
    # context_emb = (context_embs * context_alpha).sum(dim=1).unsqueeze(-1) # (b,d,1)
    context_emb = context_embs.mean(dim=1).unsqueeze(-1) # (b,d,1)

    b, c = context_ids.size()[0], context_ids.size()[1]
    all_sense_ids = []
    for i in range(self.num_sense):
      all_sense_ids.append(context_ids * self.num_sense + i)
    all_sense_ids = torch.stack(all_sense_ids, dim=-1) # (b,c,k)
    
    all_sense_embs = self.emb(all_sense_ids) # (b,c,k,d)
    all_disamb_embs = self.disamb(all_sense_ids) # (b,c,k,d)
    all_alpha = torch.matmul(all_disamb_embs.reshape(b,-1,self.emb_dim), context_emb).reshape(b,c,self.num_sense,1).softmax(dim=2) # (b,c,k,1)
    context_emb2 = (all_sense_embs * all_alpha).sum(dim=2).mean(dim=1).unsqueeze(-1) # (b,d,1)
    all_alpha2 = torch.matmul(all_disamb_embs.reshape(b,-1,self.emb_dim), context_emb2).reshape(b,c,self.num_sense,1).softmax(dim=2) # (b,c,k,1)
    context_emb3 = (all_sense_embs * all_alpha2).sum(dim=2).mean(dim=1).unsqueeze(-1) # (b,d,1)

    alpha = torch.matmul(disamb_embs, context_emb3) # (b,k,1)
    #alpha = torch.matmul(disamb_embs, context_emb) # (b,k,1)
    alpha_soft = alpha.softmax(dim=1) # (b,k,1)
    #pos_prob = torch.matmul((sense_embs * alpha_soft).sum(dim=1, keepdim=True), context_embs.transpose(1,2)).sigmoid() # (b,1,c)
    #neg_prob = torch.matmul((sense_embs * alpha_soft).sum(dim=1, keepdim=True), sample_embs.transpose(1,2)).sigmoid() # (b,1,c*n)
    #pos_loss = - torch.max(pos_prob, self.tiny_float).log().sum()
    #neg_loss = - torch.max(1 - neg_prob, self.tiny_float).log().sum()
    pos_prob = torch.matmul(sense_embs, context_embs.transpose(1,2)).sigmoid() # (b,k,c)
    neg_prob = torch.matmul(sense_embs, sample_embs.transpose(1,2)).sigmoid() # (b,k,c*n)
    pos_loss = - torch.max((pos_prob * alpha_soft).sum(dim=1), self.tiny_float).log().sum()
    neg_loss = - torch.max(1 - (neg_prob * alpha_soft).sum(dim=1), self.tiny_float).log().sum()
    loss = (pos_loss + neg_loss) / context_ids.numel()

    # batch_size = context_ids.size()[0]
    # rand2id = torch.gather(context_ids, 1, torch.randint(context_ids.size()[1], size=(batch_size, 2)).to(self.device)) # (b,2)

    # sample_sense_ids = []
    # for i in range(self.num_sense):
    #   sample_sense_ids.append(rand2id * self.num_sense + 1)
    # sample_sense_ids = torch.stack(sample_sense_ids, dim=-1) # (b,2,k)
    # sample_sense_embs = self.emb(sample_sense_ids) # (b,2,k,d)
    # sample_disamb_embs = self.disamb(sample_sense_ids) # (b,2,k,d)
    # best_ind = torch.argmax(torch.matmul(sample_disamb_embs.reshape(batch_size, -1, self.emb_dim), context_emb).reshape(batch_size, 2, self.num_sense), dim=-1) # (b,2)
    # best_sense_embs = torch.gather(sample_sense_embs, 2, best_ind.reshape(batch_size, 2, 1, 1).repeat(1, 1, 1, self.emb_dim)) # (b,2,d)
    # best_disamb_embs = torch.gather(sample_disamb_embs, 2, best_ind.reshape(batch_size, 2, 1, 1).repeat(1, 1, 1, self.emb_dim)) # (b,2,d)
    # 
    # sense_loss = - torch.max((best_sense_embs[:,1,:] * best_disamb_embs[:,0,:]).sum(dim=-1).sigmoid(), self.tiny_float).log().sum() \
    #              - torch.max((best_sense_embs[:,0,:] * best_disamb_embs[:,1,:]).sum(dim=-1).sigmoid(), self.tiny_float).log().sum()

    if return_alpha:
      return loss, (alpha/temp).softmax(dim=1).squeeze(dim=-1), 1 #sense_loss / (batch_size * 2)# context_emb.squeeze(dim=-1)
    else:
      return loss
  
  def get_alpha(self, word_ids, context_ids):
    sense_ids = []
    for i in range(self.num_sense):
      sense_ids.append(word_ids * self.num_sense + i)
    sense_ids = torch.stack(sense_ids, dim=1)
    
    # Shape written to the right
    # b = batch_size, k = num_sense, d = emb_dim, c = num_context, n = num_neg
    sense_embs = self.emb(sense_ids) # (b,k,d)
    disamb_embs = self.disamb(sense_ids) # (b,k,d)
    context_embs = self.global_emb(context_ids) # (b,c,d)
    context_emb = context_embs.mean(dim=1).unsqueeze(-1) # (b,d,1)
    alpha = torch.matmul(disamb_embs, context_emb) # (b,k,1)
    alpha_soft = alpha.softmax(dim=1) # (b,k,1)
    return alpha_soft

  def get_parameters(self):
    return list(self.emb.parameters()) + list(self.global_emb.parameters()) + list(self.disamb.parameters())


class BertCtxDataset(Dataset):
  def __init__(self, data_path, vocab_path, params, load_all=True):
    with h5py.File(data_path, 'r', libver='latest', swmr=True) as data:
      self.data_len = len(data['word'])
    
    self.word = None
    self.data_path = data_path
    self.load_all = load_all
    
    self.min_freq = params.min_freq
    freqs = self.load_vocab(vocab_path)
    self.noise_dist = torch.tensor(freqs, dtype=torch.float)**0.75
    self.noise_dist /= self.noise_dist.sum()
    
    
  def load_vocab(self, vocab_path):
    assert os.path.exists(vocab_path), f"{vocab_path} does not exist"
    # Load vocab
    freqs = []
    with open(vocab_path) as f:
      for i, line in enumerate(f):
        word, freq = line.rstrip().split()
        freqs.append(int(freq))
    logging.info("Loaded freqs from {} with {} words".format(vocab_path, len(freqs)))
     
    return freqs

  def __len__(self):
    return self.data_len

  def __getitem__(self, idx):
    if self.word is None:
      data = h5py.File(self.data_path, 'r', libver='latest', swmr=True)
      self.word = data['word'][:] if self.load_all else data['word']
      self.ctx = data['ctx'][:] if self.load_all else data['ctx']
      #self.bert = self.data['bert'][:] if self.load_all else data['bert']
      self.bertctx = data['bert_ctx'][:] if self.load_all else data['bert_ctx']
    sample = {}
    sample['word'] = self.word[idx]
    sample['ctx'] = self.ctx[idx]
    #sample['bert'] = self.bert[idx]
    sample['bertctx'] = self.bertctx[idx]
    return sample

def combine_samples(batch):
  """
  Combine a list of samples
  """
  words = torch.LongTensor([sample['word'] for sample in batch])#, dim=0)
  ctxs = torch.LongTensor([sample['ctx'] for sample in batch])#, dim=0)
  #berts = torch.tensor([sample['bert'] for sample in batch])#, dim=0)
  bertctxs = torch.tensor([sample['bertctx'] for sample in batch])#, dim=0)

  return words, ctxs, bertctxs

if __name__ == "__main__":
  SCRATCH = "/gpfs/u/home/BERT/BERTnksh/scratch"
  data_path = os.path.join(SCRATCH, "data/bert-sense/small_750k_bert_ctx_d5.h5")
  vocab_path = os.path.join(SCRATCH, "data/bert-sense/vocab10_small_750k.txt")
  exp_dir = os.path.join(SCRATCH, "output/bert-sense", "small-750k-noshuf-dissg-bert-ctx-d5-s3-d300-b2048-ns10-adam-lr1e3-sense-temp4-alpha3")
  if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
  
  # Set the logger
  set_logger(os.path.join(exp_dir, "train.log")) 
  
  logging.info(f"Data: {data_path}")
  logging.info(f"Exp dir: {exp_dir}")
  #restore_file = os.path.join(exp_dir, 'model-3' + '.tar')
  #restore_file = os.path.join(SCRATCH, "output/bert-sense/", "westwiki-100m-vocab250k-d5-s3-adam-lr1e3-ss3-varwindow-sense", "model-3.tar")
  restore_file = None

  bert_path = os.path.join(SCRATCH, "output/bert-sense", "small-750k-noshuf-bert-d5-s3-adam-lr1e3", "model-10.tar")
  logging.info(f"Bert weights: {bert_path}")
 
  params = Params('paramsw2vbert.json')
  with h5py.File(data_path, 'r', libver='latest', swmr=True) as data_file:
    d = {}
    for k,v in data_file.attrs.items():
      if isinstance(v, np.int64):
        d[k] = int(v)
      else:
        d[k] = v		
    params.update_dict(d)

  
  small_dataset = BertCtxDataset(data_path, vocab_path, params, load_all=False)
  bert_loader = DataLoader(small_dataset, batch_size=params.batch_size, shuffle=False, 
                          num_workers=params.num_workers, collate_fn=combine_samples)
  
  logging.info(params)
  params.save(os.path.join(exp_dir, 'params.json'))
  # sys.exit(0)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logging.info(device)
  params.device = device
  w2vmodel = W2VNet(params).to(device)
  optimizer = torch.optim.Adam(w2vmodel.get_parameters(), lr=params.learning_rate)

  params.emb_dim = params.bert_dim
  bertmodel = BertNet(params).to(device)
  bertckpt = torch.load(bert_path)
  bertmodel.load_state_dict(bertckpt['model_state_dict'])
  logging.info("Loaded Bert-sense model parameters from {}".format(bert_path))

  start = time.time()
  start_epoch = 0
  c = 0
  loss_vals = []
  temp_vals = []
  tiny = torch.tensor(torch.finfo(torch.float).tiny).to(device)
  
  if restore_file is not None:
    logging.info("Restoring model parameters from {}".format(restore_file))
    checkpoint = torch.load(restore_file)
    w2vmodel.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    start_epoch = checkpoint['epoch']
    with open(os.path.join(exp_dir, "loss.pkl"), "rb") as f:
      loss_vals = pickle.load(f)

  for e in range(start_epoch, params.num_epochs):
    for step, b in enumerate(bert_loader):
      # print(b[0].size(),'\n', b[1].size(), '\n', b[2].size())
      word_id, ctx, bert_ctx = b
      c+= len(word_id)

      neg_ids = torch.multinomial(small_dataset.noise_dist, ctx.numel() * params.num_samples, replacement=True).view(ctx.size()[0], -1)

      bertalpha = bertmodel(word_id.to(device), bert_ctx.to(device), True, params.temp)
      w2vloss, alpha, sense_loss = w2vmodel(word_id.to(device), ctx.to(device), neg_ids.to(device), True, params.temp)
      transferloss = - params.temp ** 2 * torch.sum(bertalpha * torch.log(alpha), dim=1).mean()
      #ctx_loss = - torch.max((word_ctx * bert_ctx.to(device)).sum(1).sigmoid(), tiny).log().mean()
      #if e < 8:
      loss = w2vloss + params.lambda_ * transferloss #+ params.gamma * sense_loss
      #else:
      #  loss = w2vloss
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
                f" TransferLoss: {transferloss:<,.6f}"
                f" words/sec: {c/t:<6,.2f}")
      #if (step+1) == 200: 
      #  break
    
    save_path = os.path.join(exp_dir,"model-{}.tar".format(e+1))
    torch.save({'model_state_dict': w2vmodel.state_dict(),
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

 
