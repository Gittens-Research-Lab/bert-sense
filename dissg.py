import os
import sys
import time, datetime
from collections import Counter
import logging
import pickle 

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import Params, set_logger

class W2VDataset(Dataset):
  def __init__(self, data_path, vocab_path, params):
    self.delta = params.delta
    self.window_size = 2 * self.delta + 1      
    self.min_freq = params.min_freq
    self.subsample = params.subsample
    self.varwindow = params.varwindow
    
    self.corpus = []
    corpus_size = 0
    word2id = self.create_or_load_vocab(vocab_path)
    self.vocab_size = len(word2id)
    self.noise_dist = torch.tensor(self.freqs, dtype=torch.float)**0.75
    self.noise_dist /= self.noise_dist.sum()
    if self.subsample > 0:
      self.create_subsample_table()

    with open(data_path, 'r') as f:
      logging.info("Loading data into memory...")
      for line in f:
        line_ids = [word2id.get(w,0) for w in line.rstrip().split()]
        # keep_ids = line_ids
        if self.subsample == 0:
          keep_ids = [i for i in line_ids if i > 0]
        else:
          keep_ids = []
          for i in line_ids:
            if self.sstable[i] >= 1:
              keep_ids.append(i)
            elif np.random.random() < self.sstable[i]:
              keep_ids.append(i)
        self.corpus += keep_ids
        corpus_size += len(line_ids)
        if corpus_size > 200_000_000:
          break
    logging.info("Corpus loaded into memory: {} words".format(corpus_size))

  def create_or_load_vocab(self, vocab_path):
    #ps = data_path.split('/')
    #vocab_path = os.path.join("/".join(ps[:-1]), f'vocab{self.min_freq}_' + ps[-1])
    if os.path.exists(vocab_path):
      # Load vocab
      word2id = {}
      self.freqs = []
      with open(vocab_path) as f:
        for i, line in enumerate(f):
          word, freq = line.rstrip().split()
          word2id[word] = i
          self.freqs.append(int(freq))
      logging.info("Loaded vocab from {} with {} words".format(vocab_path, len(word2id)))
    else:
      # Create vocab
      logging.info("Creating vocabulary...")
      word_freqs = Counter()
      word2id = {}
      self.freqs = [0]
      word2freq = {"UNK":0}
      with open(data_path) as f:
        for line in f:
          word_freqs.update(line.rstrip().split())
      
      for i, (word, freq) in enumerate(word_freqs.most_common()):
        if freq < self.min_freq:
          break
        word2id[word] = i
        word2freq[word] = freq
        self.freqs.append(freq)
      
      with open(vocab_path, 'w') as fw:
        for word, freq in word2freq.items():
          fw.write("{} {}\n".format(word, freq))
      logging.info("Saved vocab to {} with {} words".format(vocab_path, len(word2id)))

    return word2id

  def create_subsample_table(self):
    np_freqs = np.array(self.freqs, dtype=np.float)
    self.sstable = np.zeros_like(np_freqs)
    self.sstable[1:] = np.sqrt(self.subsample / (np_freqs[1:] / np.sum(np_freqs)))

  def __len__(self):
    return len(self.corpus) - self.window_size + 1

  def __getitem__(self, idx):
    # words = self.corpus[idx]
    # num_windows = len(words) - self.window_size + 1
    # word_ids, context_ids = [], []
    # for i in range(num_windows):
    #   word_ids.append(words[i+self.delta])
    #   context_ids.append(words[i : i+self.delta] + words[i+self.delta+1 : i+self.window_size])
    
    delta = self.delta# np.random.randint(1,self.delta+1) if self.varwindow else self.delta
    center = idx + self.delta
    word_id = self.corpus[center]
    context_ids = self.corpus[center - delta : center] + self.corpus[center+1 : center+1+delta]
    #context_ids += [0] * (2 * self.delta - len(context_ids))
    context_masks = [1.] * len(context_ids) #+ [0.] * (2 * self.delta - len(context_ids))
    # if self.subsample > 0:
    #   context_masks = []
    #   for i in context_ids:
    #     if self.sstable[i] >= 1:
    #       context_masks.append(1.)
    #     elif np.random.random() < self.sstable[i]:
    #       context_masks.append(1.)
    #     else:
    #       context_masks.append(0.)
    # else:
    #   context_masks = [1.] * (self.delta * 2)

    context_tensor = torch.LongTensor(context_ids)
    return word_id, context_tensor, torch.FloatTensor(context_masks)
    # return {'w':word_tensor, 'c':context_tensor}
    
def combine_samples(batch):
  words = torch.cat([sample['w'] for sample in batch], dim=0)
  contexts = torch.cat([sample['c'] for sample in batch], dim=0)
  return words, contexts

class Net(nn.Module):
  def __init__(self, params):
    super(Net, self).__init__()
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

  def forward(self, word_ids, context_ids, context_masks, neg_ids, return_alpha=False, temp=1):
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
    #context_embs *= context_masks.unsqueeze(-1)
    #context_nums = context_masks.sum(dim=1, keepdim=True)
    #context_emb = (context_embs.sum(dim=1) / context_nums).unsqueeze(-1) # (b,d,1)
    b, c = context_ids.size()[0], context_ids.size()[1]
    all_sense_ids = []
    for i in range(self.num_sense):
      all_sense_ids.append(context_ids * self.num_sense + i)
    all_sense_ids = torch.stack(all_sense_ids, dim=-1) # (b,c,k)
    
    all_sense_embs = self.emb(all_sense_ids) # (b,c,k,d)
    all_disamb_embs = self.disamb(all_sense_ids) # (b,c,k,d)
    
    all_alpha = torch.matmul(all_disamb_embs.reshape(b,-1,self.emb_dim), context_emb).reshape(b,c,self.num_sense,1).softmax(dim=2) # (b,c,k,1)
    context_emb2 = (all_sense_embs * all_alpha).sum(dim=2).mean(dim=1).unsqueeze(-1) # (b,d,1)
    #all_context_embs = (all_sense_embs * all_alpha).sum(dim=2) * context_masks.unsqueeze(-1) # (b,c,d)
    #second_context_emb = (all_context_embs.sum(dim=1) / context_nums).unsqueeze(-1) # (b,d,1)
    all_alpha2 = torch.matmul(all_disamb_embs.reshape(b,-1,self.emb_dim), context_emb2).reshape(b,c,self.num_sense,1).softmax(dim=2) # (b,c,k,1)
    context_emb3 = (all_sense_embs * all_alpha2).sum(dim=2).mean(dim=1).unsqueeze(-1) # (b,d,1)
    
    alpha = torch.matmul(disamb_embs, context_emb3) # (b,k,1)
    alpha_soft = alpha.softmax(dim=1)
    #pos_prob = torch.matmul((sense_embs * alpha_soft).sum(dim=1, keepdim=True), context_embs.transpose(1,2)).sigmoid() # (b,1,c)
    #neg_prob = torch.matmul((sense_embs * alpha_soft).sum(dim=1, keepdim=True), sample_embs.transpose(1,2)).sigmoid() # (b,1,c*n)
    #pos_loss = - torch.max(pos_prob, self.tiny_float).log().sum()
    #neg_loss = - torch.max(1 - neg_prob, self.tiny_float).log().sum()
    pos_prob = torch.matmul(sense_embs, context_embs.transpose(1,2)).sigmoid() # (b,k,c)
    neg_prob = torch.matmul(sense_embs, sample_embs.transpose(1,2)).sigmoid() # (b,k,c*n)
    pos_loss = - torch.max((pos_prob * alpha_soft).sum(dim=1), self.tiny_float).log().sum()
    neg_loss = - torch.max(1 - (neg_prob * alpha_soft).sum(dim=1), self.tiny_float).log().sum()
    # top2alpha, _ = torch.topk(alpha_soft.squeeze(), 2, dim=1)
    # alphamargin = torch.max(torch.tensor(0.).to(device), 0.5 - top2alpha[:,0] + top2alpha[:,1]).mean()
    # top2pos, _ = torch.topk(pos_prob, 2, dim=1)
    # probmargin = torch.max(torch.tensor(0.).to(device), 0.5 - top2pos[:,0,:] + top2pos[:,1,:]).mean()
    loss = (pos_loss + neg_loss) / context_ids.numel()# , probmargin

    if return_alpha:
      return loss, (alpha/temp).softmax(dim=1).squeeze(dim=-1), context_emb.squeeze(dim=-1)
    else:
      return loss
  
  def get_parameters(self):
    return list(self.emb.parameters()) + list(self.global_emb.parameters()) + list(self.disamb.parameters())

def get_optim_fn(optim_name):
  if optim_name in ["adam", "Adam"]:
    return torch.optim.Adam
  elif optim_name in ["adagrad", "Adagrad"]:
    return torch.optim.Adagrad
  elif optim_name in ["sgd", "Sgd", "SGD"]:
    return torch.optim.SGD
  
if __name__ == "__main__":
  SCRATCH = "/gpfs/u/home/BERT/BERTnksh/scratch"
  vocab_path = os.path.join(SCRATCH, "data/bert-sense/vocab10_small_750k.txt")
  #data_path = os.path.join(SCRATCH, 'data/bert-sense', "small_250k.txt")
  data_path = os.path.join(SCRATCH, 'data/bert-sense', "westwiki-para2")
  exp_dir = os.path.join(SCRATCH, "output/bert-sense/", "westwiki-200m-vocab750k-d5-s3-d300-adam-lr1e3-ss0-b2048-ns10-sense-alpha3")
  #exp_dir = os.path.join(SCRATCH, "output/bert-sense/", "small-250k-d5-s3-adam-lr1e3-ss0-sense-alpha2")
  if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

  #restore_file = os.path.join(exp_dir, 'model-1' + '.tar')
  restore_file = None
  # Set the logger
  set_logger(os.path.join(exp_dir, "train.log"))
  
  params = Params('paramsdissg.json')
  small_dataset = W2VDataset(data_path, vocab_path, params)
  logging.info("Dataset size: {}".format(len(small_dataset)))
  w2v_loader = DataLoader(small_dataset, batch_size=params.batch_size, shuffle=True, 
                          num_workers=params.num_workers)#, collate_fn=combine_samples)
  
  # for b in w2v_loader:
  #   print(b)
  #   sys.exit()
  params.update_dict({'vocab_size': small_dataset.vocab_size})
  logging.info(params)
  params.save(os.path.join(exp_dir, 'params.json'))
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logging.info(device)

  params.device = device
  model = Net(params).to(device)
  optimizer = get_optim_fn(params.optimizer)(model.get_parameters(), lr=params.learning_rate)

  start_epoch = 0
  start = time.time()
  c = 0
  loss_vals = []
  temp_vals = []
  
  if restore_file is not None:
    logging.info("Restoring model parameters from {}".format(restore_file))
    checkpoint = torch.load(os.path.join(exp_dir, restore_file))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    start_epoch = checkpoint['epoch']
    with open(os.path.join(exp_dir, "loss.pkl"), "rb") as f:
      loss_vals = pickle.load(f)

  for e in range(start_epoch, params.num_epochs):
    for step, b in enumerate(w2v_loader):
      word_id, ctx, ctx_mask = b
      c+= len(word_id)

      neg_ids = torch.multinomial(small_dataset.noise_dist, ctx.numel() * params.num_samples, replacement=True).view(ctx.size()[0], -1)
    
      # log_loss, margin_loss = model(word_id.to(device), ctx.to(device), neg_ids.to(device))
      # if e > 1:
      # loss = log_loss + margin_loss
      # else:
      #   loss =  log_loss
      loss = model(word_id.to(device), ctx.to(device), ctx_mask.to(device), neg_ids.to(device))
      temp_vals.append(loss.item())
      
      # Zero gradients, perform a backward pass, and update the weights.
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (step+1) % 100 == 0:
        loss_vals.append(np.mean(temp_vals))
        temp_vals.clear()
        t = time.time() - start
        logging.info(f"{datetime.timedelta(seconds=int(t))}"
                f"\tE{e+1}:s{step+1:<6,d}"
                f"\tLoss: {loss:<,.6f}"
                f"\twords/sec: {c/t:<6,.2f}")
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
  plt.savefig(os.path.join(exp_dir, "plot{}.png".format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))), dpi=200)
  #plt.show()

  with open(os.path.join(exp_dir, "loss.pkl"), "wb") as f:
    pickle.dump(loss_vals, f)
