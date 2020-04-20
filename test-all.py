import numpy as np
import sys
import os
import csv
import itertools
from sklearn.metrics import adjusted_rand_score,v_measure_score, adjusted_mutual_info_score

from semeval import write_sense

def get_pairs(labels):
  result = []
  for label in np.unique(labels):
    ulabels = np.where(labels==label)[0]
    for p in itertools.combinations(ulabels, 2):
      result.append(p)
  return result

def compute_fscore(true, pred):
  true_pairs = get_pairs(true)
  pred_pairs = get_pairs(pred)
  int_size = len(set(true_pairs).intersection(pred_pairs))
  p = int_size / float(len(pred_pairs))
  r = int_size / float(len(true_pairs))
  return 2*p*r/float(p+r)

def read_answers(filename):
  with open(filename, 'r') as f:
    keys = []
    instances = []
    senses = []
    senses_id = {}
    sense_count = 0
    for line in f.readlines():
      key, instance, sense = line.strip().split(' ')
      num = int(instance.split('.')[-1])
      keys.append(key)
      instances.append(num)
      senses.append(sense)
      if sense not in senses_id:
        senses_id[sense] = sense_count
        sense_count += 1
    answers = {}
    for k,i,s in zip(keys, instances, senses):
      if k not in answers:
        answers[k] = ([],[])
      answers[k][0].append(i)
      answers[k][1].append(senses_id[s])
    return answers

def compute_metrics(answers, predictions):
  aris = []
  amis = []
  vscores = []
  fscores = []
  weights = []
  for k in answers.keys():
    idx = np.argsort(np.array(answers[k][0]))
    true = np.array(answers[k][1])[idx]
    pred = np.array(predictions[k][1])
    weights.append(pred.shape[0])
    if len(np.unique(true)) > 1:
      aris.append(adjusted_rand_score(true, pred))
      # print(k, aris[-1])
    vscores.append(v_measure_score(true, pred))
    fscores.append(compute_fscore(true, pred))
    #amis.append(adjusted_mutual_info_score(true, pred))
  
#        print '%s: ari=%f, vscore=%f, fscore=%f' % (k, aris[-1], vscores[-1], fscores[-1])
  #amis = np.array(amis)
  aris = np.array(aris)
  vscores = np.array(vscores)
  fscores = np.array(fscores)
  weights = np.array(weights)
  ari = np.mean(aris)
  print('number of one-sense words: %d' % (len(vscores) - len(aris)))
  print('mean ari: %f' % ari)
  print('mean vscore: %f' % np.mean(vscores))
  print('weighted vscore: %f' % np.sum(vscores * (weights / float(np.sum(weights)))))
  print('mean fscore: %f' % np.mean(fscores))
  print('weighted fscore: %f' % np.sum(fscores * (weights / float(np.sum(weights)))))
  #print('mean ami: %f' % np.mean(amis))
  return ari
  
if __name__ == '__main__':
  from load_model import Embedder, load_vocab
  from utils import Params
  from train import Net
  #from dissg import Net
  from transformers import BertModel, BertTokenizer
  import torch

  # model = sys.argv[1]
  # epochs = int(sys.argv[2])

  SCRATCH = "/gpfs/u/home/BERT/BERTnksh/scratch/"
  vocab = load_vocab(os.path.join(SCRATCH, 'data/bert-sense/vocab10_small_750k.txt'))
  #vocab = load_vocab(os.path.join(SCRATCH, 'data/bert-sense/vocab100_westwiki-para2'))
  #model_dir = os.path.join(SCRATCH, 'output/bert-sense/small-750k-split-dissg-bert-ctx-d5-s3-d300-b2048-ns10-adam-lr1e3-sense-temp4-alpha2')
  model_dir = os.path.join(SCRATCH, 'output/bert-sense/small-750k-split-bert-d5-s3-adam-lr1e3')
  #model_dir = os.path.join(SCRATCH, 'output/bert-sense/westwiki-100m-vocab250k-d5-s3-d100-adam-lr1e3-ss0-ns10-sense-alpha2')
  #model_dir = os.path.join(SCRATCH, 'output/bert-sense/small-250k-d5-s3-adam-lr1e3-ss3-varwindow')

  params = Params(os.path.join(model_dir, 'params.json'))
  params.device = torch.device('cuda:0')
  model = Net(params)
  
  bert = {'use_bert': True}
  if bert['use_bert']:
    bert['last4'] = False 
    bert['tokenizer'] = BertTokenizer.from_pretrained('bert-base-uncased-vocab.txt')
    bert['model'] = BertModel.from_pretrained('bert_base_weights', output_hidden_states=True)

  data_dir = os.path.join(SCRATCH, "data/bert-sense/wsi-eval")
  # model = sys.argv[1]
  # epochs = int(sys.argv[2])

  #datasets = ['semeval-2013'] 
  datasets = ['semeval-2007', 'semeval-2010', 'semeval-2013']
  results = {}

  e = 2 
  for epoch in range(e,e+1):#range(1,epochs+1):
    # msvec = MSVec(model, epoch)
    ckpt_file = os.path.join(model_dir, 'model-{}.tar'.format(epoch))
    checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    embed = Embedder(model, vocab, params)
    print("Loaded model parameters from", ckpt_file)

 
    results[epoch] = []

    for dataset in datasets:
      predictions = write_sense(embed, bert, os.path.join(data_dir, "%s/dataset.txt" %(dataset)))#, "__result__.tmp")
      true_answers = read_answers(os.path.join(data_dir, "%s/key.txt" %(dataset)))
      # predictions = read_answers('__result__.tmp')
      print('DATASET %s:' % dataset)
      # print([len(set(k[1])) for k in true_answers.values()])
      # print([len(set(k[1])) for k in predictions.values()])
      results[epoch].append(compute_metrics(true_answers, predictions))
      # os.remove('__result__.tmp')
      print('\n')

    # subprocess.call('./run.sh benchmark/test_wwsi.jl %s __result__.tmp' % model, shell=True)
    # os.remove('__result__.tmp')

  csv_headers = ["epoch"] + datasets
  model_name = model_dir.split("/")[-1]
  with open("results/" + model_name + "_semeval.csv", "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)
    for e in results:
      writer.writerow([e] + results[e])
    
