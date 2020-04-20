import sys
import os
import numpy as np
import torch

def cosine_sim(a,b):
  dot_product = np.dot(a, b)
  norm_a = np.linalg.norm(a)
  norm_b = np.linalg.norm(b)
  return dot_product / (norm_a * norm_b)

def disambiguate(model, bert, word_id, word_pos, context):
  sense_disamb = model.sense_disamb[word_id]# / np.linalg.norm(model.sense_disamb[word_id], axis=-1, keepdims=True)
  if bert['use_bert']:
    if model.M is not None:
      sense_disamb = np.dot(sense_disamb, model.M)
    
    context_token_map = []
    tokens = []
    for w in context:
      context_token_map.append(len(tokens)+1)
      tokens += bert['tokenizer'].tokenize(w)
    context_token_map.append(len(tokens)+1)

    context_ids = [[bert['tokenizer'].cls_token_id] \
                  + bert['tokenizer'].convert_tokens_to_ids(tokens) \
                  + [bert['tokenizer'].sep_token_id]]

    with torch.no_grad():
      output_states, _, hidden_states = bert['model'](torch.tensor(context_ids))
   
    if bert['last4']:
      out_state = output_states[0]
    else:
      out_state = sum(hidden_states[-4:])[0]
    context_states = np.zeros((len(context), output_states.shape[-1]), dtype=np.float32)
    for i in range(len(context)):
      context_states[i] = torch.mean(out_state[context_token_map[i]:context_token_map[i+1]], dim=0)
    
    ctx = np.mean(np.concatenate([context_states[:word_pos,:], context_states[word_pos+1:,:]], axis=0), axis=0)
    
  else:
    ctx = model.get_ctx_emb(context)
  
  scores = np.dot(sense_disamb, ctx)
    
  cl_max = np.argmax(scores)
  return cl_max + 1

def prepare_context(context, word2id):
  context_ids = []
  for w in context.split():
    i = word2id.get(w, -1)
    if i > 0:# and np.random.uniform() <= np.sqrt(0.001 * total_freq / word2freq[w]):
      context_ids.append(i)

  return context_ids

# word2freq = dict()
# for line in open(os.path.join(modelpath, 'vocab.txt')):
#     word, freq = line.strip().split()
#     word2freq[word] = int(freq)
# total_freq = sum(word2freq.values())

def write_sense(model, bert, data_file, out_file=None):
  print("Loading words and contexts from", data_file)
  window_size = 5
  senses = {}
  # arg3 is semeval data file
  if out_file:
    fw = open(out_file, 'w')
    orig_stdout = sys.stdout
    sys.stdout = fw
  with open(data_file) as f:
    while(True):
      line = f.readline().strip()
      if line == '':
        break
      try:
        word_full, context_number = line.split()        
      except:
        print('word-line', line)
        break
      
      if word_full not in senses:
        senses[word_full] = ([],[])
      context_number = int(context_number)
      word = word_full.split('.')[0]
      word_id = model.word2id.get(word,-1)
      if word_id == -1:
        print("{} not in vocabulary".format(word))
      elif out_file:
        model.nearby(word)
        
      # if model.mask is None:
      #   num_senses = model.num_senses
      # else:
      #   num_senses = np.sum(model.mask[model.word2id[word]], dtype=np.int32)
      # print(word, num_senses)

      for i in range(context_number):
        l = f.readline().strip()
        if word_id == -1:
          s=1
          senses[word_full][0].append(i+1)
          senses[word_full][1].append(s)
          continue
        try:
          left, right = l.split('\t')
          left_context, right_context = left[2:], right[2:]
        except:
          if l[0] == "L":
            left_context = l[2:]
            right_context = ""
          elif l[0] == "R":
            left_context = ""
            right_context = l[2:]
          else:
            print(l)
          # continue

        if out_file:
          print(f"Word: {word}, L:{left_context.split()[-10:]}, R:{right_context.split()[:10]}")
        
        if bert['use_bert']:
          left_context = left_context.split()[-window_size:]
          right_context = right_context.split()[:window_size]
          context = left_context + [word] + right_context
        else:
          left_context = prepare_context(left_context, model.word2id)[-window_size:]
          right_context = prepare_context(right_context, model.word2id)[:window_size]
          context = left_context + right_context
        
        s = disambiguate(model, bert, word_id, len(left_context), context)
        if out_file:
          print(f"sense: {s}")
        
        senses[word_full][0].append(i+1)
        senses[word_full][1].append(s)
        # fw.write("{} {} {}\n".format(word_full, word_full + '.' + str(i+1), word_full + '.sense.' + str(s)))
      
      print("Induced sense of {} words".format(len(senses)), end='\r') 
      # break
    if out_file:
      sys.stdout = orig_stdout
    print()
    return senses

if __name__ == '__main__':
  from load_model import Embedder, load_vocab
  from utils import Params
  from train import Net
  from transformers import BertModel, BertTokenizer

  # model = sys.argv[1]
  # epochs = int(sys.argv[2])

  datasets = ['semeval-2013']#, 'semeval-2013'] #'semeval-2007',
  
  # for epoch in range(1,epochs+1):
  vocab = load_vocab('vocab_small_50k.txt')
  model_dir = 'experiments/small-50k-bert-d5-s3-adam-lr1e3-disamb-mse'

  params = Params(os.path.join(model_dir, 'params.json'))
  model = Net(params)
  checkpoint = torch.load(os.path.join(model_dir, 'model-10.tar'))
  model.load_state_dict(checkpoint['model_state_dict'])
  embed = Embedder(model, vocab, params)

  bert_name = 'bert-base-uncased'
  bert_tokenizer = BertTokenizer.from_pretrained(bert_name)
  bert_model = BertModel.from_pretrained(bert_name)
  
  data_dir = "/home/anik/Documents/f18/embeddings/cluster-multi-sense/wsi-eval/datasets/"
  for dataset in datasets:
    predictions = write_sense(embed, bert_tokenizer, bert_model, os.path.join(data_dir, "%s/dataset.txt" %(dataset)), "_tmp_")#, "__result__.tmp")
