# Word Sense Induction with Knowledge Distillation from BERT

Code for the paper
https://arxiv.org/pdf/2304.10642.pdf


## Training

### Sense Embeddings

`dissg.py`

Train sense embeddings from text. Set the path to the input text file and the output model file. The parent folder is `SCRATCH`. The text file and vocab file path are set in `data_path` and `vocab_path` respectively. The output directory is `exp_dir`.  

The parameters for training are set in `paramsdissg.json` file.

```
python dissg.py
```

### BERT Sense Model

`save_dataset.py`

First, precompute BERT embeddings for all context windows in the input text file. The text file path is set in the `data_path` variable while the output BERT embeddings file is set in the `bert_path` variable. Change the BERT model by pointing the `bert_vocab` and `bert_weights` variables to model name (i.e. bert-base-cased) or path to the model files. This will create a **huge** file (100GB for 250k windows) since we are saving 512 dimensional embeddings for each context window. 

```
python save_dataset.py
```

`train.py`

Then, train a sense embedding model using the saved BERT context embeddings. The `data_path` variable points to the precomputed BERT embeddings file. The hyperparameters for this model are saved in `params.json` file.

```
python train.py
```

### Model Distillation from BERT

`w2v_bert.py`

Train sense embeddings using model distillation from the BERT sense embeddings trained in the previous step. `vocab_path` corresponds to the vocabulary file saved from training sense embeddings with `dissg.py`. The path to BERT context embeddings saved from `saved_dataset.py` is set in `data_path` variable. The path BERT sense model from the previous step is set in `bert_path` variable. The hyperparameters for this step is saved in `parasmw2vbert.json` file. 

```
python w2v_bert.py
```


## Evaluation

The SCWS data set is available from http://www-nlp.stanford.edu/~ehhuang/SCWS.zip

`eval_scws_w2v.py`

For evaluation on contextual similarity task, set the `vocab` and `model_dir` variables to the corresponding files from the Sense Embeddings model or the Distilled Sense Embeddings model. Set the correct path to the SCWS files in the `process_huang` function.

```
python eval_scws_w2v.py
```