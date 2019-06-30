Code for paper "Merge and Label: A Novel Neural Network Architecture for Nested NER" in ACL 2019


## Notes on reproducing results

1. Download glove embeddings glove.840B.300d.zip from https://nlp.stanford.edu/projects/glove/ and extract into 
directory ./data/pre_trained_embeddings/glove/

2. Get access to ACE 2005 corpus (https://catalog.ldc.upenn.edu/LDC2006T06) and/or the OntoNotes corpus(https://catalog.ldc.upenn.edu/LDC2013T19). Preprocess data to same format as example datafiles in ./data/problems/ACE05/data/. Train/val/test splits for the OntoNotes corpus are standard. Splits for ACE 2005 corpus follow https://arxiv.org/abs/1810.09073 and are provided in ./data/problems/ACE05/splits

## Training a model

Models can be trained from the command line:

```python3 train_script.py merge_label```

The default arguments train the model on the ACE05 dataset, with Glove embeddings. The embeddings can be changed to BERT, and the problem to ontonotes as follows:

```python3 train_script.py merge_label --problems ontonotes --embeddings flair_bert```

There are a wide range of potential command line arguments which control the network parameters, learning rate etc. All available options can be found in ./mg_lb/training/args.py. 

## Evaluating trained models from paper

We include pretrained models for the results achieved in the paper. These are in directory ./paper_models/

There are four models available, two for each dataset, with each type of embedding:

| Model Name | Dataset | Embeddings | Test F1 |
| ------------- | ------------- | ------------- | ------------- |
| ACE05_74.87 | ACE05 | Glove| 74.87 |
| ACE05_83.10 | ACE05 | BERT | 83.10  |
| ontonotes_87.59 | OntoNotes  | Glove  | 87.59 |
| ontonotes_89.25 | OntoNotes  | BERT  | 89.25 |
 

Note that in the case of the ACE05 models, the test F1 scores are slightly higher than those reported in the paper, as we provide the best model from the training repetitions (chosen on dev set), whereas the average is reported in the paper



```
from mg_lb.eval.prep import prep_for_eval, eval_data_set, network_sentences
from mg_lb.eval.visualize import sort_results
from mg_lb.eval.losses import f1_multi_layer, f1_ace

problem = 'ACE05' # one of ['ontonotes', 'ACE05']
dset = 'test.txt' # one of ['train.txt', 'val.txt', 'test.txt']
save_name = 'ACE05_83.10' # one of ['ACE05_83.10', 'ACE05_74.87', 'ontonotes_87.59', 'ontonotes_89.25']

# Load the model
model, vocab, args, prob_dict = prep_for_eval(save_name, paper_model=True, gpu=True)

# Run forward pass on dataset
res_all, iter_all = eval_data_set(problem, dset, args, vocab, model, prob_dict, batch_size=2000)

# Process the returns from the network
res_out = sort_results(res_all, iter_all, vocab, problem, cutoff=0.25)

# Calculate F1 score
if problem == 'ACE05':
    prec, rec, f1, wrong, acc_wrong, cl_wrong = f1_ace(res_out['cluster_preds'], res_out['labels'], 
                                                       remove_np=True)
elif problem == 'ontonotes':
    prec, rec, f1, wrong = f1_multi_layer(res_out['cluster_preds'], res_out['labels'], add_bs=True,
                                          remove_np=True)
    
print('F1:', "{:.4f}".format(f1))

```

## Running a trained model on list of user defined sentences

```
problem = 'ACE05' # one of ['ontonotes', 'ACE05']
save_name = 'ACE05_83.10' # one of ['ACE05_83.10', 'ACE05_74.87', 'ontonotes_87.59', 'ontonotes_89.25']

# Load the model
model, vocab, args, prob_dict = prep_for_eval(save_name, paper_model=True, gpu=True)

# Input sentences
sentences = ['The Prime Minister of the UK met with Donald Trump on Thursday.']

# Run forward pass
res_sents, iter_sents = network_sentences(sentences, problem, args, vocab, model, prob_dict=prob_dict, gpu=True)

# Process returns
res_out = sort_results(res_sents, iter_sents, vocab, problem, cutoff=0.25)

```

Clustered entities are in res_out['cluster_words'], and predictions of labels in res_out['cluster_preds']. NB the predictions will all be in format 'I-PERS' even if first word in entitity. These can be converted to 'B-s' if required, as done in the F1 calculation functions


## Evaluating a new trained model/ new dataset

```
# Examples of args for evaluating a model trained by the user, which will be saved in ./saved_models/
# NB need to train a model first as per instructions in ReadMe for this to work 

problem = 'ACE05' # Name  of problem model was trained on
save_id = '1' # Save id for the training run - set in command line args, default value is '1'

# Same as above for data set vs sentences but change paper_model argument in prep_for_eval to False
model, vocab, args, prob_dict = prep_for_eval(problem, paper_model=False, save_id=save_id, gpu=True)

```

Then follows as above two examples for dataset evaluation or forward pass on new sentences
