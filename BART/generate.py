import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BartForConditionalGeneration, AutoTokenizer

from dataset import KoBARTSummaryDataset


data_path='./data/'
model_path='./LG_model/'
sub_path='./LG_sub/'
tokenizer=AutoTokenizer.from_pretrained("hyunwoongko/kobart")

# Args
args={}
args['test_path']=data_path+'test_evi_final.csv'
args['weight_path']=model_path
args['max_len']=1024
args['batch_size']=4
args['num_workers']=2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# Test dataloader	
test_data = KoBARTSummaryDataset(args['test_path'],tok=AutoTokenizer.from_pretrained("hyunwoongko/kobart"),max_len=args['max_len'],infer=True)

test_dataloader = DataLoader(test_data,
                              batch_size=args['batch_size'], 
                              shuffle=False,
                              num_workers=args['num_workers'], 
                              pin_memory=True)


# Generation
test_sum=[]
test_model=BartForConditionalGeneration.from_pretrained(os.path.join(args['weight_path'], "kobart_title_evi_concat_final_epoch_2")).to(device)

for i,batch in tqdm(enumerate(test_dataloader)):
  sets=batch['input_ids'].to(device)
  batch_sum=test_model.generate(sets, max_length=150,no_repeat_ngram_size=3)
  test_sum=[*test_sum,*batch_sum]

# decode outputs
test_sum_sent=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in test_sum]

sub=pd.read_csv(data_path+'sample_submission.csv')
sub['summary']=test_sum_sent


# view result
test_real=pd.read_csv(data_path+'test_evi_final.csv')

id=35
print('Context:\n')
display(test_real['title_context'][id])

print('\nsummary:\n')
display(sub['summary'][id])


id=3
print('Context:\n')
display(test_real['title_context'][id])

print('\nsummary:\n')
display(sub['summary'][id])


