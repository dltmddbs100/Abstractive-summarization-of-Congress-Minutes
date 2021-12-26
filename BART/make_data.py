import os
import json

import numpy as np
import pandas as pd

import re
from utils import process_basic, process_detail, election, monetary

from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("hyunwoongko/kobart")


data_path='./data/'
model_path='./LG_model/'
sub_path='./LG_sub/'

# 1. Make train data
# making title + context data
train=pd.read_json(data_path+'train.json')

real_id=[]
for i in range(0,len(train)):
  for j in range(1,train.iloc[i,3]+1):
    real_id.append('id'+str(train['id'][i])+'-AGENDA_'+str(j))

real_context=[]
for i in range(0,len(train)):
  for j in range(1,train.iloc[i,3]+1):
    real_context.append(' '.join(list(train['context'][i][f'AGENDA_{j}'].values())))

real_title=[]
for i in range(0,len(train)):
  for j in range(1,train.iloc[i,3]+1):
    real_title.append(train['title'][i])

real_label=[]
for i in range(0,len(train)):
  for j in range(1,train.iloc[i,3]+1):
    real_label.append(train['label'][i][f'AGENDA_{j}']['summary'])


train_real=pd.DataFrame({'id':real_id,'title':real_title,'context':real_context,'summary':real_label})

train_real=process_basic(train_real,'context')
train_real=process_detail(train_real,'context')
train_real['title_context']=train_real['title']+': '+train_real['context']

# Special case preprocessing previously defined
train_real=election(train_real, 'title_context')
train_real=monetary(train_real, 'title_context',tokenizer)

for i in range(len(train_real)):
  train_real.at[i,'title_context']=re.sub(' +',' ',train_real['title_context'][i])

train_real.to_csv(data_path+'train_real_final.csv',index=False)


# making title + evidence text data
# will be used for additional train data
real_id=[]
for i in range(0,len(train)):
  for j in range(1,train.iloc[i,3]+1):
    real_id.append('id'+str(train['id'][i])+'-AGENDA_'+str(j))

real_title=[]
for i in range(0,len(train)):
  for j in range(1,train.iloc[i,3]+1):
    real_title.append(train['title'][i])

real_label=[]
for i in range(0,len(train)):
  for j in range(1,train.iloc[i,3]+1):
    real_label.append(train['label'][i][f'AGENDA_{j}']['summary'])

real_evi=[]
for i in range(0,len(train)):
  for j in range(1,train.iloc[i,3]+1):
    real_evi.append(train['label'][i][f'AGENDA_{j}']['evidence'].values())

real_num=[]
for i in range(0,len(train)):
  for j in range(1,train.iloc[i,3]+1):
    real_num.append(train['label'][i][f'AGENDA_{j}']['evidence'].keys())

evi_real=pd.DataFrame({'id':real_id,'title':real_title,'evidence':real_evi, 'evi_num':real_num,'summary':real_label})
evi_real['evi_num']=evi_real['evi_num'].apply(len)

# link the evidence text to single paragraph
evidence_text=[]

for i in range(0,len(evi_real)):
  l=[]
  for j in range(0,evi_real['evi_num'][i]):
    l.append(' '.join(list(evi_real['evidence'][i])[j]))

  evidence_text.append(' '.join(l))

evi_real['evidence_text']=evidence_text
evi_real=evi_real.drop(['evidence','evi_num'],axis=1)


evi_real=process_basic(evi_real,'evidence_text')
evi_real=process_detail(evi_real,'evidence_text')

evi_real['title_evidencetext']=evi_real['title']+': '+evi_real['evidence_text']
evi_real.columns=['id','title','summary','context','title_context']

# concat all train data
train_real=pd.read_csv(data_path+'train_real_final.csv')
train_evi_concat=pd.concat([train_real,evi_real]).reset_index(drop=True)

train_evi_concat.to_csv(data_path+'train_evi_concat_final.csv',index=False)


# 2. Make test data
# same as train data format
test=pd.read_json(data_path+'test.json')

real_id=[]
for i in range(0,len(test)):
  for j in range(1,test.iloc[i,3]+1):
    real_id.append('id_'+str(test['id'][i])+'-AGENDA_'+str(j))

real_title=[]
for i in range(0,len(test)):
  for j in range(1,test.iloc[i,3]+1):
    real_title.append(test['title'][i])

real_context=[]
for i in range(0,len(test)):
  for j in range(1,test.iloc[i,3]+1):
    real_context.append(' '.join(list(test['context'][i][f'AGENDA_{j}'].values())))

test_real=pd.DataFrame({'id':real_id,'title':real_title,'context':real_context})

test_real=process_basic(test_real,'context')
test_real=process_detail(test_real,'context')
test_real['title_context']=test_real['title']+': '+test_real['context']

test_real=election(test_real, 'title_context')
test_real=monetary(test_real, 'title_context', tokenizer)

test_real.to_csv(data_path+'test_evi_final.csv',index=False)
