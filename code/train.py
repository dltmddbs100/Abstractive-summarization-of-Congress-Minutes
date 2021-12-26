import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import time

import torch
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader

from transformers import BartForConditionalGeneration, AutoTokenizer
from transformers.optimization import AdamW

from dataset import KoBARTSummaryDataset


data_path='./data/'
model_path='./LG_model/'
sub_path='./LG_sub/'
tokenizer=AutoTokenizer.from_pretrained("hyunwoongko/kobart")

# Args
args={}
args['train_path']=data_path+'train_evi_concat_final.csv'
args['weight_path']=model_path
args['max_len']=1024
args['batch_size']=4
args['num_workers']=2
args['max_epochs']=2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# Train dataloader
train_data = KoBARTSummaryDataset(args['train_path'],tok=AutoTokenizer.from_pretrained("hyunwoongko/kobart"),max_len=args['max_len'])

train_dataloader = DataLoader(train_data,
                              batch_size=args['batch_size'], 
                              shuffle=True,
                              num_workers=args['num_workers'], 
                              pin_memory=True)


# Training
model=BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart").to(device)


optimizer = AdamW(model.parameters(),lr=2e-5, weight_decay=1e-4,correct_bias=False)
scaler = amp.GradScaler()


for epoch_i in range(0, args['max_epochs']):
  
  # ========================================
  #               Training
  # ========================================

  print("")
  print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args['max_epochs']))
  model.train()
  t0 = time.time()
  total_train_loss = 0
  total_batch=len(train_dataloader)

  for i, batch in enumerate(train_dataloader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    decoder_input_ids = batch['decoder_input_ids'].to(device)
    decoder_attention_mask = batch['decoder_attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    optimizer.zero_grad()

    with amp.autocast():
      loss = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   decoder_input_ids=decoder_input_ids,
                   decoder_attention_mask=decoder_attention_mask,
                   labels=labels)[0]
      
    total_train_loss += loss.item()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
      
    training_time = time.time() - t0

    print(f"\rTotal Batch {i+1}/{total_batch} , elapsed time : {training_time/(i+1):.1f}s , train_loss : {total_train_loss/(i+1):.2f}", end='')
  print("")

  model.save_pretrained(os.path.join(args['weight_path'], "kobart_title_evi_concat_final_epoch_{}".format(epoch_i+1)))