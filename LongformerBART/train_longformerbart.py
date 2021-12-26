import os
from tqdm import tqdm, trange
import time
import re
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader

from transformers import BartForConditionalGeneration, AutoTokenizer, BartConfig
from transformers.optimization import AdamW

from dataset import KoBARTSummaryDataset
from longformerbart import LongformerEncoderDecoderConfig, LongformerSelfAttentionForBart


data_path='./data/'
model_path='./LG_model/'
sub_path='./LG_sub/'

# Args
args={}
args['train_path']=data_path+'train_evi_concat_final.csv'
args['test_path']=data_path+'test_evi_final.csv'
args['weight_path']=model_path
args['batch_size']=4
args['num_workers']=4
args['max_epochs']=4
args['attention_window'] = 512
args['max_pos'] = 2052
args['max_seq_len'] = 2048

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# load tokenizer, model
tokenizer = AutoTokenizer.from_pretrained("hyunwoongko/kobart", model_max_length=args['max_pos'])
model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")
config = LongformerEncoderDecoderConfig.from_pretrained("hyunwoongko/kobart")

model.config = config

# in BART attention_probs_dropout_prob is attention_dropout, but LongformerSelfAttention
# expects attention_probs_dropout_prob, so set it here
config.attention_probs_dropout_prob = config.attention_dropout
config.architectures = ['LongformerEncoderDecoderForConditionalGeneration']

# extend position embeddings
tokenizer.model_max_length = args['max_pos']
tokenizer.init_kwargs['model_max_length'] = args['max_pos']
current_max_pos, embed_size = model.model.encoder.embed_positions.weight.shape
assert current_max_pos == config.max_position_embeddings + 2

config.max_encoder_position_embeddings = args['max_pos']
config.max_decoder_position_embeddings = config.max_position_embeddings
del config.max_position_embeddings
args['max_pos'] += 2  # NOTE: BART has positions 0,1 reserved, so embedding size is max position + 2
assert args['max_pos'] >= current_max_pos

# allocate a larger position embedding matrix for the encoder
new_encoder_pos_embed = model.model.encoder.embed_positions.weight.new_empty(args['max_pos'], embed_size)
# copy position embeddings over and over to initialize the new position embeddings
k = 2
step = current_max_pos - 2
while k < args['max_pos'] - 1:
    new_encoder_pos_embed[k:(k + step)] = model.model.encoder.embed_positions.weight[2:]
    k += step
model.model.encoder.embed_positions.weight.data = new_encoder_pos_embed

# allocate a larger position embedding matrix for the decoder
# new_decoder_pos_embed = model.model.decoder.embed_positions.weight.new_empty(args['max_pos'], embed_size)
# # copy position embeddings over and over to initialize the new position embeddings
# k = 2
# step = current_max_pos - 2
# while k < args['max_pos'] - 1:
#     new_decoder_pos_embed[k:(k + step)] = model.model.decoder.embed_positions.weight[2:]
#     k += step
# model.model.decoder.embed_positions.weight.data = new_decoder_pos_embed

# replace the `modeling_bart.SelfAttention` object with `LongformerSelfAttention`
config.attention_window = [args['attention_window']] * config.num_hidden_layers
config.attention_dilation = [1] * config.num_hidden_layers

for i, layer in enumerate(model.model.encoder.layers):
    longformer_self_attn_for_bart = LongformerSelfAttentionForBart(config, layer_id=i)

    longformer_self_attn_for_bart.longformer_self_attn.query = layer.self_attn.q_proj
    longformer_self_attn_for_bart.longformer_self_attn.key = layer.self_attn.k_proj
    longformer_self_attn_for_bart.longformer_self_attn.value = layer.self_attn.v_proj

    longformer_self_attn_for_bart.longformer_self_attn.query_global = copy.deepcopy(layer.self_attn.q_proj)
    longformer_self_attn_for_bart.longformer_self_attn.key_global = copy.deepcopy(layer.self_attn.k_proj)
    longformer_self_attn_for_bart.longformer_self_attn.value_global = copy.deepcopy(layer.self_attn.v_proj)

    longformer_self_attn_for_bart.output = layer.self_attn.out_proj

    layer.self_attn = longformer_self_attn_for_bart


# data loader
torch.manual_seed(1514)

train_data = KoBARTSummaryDataset(args['train_path'],tok=AutoTokenizer.from_pretrained("hyunwoongko/kobart"),max_len=args['max_seq_len'])
train_data, eval_data = torch.utils.data.random_split(train_data,[2400,594])

train_dataloader = DataLoader(train_data,
                              batch_size=args['batch_size'], 
                              shuffle=True,
                              num_workers=args['num_workers'], 
                              pin_memory=True)

eval_dataloader = DataLoader(eval_data,
                              batch_size=args['batch_size'], 
                              shuffle=False,
                              num_workers=args['num_workers'], 
                              pin_memory=True)


# function
def accuracy_function(real, pred):
    accuracies = torch.eq(real, torch.argmax(pred, dim=2))
    mask = torch.logical_not(torch.eq(real, -100))
    accuracies = torch.logical_and(mask, accuracies)
    accuracies = accuracies.clone().detach()
    mask = mask.clone().detach()

    return torch.sum(accuracies)/torch.sum(mask)

def loss_function(real, pred):
    mask = torch.logical_not(torch.eq(real, -100))
    loss_ = criterion(pred.permute(0,2,1), real)
    mask = mask.clone().detach()
    loss_ = mask * loss_

    return torch.sum(loss_)/torch.sum(mask)


# Training
model=model.to(device)

criterion = nn.CrossEntropyLoss()
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
  total_train_acc = 0
  total_batch=len(train_dataloader)

  for i, batch in enumerate(train_dataloader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    decoder_input_ids = batch['decoder_input_ids'].to(device)
    decoder_attention_mask = batch['decoder_attention_mask'].to(device)
    labels = batch['labels'].to(device)

    model.model.encoder.config.gradient_checkpointing = True
    model.model.decoder.config.gradient_checkpointing = True
    
    optimizer.zero_grad()

    with amp.autocast():
      output = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   decoder_input_ids=decoder_input_ids,
                   decoder_attention_mask=decoder_attention_mask,
                   labels=labels)
      loss = output.loss

    acc = accuracy_function(labels, output.logits)

    total_train_acc += acc  
    total_train_loss += loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
      
    training_time = time.time() - t0

    print(f"\rTotal Batch {i+1}/{total_batch} , elapsed time : {training_time/(i+1):.1f}s , train_loss : {total_train_loss/(i+1):.2f}, train_acc: {total_train_acc/(i+1):.3f}", end='')
  print("")

  model.eval()
  total_eval_loss=0
  total_val_acc=0
  total_val_batch=len(eval_dataloader)

  for i,batch in enumerate(eval_dataloader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    decoder_input_ids = batch['decoder_input_ids'].to(device)
    decoder_attention_mask = batch['decoder_attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
      output = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   decoder_input_ids=decoder_input_ids,
                   decoder_attention_mask=decoder_attention_mask,
                   labels=labels)
      loss = output.loss
    
    acc = accuracy_function(labels, output.logits)
    
    total_val_acc +=acc
    total_eval_loss += loss
    
    print(f"\rValidation Batch {i+1}/{total_val_batch} , validation loss: {total_eval_loss/(i+1):.2f}, val_acc: {total_val_acc/(i+1):.3f}", end='')

  torch.save(model.state_dict(), args['weight_path']+'LongformerKoBART_2048_epoch {} weight.ckpt'.format(epoch_i + 1))


# Generate
model.load_state_dict(torch.load(args['weight_path']+'LongformerKoBART_2048_epoch 2 weight.ckpt'))
model.eval()

# test data load
test_data = KoBARTSummaryDataset(args['test_path'],tok=AutoTokenizer.from_pretrained("hyunwoongko/kobart"),max_len=args['max_len'],infer=True)

test_dataloader = DataLoader(test_data,
                              batch_size=args['batch_size'], 
                              shuffle=False,
                              num_workers=args['num_workers'], 
                              pin_memory=True)

# Generation
test_sum=[]

for i,batch in tqdm(enumerate(test_dataloader)):
  sets=batch['input_ids'].to(device)
  batch_sum=model.generate(sets, max_length=200,no_repeat_ngram_size=3)
  test_sum=[*test_sum,*batch_sum]

test_sum_sent=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in test_sum]

sub=pd.read_csv(data_path+'sample_submission.csv')
sub['summary']=test_sum_sent


# view results
test_real=pd.read_csv(data_path+'test_evi_final.csv')

id=27
print('Context:\n')
display(test_real['title_context'][id])

print('\nsummary:\n')
display(sub['summary'][id])


id=66
print('Context:\n')
display(test_real['title_context'][id])

print('\nsummary:\n')
display(sub['summary'][id])
