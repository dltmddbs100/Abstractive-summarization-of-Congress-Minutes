import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class KoBARTSummaryDataset(Dataset):
    def __init__(self, file, tok, max_len, ignore_index=-100,infer=False):
        super().__init__()
        self.tok = tok
        self.max_len = max_len
        self.dec_max_len = 150
        self.docs = pd.read_csv(file)
        self.len = self.docs.shape[0]
        self.pad_index = self.tok.pad_token_id
        self.ignore_index = ignore_index
        self.infer=infer

    def add_padding_data(self, inputs, dec=False):
        if dec:
          if len(inputs) < self.dec_max_len:
            pad = np.array([self.pad_index] *(self.dec_max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
          else:
            inputs = inputs[:self.dec_max_len]
        else:
          if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
          else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.dec_max_len:
            pad = np.array([self.ignore_index] *(self.dec_max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.dec_max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_ids = self.tok.encode(instance['title_context'])
        input_ids.append(self.tok.eos_token_id)
        input_ids.insert(0,0)
        input_ids = self.add_padding_data(input_ids)
        
        if not self.infer:
          label_ids = self.tok.encode(instance['summary'])
          label_ids.append(self.tok.eos_token_id)

          dec_input_ids = [0]
          dec_input_ids += label_ids[:]
          dec_input_ids = self.add_padding_data(dec_input_ids,True)
          label_ids = self.add_ignored_data(label_ids)

          return {'input_ids': torch.tensor(input_ids,dtype=torch.long),
                'attention_mask':torch.tensor(input_ids).ne(self.pad_index).int(),
                'decoder_input_ids': torch.tensor(dec_input_ids,dtype=torch.long),
                'decoder_attention_mask': torch.tensor(dec_input_ids).ne(self.pad_index).int(),
                'labels': torch.tensor(label_ids,dtype=torch.long)}
        else:
          
          return {'input_ids': torch.tensor(input_ids,dtype=torch.long),
                'attention_mask':torch.tensor(input_ids).ne(self.pad_index).int()}
    
    def __len__(self):
        return self.len
