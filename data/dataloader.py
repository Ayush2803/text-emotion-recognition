import os
import logging
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer
from transformers import RobertaTokenizer


__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')


class MMDataset(Dataset):
    def __init__(self, args, labels, _past=2):
        self.args = args
        self.labels = labels
        self.save = []
        self.inputs=[]
        self.past = _past
        print(os.path.join(args.data_dir, labels))

        # args.data_dir = args.data_dir + '/IEMOCAP/'
        self.df = pd.read_csv(os.path.join(args.data_dir, labels),
                              dtype={'session': str, 'video': str, 'segment': str, 'text': str, 'label': int,
                                     'annotation': str})

        # self.pretrainedBertPath = 'pretrained_model/bert_en'
        # self.tokenizer = BertTokenizer.from_pretrained(self.pretrainedBertPath, do_lower_case=True)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.create_inputs()

    # def __len__(self):
    #     return len(self.df)

    # def __getitem__(self, index):
    #     session, video, segment, text, label, annotation = self.df.loc[index]
    #     text_tokens = self.tokenizer(text, max_length=50, add_special_tokens=True, truncation=True, padding='max_length', return_token_type_ids=True,
    #                                  return_tensors="pt")
    #     return text_tokens['input_ids'], text_tokens['token_type_ids'], text_tokens['attention_mask'], label

    def create_inputs(self):
      for i in range(len(self.df)):
        session, cur_video, segment, cur_text, label, annotation = self.df.loc[i]
        final_text = " "
        for past_idx in range(i-self.past,i):
          if (past_idx<0):
            continue
          session, past_video, segment, past_text, label, annotation = self.df.loc[past_idx]
          if (past_video==cur_video):
            final_text=final_text+past_text
        if (final_text==" "):
          final_text=""
        final_text=final_text+"</s></s>"+cur_text
        self.inputs.append({'text':final_text, 'label':label})

    def __len__(self):
      return len(self.inputs)
    def __getitem__(self,index):
      text=self.inputs[index]['text']
      label=self.inputs[index]['label']
      text_tokens = self.tokenizer(text, max_length=50, add_special_tokens=True, truncation=True, padding='max_length', return_token_type_ids=True,
                                      return_tensors="pt")
      return text_tokens['input_ids'], text_tokens['token_type_ids'], text_tokens['attention_mask'], label
          

def MMDataLoader(args):

    train_set = MMDataset(args, 'train.csv')
    valid_set = MMDataset(args, 'valid.csv')
    test_set = MMDataset(args, 'test.csv')

    print("Train Dataset: ", len(train_set))
    print("Valid Dataset: ", len(valid_set))
    print("Test Dataset: ", len(test_set))

    # print(args.num_workers, args.batch_size)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=False, pin_memory=False, drop_last=True)
    valid_loader = DataLoader(valid_set,  batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=False, pin_memory=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=False, drop_last=True)

    return train_loader, valid_loader, test_loader
