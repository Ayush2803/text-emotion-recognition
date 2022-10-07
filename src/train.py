"""
AIO -- All Trains in One
"""
import logging
from torch.autograd import Variable
import math
import numpy as np
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from src.utils.functions import dict_to_str
from src.utils.metricsTop import MetricsTop

__all__ = ['WMSATrain']

logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger('MSA')


class WMSATrain():
    def __init__(self, args):
        self.args = args

        self.metrics = MetricsTop().getMetics(args.dataset)
        self.train_epoch = 0
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()  # reduction='none'
        self.entropy = nn.CrossEntropyLoss(weight=torch.tensor(
            [4290/1103, 4290/1084, 4290/1636, 4290/1708]).to(self.args.device))
        # self.entropy = nn.CrossEntropyLoss(weight=torch.tensor([4290 / 1103, 4290 / 1084, 4290 / 1636, 4290 / 1708]).cuda())

    def do_train(self, model, dataloader):
        tfm_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        text_bert_params = list(model.text_model.named_parameters())
        text_bert_params_decay = [
            p for n, p in text_bert_params if not any(nd in n for nd in tfm_no_decay)]
        text_bert_params_no_decay = [
            p for n, p in text_bert_params if any(nd in n for nd in tfm_no_decay)]
        model_params_other = [p for n, p in list(
            model.named_parameters()) if ('text' not in n)]

        optimizer_grouped_parameters = [
            # {'params': text_bert_params_decay,
            #     'weight_decay': self.args.weight_decay_text, 'lr': self.args.lr_bert},
            {'params': text_bert_params_decay,
                'weight_decay': self.args.weight_decay_text, 'lr': self.args.lr_roberta},
            {'params': text_bert_params_no_decay,
                'weight_decay': 0.0, 'lr': self.args.lr_roberta},
            {'params': model_params_other,
                'weight_decay': self.args.weight_decay_other, 'lr': self.args.lr_other}
        ]

        optimizer = optim.Adam(optimizer_grouped_parameters)
        # optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_loader, valid_loader, test_loader = dataloader

        print("train_loader:", len(train_loader))

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else -1e8
        # loop util earlystop
        while True:
            epochs += 1
            # train
            y_pred = []
            y_true = []

            model.train()
            train_loss = 0.0
            # left_epochs = self.args.update_epochs
            with tqdm(train_loader) as td:
                for text_input_ids, text_token_type_ids, text_attention_mask, batch_label in td:
                # for text_input_ids, text_attention_mask, batch_label in td:
                # for text_tokens, batch_label in td:
                    text = text_input_ids.to(self.args.device), text_token_type_ids.to(self.args.device), text_attention_mask.to(
                        self.args.device)
                    # text = text_input_ids.to(
                    #     self.args.device),  text_attention_mask.to(self.args.device)
                    # text = text_tokens.to(self.args.device)
                    label_t = batch_label.to(self.args.device).view(-1)

                    optimizer.zero_grad()
                    # forward
                    outputs = model(text=text)

                    y_pred.append(outputs.cpu())
                    y_true.append(label_t.cpu())

                    # compute loss

                    loss = self.weighted_loss(outputs, label_t)

                    loss.backward()
                    train_loss += loss.item()

                    # update parameters
                    optimizer.step()

            train_loss = train_loss / len(train_loader)
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName,
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss))

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info('Text ER: >> ' + dict_to_str(train_results))

            if epochs < self.args.MIN_Epoch:
                continue

            # validation
            val_results = self.do_test(model, valid_loader, mode="VAL")

            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (
                best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            print(isBetter)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_to_save = model.module if hasattr(
                    model, 'module') else model
                # torch.save(model_to_save.state_dict(), args.model_save_path)
                torch.save(model_to_save.state_dict(),
                           self.args.best_model_save_path)
                model.to(self.args.device)

            if epochs > self.args.MAX_Epoch:
                model_to_save = model.module if hasattr(
                    model, 'module') else model
                torch.save(model_to_save.state_dict(),
                           self.args.best_model_save_path)
                return self.args.MAX_Epoch

            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                self.train_epoch = best_epoch
                return self.train_epoch

    def do_test(self, model, dataloader, mode="VAL"):

        model.eval()
        y_pred = []
        y_true = []

        eval_loss = 0.0
        # criterion = nn.L1Loss()
        with torch.no_grad():
            with tqdm(dataloader) as td:

                for text_input_ids, text_token_type_ids, text_attention_mask, batch_label in td:
                # for text_input_ids, text_attention_mask, batch_label in td:
                    text = text_input_ids.to(self.args.device), text_token_type_ids.to(self.args.device), text_attention_mask.to(
                        self.args.device)
                    # text = text_input_ids.to(self.args.device), text_attention_mask.to(
                    #     self.args.device)
                    labels_m = batch_label.to(self.args.device).view(-1)

                    outputs = model(text=text)

                    loss = self.weighted_loss(outputs, labels_m)
                    eval_loss += loss.item()

                    y_pred.append(outputs.cpu())
                    y_true.append(labels_m.cpu())

        eval_loss = eval_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName +
                    " >> loss: %.4f " % eval_loss)

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        logger.info('Text ER: >> ' + dict_to_str(eval_results))
        eval_results_m = self.metrics(torch.cat(y_pred), torch.cat(y_true))
        len(torch.cat(y_true))

        eval_results_m['EPOCH'] = self.train_epoch
        eval_results_m['Loss'] = eval_loss

        return eval_results_m

    def weighted_loss(self, y_pred, y_true):
        # + self.criterion_mml(y_pred, y_true)
        loss = self.entropy(y_pred, y_true.long())

        return loss  # .sum()
