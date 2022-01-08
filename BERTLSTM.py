import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
#import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, AdamW, BertConfig
import gc
from transformers import BertModel
from sklearn.metrics import roc_auc_score,f1_score
import time
import datetime
from transformers import BertTokenizer

import re
from string import punctuation

from torch.utils.data import TensorDataset, random_split, Subset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report

class BertLstmClassifier(nn.Module):
    def __init__(self, model_tune):
        super().__init__()
        self.bert = model_tune.bert
        self.lstm = nn.LSTM(input_size = 768, 
                            hidden_size = 768, 
                            num_layers = 1, 
                            batch_first = True, 
                            bidirectional = True)
        self.classifier = nn.Linear(768 * 2, 2)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        out, _ = self.lstm(bert_output[0])
        logits = self.classifier(out[:, 1, :])
        return self.softmax(logits)


################### Data loader Begin ##########################

def preprocess(data):
    #remove url and hashtag
    for i in range(data.shape[0]):
        text=data[i].lower()
        text1=''.join([word+" " for word in text.split()])
        data[i]=text1
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    space_pattern = '\s+'

    for i in range(data.shape[0]):
        text_string = data[i]
        parsed_text = re.sub(hashtag_regex, '', text_string)
        parsed_text = re.sub(giant_url_regex, '', parsed_text)
        parsed_text = re.sub(mention_regex, '', parsed_text) 
        #remove punctuation
        parsed_text = re.sub(r"[{}]+".format(punctuation), '', parsed_text) 
        parsed_text = re.sub(space_pattern, ' ', parsed_text)
        data[i] = parsed_text
    return data


torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    torch.cuda.manual_seed(0)
    
print("Using GPU: {}".format(use_cuda))

path = "Constraint-English/"
train_filename= path + "Constraint_English_Train.xlsx"
val_filename = path + "Constraint_English_Val.xlsx"
test_filename = path + "Constraint_English_Test.xlsx"
train = pd.read_excel(train_filename)
val = pd.read_excel(val_filename)
test = pd.read_excel(test_filename)

train["label"] = train["label"].map({"real": 1, "fake": 0})
val["label"] = val["label"].map({"real": 1, "fake": 0})
test["label"] = test["label"].map({"real": 1, "fake": 0})

# all the data
data = pd.concat([train, val, test], axis=0, ignore_index=True).drop(["id"], axis=1)

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tweets = data.tweet.values
labels = data.label.values

tweets = preprocess(tweets)

# Print the original sentence.
print(' Original: ', tweets[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(tweets[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets[0])))

# Length of the sentences in dataset
max_len = 0
ind = [100,200,300,400,500,512]
for i in ind:
  count = 0
  for tweet in tweets:
      max_len = max(max_len, len(tweet))
      if len(tweet)>i:
        count+=1
  print("Count of sentence length over {} is: ".format(i), count)
print('Max sentence length: ', max_len)

input_ids = []
attention_masks = []
for tweet in tweets:
    encoded_dict = tokenizer.encode_plus(
                        tweet,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation=True
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])
# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', tweets[0])
print('Token IDs:', input_ids[0])


# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 80-10-10 train-validation-test split.
train_size = len(train)
val_size = len(val)
test_size = len(test)

# random split dataset
#train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],generator=torch.Generator().manual_seed(42))

train_dataset = Subset(dataset,range(train_size))
# bagging
bagging = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=train_size, generator=torch.Generator().manual_seed(256))
val_dataset = Subset(dataset,range(train_size, train_size+val_size))
test_dataset = Subset(dataset,range(train_size+val_size,len(dataset)))

print('{:>5,} training samples'.format(len(train_dataset)))
print('{:>5,} validation samples'.format(len(val_dataset)))
print('{:>5,} testing samples'.format(len(test_dataset)))

# dataloader
batch_size = 16

train_dataloader = DataLoader(
            train_dataset,  
            shuffle = True,
            batch_size = batch_size 
        )

val_dataloader = DataLoader(
            val_dataset,
            shuffle = False,
            batch_size = batch_size 
        )


test_dataloader = DataLoader(
            test_dataset,
            shuffle = False,
            batch_size = batch_size 
        )


################### Data loader End ##########################


PATH1 = "bert_finetune.pt"
the_best_model = torch.load(PATH1,map_location='cpu')

# Initializing model
model = BertLstmClassifier(the_best_model).cuda()
for param in model.bert.parameters():
   param.requires_grad = False
# set parameters
epochs = 4
learning_rate = 5e-5
optimizer = AdamW(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()


import random
import numpy as np

seed_val = 2020

random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []
total_t0 = time.time()
best_accuracy = 0
best_f1 = 0
best_roc = 0

for epoch_i in range(0, epochs):
    #Training
    print("")
    print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()
    for step, batch in enumerate(train_dataloader):

        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)

        model.zero_grad()        
        out = model(input_ids = input_ids, attention_mask = input_mask, token_type_ids = None)
        loss = criterion(out, labels)
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pred = torch.argmax(out, dim = 1)
        total_train_accuracy +=  torch.sum(pred == labels).item()
        
    avg_train_accuracy = total_train_accuracy / len(train_dataloader.dataset)
    avg_train_loss = total_train_loss / len(train_dataloader.dataset)            
    print("  Accuracy: {}".format(avg_train_accuracy))
    print("  Training loss: {}".format(avg_train_loss))


    # Validation
    print("")
    print("Validation...")
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    y_true = []
    y_pred = []

    for batch in val_dataloader:
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        with torch.no_grad():        
            out = model(input_ids = input_ids, attention_mask = input_mask, token_type_ids = None)
        loss = criterion(out, labels)
        total_eval_loss += loss.item()
        pred = torch.argmax(out, dim = 1)
        total_eval_accuracy += torch.sum(pred == labels).item()
        y_true.append(labels.flatten())
        y_pred.append(pred.flatten())
        
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader.dataset)
    print("  Accuracy: {}".format(avg_val_accuracy))
    avg_val_loss = total_eval_loss / len(test_dataloader.dataset)
    print("  Test loss: {}".format(avg_val_loss))
    print()
    
    y_true = torch.cat(y_true).tolist()
    y_pred = torch.cat(y_pred).tolist()
    target_names = ['real','fake']
    print(classification_report(y_true, y_pred, target_names=target_names))
    print('roc_auc score: ', roc_auc_score(y_true,y_pred))
    print('F1 score:',f1_score(y_true, y_pred))
    print()

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Train Accur.': avg_train_accuracy,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
        }
    )
    print()

    if avg_val_accuracy > best_accuracy:
        best_accuracy = avg_val_accuracy
        best_f1 = f1_score(y_true, y_pred)
        best_roc = roc_auc_score(y_true,y_pred)
        best_model = model

print()
print("="*10)

print("Testing...")

best_model.eval()
total_eval_accuracy = 0
total_eval_loss = 0
y_true = []
y_pred = []

for batch in test_dataloader:
    input_ids = batch[0].to(device)
    input_mask = batch[1].to(device)
    labels = batch[2].to(device)
        
    with torch.no_grad():        
        out = best_model(input_ids = input_ids, attention_mask = input_mask, token_type_ids = None)
    loss = criterion(out, labels)
    total_eval_loss += loss.item()
    pred = torch.argmax(out, dim = 1)
    total_eval_accuracy += torch.sum(pred == labels).item()
    y_true.append(labels.flatten())
    y_pred.append(pred.flatten())
        
avg_val_accuracy = total_eval_accuracy / len(test_dataloader.dataset)
print(" Accuracy: {}".format(avg_val_accuracy))
avg_val_loss = total_eval_loss / len(test_dataloader.dataset)
print("  Test loss: {}".format(avg_val_loss))
print()

y_true = torch.cat(y_true).tolist()
y_pred = torch.cat(y_pred).tolist()
target_names = ['real','fake']
print(classification_report(y_true, y_pred, target_names=target_names))
print('roc_auc score: ', roc_auc_score(y_true,y_pred))
print('F1 score:',f1_score(y_true, y_pred))
print()

# save fine tune model for voting
PATH1 = "BERTLSTM_finetune.pt"
torch.save(best_model, PATH1)

