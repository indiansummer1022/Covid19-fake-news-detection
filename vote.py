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

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import tree

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


class BertCNNClassifier(nn.Module):
    def __init__(self, tuned_model, embed_num = 512, embed_dim = 768, dropout=0.2, kernel_num=3, kernel_sizes=[1,2], num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.softmax = nn.functional.softmax

        self.bert = tuned_model.bert
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (k, self.embed_dim)) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(len(self.kernel_sizes)*self.kernel_num, self.num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids = None):
        output = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids) #16,512,768
        output = output[0].unsqueeze(1) #16,1,512,768
        output = [nn.functional.relu(conv(output)).squeeze(3) for conv in self.convs] #16,3,508,1 => #16,3,508
        output = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in output] #=> 16,3
        output = torch.cat(output, 1)
        output = self.dropout(output)
        logits = self.classifier(output)
        return self.softmax(logits, 1)

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


# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 80-10-10 train-validation-test split.
# train_size = int(0.8 * len(dataset))
# val_size = int(0.1 * len(dataset))
# test_size = int(0.1 * len(dataset))

test_dataset = Subset(dataset,range(len(train)+len(val),len(dataset)))

print('{:>5,} testing samples'.format(len(test_dataset)))



# dataloader
batch_size = 16

test_dataloader = DataLoader(
            test_dataset,
            shuffle = False,
            batch_size = batch_size 
        )


################### Data loader End ##########################

PATH1 = "bert_finetune.pt"
model1 = torch.load(PATH1,map_location='cpu')
model1 = model1.cuda()

PATH2 = "BERTCNN_finetune.pt"
model2 = torch.load(PATH2,map_location='cpu').cuda()

PATH3 = "BERTLSTM_finetune.pt"
model3 = torch.load(PATH3,map_location='cpu').cuda()

model4 = Pipeline([
        ('bow', CountVectorizer()),  
        ('tfidf', TfidfTransformer()),  
        ("svm_clf", SVC(kernel="rbf", gamma=1, C=10))
    ])
fit = model4.fit(train['tweet'],train['label'])
pred_svm=model4.predict(test['tweet'])

print("Testing...")

model1.eval()
model2.eval()
model3.eval()
total_eval_accuracy = 0
y_true = []
y_pred = []
cnt = 0

for batch in test_dataloader:
    input_ids = batch[0].to(device)
    input_mask = batch[1].to(device)
    labels = batch[2].to(device)
        
    with torch.no_grad():
        #out1 = model1(input_ids, token_type_ids=None, attention_mask=input_mask,labels=labels)
        out2 = model2(input_ids = input_ids, attention_mask = input_mask, token_type_ids = None)
        out3 = model3(input_ids = input_ids, attention_mask = input_mask, token_type_ids = None)

    #pred1 = torch.argmax(out1[1], dim = 1)
    pred1 = np.array(pred_svm[cnt:min(cnt+batch_size,len(pred_svm))])
    pred1 = torch.tensor(pred1).to(device)
    cnt += batch_size
    pred2 = torch.argmax(out2, dim = 1)
    pred3 = torch.argmax(out3, dim = 1)

    pred = pred1 + pred2 + pred3
    pred[pred < 2] = 0
    pred[pred >= 2] = 1


    print(pred1)
    print(pred2)
    print(pred3)
    print(pred)
    print(labels)
    print("=======================================")
    total_eval_accuracy += torch.sum(pred == labels).item()
    y_true.append(labels.flatten())
    y_pred.append(pred.flatten())

        
avg_val_accuracy = total_eval_accuracy / len(test_dataloader.dataset)
print(" Accuracy: {}".format(avg_val_accuracy))
print()

y_true = torch.cat(y_true).tolist()
y_pred = torch.cat(y_pred).tolist()
target_names = ['real','fake']
print(classification_report(y_true, y_pred, target_names=target_names))
print('roc_auc score: ', roc_auc_score(y_true,y_pred))
print('F1 score:',f1_score(y_true, y_pred))
print()
