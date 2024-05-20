# Importing stock ml libraries
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel
import logging
logging.basicConfig(level=logging.ERROR)
from torch import cuda
from pyplutchik import plutchik
from torch.utils.data import DataLoader



device = 'cuda' if cuda.is_available() else 'cpu'
#Parameters for fine-tuning distilbert
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 2e-05


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)


class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.tweets
        self.average_score = dataframe.average_score
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        average_score = torch.tensor(self.average_score[index], dtype=torch.float)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'average_score': average_score,
        }
train_params = {'batch_size': 1,
                'shuffle': True,
                'num_workers': 0
                }

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768+8,8)

    def forward(self, input_ids, attention_mask, token_type_ids, average_score):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        # Concatenate tokenized inputs with average_score
        combined = torch.cat((pooler,average_score), dim=1)
        output = self.classifier(combined)
        return output

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def optimizer(model):
    return torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def train(epoch,model):
    model.train()
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        average_score = data['average_score'].to(device, dtype=torch.float)
        outputs = model(ids, mask, token_type_ids, average_score)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        loss.backward()
        optimizer.step()

from sklearn.metrics import classification_report
import numpy as np

def validation(testing_loader,model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            average_score = data['average_score'].to(device, dtype=torch.float)  
    
            outputs = model(ids, mask, token_type_ids, average_score)  # Pass emotion_scores to the model

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    # Convert lists to numpy arrays for easier processing
    fin_targets = np.array(fin_targets)
    fin_outputs = np.array(fin_outputs)

    # Convert binary outputs to class labels
    final_outputs = np.where(fin_outputs >= 0.5, 1, 0)

    # Calculate precision, recall, and F1 score
    report = classification_report(fin_targets, final_outputs, output_dict=True)
    precision = report['weighted avg']['precision']
    f1_score = report['weighted avg']['f1-score']

    # Calculate accuracy manually
    accuracy = np.mean(fin_targets == final_outputs)

    print(f"Precision: {precision}")
    print(f"F1 Score: {f1_score}")
    print(f"Accuracy: {accuracy}")

    return final_outputs, fin_targets

class Emotion:
    def __init__(self, scores):
        # Initialize the emotions dictionary with the given scores
        self.emotions = {
            'anger': scores[0],
            'anticipation': scores[1],
            'disgust': scores[2],
            'fear': scores[3],
            'joy': scores[4],
            'sadness': scores[5],
            'surprise': scores[6],
            'trust': scores[7]
        }


def evaluate_tweet(data,model):
    ids = data['ids'].to(device, dtype=torch.long)
    mask = data['mask'].to(device, dtype=torch.long)
    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long) 
    average_score = data['average_score'].to(device, dtype=torch.float)  # Extract emolex
    outputs = model(ids, mask, token_type_ids,average_score)

    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids,average_score)
        probabilities = torch.sigmoid(outputs).cpu().numpy()
    return probabilities



def convert_percentage(probabilities):
    # Define the emotions
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

    # Print the percentages for each emotion
    for i, emotion in enumerate(emotions):
        print(f"{emotion}: {probabilities[0][i] * 100:.2f}%")
    
def evaluate_data_loader(data_loader,model):
    probability = []
    for item in data_loader:
        probability.append(evaluate_tweet(item,model))
    return probability
