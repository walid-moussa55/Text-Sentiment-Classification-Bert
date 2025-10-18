
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
model_name = 'bert-base-uncased'
import re

def preprocess_text(text):
    text = text.lower()
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = re.sub(r'[^a-z0-9\s\']', '', text)
    return text
    
class Bert_model(nn.Module):
    def __init__(self, output_size=1, dropout=0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Linear(self.bert.config.hidden_size, output_size)

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
    def prepare_input(self, text):
        text = preprocess_text(text)
        encoded_input = self.tokenizer(
            text,
            return_tensors='pt',  # Return PyTorch tensors
            padding=True,
            truncation=True,
            max_length=128
        )
        return encoded_input['input_ids'], encoded_input['attention_mask']
    def load_model(self, model_path, device='cpu'):
        state_dict = torch.load(model_path, map_location=device)
        self.load_state_dict(state_dict)
    def forward(self, ids, mask):
        output = self.bert(input_ids=ids, attention_mask=mask)
        output_cls = output.last_hidden_state[:, 0, :]
        output_cls = self.dropout(output_cls)
        out = self.net(output_cls)
        out = torch.sigmoid(out)
        return out

    def predict_sentiment(self, text):
        ids, masks = self.prepare_input(text)
        with torch.no_grad():
            output = self.forward(ids,masks)
            pred_labels = (output >= 0.5).int().item()
        print(f"Text : {text}\n Predicted sentiment : { 'positive' if pred_labels ==1 else 'negative' }.")
        return pred_labels
