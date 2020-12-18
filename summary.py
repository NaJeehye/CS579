import os
import re
import time
import math
import random
import unicodedata

import numpy as np
import pandas as pd

from tqdm import tqdm

import spacy

from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 28

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def read_data():
    df=pd.read_csv('../archive/news_summary_more.csv')
    
    return df.reset_index(drop=True)

data_df = read_data()

seq_len_headline = 15
seq_len_text = 60

train_df, valid_df = train_test_split(data_df, test_size=0.1, shuffle=True, random_state=28)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)


class Vocabulary:
    def __init__(self, freq_threshold=2, language='en', preprocessor=None, reverse=False):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.tokenizer = spacy.load(language)
        self.freq_threshold = freq_threshold
        self.preprocessor = preprocessor
        self.reverse = reverse

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        if self.reverse:
            return [token.text.lower() for token in self.tokenizer.tokenizer(text)][::-1]
        else:
            return [token.text.lower() for token in self.tokenizer.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = len(self.itos)

        for sentence in sentence_list:
            # Preprocess the sentence using given preprocessor.
            if self.preprocessor:
                sentence = self.preprocessor(sentence)

            for word in self.tokenize(sentence):
                if word in frequencies:
                    frequencies[word] += 1
                else:
                    frequencies[word] = 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(text):
    text = unicode_to_ascii(text.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)

    text = text.strip()
    
    text = re.sub("(\\t)", ' ', text)  #remove escape charecters
    text = re.sub("(\\r)", ' ', text)
    text = re.sub("(\\n)", ' ', text)
    text = re.sub("(__+)", ' ', text)   #remove _ if it occors more than one time consecutively
    text = re.sub("(--+)", ' ', text)   #remove - if it occors more than one time consecutively
    text = re.sub("(~~+)", ' ', text)   #remove ~ if it occors more than one time consecutively
    text = re.sub("(\+\++)", ' ', text)   #remove + if it occors more than one time consecutively
    text = re.sub("(\.\.+)", ' ', text)   #remove . if it occors more than one time consecutively
    text = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', text) #remove <>()|&©ø"',;?~*!
    text = re.sub("(mailto:)", ' ', text)  #remove mailto:
    text = re.sub(r"(\\x9\d)", ' ', text)  #remove \x9* in text
    text = re.sub("([iI][nN][cC]\d+)", 'INC_NUM', text)  #replace INC nums to INC_NUM
    text = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', text)  #replace CM# and CHG# to CM_NUM
    text = re.sub("(\.\s+)", ' ', text)  #remove full stop at end of words(not between)
    text = re.sub("(\-\s+)", ' ', text)  #remove - at end of words(not between)
    text = re.sub("(\:\s+)", ' ', text)  #remove : at end of words(not between)
    text = re.sub("(\s+.\s+)", ' ', text)  #remove any single charecters hanging between 2 spaces

    #Replace any url as such https://abc.xyz.net/browse/sdf-5327 ====> abc.xyz.net
    try:
        url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', text)
        repl_url = url.group(3)
        text = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)',repl_url, text)
    except:
        pass #there might be emails with no url in them

    text = re.sub("(\s+)",' ',text) #remove multiple spaces
    text = re.sub("(\s+.\s+)", ' ', text) #remove any single charecters hanging between 2 spaces
    return text
    return w

# Build vocab using training data
freq_threshold = 1
headline_vocab = Vocabulary(freq_threshold=freq_threshold, language="en", preprocessor=preprocess_sentence)
text_vocab = Vocabulary(freq_threshold=freq_threshold, language="en", preprocessor=preprocess_sentence)

# build vocab for both english and german
headline_vocab.build_vocabulary(train_df["headlines"].tolist())
text_vocab.build_vocabulary(train_df["text"].tolist())

class CustomTranslationDataset(Dataset):    
    def __init__(self, df, headline_vocab, text_vocab):
        super().__init__()
        self.df = df
        self.headline_vocab = headline_vocab
        self.text_vocab = text_vocab
        
    def __len__(self):
        return len(self.df)
    
    def _get_numericalized(self, sentence, vocab):
        """Numericalize given text using prebuilt vocab."""
        numericalized = [vocab.stoi["<sos>"]]
        numericalized.extend(vocab.numericalize(sentence))
        numericalized.append(vocab.stoi["<eos>"])
        return numericalized

    def __getitem__(self, index):
        headline_numericalized = self._get_numericalized(self.df.iloc[index]["headlines"], self.headline_vocab)
        text_numericalized = self._get_numericalized(self.df.iloc[index]["text"], self.text_vocab)

        return torch.tensor(text_numericalized), torch.tensor(headline_numericalized)

class CustomCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src = [item[0] for item in batch]
        src = pad_sequence(src, batch_first=False, padding_value=self.pad_idx)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return src, targets
BATCH_SIZE = 256

# Define dataset and dataloader
train_dataset = CustomTranslationDataset(train_df, headline_vocab, text_vocab)
valid_dataset = CustomTranslationDataset(valid_df, headline_vocab, text_vocab)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    shuffle=False,
    collate_fn=CustomCollate(pad_idx=headline_vocab.stoi["<pad>"])
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    shuffle=False,
    collate_fn=CustomCollate(pad_idx=headline_vocab.stoi["<pad>"])
)


fun_text = np.vectorize(lambda x: text_vocab.itos[x])
fun_headline = np.vectorize(lambda x: headline_vocab.itos[x])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        outputs, (hidden_state, cell_state) = self.lstm(x)
        
        return hidden_state, cell_state

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout=0.2):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden_state, cell_state):
        x = x.unsqueeze(0)
        x = self.embedding(x)
        x = self.dropout(x)
        outputs, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        preds = self.fc(outputs.squeeze(0))
        return preds, hidden_state, cell_state

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        assert self.encoder.hidden_dim == decoder.hidden_dim
        assert self.encoder.n_layers == decoder.n_layers
    
    def forward(self, x, y, teacher_forcing_ratio=0.75):
        
        target_len = y.shape[0]
        batch_size = y.shape[1]
        target_vocab_size = self.decoder.output_dim  # Output dim
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        
        # Encode the source text using encoder
        hidden_state, cell_state = self.encoder(x)
        
        # First input is <sos>
        input = y[0,:]
        
        # Decode the encoded vector using decoder
        for t in range(1, target_len):
            output, hidden_state, cell_state = self.decoder(input, hidden_state, cell_state)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            pred = output.argmax(1)
            input = y[t] if teacher_force else pred
        
        return outputs
    
# Initialize all models
input_dim = len(text_vocab)
output_dim = len(headline_vocab)
emb_dim = 128
hidden_dim = 256
n_layers = 2
dropout = 0.5

encoder = Encoder(input_dim, emb_dim, hidden_dim, n_layers, dropout)
decoder = Decoder(output_dim, emb_dim, hidden_dim, n_layers, dropout)
model = EncoderDecoder(encoder, decoder).to(device)

# Initialized weights as defined in paper
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)

def inference(model, sentence):
    model.eval()
    result = []

    with torch.no_grad():
        sentence = sentence.to(device)
        
        hidden_state, cell_state = model.encoder(sentence)

        # First input to decoder is "<sos>"
        inp = torch.tensor([headline_vocab.stoi["<sos>"]]).to(device)

        # Decode the encoded vector using decoder until max length is reached or <eos> is generated.
        for t in range(1, seq_len_headline):
            output, hidden_state, cell_state = model.decoder(inp, hidden_state, cell_state)
            pred = output.argmax(1)
            if pred == headline_vocab.stoi["<eos>"]:
                break
            result.append(headline_vocab.itos[pred.item()])
            inp = pred
            
    return " ".join(result)

for sample_batch in valid_loader:
    break
    
# Load the best model.
#model_path = "./best_model.pt"
#model.load_state_dict(torch.load(model_path))


def summary(sentence):
    valid_df.iloc[0]["headlines"] =  "TEST"
    valid_df.iloc[0]["text"] = sentence
    return str(inference(model, sample_batch[0][:, 0].reshape(-1, 1)))

if __name__=="__main__":
    print(summary("""Mia is a famous CEO. Vincent is a CEO. \n Mia knows a woman with a weapon."""))
