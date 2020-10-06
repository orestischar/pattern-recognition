
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#step 1b

import os
#print(os.listdir('/kaggle/input/multitask-music-classification-2020/patreco3-affective-multitask-music/'))
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import librosa.display


# In[ ]:


#step 1

import matplotlib.pyplot as plt
#label-> blues
spec1 = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/1043.fused.full.npy')
print(spec1.shape)
plt.figure(1)
mel1, chroma1 = spec1[:128], spec1[128:]
librosa.display.specshow(mel1,y_axis='mel')
#label->classical
spec2 = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/10035.fused.full.npy')
print(spec2.shape)
mel2, chroma2 = spec2[:128], spec2[128:]
plt.figure(2)
librosa.display.specshow(mel2,y_axis='mel')


# In[ ]:


#step 2

#label-> blues
spec1 = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/1043.fused.full.npy')
print(spec1.shape)
plt.figure(1)
mel1, chroma1_beat = spec1[:128], spec1[128:]
librosa.display.specshow(mel1,y_axis='mel')
#label->classical
spec2 = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/10035.fused.full.npy')
print(spec2.shape)
mel2, chroma2_beat = spec2[:128], spec2[128:]
plt.figure(2)
librosa.display.specshow(mel2,y_axis='mel')


# In[ ]:


#step 3
plt.figure(3)
librosa.display.specshow(chroma1,y_axis='chroma')
plt.figure(4)
librosa.display.specshow(chroma2,y_axis='chroma')
plt.figure(5)
librosa.display.specshow(chroma1_beat,y_axis='chroma')
plt.figure(6)
librosa.display.specshow(chroma2_beat,y_axis='chroma')


# In[6]:


#step 4

#custom pytorch dataset
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
import re

#split dataset --create train and validation set 
def torch_train_val_split(
        dataset, batch_train, batch_eval,
        val_size=.2, shuffle=True, seed=None):
    #validation->20% of test data
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    #SubsetRandomSampler samples elements randomly from a given list of indices, without replacement.
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    #we use the custom sampler
    train_loader = DataLoader(dataset,
                              batch_size=batch_train,
                              drop_last=True,
                              sampler=train_sampler)
    val_loader = DataLoader(dataset,
                            batch_size=batch_eval,
                            drop_last=True,
                            sampler=val_sampler)
    return train_loader, val_loader

def read_fused_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)
    return spectrogram.T

def read_mel_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[:128]
    return spectrogram.T

    
def read_chromagram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[128:]
    return spectrogram.T


#padding is needed so that all samples have the same length
class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[:self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1

class SpectrogramDataset(Dataset):
    def __init__(self, path, task=None, train=True, max_length=-1, read_spec_fn=read_mel_spectrogram):
        t = 'train' if train else 'test'
        p = os.path.join(path, t)
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, task)
        self.feats = [read_spec_fn(os.path.join(p, f)) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(np.array(labels).astype('float'))

    def get_files_labels(self, txt, task):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split(',') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            if task=='valence':
                label = l[1]
            elif task=='energy':
                label = l[2]
            else:
                label = l[3]
            # Kaggle automatically unzips the npy.gz format so this hack is needed
            _id = l[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
            labels.append(label)
        return files, labels

    def __getitem__(self, item):
        # TODO: Inspect output and comment on how the output is formatted
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], l

    def __len__(self): #returns lenght
        return len(self.labels)


# In[7]:


#custom pytorch dataset
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
import re

# Combine similar classes and remove underrepresented classes
class_mapping = {
    'Rock': 'Rock',
    'Psych-Rock': 'Rock',
    'Indie-Rock': None,
    'Post-Rock': 'Rock',
    'Psych-Folk': 'Folk',
    'Folk': 'Folk',
    'Metal': 'Metal',
    'Punk': 'Metal',
    'Post-Punk': None,
    'Trip-Hop': 'Trip-Hop',
    'Pop': 'Pop',
    'Electronic': 'Electronic',
    'Hip-Hop': 'Hip-Hop',
    'Classical': 'Classical',
    'Blues': 'Blues',
    'Chiptune': 'Electronic',
    'Jazz': 'Jazz',
    'Soundtrack': None,
    'International': None,
    'Old-Time': None
}



class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


class SpectrogramDatasetCategory(Dataset):
    def __init__(self, path, class_mapping=None, train=True, max_length=-1, read_spec_fn=read_mel_spectrogram):
        t = 'train' if train else 'test'
        p = os.path.join(path, t)
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, class_mapping)
        self.feats = [read_spec_fn(os.path.join(p, f)) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(self.label_transformer.fit_transform(labels)).astype('int64')

    def get_files_labels(self, txt, class_mapping):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split('\t') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            label = l[1]
            if class_mapping:
                label = class_mapping[l[1]]
            if not label:
                continue
            # Kaggle automatically unzips the npy.gz format so this hack is needed
            _id = l[0].split('.')[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
            labels.append(label)
        return files, labels

    def __getitem__(self, item):
        # TODO: Inspect output and comment on how the output is formatted
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], l

    def __len__(self):
        return len(self.labels)
    


# In[8]:


#step 4c
import numpy as np
import matplotlib.pyplot as plt

#class histogram before and after class mapping
classes_labels=(np.genfromtxt('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train_labels.txt',dtype='str'))[1:]
unique_classes=set(classes_labels[:,1])
unique_classes_am=set(class_mapping.values())
unique_classes_am.remove(None)
freq_dict= {s:0 for s in unique_classes}
freq_dict_am={s:0 for s in unique_classes_am}
for i in classes_labels[:,1]:
    freq_dict[i]+=1
    if (class_mapping[i]):
        freq_dict_am[class_mapping[i]]+=1
#before the mapping 
print(freq_dict)
fig = plt.figure(figsize = (24,16))
plt.bar(freq_dict.keys(),freq_dict.values(),align='center',color='green',alpha=0.5)
#after mapping
fig = plt.figure(figsize = (24,10))
plt.bar(freq_dict_am.keys(),freq_dict_am.values(),align='center',color='blue',alpha=0.5)


# In[10]:


#for step 5a
#create a pytorch dataset for the mel spectogram

mel_specs = SpectrogramDatasetCategory(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_mel, val_loader_mel = torch_train_val_split(mel_specs, 32 ,32, val_size=.33)
test_loader_mel = SpectrogramDatasetCategory(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
test_loader_mel = DataLoader(test_loader_mel,
                              batch_size=5,
                              drop_last=True)


# In[9]:


#for step 5b
#create a pytorch dataset for the beat-synced mel spectogramm

beat_mel_specs = SpectrogramDatasetCategory(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_beat_mel, val_loader_beat_mel = torch_train_val_split(beat_mel_specs, 32 ,32, val_size=.33)
test_loader_beat_mel = SpectrogramDatasetCategory(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
test_loader_beat_mel = DataLoader(test_loader_beat_mel,
                              batch_size=5,
                              drop_last=True)





# In[11]:


#for step 5c
# create a pytorch dataset for the chromograms
chroma = SpectrogramDatasetCategory(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_chromagram)
train_loader_chroma, val_loader_chroma = torch_train_val_split(chroma, 32 ,32, val_size=.33)
test_loader_chroma = SpectrogramDatasetCategory(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_chromagram)
test_loader_chroma = DataLoader(test_loader_chroma,
                              batch_size=5,
                              drop_last=True)


# In[11]:


#for step 5d
# create a pytorch dataset for the concatenated spectogram & chromogram

fused = SpectrogramDatasetCategory(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_fused_spectrogram)
train_loader_fused, val_loader_fused = torch_train_val_split(fused, 32 ,32, val_size=.33)
test_loader_fused = SpectrogramDatasetCategory(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_fused_spectrogram)
test_loader_fused = DataLoader(test_loader_fused,
                              batch_size=5,
                              drop_last=True)


# In[13]:


#for step 8a
#create a pytorch dataset for the mel spectogram
mel_specs_energy = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset_beat/',
         train=True,
         task='energy',
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_energy, test_loader_energy= torch_train_val_split(mel_specs_energy, 32 ,32, val_size= 0.2 )

mel_specs_valence = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset_beat/',
         train=True,
         task='valence',
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_valence, test_loader_valence = torch_train_val_split(mel_specs_valence, 32 ,32, val_size= 0.2 )

mel_specs_danceability = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset_beat/',
         train=True,
         task='danceability',
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_danceability, test_loader_danceability = torch_train_val_split(mel_specs_danceability, 32 ,32, val_size= 0.2 )


# In[14]:


#step 5

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers,drop_prob=0.2, bidirectional=False):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.direction = 2 if self.bidirectional else 1
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.rnn_size = rnn_size
        #lstm initialization
        self.lstm = nn.LSTM(input_dim,self.rnn_size, num_layers, dropout=drop_prob, batch_first=True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(drop_prob)
        #linear layer /output_dimension = number of classes(10)
        self.fc = nn.Linear(self.feature_size, output_dim)

        
        # Initialize the LSTM, Dropout, Output layers
        

      
    def forward(self, x, lengths,hidden):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
         """
        
        
        
        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network

        lstm_out, hidden = self.lstm(x, hidden)
        batch_size=x.shape[0]
        lstm_out = lstm_out.contiguous().view(batch_size,-1, self.feature_size)
        out = self.fc(lstm_out)
        #get last output for each sequence 
        last_outputs = self.last_timestep(out,lengths,self.bidirectional)
        return last_outputs,hidden

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)
  

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        new_lengths=np.array([length-1 for length in lengths],dtype='long')
        idx = (torch.from_numpy(new_lengths)).view(-1, 1)
        idx = idx.expand(outputs.size(0),outputs.size(2)).unsqueeze(1).to(device)
        return outputs.gather(1, idx).squeeze().to(device)

    def init_hidden(self, batch_size):
        #initialize hidden size 
        weight = next(self.parameters()).data
        hidden = (weight.new(self.direction*self.num_layers, batch_size, self.rnn_size).zero_().to(device),
                      weight.new(self.direction*self.num_layers, batch_size, self.rnn_size).zero_().to(device))
        return hidden


# In[15]:


#we initialize and train our LSTM model--GPU has to be turned on to allow for faster computation times

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train_model(input_dim,output_dim,save_string,data_loader,val_loader=None,category=True,val=True):
    batch_size=32
    rnn_size = 128 #hidden size
    num_layers = 2 

    model = BasicLSTM(input_dim,rnn_size,output_dim,num_layers)
    model.to(device)
    lr=0.0001
    if (category):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.001)
    epochs = 30
    print_every = 200
    valid_loss_min = np.Inf
    loss_values_train = []
    loss_values_val = []
    model.train()
    counter=0



    for i in range(epochs):
        running_loss_train=0.0
        running_loss_val=0.0
        h = model.init_hidden(batch_size)
        train_losses=[]
        for inputs, labels,lengths in data_loader:
            inputs,labels,lengths= inputs.to(device), labels.to(device),lengths.to(device)
            counter += 1
            h = tuple([e.data for e in h])
            model.zero_grad()
            output, h = model(inputs.float(),lengths,h)
            if (category):
                loss = criterion(output.squeeze(), labels.long())
            else:
                loss = criterion(output.squeeze(), labels.float())
            running_loss_train =+ loss.item() * batch_size
            train_losses.append(loss.item())

            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if (counter%print_every == 0 and val):
              val_h = model.init_hidden(batch_size)
              val_losses = []
              model.eval()
              for inp, lab,lens in val_loader:
                  inp,lab,lens=inp.to(device),lab.to(device),lens.to(device)
                  val_h = tuple([each.data for each in val_h])
                  out, val_h = model(inp.float(),lens, val_h)
                  val_loss = criterion(out.squeeze(), lab.long())
                  val_losses.append(val_loss.item())
                  #running_loss_val=+ val_loss.item() * batch_size
              model.train()
              #'./state_mel_beat_dict.pt'
              if np.mean(val_losses) <= valid_loss_min:
                  torch.save(model.state_dict(), save_string)
                  print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                  valid_loss_min = np.mean(val_losses)
        if val:
            model.eval()
            val_h = model.init_hidden(batch_size) 
            validation_losses=[]
            for inp, lab,lens in val_loader:
                      inp,lab,lens=inp.to(device),lab.to(device),lens.to(device)
                      val_h = tuple([each.data for each in val_h])
                      out, val_h = model(inp.float(),lens, val_h)
                      val_loss = criterion(out.squeeze(), lab.long())
                      validation_losses.append(val_loss.item())
                      running_loss_val=+ val_loss.item() * batch_size

            model.train()
        print("Epoch: {}/{}...".format(i+1, epochs),
                "Step: {}...".format(counter),
                "Train Loss: {:.6f}...".format(loss.item()))
        loss_values_train.append(np.mean(train_losses))
        if (val):
            loss_values_val.append(np.mean(validation_losses))    
    plt.plot(range(epochs),loss_values_train)
    if (val):
        plt.plot(range(epochs),loss_values_val)
    if(not val):
        torch.save(model.state_dict(), save_string)


# In[17]:


#step 5a
train_model(128,10,'./state_mel_dict.pt',train_loader_mel,val_loader_mel)


# In[16]:


#step 5b
train_model(128,10,'./state_mel_beat_dict.pt',train_loader_beat_mel,val_loader_beat_mel)


# In[18]:


#step 5c
train_model(12,10,'./chroma_dict.pt',train_loader_chroma,val_loader_chroma)


# In[17]:


#step 5d
train_model(140,'./fused_dict.pt',train_loader_fused,val_loader_fused)


# In[20]:


#step 6
#we test our model
def test(input_dim,save_file,test_loader):
    model = BasicLSTM(input_dim,128,10,2)
    model.to(device)
    criterion=nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(save_file))
    batch_size=5
    test_losses = []
    num_correct = 0
    h = model.init_hidden(batch_size)
    correct=0
    y_pred_test=[]
    y_true=[]
    model.eval()
    for inputs, labels,lengths in test_loader:
          inputs,labels,lengths= inputs.to(device), labels.to(device),lengths.to(device)
          h = tuple([each.data for each in h])
          output, h = model(inputs.float(),lengths, h)
          test_loss = criterion(output.squeeze(), labels.long())
          test_losses.append(test_loss.item())
          pred = output.data.max(1)[1]  # get the index of the max log-probability
          y_pred_test.append(pred.tolist())
          y_true.append(labels.tolist())
          correct += pred.eq(labels.data).sum()

    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
              np.mean(test_losses), correct, len(test_loader.dataset),
              100. * correct / (len(test_loader.dataset))))

    #test on different metrics
    #precision,recall, F1 score for each class
    print(np.array(y_true).shape)
    print(np.array(y_pred_test).shape)
    from sklearn.metrics import classification_report
    print(classification_report(np.array(y_true).flatten(),np.array(y_pred_test).flatten()))


# In[22]:


#model with spectograms as input
test(128,'./state_mel_dict.pt',test_loader_mel)


# In[21]:


#model with beat-synced spectograms as input
test(128,'./state_mel_beat_dict.pt',test_loader_beat_mel)


# In[23]:


#model with chromagrams as input
test(12,'./chroma_dict.pt',test_loader_chroma)


# In[22]:


#model with concatenated spectograms and chromograms as input
test(140,'./fused_dict.pt',test_loader_fused)


# In[25]:


#for step 8c
#We train our model on the three different tasks of the multitask dataset

train_model(128,1,'./state_mel_energy_multi.pt',test_loader_energy,val=False,category=False)


# In[26]:


#for step 8b
train_model(128,1,'./state_mel_valence_multi.pt',test_loader_valence,val=False,category=False)


# In[27]:


#for step 8d
train_model(128,1,'./state_mel_danceability_multi.pt',test_loader_danceability,val=False,category=False)


# In[28]:


#for step 8e
import scipy.stats


def test_lstm_model_multitask(test_loader,save_file):
    model_lstm = BasicLSTM (128,128,1,2)
    model_lstm.to(device)
    model_lstm.load_state_dict(torch.load(save_file))
    batch_size=32
    h = model_lstm.init_hidden(batch_size)
    test_losses = []
    y_pred_test=[]
    y_true=[]
    model_lstm.eval()
    for inputs, labels,lengths in test_loader:
          inputs,labels,lengths= inputs.to(device), labels.to(device),lengths.to(device)
          h = tuple([each.data for each in h])
          output, h = model_lstm(inputs.float(),lengths, h)
          y_pred_test.append(output.data.tolist())
          y_true.append(labels.tolist())
    rho= scipy.stats.spearmanr(np.array(y_true).flatten(),np.array(y_pred_test).flatten()).correlation
    print('\nTest set: Spearman Correlation: {:.6f} \n'.format(rho))
    return rho

rho_energy=test_lstm_model_multitask(test_loader_energy,'./state_mel_energy_multi.pt')
rho_valence=test_lstm_model_multitask(test_loader_valence,'./state_mel_valence_multi.pt')
rho_danceability = test_lstm_model_multitask(test_loader_danceability,'./state_mel_danceability_multi.pt')
print((rho_energy+rho_valence+rho_danceability)/3)


# In[2]:


import torch
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# In[5]:


#defined again

mel_specs_category = SpectrogramDatasetCategory(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_mel, val_loader_mel = torch_train_val_split(mel_specs_category, 16 ,16, val_size= 0 )
test_loader_mel = SpectrogramDatasetCategory(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
test_loader_mel = DataLoader(test_loader_mel,
                              batch_size=5,
                              drop_last=True)


# In[5]:


mel_specs_category = SpectrogramDatasetCategory(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_mel, val_loader_mel = torch_train_val_split(mel_specs_category, 16 ,16, val_size= 0 )
test_loader_mel = SpectrogramDatasetCategory(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
test_loader_mel = DataLoader(test_loader_mel,
                              batch_size=5,
                              drop_last=True)


# In[7]:


#step 7 (b)

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

#CNN parameters initinialization
class FancyCNN(nn.Module):
    def __init__(self):
        super(FancyCNN,self).__init__()
        self._cnn_module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            Flatten()
        
        )
        self._fc_module = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=10240, out_features=1)
        )
        #applies initialize weights to every submodule of nn.Sequential--we are only interested in linear and convolutional layer
        #self.apply(self.initialize_weights)

    def forward(self, x):
        #we want x to have dimensions [batch_size,1,]
        x=x.transpose(1,2)
        x=torch.unsqueeze(x,1)
        for layer in self._cnn_module:
            x = layer(x)
        
        for layer in self._fc_module:
            x=layer(x)
        return x
    def initialize_weights(self,layer)->None:
        if isinstance(layer, nn.Conv2d):
            #weights are drawn from a zero mean Gaussian with std sqrt(2/nj) nj-> number of neurons (kaiming he initialization)
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            #we use xavier initialization
            nn.init.xavier_uniform_(layer.weight)


# In[8]:


import matplotlib.pyplot as plt
def train_cnn(data_loader,save_string,epochs,category=False):
    model_cnn = FancyCNN()
    if category:
        model_cnn._fc_module = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=10240, out_features=10)
        )
    
    batch_size=16
    output_dim = 1

    model_cnn.to(device)
    lr=0.001
    if category:
        criterion=nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_cnn.parameters(), lr=lr,weight_decay=0.0001)

    train_loss_min = np.Inf
    loss_values_train = []
    loss_values_val = []
    model_cnn.train()
    counter=0
    


    model_cnn.train()    
    for i in range(epochs):
        running_loss_train=0.0
        running_loss_val=0.0
        train_losses=[]
        for inputs, labels,lengths in data_loader:
            inputs,labels,lengths= inputs.to(device), labels.to(device),lengths.to(device)
            counter += 1
            model_cnn.zero_grad()
            output =  model_cnn(inputs.float())
            if category:
                loss = criterion(output.squeeze(), labels.long())
            else:
                loss = criterion(output.squeeze(), labels.float())
            running_loss_train =+ loss.item() * batch_size
            train_losses.append(loss.item())

            loss.backward()

            optimizer.step()


        model_cnn.eval()
        model_cnn.train()
        print("Epoch: {}/{}...".format(i+1, epochs),
                "Step: {}...".format(counter),
                "Train Loss: {:.6f}...".format(loss.item()))
        loss_values_train.append(np.mean(train_losses))
        if np.mean(train_losses)<train_loss_min:
            torch.save(model_cnn.state_dict(),save_string)
    plt.plot(range(epochs),loss_values_train)
    #torch.save(model_cnn.state_dict(),save_string)
    




# In[9]:


# we train our cnn model 

train_cnn(train_loader_mel,'state_mel_cnn.pt',epochs=45,category=True)


# In[10]:


#step 7d

from sklearn.metrics import classification_report
# accuracy for simple CNN model

def cnn_test(save_file,test_loader):
    criterion=nn.CrossEntropyLoss()
    model_cnn = FancyCNN()
    model_cnn._fc_module = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=10240, out_features=10)
    )
    model_cnn.to(device)
    model_cnn.load_state_dict(torch.load(save_file))
    batch_size=5
    test_losses = []
    num_correct = 0
    correct=0
    y_pred_test=[]
    y_true=[]
    model_cnn.eval()
    for inputs, labels,lengths in test_loader:
          inputs,labels,lengths= inputs.to(device), labels.to(device),lengths.to(device)
          output = model_cnn(inputs.float())
          test_loss = criterion(output.squeeze(), labels.long())
          test_losses.append(test_loss.item())
          pred = output.data.max(1)[1]  # get the index of the max log-probability
          y_pred_test.append(pred.tolist())
          y_true.append(labels.tolist())
          correct += pred.eq(labels.data).sum()

    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
              np.mean(test_losses), correct, len(test_loader.dataset),
              100. * correct / (len(test_loader.dataset))))
    print(classification_report(np.array(y_true).flatten(),np.array(y_pred_test).flatten()))
cnn_test('./state_mel_cnn.pt',test_loader_mel)


# In[11]:


#for step 8b,c,d
# we train our cnn model on the 3 seperate learning tasks

train_cnn(train_loader_energy,'state_mel_energy_cnn.pt',epochs=30)

train_cnn(train_loader_valence,'state_mel_valence_cnn.pt',epochs=36)

train_cnn(train_loader_danceability,'state_mel_danceability_cnn.pt',epochs=30)



# In[12]:


#step 8e
#we will now test our cnn model for the mutitask dataset
import scipy.stats


def test_cnn_model_multitask(test_loader,save_file):
    model_cnn=FancyCNN()
    model_cnn.to(device)
    model_cnn.load_state_dict(torch.load(save_file))
    batch_size=5
    test_losses = []
    y_pred_test=[]
    y_true=[]
    model_cnn.eval()
    for inputs, labels,lengths in test_loader:
          inputs,labels,lengths= inputs.to(device), labels.to(device),lengths.to(device)
          output = model_cnn(inputs.float())
          y_pred_test.append(output.data.tolist())
          y_true.append(labels.tolist())
    rho= scipy.stats.spearmanr(np.array(y_true).flatten(),np.array(y_pred_test).flatten()).correlation
    print('\nTest set: Spearman Correlation: {:.6f} \n'.format(rho))
    return rho

rho_energy=test_cnn_model_multitask(test_loader_energy,'./state_mel_energy_cnn.pt')
rho_valence=test_cnn_model_multitask(test_loader_valence,'./state_mel_valence_cnn.pt')
rho_danceability = test_cnn_model_multitask(test_loader_danceability,'./state_mel_danceability_cnn.pt')
print((rho_energy+rho_valence+rho_danceability)/3)



# In[ ]:


#step 8 has accumulatively been executed among the previous cells

#step 9a_c has already been executed along with step 7b


# In[13]:


#step 9a_d

#for the transfer learning part we will employ a fine tuning strategy

model_ft = FancyCNN()
model_ft.to(device)
model_ft._fc_module = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=10240, out_features=10)
    )
model_ft.load_state_dict(torch.load('./state_mel_cnn.pt'))

#we can try a freeze approach
for param in model_ft.parameters():
    param.requires_grad = False
model_ft._fc_module = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=10240, out_features=1)
    )

#then we will train our model for a few epochs using the multitask dataset

# we train our cnn model on the 3 seperate learning tasks

train_cnn(train_loader_energy,'state_mel_energy_cnn.pt',epochs=20)

#train_cnn(train_loader_valence,'state_mel_valence_cnn.pt',epochs=20)

#train_cnn(train_loader_danceability,'state_mel_danceability_cnn.pt',epochs=30)


rho_energy=test_cnn_model_multitask(test_loader_energy,'./state_mel_energy_cnn.pt')
#rho_valence=test_cnn_model_multitask(test_loader_valence,'./state_mel_valence_cnn.pt')
#rho_danceability = test_cnn_model_multitask(test_loader_danceability,'./state_mel_danceability_cnn.pt')
#print((rho_energy+rho_valence+rho_danceability)/3)



# In[7]:


#step 9b_b
import matplotlib.pyplot as plt
def train_cnn(data_loader,save_string,epochs):
    model_cnn = FancyCNN()
    batch_size=16
    output_dim = 1

    model_cnn.to(device)
    lr=0.001
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_cnn.parameters(), lr=lr,weight_decay=0.0001)

    train_loss_min = np.Inf
    loss_values_train = []
    loss_values_val = []
    model_cnn.train()
    counter=0
    


    model_cnn.train()    
    for i in range(epochs):
        running_loss_train=0.0
        running_loss_val=0.0
        train_losses=[]
        for inputs, labels,lengths in data_loader:
            inputs,labels,lengths= inputs.to(device), labels.to(device),lengths.to(device)
            counter += 1
            model_cnn.zero_grad()
            output =  model_cnn(inputs.float())
            loss1= criterion(output.squeeze()[:,0], labels.float()[:,0])
            loss2= criterion(output.squeeze()[:,1], labels.float()[:,1])
            loss3= criterion(output.squeeze()[:,2], labels.float()[:,2])
            loss=0.4*loss1+0.3*loss2+0.3*loss3
            running_loss_train =+ loss.item() * batch_size
            train_losses.append(loss.item())

            loss.backward()

            optimizer.step()


        model_cnn.eval()
        model_cnn.train()
        print("Epoch: {}/{}...".format(i+1, epochs),
                "Step: {}...".format(counter),
                "Train Loss: {:.6f}...".format(loss.item()))
        loss_values_train.append(np.mean(train_losses))
        if np.mean(train_losses)<train_loss_min:
            torch.save(model_cnn.state_dict(),save_string)
    plt.plot(range(epochs),loss_values_train)
    #torch.save(model_cnn.state_dict(),save_string)
    
train_cnn(train_loader_multi,'state_mel_dict.pt',50)


# In[8]:


#we will now test our cnn model for the mutitask dataset
import scipy.stats


def test_cnn_model_multitask(test_loader,save_file):
    model_cnn=FancyCNN()
    model_cnn.to(device)
    model_cnn.load_state_dict(torch.load(save_file))
    batch_size=5
    test_losses = []
    y_pred_test_1,y_pred_test_2,y_pred_test_3 = [], [], []
    y_true_1,y_true_2,y_true_3 = [], [], []
    model_cnn.eval()
    for inputs, labels,lengths in test_loader:
          inputs,labels,lengths= inputs.to(device), labels.to(device),lengths.to(device)
          output = model_cnn(inputs.float())
          y_pred_test_1.append(output.data[:,0].tolist())
          y_true_1.append(labels[:,0].tolist())
          y_pred_test_2.append(output.data[:,1].tolist())
          y_true_2.append(labels[:,1].tolist())
          y_pred_test_3.append(output.data[:,2].tolist())
          y_true_3.append(labels[:,2].tolist())
    rho1= scipy.stats.spearmanr(np.array(y_true_1).flatten(),np.array(y_pred_test_1).flatten()).correlation
    rho2= scipy.stats.spearmanr(np.array(y_true_2).flatten(),np.array(y_pred_test_2).flatten()).correlation
    rho3= scipy.stats.spearmanr(np.array(y_true_3).flatten(),np.array(y_pred_test_3).flatten()).correlation
    print('\nTest set: Spearman Correlation: {:.6f} \n'.format((rho1+rho2+rho3)/3))
    print(rho1,rho2,rho3)
test_cnn_model_multitask(test_loader_multi,'state_mel_dict.pt')


# In[9]:


#step 10

class CustomDataset(Dataset):
    def __init__(self, path,max_length=-1, read_spec_fn=read_mel_spectrogram):
        p = os.path.join(path, 'test')
        self.files = self.get_files()
        self.feats = [read_spec_fn(os.path.join(p, f)) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)

    def get_files(self):
        files = []
        for dirname,_,filenames in os.walk('/kaggle/input/patreco3-multitask-affective-music/data/multitask_dataset/test'):
            for filename in filenames:
                npy_file = filename
                files.append(npy_file)
        return files

    def __getitem__(self, item):
        # TODO: Inspect output and comment on how the output is formatted
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.files[item], l

    def __len__(self):
        return len(self.files)


# In[10]:



model_cnn=FancyCNN()
model_cnn.to(device)
model_cnn.load_state_dict(torch.load('state_mel_dict.pt'))
batch_size=1
out = open('solution.txt', "w")

dataset = CustomDataset('../input/patreco3-multitask-affective-music/data/multitask_dataset/')

custom_test_loader = DataLoader(dataset,
                                batch_size=1,
                                drop_last=False)

out.write("Id.fused.full.npy.gz,valence,energy,danceability\n") #Id.fused.full.npy.gz was replaced with Id in the csv file
model_cnn.eval()
for inputs,files,lengths in custom_test_loader:
    inputs,lengths= inputs.to(device),lengths.to(device)
    output = model_cnn(inputs.float())
    out.write('%s,%f,%f,%f\n' % (files[0],output.data[:,0].tolist()[0],output.data[:,1].tolist()[0],output.data[:,2].tolist()[0]))
out.close()
              
    
    

