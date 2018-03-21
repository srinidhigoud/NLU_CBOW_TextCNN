
# coding: utf-8

# # Exercise 6 + Homework 3: MLPs + Dropout + CNNs

# Much of this was taken from DS-GA 1011 course from last semester.

# ### Data things

# We're doing a sentiment classification task. So first load the Stanford Sentiment Treebank data.

# In[1]:

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import re
import random

import pprint
import pickle

pp = pprint.PrettyPrinter()

random.seed(1)
sst_home = 'data/trees'

# Let's do 2-way positive/negative classification instead of 5-way
easy_label_map = {0:0, 1:0, 2:None, 3:1, 4:1}
    # so labels of 0 and 1 in te 5-wayclassificaiton are 0 in the 2-way. 3 and 4 are 1, and 2 is none
    # because we don't have a neautral class. 

PADDING = "<PAD>"
UNKNOWN = "<UNK>"
max_seq_length = 20

def load_sst_data(path):
    data = []
    with open(path) as f:
        for i, line in enumerate(f): 
            example = {}
            example['label'] = easy_label_map[int(line[1])]
            if example['label'] is None:
                continue
            
            # Strip out the parse information and the phrase labels---we don't need those here
            text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            example['text'] = text[1:]
            data.append(example)

    random.seed(1)
    random.shuffle(data)
    return data
     
training_set = load_sst_data(sst_home + '/train.txt')
dev_set = load_sst_data(sst_home + '/dev.txt')
test_set = load_sst_data(sst_home + '/test.txt')


# And extract bag-of-words feature vectors. For speed, we'll only use words that appear at least 25 times in the training set, leaving us with |V|=1254.

# In[2]:

import collections
import numpy as np

def tokenize(string):
    return string.split()

def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['text']))
        
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices, len(vocabulary)

def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for i, dataset in enumerate(datasets):
        for example in dataset:
            example['text_index_sequence'] = torch.zeros(max_seq_length)

            token_sequence = tokenize(example['text'])
            padding = max_seq_length - len(token_sequence)

            for i in range(max_seq_length):
                if i >= len(token_sequence):
                    index = word_indices[PADDING]
                    pass
                else:
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[UNKNOWN]
                example['text_index_sequence'][i] = index

            example['text_index_sequence'] = example['text_index_sequence'].long().view(1,-1)
            example['label'] = torch.LongTensor([example['label']])


word_to_ix, vocab_size = build_dictionary([training_set])
sentences_to_padded_index_sequences(word_to_ix, [training_set, dev_set, test_set])


# We want to feed data to our model in mini-batches so we need a data iterator that will "batchify" the data. We 

# In[4]:

# This is the iterator we'll use during training. 
# It's a generator that gives you one batch at a time.
def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)   
        batch_indices = order[start:start + batch_size]
        yield [source[index] for index in batch_indices]

# This is the iterator we use when we're evaluating our model. 
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        batches.append(batch)
        
    return batches

# The following function gives batches of vectors and labels, 
# these are the inputs to your model and loss function
def get_batch(batch):
    vectors = []
    labels = []
    for dict in batch:
        vectors.append(dict["text_index_sequence"])
        labels.append(dict["label"])
    return vectors, labels


# ### Model time!

# We need to define an evaluation function,

# In[5]:

def evaluate(model, data_iter):
    model.eval()
    correct = 0
    total = 0
    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])
        vectors = Variable((torch.stack(vectors).squeeze()).cuda())
        labels = (torch.stack(labels).squeeze()).cuda()
        output = model(vectors)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return correct / float(total)


# Below is a multilayer perceptron classifier.
# 
# What hyperparameters do you think would work well?

# In[6]:

# A Multi-Layer Perceptron (MLP)
class MLPClassifier(nn.Module): # inheriting from nn.Module!
    
    def __init__(self, input_size, embedding_dim, hidden_dim, num_labels, dropout_prob):
        super(MLPClassifier, self).__init__()
        
        self.embed = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout_prob)
            
        self.linear_1 = nn.Linear(embedding_dim, hidden_dim) 
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, num_labels)
        self.init_weights()
        
    def forward(self, x):
        # Pass the input through your layers in order
        out = self.embed(x)
        out = self.dropout(out)
        out = torch.sum(out, dim=1)
        out = F.relu(self.linear_1(out))
        out = F.relu(self.linear_2(out))
        out = self.dropout(self.linear_3(out))
        return out

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.linear_1, self.linear_2]
        em_layer = [self.embed]
     
        for layer in lin_layers+em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)


# We now define our training loop,

# In[7]:
cudnn.benchmark = True

def training_loop(model, loss, optimizer, training_iter, dev_iter, train_eval_iter):
    step = 0
    legend = []
    for i in range(num_train_steps):
        model.train()
        vectors, labels = get_batch(next(training_iter))
        vectors = Variable((torch.stack(vectors).squeeze()).cuda())
        labels = Variable((torch.stack(labels).squeeze()).cuda())

        model.zero_grad()
        output = model(vectors)

        lossy = loss(output, labels)
        lossy.backward()
        optimizer.step()

        if step % 100 == 0:
            legend.append([evaluate(model, train_eval_iter),evaluate(model, dev_iter)])
            print( "Step %i; Loss %f; Train acc: %f; Dev acc %f" 
                %(step, lossy.data[0], evaluate(model, train_eval_iter), evaluate(model, dev_iter)))

        step += 1
    return legend


# Let's define our hyperparameters

# In[8]:

# Hyper Parameters 
input_size = vocab_size
num_labels = 2
batch_size = 32
num_train_steps = 1000


# Modify these hyperparameters to try to achieve approximately 80% dev accuracy.

# In[9]:

# hidden_dim = 20
# embedding_dim = 300
# learning_rate = 0.001
# dropout_prob = 0.5
hidden_dim = 20
embedding_dim = 100
learning_rate = 0.1
dropout_prob = 0.0


# In[ ]:




# Finally, we can build and train our model!

# In[11]:



def hyperTuningCBOW(hidden_dim,embedding_dim,learning_rate,dropout_prob,legendCBOW,iteratingParam,Param):
    
    model = MLPClassifier(input_size, embedding_dim, hidden_dim, num_labels, dropout_prob)
    model.cuda()
    # Loss and Optimizer
    loss = nn.CrossEntropyLoss().cuda()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    training_iter = data_iter(training_set, batch_size)
    train_eval_iter = eval_iter(training_set[0:500], batch_size)
    dev_iter = eval_iter(dev_set[0:500], batch_size)
    legendCBOW[iteratingParam][Param] = training_loop(model, loss, optimizer, training_iter, dev_iter, train_eval_iter)


# In[137]:

legendCBOW = collections.defaultdict(dict)

print("iterating hidden_dim CBOW")
for h in range(5,50,5):#20
    hyperTuningCBOW(h,embedding_dim,learning_rate,dropout_prob,legendCBOW,'hidden_dim',h)
    pp.pprint(legendCBOW)
print("iterating embedding_dim CBOW")
for e in range(50,500,10):#300
    hyperTuningCBOW(hidden_dim,e,learning_rate,dropout_prob,legendCBOW,'embedding_dim',e)
    pp.pprint(legendCBOW)
print("iterating learning_rate CBOW")
for l in np.linspace(0, 0.1, num=101, endpoint=True):
    hyperTuningCBOW(hidden_dim,embedding_dim,l,dropout_prob,legendCBOW,'learning_rate',l)
    pp.pprint(legendCBOW)
print("iterating dropout_prob CBOW")
for d in np.linspace(0, 0.9, num=10, endpoint=True):
    hyperTuningCBOW(hidden_dim,embedding_dim,learning_rate,d,legendCBOW,'dropout_prob',d)
    pp.pprint(legendCBOW)


# In[ ]:

file_Name = "CBOW"
# open the file for writing
fileObject = open(file_Name,'wb') 

# this writes the object a to the
# file named 'testfile'
pickle.dump(legendCBOW,fileObject)   

# here we close the fileObject
fileObject.close()




# Let's see how it performs on the held out test set,

# In[86]:

# Test the model
# test_iter = eval_iter(test_set, batch_size)
# test_acc = evaluate(model, test_iter)
# print('Accuracy of the CBOW on the test data: %f' % (test_acc))


# <br>
# This model does okay. It doesn't do that well. Lets try and define a Convolutional Neural Network to try and improve performance.

# In[12]:

class TextCNN(nn.Module):
    def __init__(self, input_size, embedding_dim, window_size, n_filters, num_labels, dropout_prob):
        super(TextCNN, self).__init__()
        
        self.embed = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p = dropout_prob)
        self.dropout2 = nn.Dropout(p = dropout_prob)
        self.conv1 = nn.Conv2d(1, n_filters, (window_size, embedding_dim)) 
        self.fc1 = nn.Linear(n_filters, num_labels)
        self.init_weights()
        
    def forward(self, x):
        # Pass the input through your layers in order
        out = self.embed(x)
        out = self.dropout(out)
        out = out.unsqueeze(1)
        out = self.conv1(out).squeeze(3)
        out = F.relu(out)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        out = self.fc1(self.dropout2(out))
        return out

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.fc1]
        em_layer = [self.embed]
     
        for layer in lin_layers+em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)


# Lets train our Conv Net. Lets redefine the hyperparameters here. You need to modify these as well! Try to achieve approximately 80% dev accuracy.

# In[13]:

# window_size = 5
# n_filters = 10
# embedding_dim = 100
# learning_rate = 0.001
# dropout_prob = 0.5

window_size = 2
n_filters = 5
embedding_dim = 100
learning_rate = 0.1
dropout_prob = 0.0


# Lets build and train this model:

def hyperTuningCNN(window_size,n_filters,embedding_dim,learning_rate,dropout_prob,legendCNN,iteratingParam,Param):

    cnn_model = TextCNN(input_size, embedding_dim, window_size, n_filters, num_labels, dropout_prob)
    cnn_model.cuda()
    # Loss and Optimizer
    loss = nn.CrossEntropyLoss().cuda()  
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)

    # Train the model
    training_iter = data_iter(training_set, batch_size)
    train_eval_iter = eval_iter(training_set[0:500], batch_size)
    dev_iter = eval_iter(dev_set[0:500], batch_size)
    legendCNN[iteratingParam][Param] = training_loop(cnn_model, loss, optimizer, training_iter, dev_iter, train_eval_iter)


# In[ ]:

legendCNN = collections.defaultdict(dict)
print("iterating window_size CNN")
for w in range(0,10,1):#range(1,11):
    hyperTuningCNN(w,n_filters,embedding_dim,learning_rate,dropout_prob,legendCNN,'window_size',w)
    pp.pprint(legendCNN)
print("iterating n_filters CNN")
for n in range(0,50,5):#range(10,21):
    hyperTuningCNN(window_size,n,embedding_dim,learning_rate,dropout_prob,legendCNN,'n_filters',n)
    pp.pprint(legendCNN)
print("iterating embedding_dim CNN")
for e in range(50,500,10):#range(50,300,50):
    hyperTuningCNN(window_size,n_filters,e,learning_rate,dropout_prob,legendCNN,'embedding_dim',e)
    pp.pprint(legendCNN)
print("iterating learning_rate CNN")
for l in np.linspace(0, 0.1, num=101, endpoint=True):
    hyperTuningCNN(window_size,n_filters,embedding_dim,l,dropout_prob,legendCNN,'learning_rate',l)
    pp.pprint(legendCNN)
print("iterating dropout_prob CNN")
for d in np.linspace(0, 0.9, num=10, endpoint=True):
    hyperTuningCNN(window_size,n_filters,embedding_dim,learning_rate,d,legendCNN,'dropout_prob',d)
    pp.pprint(legendCNN)


file_Name = "CNN"
# open the file for writing
fileObject = open(file_Name,'wb') 

# this writes the object a to the
# file named 'testfile'
pickle.dump(legendCNN,fileObject)

# here we close the fileObject
fileObject.close()



# Lets evaluate this on the held out test set

# In[111]:

# Test the model
# test_iter = eval_iter(test_set, batch_size)
# test_acc = evaluate(cnn_model, test_iter)
# print('Accuracy of the CNN model on the test data: %f' % (test_acc))


# # Homework 3 (10pts)

# ### Please construct all of your plots in the ipython notebook using something like matplotlib. Provide all answers in the ipython notebook. We will not grade anything other than the ipython notebook

# Questions:
# 
# 1. Provide plots of varying hidden_dim, embedding_dim, LR, and dropout for deep CBOW (0.75pts each). 
# 2. Describe how each hyperparameter affects performance on train and dev (1.5pts total).
# 3. Provide plots of varying embedding_dim, window_size, num_filters, LR, and dropout for CNN (0.6pts for each HP).
# 4. Describe how each hyperparameter affects performance on train and dev (1.5pts total).
# 5. Write down an hyperparameter configuration for CBOW that achieves 80 dev within the first 1000 train steps. Make sure this configuration is run in your ipython notebook when it is submitted (0.5pts).
# 6. Write down an hyperparameter configuration for CNN that achieves 80 dev within the first 1000 train steps. Make sure this configuration is run in your ipython notebook when it is submitted (0.5pts).

# In[18]:


