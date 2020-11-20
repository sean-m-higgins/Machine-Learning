import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from argparse import Namespace

def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))


# -----------------------------------------------------------------
# x = torch.rand(5, 3)			# uniform random

# describe(torch.Tensor(2,3))
# describe(torch.randn(2,3))		# random normal

# describe(torch.zeros(2,3))

# x = torch.ones(2, 3)
# describe(x)

# x.fill_(5)
# describe(x)

# x = torch.Tensor([[1, 2, 3], [4, 5, 6]]) # can be list or numpy array
# describe(x)


# -----------------------------------------------------------------
# convert from numpy to torch

# npy = np.random.rand(2, 3)
# describe(torch.from_numpy(npy))


# -----------------------------------------------------------------
# typecast tensor

# x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# describe(x)

# x = x.long()
# describe(x)

# x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
# describe(x)

# x = x.float()
# describe(x)


# -----------------------------------------------------------------
# tensor operations

# x = torch.randn(2,3)
# describe(x)
# describe(torch.add(x, x))
# describe( x + x )

# # dimension-based tensor operations
# x = torch.arange(6)
# describe(x)

# x = x.view(2,3)
# describe(x)

# describe(torch.sum(x, dim=0))

# describe(torch.sum(x, dim=1))

# describe(torch.transpose(x, 0, 1))


# -----------------------------------------------------------------
# indexing, slicing

# x = torch.arange(6).view(2,3)
# describe(x)

# describe(x[:1, :2])

# describe(x[0,1])

# # get first and third column
# indices = torch.LongTensor([0, 2])		# indices must be LongTensor
# describe(torch.index_select(x, dim=1, index=indices))

# # get first row and first row
# indices = torch.LongTensor([0, 0])
# describe(torch.index_select(x, dim=0, index=indices))

# row_indices = torch.arange(2).long()	#[0,1]

# print("row: " + str(row_indices))
# col_indices = torch.LongTensor([0, 1])	# [0, 0] [1, 1]
# describe(x[row_indices, col_indices])


# -----------------------------------------------------------------
# joining

# x = torch.arange(6).view(2,3)
# describe(x)

# describe(torch.cat([x, x], dim=0))	# like union

# describe(torch.cat([x, x], dim=1))  # like left join

# describe(torch.stack([x, x]))  # becomes 3-d [ x, x ]	x= [ [0, 1, 2], [3, 4, 5] ]


# -----------------------------------------------------------------
# linear algebra operations

# x1 = torch.arange(6).view(2, 3)
# describe(x1)

# x2 = torch.ones(3, 2, dtype=torch.long)
# x2[:, 1] += 1
# describe(x2)

# describe(torch.mm(x1, x2))


# -----------------------------------------------------------------
# Tensors and Computational Graphs

# x = torch.ones(2, 2, requires_grad=True)
# describe(x)
# print(x.grad is None)

# y = (x + 2) * (x + 5) + 3 
# print(x.grad is None)

# z = y.mean() 
# describe(z) 
# z.backward() 
# print(x.grad is None)


# -----------------------------------------------------------------
# CUDA Tensors

# print (torch.cuda.is_available())

# # preferred method: device agnostic tensor instantiation
# # To operate on CUDA and non­CUDA objects, we need to ensure that they are on the same device. 
# # ...If we don’t, the computations will break
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# print (device)

# x = torch.rand(3, 3).to(device) 
# describe(x)

# If you have several CUDA­visible devices (i.e., multiple GPUs), the best practice ...
# ...is to use the CUDA_VISIBLE_DEVICES environment variable when executing the program 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py


# -----------------------------------------------------------------
# Activation Functions

# x = torch.range(-5., 5., 0.1) 
# y = torch.sigmoid(x) 
# plt.plot(x.numpy(), y.numpy()) 
# plt.show()

# x = torch.range(-5., 5., 0.1) 
# y = torch.tanh(x) 
# plt.plot(x.numpy(), y.numpy()) 
# plt.show()

# relu = torch.nn.ReLU()
# x = torch.range(-5., 5., 0.1) 
# y = relu(x)
# plt.plot(x.numpy(), y.numpy()) 
# plt.show()

# prelu = torch.nn.PReLU(num_parameters=1) 
# x = torch.range(-5., 5., 0.1)
# y = prelu(x)
# plt.plot(x.numpy(), y.detach().numpy()) 
# plt.show()

# softmax = nn.Softmax(dim=1) 
# x_input = torch.randn(1, 3) 
# y_output = softmax(x_input) 
# print(x_input)
# print(y_output) 
# print(torch.sum(y_output, dim=1))


# -----------------------------------------------------------------
# Loss Functions

# mse_loss = nn.MSELoss()
# outputs = torch.randn(3, 5, requires_grad=True) 
# targets = torch.randn(3, 5)
# loss = mse_loss(outputs, targets)
# print(loss)

# ce_loss = nn.CrossEntropyLoss()
# outputs = torch.randn(3, 5, requires_grad=True) 
# targets = torch.tensor([1, 0, 3], dtype=torch.int64) 
# loss = ce_loss(outputs, targets)
# print(loss)

# bce_loss = nn.BCELoss()
# sigmoid = nn.Sigmoid()
# probabilities = sigmoid(torch.randn(4, 1, requires_grad=True)) 
# targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32).view(4, 1)
# loss = bce_loss(probabilities, targets) 
# print(probabilities)
# print(loss)



# -----------------------------------------------------------------
# Adam Optimizer

# input_dim = 2 
# lr = 0.001

# Example: DOESNT BELOW WORK!

# perceptron = Perceptron(input_dim=input_dim)
# bce_loss = nn.BCELoss()
# optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)

# # each epoch is a complete pass over the training data 
# for epoch_i in range(n_epochs):
#     # the inner loop is over the batches in the dataset 
#     for batch_i in range(n_batches):
    
#         # Step 0: Get the data
#         x_data, y_target = get_toy_data(batch_size)
        
#         # Step 1: Clear the gradients 
#         perceptron.zero_grad()
        
#         # Step 2: Compute the forward pass of the model
#         y_pred = perceptron(x_data, apply_sigmoid=True)
        
#         # Step 3: Compute the loss value that we wish to optimize 
#         loss = bce_loss(y_pred, y_target)
        
#         # Step 4: Propagate the loss signal backward 
#         loss.backward()
        
#         # Step 5: Trigger the optimizer to perform one update 
#         optimizer.step()



# -----------------------------------------------------------------
# Pytorch Datasets

# class ReviewDataset(Dataset):
#     def __init__(self, review_df, vectorizer):
#         """ 
#         Args:
#             review_df (pandas.DataFrame): the dataset
#             vectorizer (ReviewVectorizer): vectorizer instantiated from dataset
#         """
#         self.review_df = review_df self._vectorizer = vectorizer
#         self.train_df = self.review_df[self.review_df.split=='train'] self.train_size = len(self.train_df)
#         self.val_df = self.review_df[self.review_df.split=='val'] self.validation_size = len(self.val_df)
#         self.test_df = self.review_df[self.review_df.split=='test'] self.test_size = len(self.test_df)
#         self._lookup_dict = {'train': (self.train_df, self.train_size), 
#             'val': (self.val_df, self.validation_size),
#             'test': (self.test_df, self.test_size)} self.set_split('train')
#         self.set_split('train')

#     @classmethod
#     def load_dataset_and_make_vectorizer(cls, review_csv):
#         """Load dataset and make a new vectorizer from scratch
#         Args:
#             review_csv (str): location of the dataset
#         Returns:
#             an instance of ReviewDataset
#         """
#         review_df = pd.read_csv(review_csv)
#         return cls(review_df, ReviewVectorizer.from_dataframe(review_df))

#     def get_vectorizer(self):
#         """ returns the vectorizer """
#         return self._vectorizer
    
#     def set_split(self, split="train"):
#         """ selects the splits in the dataset using a column in the dataframe
#         Args:
#             split (str): one of "train", "val", or "test"
#         """
#         self._target_split = split
#         self._target_df, self._target_size = self._lookup_dict[split]
    
#     def __len__(self):
#         return self._target_size
    
#     def __getitem__(self, index):
#         """the primary entry point method for PyTorch datasets
#         Args:
#             index (int): the index to the data point
#         Returns:
#             a dict of the data point's features (x_data) and label (y_target)
#         """
#         row = self._target_df.iloc[index]
        
#         review_vector = \ 
#             self._vectorizer.vectorize(row.review)
        
#         rating_index = \ 
#             self._vectorizer.rating_vocab.lookup_token(row.rating)
        
#         return {'x_data': review_vector, 'y_target': rating_index}

#     def get_num_batches(self, batch_size):
#         """Given a batch size, return the number of batches in the dataset
#         Args:
#             batch_size (int)
#         Returns:
#             number of batches in the dataset
#         """
#         return len(self) // batch_size

# The Vocabulary class maintains token to integer mapping needed for the rest of the machine learning pipeline
# class Vocabulary(object):
#     """Class to process text and extract Vocabulary for mapping"""
    
#     def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
#         """ 
#         Args:
#             token_to_idx (dict): a pre­existing map of tokens to indices 
#             add_unk (bool): a flag that indicates whether to add the UNK token 
#             unk_token (str): the UNK token to add into the Vocabulary
#         """
#         if token_to_idx is None: 
#             token_to_idx = {}
#         self._token_to_idx = token_to_idx
        
#         self._idx_to_token = {idx: token
#             for token, idx in self._token_to_idx.items()}
        
#         self._add_unk = add_unk 
#         self._unk_token = unk_token
        
#         self.unk_index = -1 
#         if add_unk:
#             self.unk_index = self.add_token(unk_token)
        
#     def to_serializable(self):
#         """ returns a dictionary that can be serialized """
#         return {'token_to_idx': self._token_to_idx, 
#                 'add_unk': self._add_unk, 
#                 'unk_token': self._unk_token}

#     @classmethod
#     def from_serializable(cls, contents):
#         """ instantiates the Vocabulary from a serialized dictionary """
#         return cls(**contents)

#     def add_token(self, token):
#         """Update mapping dicts based on the token.
#         Args:
#             token (str): the item to add into the Vocabulary
#         Returns:
#             index (int): the integer corresponding to the token
#         """
#         if token in self._token_to_idx:
#             index = self._token_to_idx[token]
#         else:
#             index = len(self._token_to_idx) 
#             self._token_to_idx[token] = index 
#             self._idx_to_token[index] = token
#         return index

#     def lookup_token(self, token):
#         """Retrieve the index associated with the token 
#             or the UNK index if token isn't present.
#         Args:
#             token (str): the token to look up
#         Returns:
#             index (int): the index corresponding to the token
#         Notes:
#             `unk_index` needs to be >=0 (having been added into the Vocabulary)
#             for the UNK functionality
#         """
#         if self.add_unk:
#             return self._token_to_idx.get(token, self.unk_index)
#         else:
#             return self._token_to_idx[token]

#     def lookup_index(self, index):
#         """Return the token associated with the index
#         Args:
#             index (int): the index to look up
#         Returns:
#             token (str): the token corresponding to the index
#         Raises:
#             KeyError: if the index is not in the Vocabulary
#         """
#         if index not in self._idx_to_token:
#             raise KeyError("the index (%d) is not in the Vocabulary" % index)
#         return self._idx_to_token[index]
    
#     def __str__(self):
#         return "<Vocabulary(size=%d)>" % len(self)
    
#     def __len__(self):
#         return len(self._token_to_idx)


# The Vectorizer class converts text to numeric vectors
# class ReviewVectorizer(object):
#     """ The Vectorizer which coordinates the Vocabularies and puts them to use"""

#     def __init__(self, review_vocab, rating_vocab):
#         """ 
#         Args:
#             review_vocab (Vocabulary): maps words to integers 
#             rating_vocab (Vocabulary): maps class labels to integers
#         """
#         self.review_vocab = review_vocab 
#         self.rating_vocab = rating_vocab
    
#     def vectorize(self, review):
#         """Create a collapsed one­hit vector for the review
#         Args:
#             review (str): the review
#         Returns:
#             one_hot (np.ndarray): the collapsed one­hot encoding
#         """
#         one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)
#         for token in review.split(" "):
#             if token not in string.punctuation:
#                 one_hot[self.review_vocab.lookup_token(token)] = 1 
#         return one_hot
    
#     @classmethod
#     def from_dataframe(cls, review_df, cutoff=25):
#         """Instantiate the vectorizer from the dataset dataframe
#         Args:
#             review_df (pandas.DataFrame): the review dataset
#             cutoff (int): the parameter for frequency­based filtering
#         Returns:
#             an instance of the ReviewVectorizer
#         """
#         review_vocab = Vocabulary(add_unk=True) 
#         rating_vocab = Vocabulary(add_unk=False)
        
#         # Add ratings
#         for rating in sorted(set(review_df.rating)):
#             rating_vocab.add_token(rating)
        
#         # Add top words if count > provided count 
#         word_counts = Counter()
#         for review in review_df.review:
#             for word in review.split(" "):
#                 if word not in string.punctuation:
#                     word_counts[word] += 1
        
#         for word, count in word_counts.items(): 
#             if count > cutoff:
#                 review_vocab.add_token(word) 
#         return cls(review_vocab, rating_vocab)
    
#     @classmethod
#     def from_serializable(cls, contents):
#         """Intantiate a ReviewVectorizer from a serializable dictionary
#         Args:
#             contents (dict): the serializable dictionary
#         Returns:
#             an instance of the ReviewVectorizer class
#         """
#         review_vocab = Vocabulary.from_serializable(contents['review_vocab']) 
#         rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])
#         return cls(review_vocab=review_vocab, rating_vocab=rating_vocab) 
    
#     def to_serializable(self):
#         """Create the serializable dictionary for caching
#         Returns:
#             contents (dict): the serializable dictionary
#         """
#         return {'review_vocab': self.review_vocab.to_serializable(), 
#             'rating_vocab': self.rating_vocab.to_serializable()}


# -----------------------------------------------------------------
# The DataLoader class is instantiated by providing a PyTorch Dataset 
# ... (such as the ReviewDataset defined for this example), 
# ... a batch_size, and a handful of other keyword arguments.

# def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
#     """
#     A generator function which wraps the PyTorch DataLoader. It will
#         ensure each tensor is on the write device location. 
#     """
#     dataloader = DataLoader(dataset=dataset, batch_size=batch_size, 
#         shuffle=shuffle, drop_last=drop_last)
    
#     for data_dict in dataloader:
#         out_data_dict = {}
#         for name, tensor in data_dict.items():
#             out_data_dict[name] = data_dict[name].to(device) 
#         yield out_data_dict


# A perceptron classifier for classifying Yelp reviews
# class ReviewClassifier(nn.Module):
#     """ a simple perceptron­based classifier """
    
#     def __init__(self, num_features):
#         """ Args:
#             num_features (int): the size of the input feature vector
#         """
#         super(ReviewClassifier, self).__init__() 
#         self.fc1 = nn.Linear(in_features=num_features, out_features=1)
    
#     def forward(self, x_in, apply_sigmoid=False):
#         """The forward pass of the classifier
#         Args:
#             x_in (torch.Tensor): an input data tensor
#                 x_in.shape should be (batch, num_features) 
#             apply_sigmoid (bool): a flag for the sigmoid activation
#                 should be false if used with the cross­entropy losses 
#         Returns:
#             the resulting tensor. tensor.shape should be (batch,).
#         """
#         y_out = self.fc1(x_in).squeeze() 
#         if apply_sigmoid:
#             y_out = F.sigmoid(y_out) 
#         return y_out


# -----------------------------------------------------------------
# Hyperparameters

# args = Namespace(
#     # Data and path information
#     frequency_cutoff=25,
#     model_state_file='model.pth', 
#     review_csv='data/yelp/reviews_with_splits_lite.csv', 
#     save_dir='model_storage/ch3/yelp/', 
#     vectorizer_file='vectorizer.json',
#     # No model hyperparameters
#     # Training hyperparameters
#     batch_size=128,
#     early_stopping_criteria=5,
#     learning_rate=0.001,
#     num_epochs=100,
#     seed=1337,
#     # Runtime options omitted for space
# )

# # BELOW EXAMPLE DOESNT WORK!
# # Instantiating the dataset, model, loss, optimizer, and training state

# def make_train_state(args): 
#         return {'epoch_index': 0, 
#                 'train_loss': [],
#                 'train_acc': [], 
#                 'val_loss': [], 
#                 'val_acc': [], 
#                 'test_loss': ­1, 
#                 'test_acc': ­1}
# train_state = make_train_state(args)

# if not torch.cuda.is_available(): 
#     args.cuda = False
# args.device = torch.device("cuda" if args.cuda else "cpu")

# # dataset and vectorizer
# dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv) 
# vectorizer = dataset.get_vectorizer()

# # model
# classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab)) 
# classifier = classifier.to(args.device)

# # loss and optimizer
# loss_func = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

# # A bare-bones training loop
# for epoch_index in range(args.num_epochs): 
#     train_state['epoch_index'] = epoch_index

#     # Iterate over training dataset

#     # setup: batch generator, set loss and acc to 0, set train mode on 
#     dataset.set_split('train')
#     batch_generator = generate_batches(dataset,
#                                         batch_size=args.batch_size, 
#                                         device=args.device)
#     running_loss = 0.0 
#     running_acc = 0.0 
#     classifier.train()

#     for batch_index, batch_dict in enumerate(batch_generator): 
#         # the training routine is 5 steps:

#         # step 1. zero the gradients 
#         optimizer.zero_grad()

#         # step 2. compute the output
#         y_pred = classifier(x_in=batch_dict['x_data'].float())

#         # step 3. compute the loss
#         loss = loss_func(y_pred, batch_dict['y_target'].float()) 
#         loss_batch = loss.item()
#         running_loss += (loss_batch ­ running_loss) / (batch_index + 1)

#         # step 4. use loss to produce gradients 
#         loss.backward()

#         # step 5. use optimizer to take gradient step 
#         optimizer.step()

#         # ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­----------------------------
#         # compute the accuracy
#         acc_batch = compute_accuracy(y_pred, batch_dict['y_target']) 
#         running_acc += (acc_batch ­ running_acc) / (batch_index + 1)

# train_state['train_loss'].append(running_loss) 
# train_state['train_acc'].append(running_acc)

# # Iterate over val dataset
# # setup: batch generator, set loss and acc to 0, set eval mode on 
# dataset.set_split('val')

# batch_generator = generate_batches(dataset, batch_size=args.batch_size,
#                                     device=args.device)

# running_loss = 0. 
# running_acc = 0. 
# classifier.eval()

# for batch_index, batch_dict in enumerate(batch_generator):

#     # step 1. compute the output
#     y_pred = classifier(x_in=batch_dict['x_data'].float())

#     # step 2. compute the loss
#     loss = loss_func(y_pred, batch_dict['y_target'].float()) 
#     loss_batch = loss.item()
#     running_loss += (loss_batch ­ running_loss) / (batch_index + 1)

#     # step 3. compute the accuracy
#     acc_batch = compute_accuracy(y_pred, batch_dict['y_target']) 
#     running_acc += (acc_batch ­ running_acc) / (batch_index + 1)

# train_state['val_loss'].append(running_loss) 
# train_state['val_acc'].append(running_acc)

# # TEST SET EVALUATION
# dataset.set_split('test')
# batch_generator = generate_batches(dataset,
#                                     batch_size=args.batch_size, 
#                                     device=args.device)

# running_loss = 0. 
# running_acc = 0. 
# classifier.eval()

# for batch_index, batch_dict in enumerate(batch_generator): 
#     # compute the output
#     y_pred = classifier(x_in=batch_dict['x_data'].float())

#     # compute the loss
#     loss = loss_func(y_pred, batch_dict['y_target'].float()) 
#     loss_batch = loss.item()
#     running_loss += (loss_batch ­ running_loss) / (batch_index + 1)

#     # compute the accuracy
#     acc_batch = compute_accuracy(y_pred, batch_dict['y_target']) 
#     running_acc += (acc_batch ­ running_acc) / (batch_index + 1)

# train_state['test_loss'] = running_loss 
# train_state['test_acc'] = running_acc

# print("Test loss: {:.3f}".format(train_state['test_loss'])) 
# print("Test Accuracy: {:.2f}".format(train_state['test_acc']))

# # Printing prediction for a sample review
# def predict_rating(review, classifier, vectorizer, decision_threshold=0.5):
#     """Predict the rating of a review
#     Args:
#         review (str): the text of the review
#         classifier (ReviewClassifier): the trained model
#         vectorizer (ReviewVectorizer): the corresponding vectorizer 
#         decision_threshold (float): The numerical boundary which
#             separates the rating classes
#     """
#     review = preprocess_text(review)
#     vectorized_review = torch.tensor(vectorizer.vectorize(review)) 
#     result = classifier(vectorized_review.view(1, ­1))
    
#     probability_value = F.sigmoid(result).item()
    
#     index = 1
#     if probability_value < decision_threshold:
#         index = 0
    
#     return vectorizer.rating_vocab.lookup_index(index)

# test_review = "this is a pretty awesome book"
# prediction = predict_rating(test_review, classifier, vectorizer) 
# print("{} ­> {}".format(test_review, prediction)

# # Inspecting what the classifier learned
# # Sort weights
# fc1_weights = classifier.fc1.weight.detach()[0]
# _, indices = torch.sort(fc1_weights, dim=0, descending=True) 
# indices = indices.numpy().tolist()

# # Top 20 words
# print("Influential words in Positive Reviews:") 
# print("­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­--------------------------------------") 
# for i in range(20):
#     print(vectorizer.review_vocab.lookup_index(indices[i]))

# # Top 20 negative words
# print("Influential words in Negative Reviews:") 
# print("­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­--------------------------------------") 
# indices.reverse()
# for i in range(20):
#     print(vectorizer.review_vocab.lookup_index(indices[i]))




# -----------------------------------------------------------------
# Multilayer perceptron using PyTorch

# class MultilayerPerceptron(nn.Module):

#     def __init__(self, input_dim, hidden_dim, output_dim):
#         """ 
#         Args:
#             input_dim (int): the size of the input vectors
#             hidden_dim (int): the output size of the first Linear layer 
#             output_dim (int): the output size of the second Linear layer
#         """
#         super(MultilayerPerceptron, self).__init__() 
#         self.fc1 = nn.Linear(input_dim, hidden_dim) 
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x_in, apply_softmax=False):
#         """The forward pass of the MLP
#         Args:
#             x_in (torch.Tensor): an input data tensor
#                 x_in.shape should be (batch, input_dim) 
#             apply_softmax (bool): a flag for the softmax activation
#                 should be false if used with the cross­entropy losses 
#             Returns:
#                 the resulting tensor. tensor.shape should be (batch, output_dim)
#         """
#         intermediate = F.relu(self.fc1(x_in)) 
#         output = self.fc2(intermediate)

#         if apply_softmax:
#             output = F.softmax(output, dim=1)
#         return output

# batch_size = 2 # number of samples input at once 
# input_dim = 3
# hidden_dim = 100
# output_dim = 4

# # Initialize model
# mlp = MultilayerPerceptron(input_dim, hidden_dim, output_dim) 
# print(mlp)


# # Testing the MLP with random inputs
# x_input = torch.rand(batch_size, input_dim) 
# describe(x_input)

# y_output = mlp(x_input, apply_softmax=False) 
# describe(y_output)

# # Producing probabilistic outputs with a multilayer perceptron classifier
# y_output = mlp(x_input, apply_softmax=True) 
# describe(y_output)



# -----------------------------------------------------------------
# Example: Surname Classification with an MLP

# Implementing SurnameDataset.__getitem__()
# class SurnameDataset(Dataset):
#     # Implementation is nearly identical to Example 3­14
#     def __getitem__(self, index):
#         row = self._target_df.iloc[index] 
#         surname_vector = self._vectorizer.vectorize(row.surname) 
#         nationality_index = self._vectorizer.nationality_vocab.lookup_token(row.nationality)
#         return {'x_surname': surname_vector, 'y_nationality': nationality_index}

#     @classmethod
#     def load_dataset_and_make_vectorizer(cls, surname_csv):
#         """Load dataset and make a new vectorizer from scratch
#         Args:
#             surname_csv (str): location of the dataset
#         Returns:
#             an instance of SurnameDataset
#         """
#         surname_df = pd.read_csv(surname_csv)
#         train_surname_df = surname_df[surname_df.split=='train']
#         return cls(surname_df, SurnameVectorizer.from_dataframe(train_surname_df))

#     def get_vectorizer(self):
#         """ returns the vectorizer """
#         return self._vectorizer
    
#     def set_split(self, split="train"):
#         """ selects the splits in the dataset using a column in the dataframe
#         Args:
#             split (str): one of "train", "val", or "test"
#         """
#         self._target_split = split
#         self._target_df, self._target_size = self._lookup_dict[split]
    
#     def __len__(self):
#         return self._target_size
    
#     def __getitem__(self, index):
#         """the primary entry point method for PyTorch datasets
#         Args:
#             index (int): the index to the data point
#         Returns:
#             a dict of the data point's features (x_data) and label (y_target)
#         """
#         row = self._target_df.iloc[index]
        
#         review_vector = \
#             self._vectorizer.vectorize(row.review)
        
#         rating_index = \
#             self._vectorizer.rating_vocab.lookup_token(row.rating)
        
#         return {'x_data': review_vector, 'y_target': rating_index}

#     def get_num_batches(self, batch_size):
#         """Given a batch size, return the number of batches in the dataset
#         Args:
#             batch_size (int)
#         Returns:
#             number of batches in the dataset
#         """
#         return len(self) // batch_size

# # Implementing SurnameVectorizer
# class SurnameVectorizer(object):
#     """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
#     def __init__(self, surname_vocab, nationality_vocab): 
#         self.surname_vocab = surname_vocab 
#         self.nationality_vocab = nationality_vocab

#     def vectorize(self, surname):
#         """Vectorize the provided surname
#         Args:
#             surname (str): the surname
#         Returns:
#             one_hot (np.ndarray): a collapsed one­hot encoding
#         """
#         vocab = self.surname_vocab
#         one_hot = np.zeros(len(vocab), dtype=np.float32) 
#         for token in surname:
#             one_hot[vocab.lookup_token(token)] = 1 
#         return one_hot
    
#     @classmethod
#     def from_dataframe(cls, surname_df):
#         """Instantiate the vectorizer from the dataset dataframe
#         Args:
#             surname_df (pandas.DataFrame): the surnames dataset
#         Returns:
#             an instance of the SurnameVectorizer
#         """
#         surname_vocab = Vocabulary(unk_token="@") 
#         nationality_vocab = Vocabulary(add_unk=False)
#         for index, row in surname_df.iterrows(): 
#             for letter in row.surname:
#                 surname_vocab.add_token(letter) 
#             nationality_vocab.add_token(row.nationality)

#         return cls(surname_vocab, nationality_vocab)

# # The SurnameClassifier using an MLP
# class SurnameClassifier(nn.Module):
#     """ A 2­layer multilayer perceptron for classifying surnames """

#     def __init__(self, input_dim, hidden_dim, output_dim):
#         """ 
#         Args:
#             input_dim (int): the size of the input vectors
#             hidden_dim (int): the output size of the first Linear layer 
#             output_dim (int): the output size of the second Linear layer
#         """
#         super(SurnameClassifier, self).__init__() 
#         self.fc1 = nn.Linear(input_dim, hidden_dim) 
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x_in, apply_softmax=False):
#         """The forward pass of the classifier
#         Args:
#             x_in (torch.Tensor): an input data tensor
#                 x_in.shape should be (batch, input_dim) 
#             apply_softmax (bool): a flag for the softmax activation
#                 should be false if used with the cross­entropy losses 
#         Returns:
#             the resulting tensor. tensor.shape should be (batch, output_dim).
#         """
#         intermediate_vector = F.relu(self.fc1(x_in)) 
#         prediction_vector = self.fc2(intermediate_vector)
#         if apply_softmax:
#             prediction_vector = F.softmax(prediction_vector, dim=1)

#         return prediction_vector

# # Hyperparameters and program options for the MLP­based Yelp review classifier
# args = Namespace(
#     # Data and path information
#     surname_csv="data/surnames/surnames_with_splits.csv", 
#     vectorizer_file="vectorizer.json", 
#     model_state_file="model.pth", 
#     save_dir="model_storage/ch4/surname_mlp",
#     # Model hyper parameters hidden_dim=300
#     # Training hyper parameters 
#     seed=1337,
#     num_epochs=100, 
#     early_stopping_criteria=5, 
#     learning_rate=0.001,
#     batch_size=64,
#     # Runtime options omitted for space
# )

# # Instantiating the dataset, model, loss, and optimizer
# dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv)
# vectorizer = dataset.get_vectorizer()
# classifier = SurnameClassifier(input_dim=len(vectorizer.surname_vocab), 
#                                             hidden_dim=args.hidden_dim,
#                                             output_dim=len(vectorizer.nationality_vocab)) 

# classifier = classifier.to(args.device)

# loss_func = nn.CrossEntropyLoss(dataset.class_weights)
# optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

# # A snippet of the training loop 
# # the training routine is these 5 steps:

# # step 1. zero the gradients
# optimizer.zero_grad()

# # step 2. compute the output
# y_pred = classifier(batch_dict['x_surname'])

# # step 3. compute the loss
# loss = loss_func(y_pred, batch_dict['y_nationality']) 
# loss_batch = loss.to("cpu").item()
# running_loss += (loss_batch - running_loss) / (batch_index + 1)

# # step 4. use loss to produce gradients 
# loss.backward()

# # step 5. use optimizer to take gradient step 
# optimizer.step()

# # Inference using an existing model (classifier): Predicting the nationality given a name
# def predict_nationality(name, classifier, vectorizer): 
#     vectorized_name = vectorizer.vectorize(name) 
#     vectorized_name = torch.tensor(vectorized_name).view(1, -1) 
#     result = classifier(vectorized_name, apply_softmax=True)

#     probability_values, indices = result.max(dim=1) 
#     index = indices.item()

#     predicted_nationality = vectorizer.nationality_vocab.lookup_index(index) 
#     probability_value = probability_values.item()

#     return {'nationality': predicted_nationality, 'probability': probability_value}

# # Predicting the top­k nationalities
# def predict_topk_nationality(name, classifier, vectorizer, k=5): 
#     vectorized_name = vectorizer.vectorize(name)
#     vectorized_name = torch.tensor(vectorized_name).view(1, -1) 
#     prediction_vector = classifier(vectorized_name, apply_softmax=True) 
#     probability_values, indices = torch.topk(prediction_vector, k=k)

#     # returned size is 1,k
#     probability_values = probability_values.detach().numpy()[0] 
#     indices = indices.detach().numpy()[0]

#     results = []
#     for prob_value, index in zip(probability_values, indices):
#         nationality = vectorizer.nationality_vocab.lookup_index(index) 
#         results.append({'nationality': nationality, 'probability': prob_value})

#     return results

# # MLP with dropout
# class MultilayerPerceptron(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         """ 
#         Args:
#             input_dim (int): the size of the input vectors
#             hidden_dim (int): the output size of the first Linear layer 
#             output_dim (int): the output size of the second Linear layer
#         """
#         super(MultilayerPerceptron, self).__init__() 
#         self.fc1 = nn.Linear(input_dim, hidden_dim) 
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x_in, apply_softmax=False):
#         """The forward pass of the MLP
#         Args:
#             x_in (torch.Tensor): an input data tensor
#                 x_in.shape should be (batch, input_dim) 
#             apply_softmax (bool): a flag for the softmax activation
#                 should be false if used with the cross­entropy losses 
#         Returns:
#             the resulting tensor. tensor.shape should be (batch, output_dim).
#         """
#         intermediate = F.relu(self.fc1(x_in))
#         output = self.fc2(F.dropout(intermediate, p=0.5))

#         if apply_softmax:
#             output = F.softmax(output, dim=1)
#         return output



# -----------------------------------------------------------------
# Convolutional Neural Networks

# # Artificial data and using a Conv1d class
# batch_size = 2
# one_hot_size = 10
# sequence_width = 7
# data = torch.randn(batch_size, one_hot_size, sequence_width) 
# conv1 = Conv1d(in_channels=one_hot_size, out_channels=16, kernel_size=3) 
# intermediate1 = conv1(data)
# print(data.size()) 
# print(intermediate1.size())

# conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3) 
# conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
# intermediate2 = conv2(intermediate1) 
# intermediate3 = conv3(intermediate2)
# print(intermediate2.size()) 
# print(intermediate3.size())

# y_output = intermediate3.squeeze()  # This method will drop any dimensions that have size=1 and return the result
# print(y_output.size())

# # Two additional methods for reducing to feature vectors
# # Method 2 of reducing to feature vectors 
# print(intermediate1.view(batch_size, -1).size())
# # Method 3 of reducing to feature vectors 
# print(torch.mean(intermediate1, dim=2).size()) 
# # print(torch.max(intermediate1, dim=2).size()) 
# # print(torch.sum(intermediate1, dim=2).size())


# # Example: Classifying Surnames by Using a CNN

# class SurnameDataset(Dataset):

#     def __getitem__(self, index):
#         row = self._target_df.iloc[index]
#         surname_matrix = self._vectorizer.vectorize(row.surname, self._max_seq_length)
#         nationality_index = self._vectorizer.nationality_vocab.lookup_token(row.nationality)
#         return {'x_surname': surname_matrix, 'y_nationality': nationality_index}

# class SurnameVectorizer(object):
#     """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
#     def vectorize(self, surname):
#         """ 
#         Args:
#             surname (str): the surname 
#         Returns:
#             one_hot_matrix (np.ndarray): a matrix of one­hot vectors
#         """
#         one_hot_matrix_size = (len(self.character_vocab), self.max_surname_length)
#         one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)
#         for position_index, character in enumerate(surname): 
#             character_index = self.character_vocab.lookup_token(character) 
#             one_hot_matrix[character_index][position_index] = 1
#         return one_hot_matrix
    
#     @classmethod
#     def from_dataframe(cls, surname_df):
#         """Instantiate the vectorizer from the dataset dataframe
#         Args:
#             surname_df (pandas.DataFrame): the surnames dataset
#         Returns:
#             an instance of the SurnameVectorizer
#         """
#         character_vocab = Vocabulary(unk_token="@") 
#         nationality_vocab = Vocabulary(add_unk=False) 
#         max_surname_length = 0
#         for index, row in surname_df.iterrows():
#             max_surname_length = max(max_surname_length, len(row.surname)) 
#             for letter in row.surname:
#                 character_vocab.add_token(letter) 
#             nationality_vocab.add_token(row.nationality)
#         return cls(character_vocab, nationality_vocab, max_surname_length)

# class SurnameClassifier(nn.Module):
#     def __init__(self, initial_num_channels, num_classes, num_channels):
#         """ 
#         Args:
#             initial_num_channels (int): size of the incoming feature vector 
#             num_classes (int): size of the output prediction vector 
#             num_channels (int): constant channel size to use throughout network
#         """
#         super(SurnameClassifier, self).__init__()
#         self.convnet = nn.Sequential( 
#             nn.Conv1d(in_channels=initial_num_channels, out_channels=num_channels, kernel_size=3),
#             nn.ELU(), 
#             nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
#             nn.ELU(), 
#             nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
#             nn.ELU(), 
#             nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3),
#             nn.ELU() 
#         )
#         self.fc = nn.Linear(num_channels, num_classes) 
        
#     def forward(self, x_surname, apply_softmax=False):
#         """The forward pass of the classifier
#         Args:
#             x_surname (torch.Tensor): an input data tensor
#                 x_surname.shape should be (batch, initial_num_channels, max_surname_length)
#             apply_softmax (bool): a flag for the softmax activation 
#                 should be false if used with the cross­entropy losses
#         Returns:
#             the resulting tensor. tensor.shape should be (batch, num_classes).
#         """
#         features = self.convnet(x_surname).squeeze(dim=2) 
#         prediction_vector = self.fc(features)

#         if apply_softmax:
#             prediction_vector = F.softmax(prediction_vector, dim=1) 
        
#         return prediction_vector

# args = Namespace(
#     # Data and path information 
#     surname_csv="data/surnames/surnames_with_splits.csv", 
#     vectorizer_file="vectorizer.json", 
#     model_state_file="model.pth", 
#     save_dir="model_storage/ch4/cnn",
#     # Model hyperparameters
#     hidden_dim=100,
#     num_channels=256,
#     # Training hyperparameters
#     seed=1337,
#     learning_rate=0.001,
#     batch_size=128,
#     num_epochs=100,
#     early_stopping_criteria=5,
#     dropout_p=0.1,
#     # Runtime options omitted for space
# )

# # Using the trained model to make predictions
# def predict_nationality(surname, classifier, vectorizer):
#     """Predict the nationality from a new surname
#     Args:
#         surname (str): the surname to classifier
#         classifier (SurnameClassifer): an instance of the classifier 
#         vectorizer (SurnameVectorizer): the corresponding vectorizer
#     Returns:
#         a dictionary with the most likely nationality and its probability
#     """
#     vectorized_surname = vectorizer.vectorize(surname) 
#     vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(0) 
#     result = classifier(vectorized_surname, apply_softmax=True)
    
#     probability_values, indices = result.max(dim=1) 
#     index = indices.item()
    
#     predicted_nationality = vectorizer.nationality_vocab.lookup_index(index) 
#     probability_value = probability_values.item()
#     return {'nationality': predicted_nationality, 'probability': probability_value}


# # Using a Conv1D layer with batch normalization
#     # ...
#     self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=5, stride=1)
#     self.conv1_bn = nn.BatchNorm1d(num_features=10)
#     # ...

# def forward(self, x): 
#     # ...
#     x = F.relu(self.conv1(x)) 
#     x = self.conv1_bn(x)
#     # ...



# -----------------------------------------------------------------
# ... Word Embeddings Example ... (skipped)






# ... TODO









# -----------------------------------------------------------------
# Introduction to Recurrent Neural Networks

# # An implementation of the Elman RNN using PyTorch’s RNNCell
# class ElmanRNN(nn.Module):
#     """ an Elman RNN built using RNNCell """
    
#     def __init__(self, input_size, hidden_size, batch_first=False):
#         """ 
#         Args:
#             input_size (int): size of the input vectors 
#             hidden_size (int): size of the hidden state vectors 
#             batch_first (bool): whether the 0th dimension is batch
#         """
#         super(ElmanRNN, self).__init__()

#         self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        
#         self.batch_first = batch_first 
#         self.hidden_size = hidden_size
    
#     def _initialize_hidden(self, batch_size):
#         return torch.zeros((batch_size, self.hidden_size))
    
#     def forward(self, x_in, initial_hidden=None):
#         """The forward pass of the ElmanRNN
#         Args:
#             x_in (torch.Tensor): an input data tensor.
#                 If self.batch_first: x_in.shape = (batch_size, seq_size, f
#                 Else: x_in.shape = (seq_size, batch_size, feat_size) initial_hidden (torch.Tensor): the initial hidden state for th
#         Returns:
#             hiddens (torch.Tensor): The outputs of the RNN at each time st
#                 If self.batch_first: hiddens.shape = (batch_size, seq_size, hidden_size)
#                 Else: hiddens.shape = (seq_size, batch_size, hidden_size)
#         """
#         if self.batch_first:
#             batch_size, seq_size, feat_size = x_in.size() 
#             x_in = x_in.permute(1, 0, 2)
#         else:
#             seq_size, batch_size, feat_size = x_in.size()
        
#         hiddens = []
        
#         if initial_hidden is None:
#             initial_hidden = self._initialize_hidden(batch_size) 
#             initial_hidden = initial_hidden.to(x_in.device)
        
#         hidden_t = initial_hidden
        
#         for t in range(seq_size):
#             hidden_t = self.rnn_cell(x_in[t], hidden_t) 
#             hiddens.append(hidden_t)
        
#         hiddens = torch.stack(hiddens)
        
#         if self.batch_first:
#             hiddens = hiddens.permute(1, 0, 2)
        
#         return hiddens


# # Example: Classifying Surname Nationality Using a Character RNN

# # ...
# # Implementing the SurnameDataset class

# # ...
# # A vectorizer for surnames

# # Implementing the SurnameClassifier model using an Elman RNN
# class SurnameClassifier(nn.Module):
#     """ An RNN to extract features & an MLP to classify """
    
#     def __init__(self, embedding_size, num_embeddings, num_classes, rnn_hidden_size, batch_first=True, padding_idx=0):
#         """ 
#         Args:
#             embedding_size (int): The size of the character embeddings
#             num_embeddings (int): The number of characters to embed 
#             num_classes (int): The size of the prediction vector
#                 Note: the number of nationalities
#             rnn_hidden_size (int): The size of the RNN's hidden state 
#             batch_first (bool): Informs whether the input tensors will
#                 have batch or the sequence on the 0th dimension 
#             padding_idx (int): The index for the tensor padding;
#                 see torch.nn.Embedding
#         """
#         super(SurnameClassifier, self).__init__()
#         self.emb = nn.Embedding(num_embeddings=num_embeddings, 
#                                 embedding_dim=embedding_size,
#                                 padding_idx=padding_idx) 
#         self.rnn = ElmanRNN(input_size=embedding_size,
#                             hidden_size=rnn_hidden_size,
#                             batch_first=batch_first) 
#         self.fc1 = nn.Linear(in_features=rnn_hidden_size, 
#                             out_features=rnn_hidden_size) 
#         self.fc2 = nn.Linear(in_features=rnn_hidden_size,
#                             out_features=num_classes)
    
#     def forward(self, x_in, x_lengths=None, apply_softmax=False):
#     """The forward pass of the classifier
#     Args:
#         x_in (torch.Tensor): an input data tensor
#             x_in.shape should be (batch, input_dim)
#         x_lengths (torch.Tensor): the lengths of each sequence in the
#             used to find the final vector of each sequence 
#         apply_softmax (bool): a flag for the softmax activation
#             should be false if used with the cross­entropy losses 
#     Returns:
#         out (torch.Tensor); `out.shape = (batch, num_classes)` """
#     x_embedded = self.emb(x_in) 
#     y_out = self.rnn(x_embedded)
    
#     if x_lengths is not None:
#         y_out = column_gather(y_out, x_lengths)
#     else:
#         y_out = y_out[:, ­1, :]
    
#     y_out = F.dropout(y_out, 0.5) 
#     y_out = F.relu(self.fc1(y_out)) 
#     y_out = F.dropout(y_out, 0.5) 
#     y_out = self.fc2(y_out)
    
#     if apply_softmax:
#         y_out = F.softmax(y_out, dim=1)
    
#     return y_out

# # Retrieving the final output vector in each sequence using column_gather()
# def column_gather(y_out, x_lengths):
#     """Get a specific vector from each batch data point in `y_out`.
#     Args:
#         y_out (torch.FloatTensor, torch.cuda.FloatTensor)
#             shape: (batch, sequence, feature)
#         x_lengths (torch.LongTensor, torch.cuda.LongTensor)
#             shape: (batch,)
#     Returns:
#         y_out (torch.FloatTensor, torch.cuda.FloatTensor)
#             shape: (batch, feature)
#     """
#     x_lengths = x_lengths.long().detach().cpu().numpy() - 1
    
#     out = []
#     for batch_index, column_index in enumerate(x_lengths):
#         out.append(y_out[batch_index, column_index]) 
    
#     return torch.stack(out)

# args = Namespace(
# # Data and path information 
# surname_csv="data/surnames/surnames_with_splits.csv", 
# vectorizer_file="vectorizer.json", 
# model_state_file="model.pth", 
# save_dir="model_storage/ch6/surname_classification", 
# # Model hyperparameter
# char_embedding_size=100,
# rnn_hidden_size=64,
# # Training hyperparameter
# num_epochs=100,
# learning_rate=1e­3,
# batch_size=64,
# seed=1337,
# early_stopping_criteria=5,
# # Runtime options omitted for space
# )





# -----------------------------------------------------------------
# Intermediate Sequence Modeling for Natural Language Processing

# # Example: A Character RNN for Generating Surnames


# # ...
# # The SurnameDataset.__getitem__() method for a sequence prediction task


# # ...
# # The code for SurnameVectorizer.vectorize() in a sequence prediction task


# # The unconditioned surname generation model
# class SurnameGenerationModel(nn.Module):
    
#     def __init__(self, char_embedding_size, char_vocab_size, rnn_hidden_size, batch_first=True, padding_idx=0, dropout_p=0.5):
#         """ Args:
#             char_embedding_size (int): The size of the character embeddings 
#             char_vocab_size (int): The number of characters to embed 
#             rnn_hidden_size (int): The size of the RNN's hidden state
#             batch_first (bool): Informs whether the input tensors will
#                 have batch or the sequence on the 0th dimension 
#             padding_idx (int): The index for the tensor padding;
#                 see torch.nn.Embedding
#             dropout_p (float): The probability of zeroing activations using
#                 the dropout method
#         """
#         super(SurnameGenerationModel, self).__init__()
        
#         self.char_emb = nn.Embedding(num_embeddings=char_vocab_size, 
#                                     embedding_dim=char_embedding_size,
#                                     padding_idx=padding_idx) 
#         self.rnn = nn.GRU(input_size=char_embedding_size,
#                             hidden_size=rnn_hidden_size,
#                             batch_first=batch_first) 
#         self.fc = nn.Linear(in_features=rnn_hidden_size,
#                             out_features=char_vocab_size) 
#         self._dropout_p = dropout_p
    
#     def forward(self, x_in, apply_softmax=False):
#         """The forward pass of the model
#         Args:
#             x_in (torch.Tensor): an input data tensor
#                 x_in.shape should be (batch, input_dim) 
#             apply_softmax (bool): a flag for the softmax activation
#                 should be False during training 
#         Returns:
#             the resulting tensor. tensor.shape should be (batch, output_dim).
#         """
#         x_embedded = self.char_emb(x_in) 
#         y_out, _ = self.rnn(x_embedded)
        
#         batch_size, seq_size, feat_size = y_out.shape
#         y_out = y_out.contiguous().view(batch_size * seq_size, feat_size)
        
#         y_out = self.fc(F.dropout(y_out, p=self._dropout_p))
        
#         if apply_softmax:
#             y_out = F.softmax(y_out, dim=1)
        
#         new_feat_size = y_out.shape[­1]
#         y_out = y_out.view(batch_size, seq_size, new_feat_size)
        
#         return y_out


# # The conditioned surname generation model
# class SurnameGenerationModel(nn.Module):
#     def __init__(self, char_embedding_size, char_vocab_size, num_nationalities,
#                 rnn_hidden_size, batch_first=True, padding_idx=0, dropout_p=0. )
#         # ...
#         self.nation_embedding = nn.Embedding(embedding_dim=rnn_hidden_size,
#                                             num_embeddings=num_nationalities)
    
#     def forward(self, x_in, nationality_index, apply_softmax=False): 
#         # ...
#         x_embedded = self.char_embedding(x_in)
#         # hidden_size: (num_layers * num_directions, batch_size, rnn_hidden_siz 
#         nationality_embedded = self.nation_emb(nationality_index).unsqueeze(0) 
#         y_out, _ = self.rnn(x_embedded, nationality_embedded)
#         # ...


# # Handling three­dimensional tensors and sequence­wide loss computations
# def normalize_sizes(y_pred, y_true):
#     """Normalize tensor sizes
#     Args:
#         y_pred (torch.Tensor): the output of the model
#             If a 3­dimensional tensor, reshapes to a matrix 
#         y_true (torch.Tensor): the target predictions
#             If a matrix, reshapes to be a vector
#     """
#     if len(y_pred.size()) == 3:
#         y_pred = y_pred.contiguous().view(­1, y_pred.size(2))
#     if len(y_true.size()) == 2:
#         y_true = y_true.contiguous().view(­1)
#     return y_pred, y_true

# def sequence_loss(y_pred, y_true, mask_index):
#     y_pred, y_true = normalize_sizes(y_pred, y_true)
#     return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)


# args = Namespace(
# # Data and path information 
# surname_csv="data/surnames/surnames_with_splits.csv",
# vectorizer_file="vectorizer.json",
# model_state_file="model.pth", save_dir="model_storage/ch7/model1_unconditioned_surname_generation",
# # or: save_dir="model_storage/ch7/model2_conditioned_surname_generation", 
# # Model hyperparameters
# char_embedding_size=32,
# rnn_hidden_size=32,
# # Training hyperparameters
# seed=1337,
# learning_rate=0.001,
# batch_size=128,
# num_epochs=100,
# early_stopping_criteria=5,
# # Runtime options omitted for space
# )


# # Sampling from the unconditioned generation model
# def sample_from_model(model, vectorizer, num_samples=1, sample_size=20, temperature=1.0):
#     """Sample a sequence of indices from the model
#     Args:
#         model (SurnameGenerationModel): the trained model
#         vectorizer (SurnameVectorizer): the corresponding vectorizer num_samples (int): the number of samples
#         sample_size (int): the max length of the samples
#         temperature (float): accentuates or flattens the distribution
#             0.0 < temperature < 1.0 will make it peakier
#             temperature > 1.0 will make it more uniform 
#     Returns:
#         indices (torch.Tensor): the matrix of indices 
#         shape = (num_samples, sample_size)
#     """
#     begin_seq_index = [vectorizer.char_vocab.begin_seq_index for _ in range(num_samples)]
#     begin_seq_index = torch.tensor(begin_seq_index, dtype=torch.int64).unsqueeze(dim=1)
#     indices = [begin_seq_index] 
#     h_t = None
    
#     for time_step in range(sample_size):
#         x_t = indices[time_step]
#         x_emb_t = model.char_emb(x_t)
#         rnn_out_t, h_t = model.rnn(x_emb_t, h_t)
#         prediction_vector = model.fc(rnn_out_t.squeeze(dim=1)) 
#         probability_vector = F.softmax(prediction_vector / temperature, dim=1)
#         indices.append(torch.multinomial(probability_vector, num_samples=1)) 
#     indices = torch.stack(indices).squeeze().permute(1, 0)
#     return indices


# # Mapping sampled indices to surname strings
# def decode_samples(sampled_indices, vectorizer):
#     """Transform indices into the string form of a surname
#     Args:
#         sampled_indices (torch.Tensor): the indices from `sample_from_model` 
#         vectorizer (SurnameVectorizer): the corresponding vectorizer
#     """
#     decoded_surnames = []
#     vocab = vectorizer.char_vocab
    
#     for sample_index in range(sampled_indices.shape[0]): 
#         surname = ""
#         for time_step in range(sampled_indices.shape[1]):
#             sample_item = sampled_indices[sample_index, time_step].item() 
#             if sample_item == vocab.begin_seq_index:
#                 continue
#             elif sample_item == vocab.end_seq_index:
#                 break 
#             else:
#                 surname += vocab.lookup_index(sample_item) 
#         decoded_surnames.append(surname)
#     return decoded_surnames


# # Sampling from the unconditioned model
# samples = sample_from_model(unconditioned_model, vectorizer, num_samples=10)
# decode_samples(samples, vectorizer)


# # Sampling from a sequence model
# def sample_from_model(model, vectorizer, nationalities, sample_size=20, temperature=1.0):
#     """Sample a sequence of indices from the model
#     Args:
#         model (SurnameGenerationModel): the trained model
#         vectorizer (SurnameVectorizer): the corresponding vectorizer 
#         nationalities (list): a list of integers representing nationalities 
#         sample_size (int): the max length of the samples
#         temperature (float): accentuates or flattens the distribution
#             0.0 < temperature < 1.0 will make it peakier
#             temperature > 1.0 will make it more uniform 
#     Returns:
#         indices (torch.Tensor): the matrix of indices 
#         shape = (num_samples, sample_size)
#     """
#     num_samples = len(nationalities)
#     begin_seq_index = [vectorizer.char_vocab.begin_seq_index for _ in range(num_samples)] 
#     begin_seq_index = torch.tensor(begin_seq_index, dtype=torch.int64).unsqueeze(dim=1)
    
#     indices = [begin_seq_index]
#     nationality_indices = torch.tensor(nationalities, dtype=torch.int64).unsqueeze(dim=0) 
#     h_t = model.nation_emb(nationality_indices)
    
#     for time_step in range(sample_size):
#         x_t = indices[time_step]
#         x_emb_t = model.char_emb(x_t)
#         rnn_out_t, h_t = model.rnn(x_emb_t, h_t)
#         prediction_vector = model.fc(rnn_out_t.squeeze(dim=1)) 
#         probability_vector = F.softmax(prediction_vector / temperature, dim=1) 
#         indices.append(torch.multinomial(probability_vector, num_samples=1))
#     indices = torch.stack(indices).squeeze().permute(1, 0) 
#     return indices


# # Sampling from the conditioned SurnameGenerationModel (not all outputs are shown)
# for index in range(len(vectorizer.nationality_vocab)):
#     nationality = vectorizer.nationality_vocab.lookup_index(index)
#     print("Sampled for {}: ".format(nationality))
#     sampled_indices = sample_from_model(model=conditioned_model, vectorizer=vectorizer,
#                                         nationalities=[index] * 3, temperature=0.7)
#     for sampled_surname in decode_samples(sampled_indices, vectorizer):
#         print("­ " + sampled_surname)



# # Tips and Tricks for Training Sequence Models
# # When possible, use the gated variants
# # When possible, prefer GRUs over LSTMs
# # Use Adam as your optimizer
# # Gradient clipping
# # ...Applying gradient clipping in PyTorch
# # define your sequence model 
# model = ..
# # define loss function 
# loss_function = ..
# # training loop 
# for _ in ...:
#     ...
#     model.zero_grad()
#     output, hidden = model(data, hidden)
#     loss = loss_function(output, targets)
#     loss.backward() 
#     torch.nn.utils.clip_grad_norm(model.parameters(), 0.25) 
#     ...






# -----------------------------------------------------------------
# Advanced Sequence Modeling for Natural Language Processing

# # Constructing the NMTVectorizer

# class NMTVectorizer(object):
#     """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    
#     def __init__(self, source_vocab, target_vocab, max_source_length, max_target_length):
#         """ Args:
#             source_vocab (SequenceVocabulary): maps source words to integers
#             target_vocab (SequenceVocabulary): maps target words to integers
#             max_source_length (int): the longest sequence in the source dataset
#             max_target_length (int): the longest sequence in the target dataset
#         """
#         self.source_vocab = source_vocab 
#         self.target_vocab = target_vocab
#         self.max_source_length = max_source_length 
#         self.max_target_length = max_target_length
    
#     @classmethod
#     def from_dataframe(cls, bitext_df):
#         """Instantiate the vectorizer from the dataset dataframe
#         Args:
#             bitext_df (pandas.DataFrame): the parallel text dataset
#         Returns:
#             an instance of the NMTVectorizer
#         """
#         source_vocab = SequenceVocabulary() 
#         target_vocab = SequenceVocabulary() 
#         max_source_length, max_target_length = 0, 0
        
#         for _, row in bitext_df.iterrows():
#             source_tokens = row["source_language"].split(" ") 
#             if len(source_tokens) > max_source_length:
#                 max_source_length = len(source_tokens) 
#             for token in source_tokens:
#                 source_vocab.add_token(token)
        
#             target_tokens = row["target_language"].split(" ") 
#             if len(target_tokens) > max_target_length: 
#                 max_target_length = len(target_tokens)
#             for token in target_tokens: 
#                 target_vocab.add_token(token)
        
#         return cls(source_vocab, target_vocab, max_source_length, max_target_length)


# # The vectorization functions in the NMTVectorizer
# class NMTVectorizer(object):
#     """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    
#     def _vectorize(self, indices, vector_length=­1, mask_index=0):
#         """Vectorize the provided indices
#         Args:
#             indices (list): a list of integers that represent a sequence 
#             vector_length (int): forces the length of the index vector 
#             mask_index (int): the mask_index to use; almost always 0
#         """
#         if vector_length < 0: 
#             vector_length = len(indices)
#         vector = np.zeros(vector_length, dtype=np.int64) 
#         vector[:len(indices)] = indices 
#         vector[len(indices):] = mask_index
#         return vector
    
#     def _get_source_indices(self, text):
#         """Return the vectorized source text 
#         Args:
#             text (str): the source text; tokens should be separated by spaces 
#         Returns:
#             indices (list): list of integers representing the text
#         """
#         indices = [self.source_vocab.begin_seq_index] 
#         indices.extend(self.source_vocab.lookup_token(token)
#             for token in text.split(" ")) 
#         indices.append(self.source_vocab.end_seq_index)
#         return indices
    
#     def _get_target_indices(self, text):
#         """Return the vectorized source text
#         Args:
#             text (str): the source text; tokens should be separated by spaces
#         Returns:
#             a tuple: (x_indices, y_indices)
#                 x_indices (list): list of ints; observations in target decoder 
#                 y_indices (list): list of ints; predictions in target decoder
#         """
#         indices = [self.target_vocab.lookup_token(token) for token in text.split(" ")]
#         x_indices = [self.target_vocab.begin_seq_index] + indices 
#         y_indices = indices + [self.target_vocab.end_seq_index] 
#         return x_indices, y_indices
    
#     def vectorize(self, source_text, target_text, use_dataset_max_lengths=True)
#         """Return the vectorized source and target text
#         Args:
#             source_text (str): text from the source language
#             target_text (str): text from the target language 
#             use_dataset_max_lengths (bool): whether to use the max vector lengt
#         Returns:
#             The vectorized data point as a dictionary with the keys:
#                 source_vector, target_x_vector, target_y_vector, source_length
#         """
#         source_vector_length = ­1 
#         target_vector_length = ­1
        
#         if use_dataset_max_lengths:
#             source_vector_length = self.max_source_length + 2 
#             target_vector_length = self.max_target_length + 1
        
#         source_indices = self._get_source_indices(source_text) 
#         source_vector = self._vectorize(source_indices,
#             vector_length=source_vector_length, 
#             mask_index=self.source_vocab.mask_index
        
#         target_x_indices, target_y_indices = self._get_target_indices 
#         (target_text)
#         target_x_vector = self._vectorize(target_x_indices,
#             vector_length=target_vector_length,
#             mask_index=self.target_vocab.mask_index)
#         target_y_vector = self._vectorize(target_y_indices,
#             vector_length=target_vector_length,
#             mask_index=self.target_vocab.mask_index) 
#         return {"source_vector": source_vector,
#                 "target_x_vector": target_x_vector, 
#                 "target_y_vector": target_y_vector, 
#                 "source_length": len(source_indices)}


# # Generating minibatches for the NMT example
# def generate_nmt_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
#     """A generator function which wraps the PyTorch DataLoader; NMT version """
#     dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    
#     for data_dict in dataloader:
#         lengths = data_dict['x_source_length'].numpy() 
#         sorted_length_indices = lengths.argsort()[::­1].tolist()
    
#         out_data_dict = {}
#         for name, tensor in data_dict.items():
#             out_data_dict[name] = data_dict[name][sorted_length_indices].to(dev 
#     yield out_data_dict



# # The NMTModel encapsulates and coordinates the encoder and decoder in a single forward() method
# class NMTModel(nn.Module):
#     """ A Neural Machine Translation Model """
#     def __init__(self, source_vocab_size, source_embedding_size, target_vocab_size, target_embedding_size, encoding_size, target_bos_index):
#         """ 
#         Args:
#             source_vocab_size (int): number of unique words in source language
#             source_embedding_size (int): size of the source embedding vectors
#             target_vocab_size (int): number of unique words in target language
#             target_embedding_size (int): size of the target embedding vectors
#             encoding_size (int): size of the encoder RNN 
#             target_bos_index (int): index for BEGIN­OF­SEQUENCE token
#         """
#         super(NMTModel, self).__init__()
#         self.encoder = NMTEncoder(num_embeddings=source_vocab_size,
#             embedding_size=source_embedding_size,
#             rnn_hidden_size=encoding_size)
#         decoding_size = encoding_size * 2
#         self.decoder = NMTDecoder(num_embeddings=target_vocab_size, 
#             embedding_size=target_embedding_size,
#             rnn_hidden_size=decoding_size, 
#             bos_index=target_bos_index)

#     def forward(self, x_source, x_source_lengths, target_sequence):
#         """The forward pass of the model
#         Args:
#             x_source (torch.Tensor): the source text data tensor
#                 x_source.shape should be (batch, vectorizer.max_source_length) 
#             x_source_lengths torch.Tensor): the length of the sequences in x_so
#             target_sequence (torch.Tensor): the target text data tensor
#         Returns:
#             decoded_states (torch.Tensor): prediction vectors at each output st
#         """
#         encoder_state, final_hidden_states = self.encoder(x_source, x_source_lengths)
#         decoded_states = self.decoder(encoder_state=encoder_state, 
#             initial_hidden_state=final_hidden_states,
#             target_sequence=target_sequence)
#         return decoded_states


# # The encoder embeds the source words and extracts features with a bi-GRU
# class NMTEncoder(nn.Module):
#     def __init__(self, num_embeddings, embedding_size, rnn_hidden_size):
#         """ 
#         Args:
#             num_embeddings (int): size of source vocabulary
#             embedding_size (int): size of the embedding vectors 
#             rnn_hidden_size (int): size of the RNN hidden state vectors
#         """
#         super(NMTEncoder, self).__init__()
#         self.source_embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=0)
#         self.birnn = nn.GRU(embedding_size, rnn_hidden_size, bidirectional=True batch_first=True)
    
#     def forward(self, x_source, x_lengths):
#         """The forward pass of the model
#         Args:
#             x_source (torch.Tensor): the input data tensor
#                 x_source.shape is (batch, seq_size)
#             x_lengths (torch.Tensor): vector of lengths for each item in batch
#         Returns:
#             a tuple: x_unpacked (torch.Tensor), x_birnn_h (torch.Tensor)
#                 x_unpacked.shape = (batch, seq_size, rnn_hidden_size * 2) 
#                 x_birnn_h.shape = (batch, rnn_hidden_size * 2)
#         """
#         x_embedded = self.source_embedding(x_source)
#         # create PackedSequence; x_packed.data.shape=(number_items,
#         #
#         x_lengths = x_lengths.detach().cpu().numpy()
#         x_packed = pack_padded_sequence(x_embedded, x_lengths, batch_first=True
#         # x_birnn_h.shape = (num_rnn, batch_size, feature_size) 
#         x_birnn_out, x_birnn_h = self.birnn(x_packed)
#         # permute to (batch_size, num_rnn, feature_size) 
#         x_birnn_h = x_birnn_h.permute(1, 0, 2)
#         # flatten features; reshape to (batch_size, num_rnn * feature_size)
#         # (recall: -1 takes the remaining positions,
#         #       flattening the two RNN hidden vectors into 1)
#         x_birnn_h =­ x_birnn_h.contiguous().view(x_birnn_h.size(0), ­1)
        
#         x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True)
#         return x_unpacked, x_birnn_h



# # A simple demonstration of packed_padded_sequences and pad_packed_sequences
# abcd_padded = torch.tensor([1, 2, 3, 4], dtype=torch.float32) 
# efg_padded = torch.tensor([5, 6, 7, 0], dtype=torch.float32) 
# h_padded = torch.tensor([8, 0, 0, 0], dtype=torch.float32)

# padded_tensor = torch.stack([abcd_padded, efg_padded, h_padded]) 
# describe(padded_tensor)

# lengths = [4, 3, 1]
# packed_tensor = pack_padded_sequence(padded_tensor, lengths, batch_first=True)
# packed_tensor

# unpacked_tensor, unpacked_lengths = \ 
#     pad_packed_sequence(packed_tensor, batch_first=True)
# describe(unpacked_tensor) 
# describe(unpacked_lengths)


# # The NMTDecoder constructs a target sentence from the encoded source sentence
# class NMTDecoder(nn.Module):
#     def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, bos_ind
#     """ 
#     Args:
#         num_embeddings (int): number of embeddings; also the number of unique words in the target vocabulary
#         embedding_size (int): size of the embedding vector 
#         rnn_hidden_size (int): size of the hidden RNN state 
#         bos_index(int): BEGIN­OF­SEQUENCE index
#     """
#     super(NMTDecoder, self).__init__()
#     self._rnn_hidden_size = rnn_hidden_size
#     self.target_embedding = nn.Embedding(num_embeddings=num_embeddings,
#         embedding_dim=embedding_size,
#         padding_idx=0) 
#     self.gru_cell = nn.GRUCell(embedding_size + rnn_hidden_size,
#         rnn_hidden_size)
#     self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
#     self.classifier = nn.Linear(rnn_hidden_size * 2, num_embeddings) 
#     self.bos_index = bos_index
    
#     def _init_indices(self, batch_size):
#         """ return the BEGIN­OF­SEQUENCE index vector """
#         return torch.ones(batch_size, dtype=torch.int64) * self.bos_index
    
#     def _init_context_vectors(self, batch_size):
#         """ return a zeros vector for initializing the context """
#         return torch.zeros(batch_size, self._rnn_hidden_size)
    
#     def forward(self, encoder_state, initial_hidden_state, target_sequence):
#         """The forward pass of the model
#         Args:
#             encoder_state (torch.Tensor): output of the NMTEncoder 
#             initial_hidden_state (torch.Tensor): last hidden state in the NMTEn 
#             target_sequence (torch.Tensor): target text data tensor 
#             sample_probability (float): schedule sampling parameter
#                 probability of using model's predictions at each decoder step 
#         Returns:
#             output_vectors (torch.Tensor): prediction vectors at each output st
#         """
#         # We are making an assumption here: batch is on 1st dimension
#         # The input is (Batch, Seq)
#         # We want to iterate over the sequence so we permute it to (S, B) 
#         target_sequence = target_sequence.permute(1, 0)
#         # use the provided encoder hidden state as the initial hidden state 
#         h_t = self.hidden_map(initial_hidden_state)
        
#         batch_size = encoder_state.size(0)
#         # initialize context vectors to zeros
#         context_vectors = self._init_context_vectors(batch_size) 
#         # initialize first y_t word as BOS
#         y_t_index = self._init_indices(batch_size)
#         h_t = h_t.to(encoder_state.device)
#         y_t_index = y_t_index.to(encoder_state.device) 
#         context_vectors = context_vectors.to(encoder_state.device)
        
#         output_vectors = []
#         # All cached tensors are moved from the GPU and stored for analysis 
#         self._cached_p_attn = []
#         self._cached_ht = []
#         self._cached_decoder_state = encoder_state.cpu().detach().numpy()
        
#         output_sequence_size = target_sequence.size(0) 
#         for i in range(output_sequence_size):
#             # Step 1: Embed word and concat with previous context 
#             y_input_vector = self.target_embedding(target_sequence[i]) 
#             rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)
            
#             # Step 2: Make a GRU step, getting a new hidden vector 
#             h_t = self.gru_cell(rnn_input, h_t) 
#             self._cached_ht.append(h_t.cpu().data.numpy())
            
#             # Step 3: Use current hidden vector to attend to encoder state 
#             context_vectors, p_attn, _ = \
#                 verbose_attention(encoder_state_vectors=encoder_state, query_vector=h_t)
            
#             # auxiliary: cache the attention probabilities for visualization 
#             self._cached_p_attn.append(p_attn.cpu().detach().numpy())
            
#             # Step 4: Use current hidden and context vectors
#             #   to make a prediction for the next word 
#             prediction_vector = torch.cat((context_vectors, h_t), dim=1) 
#             score_for_y_t_index = self.classifier(prediction_vector)
            
#             # auxiliary: collect the prediction scores 
#             output_vectors.append(score_for_y_t_index)


# # Attention mechanism that does element­wise multiplication and summing more explicitly
# def verbose_attention(encoder_state_vectors, query_vector):
#     """
#     encoder_state_vectors: 3dim tensor from bi­GRU in encoder 
#     query_vector: hidden state in decoder GRU
#     """
#     batch_size, num_vectors, vector_size = encoder_state_vectors.size() 
#     vector_scores = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size), dim=2)
#     vector_probabilities = F.softmax(vector_scores, dim=1)
#     weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, context_vectors = torch.sum(weighted_vectors, dim=1) 
#     return context_vectors, vector_probabilities

# def terse_attention(encoder_state_vectors, query_vector):
#     """
#     encoder_state_vectors: 3dim tensor from bi­GRU in encoder query_vector: hidden state
#     """
#     vector_scores = torch.matmul(encoder_state_vectors, query_vector.unsqueeze(dim=2)).squeeze() 
#     vector_probabilities = F.softmax(vector_scores, dim=­1)
#     context_vectors = torch.matmul(encoder_state_vectors.transpose(­2, ­1), vector_probabilities.unsqueeze(dim=2)).squeeze()
#     return context_vectors, vector_probabilities


# # The decoder with a sampling procedure (in bold) built into the forward pass
# class NMTDecoder(nn.Module):
#     def __init__(self, num_embeddings, embedding_size, rnn_size, bos_index):
#         super(NMTDecoder, self).__init__() 
#         # ... other init code here ...
#         # arbitrarily set; any small constant will be fine 
#         self._sampling_temperature = 3
    
#     def forward(self, encoder_state, initial_hidden_state, target_sequence, sample_probability=0.0):
#         if target_sequence is None: 
#             sample_probability = 1.0
#         else:
#             # We are making an assumption here: batch is on 1st dimension
#             # The input is (Batch, Seq)
#             # We want to iterate over the sequence so we permute it to (S, B) 
#             target_sequence = target_sequence.permute(1, 0) 
#             output_sequence_size = target_sequence.size(0)
        
#         # ... nothing changes from the other implementation
#         output_sequence_size = target_sequence.size(0) 
#         for i in range(output_sequence_size):
#             # new: a helper Boolean and the teacher y_t_index 
#             use_sample = np.random.random() < sample_probability 
#             if not use_sample:
#                 y_t_index = target_sequence[i]
#             # Step 1: Embed word and concat with previous context
#             # ... code omitted for space
#             # Step 2: Make a GRU step, getting a new hidden vector
#             # ... code omitted for space
#             # Step 3: Use current hidden vector to attend to the encoder state
#             # ... code omitted for space
#             # Step 4: Use current hidden and context vectors
#             #
#             prediction_vector = torch.cat((context_vectors, h_t), dim=1) 
#             score_for_y_t_index = self.classifier(prediction_vector)
#             # new: sampling if Boolean is true
#             if use_sample:
#                 # sampling temperature forces a peakier distribution 
#                 p_y_t_index = F.softmax(score_for_y_t_index * self._sampling_temperature, dim=1) 
#                 # method 1: choose most likely word
#                 # _, y_t_index = torch.max(p_y_t_index, 1)
#                 # method 2: sample from the distribution
#                 y_t_index = torch.multinomial(p_y_t_index, 1).squeeze()
            
#             # auxiliary: collect the prediction scores 
#             output_vectors.append(score_for_y_t_index)

#         output_vectors = torch.stack(output_vectors).permute(1, 0, 2) 
#         return output_vectors

