import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import os
import pandas as pd
import numpy as np
import glob
cuda_enabled = torch.cuda.is_available()
#cuda_enabled = False
CUDA_VISIBLE_DEVICES=0,1,2,3

class smile_data(Dataset):

    # okay, just a comment
    def __init__(self, path, padlen, featlen=136, fold_list_number=0):
        self.instance_list = []
        self.instance_label = []
        self.fold_list_number = fold_list_number
        self.fold_list = []
        self.pad_len = padlen
        self.feat_len = featlen
        self.__load_fold_list__(path, fold_list_number)
        self.__read_data__('/data/lwang89/Documents/smile/spontaneous/results/')
        self.__read_data__('/data/lwang89/Documents/smile/deliberate/results/')
        '''
        self.feat_len = featlen
        self.dataset = dataset   # train, test, dev
        self.__longest_vector__ = 0
        self.__load_features_and_labels__(self.__get
        '''
#         for instance in self.instance_list:
#             print(instance.shape)

    """
    # to generate a dataframe, then extract instance_list and instance_label from it
    def __read_data__(self, directory_name):
        import glob
        # deli_training_path = '/Users/sam/Documents/USF/homework/cs686-02/HCI/lab/final_project/file-video-stream/videos/deliberate/results'
        training_path = directory_name

        if 'deliberate' in directory_name:
            label = 0
        else:
            label = 1

        allFiles = glob.glob(training_path + "/*.csv")
        training_df = pd.DataFrame()
        list_ = []

        # loop files in folder
        for file_ in allFiles:
#             list_ = []
            self.instance_label.append(label)
            df = pd.read_csv(file_)

            # change the first column to label
            # df.columns.values[0] = 'label'
            df.iloc[:,0] = label
#             list_.append(df)
#             training_df = pd.concat(list_)
            df_matrix = df.iloc[:,1:137].values
#             df_matrix = self.__pad_data__(df_matrix)
            self.instance_list.append(df_matrix)
#             self.instance_list = self.__pad_data__(self.instance_list)
#             self.instance_list.append(training_df.iloc[:,1:137].values)

#         training_df = pd.concat(list_)

        # generate instance_list and instance_label
#         self.instance_label.append(training_df.iloc[:,0].values.tolist())
#         self.instance_list.append(training_df.iloc[:,1:137].values)
        # Pad the data -- does it work?
#         print(self.instance_list)
        self.instance_list = self.__pad_data__(self.instance_list)
    """
    def __load_fold_list__(self, path, fold_list_number):
        list_path = '/data/lwang89/Documents/smile/experimental_protocols/fold_all'
        #         list_path = list_directory_name
        allFiles = glob.glob(list_path + "/*.txt")
        print(len(list_path))
        whole_fold_list = []
        train_fold_list = []
        test_fold_list = []
        for file_ in allFiles:
            #             file_name = file_[len(list_path) + 1:140]
            #     print(file_name)
            with open(file_) as f:
                content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]
            whole_fold_list.append(content)

        test_fold_list = whole_fold_list[fold_list_number]
        if path == 'train':
            for x in whole_fold_list:
                if x != test_fold_list:
                    train_fold_list.extend(x)
            self.fold_list = train_fold_list
        else:
            self.fold_list = test_fold_list

    def __read_data__(self, directory_name):
        import glob
        # deli_training_path = '/Users/sam/Documents/USF/homework/cs686-02/HCI/lab/final_project/file-video-stream/videos/deliberate/results'
        """
        if 'deliberate' in directory_name:
            label = 0
        else:
            label = 1
        """

        allFiles = glob.glob(directory_name + "/*.csv")
        training_df = pd.DataFrame()
        list_ = []

        # loop files in folder, we need to check if file name is in the fold list
        for file_ in allFiles:
            if 'deliberate' in file_:
                label = 0
                file_name = file_[len(directory_name) :len(directory_name)+22]
            else:
                label = 1
                file_name = file_[len(directory_name):len(directory_name)+23]

            print("file name is %s, label is %d." %(file_name, label))
            if file_name in self.fold_list:
                self.instance_label.append(label)
                df = pd.read_csv(file_)
                df.iloc[:,0] = label
                df_matrix = df.iloc[:,1:137].values
                self.instance_list.append(df_matrix)

        self.instance_list = self.__pad_data__(self.instance_list)

                                                                                                                                                                                                                                                
    def __pad_data__(self, series):
        

        padded = []
#         print(series)
        for i in range(len(series)):
#             print(i)
#             print(len(series))
            row = np.zeros((self.feat_len, self.pad_len), dtype=np.float32)
#             print("length of series[%d] is %d" %(i, len(series[i])))
#             for j in range(len(series[i])):
            for j in range(self.feat_len):
#                 print("feature length is %d" %(self.feat_len))
                # this the number of records 
#                 print(j)
                
                for k in range(len(series[i])):
                    # this should be the number of features(136)
#                     print("length of series[%d][%d] is: %d" %(i , j, len(series[i][j])))
#                     print(series[i][j])
                    row[j][k] = series[i][k][j]
            padded.append(row)
            
        return padded


    def __getitem__(self, index):
        return self.instance_list[index], self.instance_label[index]

    
    def __len__(self):
        return len(self.instance_list)
    
    def __get_instance_label__(self):
        return self.instance_label
    
    def __get_instance_list__(self):
        return np.array(self.instance_list).shape

# Copied from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
class BiRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.is_training = False
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Dropout(p=0.5, inplace=False)
        self.linear = nn.Linear(self.hidden_size*2, self.num_classes)

        #self.fc = nn.Dropout(p=0.75, inplace=False)
        if cuda_enabled:
            self.lstm = self.lstm.cuda()
            self.fc = self.fc.cuda()
            self.linear = self.linear.cuda()

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) # 2 for bidirection
        c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        if cuda_enabled:
            h0 = h0.cuda()  # 2 for bidirection
            c0 = c0.cuda()

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode hidden state of last time step
        if self.is_training:
            out = self.fc(out[:, -1, :])
        else:
            out = out[:, -1, :]

        out = F.log_softmax(self.linear(out), dim=1)
        return out

def main_biRNN():
    import datetime
    import sys
    from sklearn.metrics import accuracy_score, confusion_matrix

    # Keep timing -- for future reporting
    timing = dict()

    # Data description
    input_size = 5000  # 2939, 20 # TODO: Determine this from the data
    test_input_size = 5000
    sequence_length = 136  # TODO: Determine this from the data
    # Data
    label = 'SvP'  

    timing['start'] = datetime.datetime.now()
    i = 1
    # train = smile_data('train', padlen=input_size, featlen=sequence_length)
    test = smile_data('train', padlen=input_size, featlen=sequence_length, fold_list_number=i)
    # print('Smile train data items: {}'.format(train.__len__()))
#     print(train.__get_instance_label__())
#     print(train.__get_instance_list__())

    # dev = smile_data('dev', label, encoder=train.get_encoder())
    # print('Smile dev data items: {}'.format(dev.__len__()))

    # test = smile_data('test',padlen=test_input_size, featlen=sequence_length)
    train = smile_data('test',padlen=test_input_size, featlen=sequence_length, fold_list_number=i)
    print('We are testing the %d th fold.' %(i))
    print('Smile test data items: {}'.format(test.__len__()))
    print('Smile train data items: {}'.format(train.__len__()))

    timing['features'] = datetime.datetime.now() - timing['start']
    print('Extracted features from speech files.')

    # Some hyperparams, etc. for the network
    batch_size = 64
    kwargs = {'num_workers': 1, 'pin_memory': True}
    if not cuda_enabled:
        kwargs['pin_memory'] = False
        batch_size = 32

    print('Starting loader -------------------------------------')
    timing['training'] = datetime.datetime.now()
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, **kwargs)
    # sys.exit()

    # The net... and training it
    hidden_size = 128
    num_layers = 2
    # num_classes = 9  # TODO: Determine this from the data
    num_classes = 2  # TODO: Determine this from the data
    learning_rate = 0.0001
    num_epochs = 300

    # The network
    rnn = BiRNN(input_size, hidden_size, num_layers, num_classes)
    rnn.is_training = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    epoch_loss = 5000000000.
    # Train it
    for epoch in range(num_epochs):
        loss_total = 0.
        iteration_count = 0.
        for i, (mfcc, labels) in enumerate(train_loader):
            iteration_count += 1.
            mfcc = Variable(mfcc.view(-1, sequence_length, input_size))
            labels = Variable(labels)
            if cuda_enabled:
                mfcc = mfcc.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(mfcc)

            loss = criterion(outputs, labels)
            loss_total += loss.data[0]
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train) // batch_size, loss.data[0]))
        current_epoch_loss = loss_total / iteration_count
        print('Epoch %d; loss = %0.4f' % (epoch, current_epoch_loss))
        # Optimise training epochs: only continue training while the loss drops
        # Leon: uncomment this when you run the full system with all the data
#         if current_epoch_loss >= epoch_loss:
#             break
        epoch_loss = current_epoch_loss

    timing['training'] = datetime.datetime.now() - timing['training']

    # Test the Model
    rnn.is_training = False
    timing['testing'] = datetime.datetime.now()
    print('Testing -----------------------------------------------')
    correct = 0.0
    total = 0.0
    predicted_list = []
    label_list = []
    for mfcc, labels in test_loader:#test_loader
        mfcc = Variable(mfcc.view(-1, sequence_length, input_size))
        if cuda_enabled:
            mfcc = mfcc.cuda()

        outputs = rnn(mfcc)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        for p, l in zip(predicted, labels):
            predicted_list.append(p)
            label_list.append(l)
            if p == l:
                correct += 1.0
    timing['testing'] = datetime.datetime.now() - timing['testing']
    print('Timing (feature extraction, training, timing)')
    print('=============================================')
    print(timing['features'])
    print(timing['training'])
    print(timing['testing'])
    print('')
    print('=============================================')
    print('')
    print('Confusion Matrix')
    print('================')
    # print(train.get_encoder().classes_)
    print(confusion_matrix(label_list, predicted_list))
    print('=============================================')
    print('Accuracy = %0.4f' % (accuracy_score(label_list, predicted_list)))
    print('=============================================')


    # Save the Model
    # torch.save(rnn.state_dict(), 'Smile.pkl')

if __name__ == '__main__':
    main_biRNN()
