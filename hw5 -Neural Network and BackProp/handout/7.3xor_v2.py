

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Net(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H1,H2,H3,D_out):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(D_in, H1)
        #hidden layer 
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        #output layer 
        self.linear4 = nn.Linear(H3, D_out)

    # Prediction    
    def forward(self, x):
        x = torch.relu(self.linear1(x))  
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        return x


def accuracy(y, yhat):
    yhat = yhat.cpu().detach().numpy()
    acc = np.argmax(yhat)
    #print(str(y.item()) + ":" + str(acc.item()))
    return (y==acc).detach().cpu().numpy()



def read_tsv(filepath): 
    data_testList = []
    
    file1 = open(filepath, 'r') 
    data_testList = file1.readlines()
    return np.asarray(data_testList)

def split_line(train_raw):
    label = []
    features = []
    a = []
    for line in  train_raw :
        temp = line.split('\n')
        a.append(temp[0])
        #print(temp[0])        
    for line in a:
        line = line.split(',')
        label.append(line[-1])
        features.append(line[1:])
        
    return np.asarray(label),np.asarray(features)

# Define the train model

def train(data_set,test_data_set, model, criterion, train_loader,test_loader, optimizer, epochs=5):
    
   # cuda = torch.cuda.is_available()
    #print(cuda)
  #  device = torch.device( 'cuda' if cuda else 'cpu' )
    for epoch in range(epochs):
        total=0
        ACC = []
        print("\n")
        print("epoch: "+ str(epoch))
        for x, y in train_loader:
            
            
            x =x.cuda()
            #x.to(device)
            y =y.cuda()
            #print(x.is_cuda)
            
            
            
            optimizer.zero_grad()
            yhat = model(x.float())
            #print(yhat)
            #print(y)
            #print(x.is_cuda)
            loss = criterion(yhat, y.long())
            loss.backward()
            optimizer.step()
            #cumulative loss 
            total+=loss.item()
            acc = accuracy(y, yhat)
            #print(np.sum(acc))
            
            ACC.append(np.sum(acc))
            
            
        print("loss: " + str(total))
        print(str(np.sum(ACC))+ " / "+ str(len(data_set)) + "  Percentage: "+ str(np.sum(ACC)/len(data_set)))

        test(data_set, test_data_set, model, criterion, test_loader, epochs=1)
    
 


def test(data_set,test_data_set, model, criterion,train_loader, epochs=1):

    ACC = []
    total=0
    
   
    for x, y in train_loader:           
        x =x.cuda()           
        y =y.cuda()
        yhat = model(x.float())

        loss = criterion(yhat, y)

        #cumulative loss 
        total+=loss.item()
        acc = accuracy(y, yhat)
        ACC.append(acc.item())
            
            
        
    print("loss_test: " + str(total))
    print("correct: " + str(np.sum(ACC))+ "/" + str(len(test_data_set)) + "  Percentage: "+ str(np.sum(ACC)/len(test_data_set)))
    

    



class Custom_dataset(Dataset):

    def __init__(self,x,y):
      self.x=x
      self.y=y
    def __getitem__(self, index):    
        return torch.from_numpy(self.x[index]),torch.tensor(self.y[index]).long()
    def __len__(self):
        return self.x.shape[0]

class Custom_dataset_test(Dataset):

    def __init__(self,x):
      self.x=x
    def __getitem__(self, index):    
        return self.x[index].long()
    def __len__(self):
        return self.x.shape[0]

train_input = '../handout/train.csv'        

train_array = read_tsv(train_input)
labels_train_main, features_train_main = split_line(train_array)
features_train_main = features_train_main[1:,:-1]
features_train_main = features_train_main.astype(np.float)
labels_train_main = labels_train_main[1:]
labels_train_main = labels_train_main.astype(np.int)    

labels_train = labels_train_main[:5001]
features_train = features_train_main[:5001]

labels_test = labels_train_main[5001:] 
features_test = features_train_main[5001:]

data_set = Custom_dataset(features_train,labels_train)
test_data_set = Custom_dataset(features_test,labels_test)


model = Net(250, 128, 64, 32, 5)
cuda = torch.cuda.is_available()
device = torch.device( 'cuda' if cuda else 'cpu' )

model.to(device)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.05)

train_loader = DataLoader(dataset=data_set, batch_size=1,shuffle = True)
test_loader = DataLoader(dataset=test_data_set, batch_size=1 , shuffle = True)

LOSS12 = train(data_set,test_data_set, model, criterion, train_loader,test_loader, optimizer, epochs=20)

torch.save(model, '../handout/weights.pt')




#testing after saving and loading the weights (check for saving )

model_new = Net(250, 128, 64, 32, 5)
optimizer_new  = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.05)

checkpoint  = torch.load('../handout/weights.pt')

model_new.load_state_dict(checkpoint.state_dict())
model.eval()



test(data_set, test_data_set, model, criterion, test_loader, epochs=1)









