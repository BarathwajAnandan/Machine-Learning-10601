import numpy as np
import sys

def read_tsv(filepath): 
    data_testList = []
    
    file1 = open(filepath, 'r') 
    data_testList = file1.readlines()
    return np.asarray(data_testList)

def index_converter(train_raw):
    a = []
    for line in  train_raw :
        temp = line.split('\n')
        #print(temp[0])
        a.append(temp[0])
    #print(a)
    
    index_conv = {}
    for index,i in enumerate(a):
        index_conv[i] = index+1
    
    return index_conv

def split_word_tag(train_raw,word_index,tag_index):
   
    after_space = []
    X_mid,Y_mid,X,Y,x_mid,y_mid = [],[],[],[],[],[]
    emission = np.ones((len(word_index),len(tag_index)))
    prior = np.ones(len(tag_index))
    a = []
    for line in  train_raw :
        temp = line.split('\n')
       #print(temp[0])
        a.append(temp[0])
          
    for line in a:
        words = line.split()
        after_space.append(words)
    
        #print(words)
    for line in after_space:
        labels = []     
        states =  []
        for i,word in enumerate(line):
            underscore_split = word.split('_')
            #print(underscore_split)
            label = underscore_split[1]
            label = tag_index[label]            
            state = underscore_split[0]
            state = word_index[state]
            if i == 0:
                prior[label-1] +=1
                
            #print(state-1 , label-1 )
            emission[state-1][label-1]+=1
            
            labels.append(label)
            states.append(state)
        X.append(states)
        Y.append(labels)    
    prior = prior/np.sum(prior)
   
#emission Matrix    
    for i in range(len(tag_index)):
        #print(np.sum(emission[:,i]))
        emission[:,i] =  emission[:,i]/np.sum(emission[:,i])
    return  np.asarray(X),np.asarray(Y),emission.T,prior

if __name__ == "__main__":  
    train_input = '../handout/trainwords.txt'
    index_to_word = '../handout/index_to_word.txt'               
    index_to_tag = '../handout/index_to_tag.txt'
    hmmprior  = '../handout/hmmprior.txt'
    hmmemit = '../handout/hmmemit.txt'
    hmmtrans = '../handout/hmmtrans.txt'
# =============================================================================
#     train_input = sys.argv[1]
#     index_to_word = sys.argv[2]              
#     index_to_tag = sys.argv[3]
#     hmmprior  = sys.argv[4]
#     hmmemit = sys.argv[5]
#     hmmtrans =sys.argv[6]   
#     
# =============================================================================
    word_index_raw = read_tsv(index_to_word)
    word_index = index_converter(word_index_raw)
    
    tag_index_raw = read_tsv(index_to_tag)
    tag_index = index_converter(tag_index_raw)            
    read_traindata = read_tsv(train_input)
    read_traindata = read_traindata[:10]
    
    XX,YY, emission,prior = split_word_tag(read_traindata,word_index,tag_index)    
#transition matrix
    temp =[]
    transition_i = []
    transition_j = []
    for y in YY:
        temp.append(y[0])
        transition_j.append(y[1:])
        transition_i.append(y[:-1])      
    transition = np.ones((len(tag_index_raw),len(tag_index_raw)))    
    for ii,jj in zip(transition_i,transition_j):
        for i,j in zip(ii,jj):    
            #print(i,j)
            transition[i-1][j-1] +=1          
    for i in range(len(tag_index_raw)):
        transition[i] =  transition[i]/np.sum(transition[i])       
    np.savetxt(hmmemit,emission)    
    np.savetxt(hmmtrans,transition)
    np.savetxt(hmmprior,prior)        
    
    
    
        
    
    
    
    
    
  