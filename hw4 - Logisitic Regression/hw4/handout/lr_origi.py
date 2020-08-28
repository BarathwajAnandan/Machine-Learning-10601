# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 00:32:19 2020

@author: KevinX
"""

import numpy as np
import sys
import csv




def load_Inputdata(path):
    labels = []
    features = []
    before_colonSplit = []
    with open(path,"r") as file:
        lines = file.readlines()
    for index, eachline in enumerate(lines):
        #print(index)
        list_words  = eachline.split("\t")
        labels.append(int(list_words[0]))
        
        #for split in list_words[1:]:
            
        features.append([int(split.split(":")[0]) for split in list_words[1:]])
            #print(split.split(":")[0])
            #adding extra 1.0 feature at the start would ruin things. So add a number greater than dic length at end
        features[index].append(39176)
    return  features, labels
        
    

    #print(before_colonSplit)





def sparse_dot(features,weights):
    
    sum = 0.0
    #for i in features:
    sum = np.sum(weights)
        
    return sum




def train(feats_index, labels, epochs, lr):
    weights = np.zeros(39177)
    for epoch in range(epochs):
        for feat_index, label in zip(feats_index, labels):
            feat_vec = np.zeros(39177)
            feat_vec[feat_index] = 1.0
            
            sparse = weights[feat_index]
            dot_product =np.sum(sparse)
            #dot_product = sparse_dot(feat_index, sparse)
            weights += lr * feat_vec * gradient_helper(dot_product,label)
    print(weights)
    return weights
            


    
        
        
def gradient_helper(dot,label):
    
    temp = label  -  np.exp(dot)/ (1 +np.exp(dot))
    return temp
    
    




if __name__ == "__main__":  
    
    #file Paths
    train_input = 'model_formatted_train.tsv'
    test_input = 'model_formatted_test.tsv'
    val_input =  'model_formatted_test.tsv'
    dict_path =  "dict.txt"
    train_out = "train_out.labels" 
    test_out = "test_out.labels"
    metrics_out = "metrix.txt"
    epoch = 60

    
    
    
    
    
    #file Paths - Command line Argument
# =============================================================================
#     train_input =  sys.argv[1]
#     val_input =  sys.argv[2]
#     test_input =  sys.argv[3]
#     dict_path =   sys.argv[4]
#     train_out =  sys.argv[5]
#     test_out=  sys.argv[6]
#     matrics_out =  sys.argv[7]
#     num_epoch =  sys.argv[8]
#     
# 
#     file = open(dict_path,"r")
#     line = file.readlines()
#     dictionary = {}
#     for i in line:          #reading Dict
#         temp = i.split()
#         dictionary[temp[0]] = temp[1]
# =============================================================================
        
        
        
    features_train,label_train = load_Inputdata(train_input)
    print(len(features_train))
    weights = train(features_train,label_train,epoch,0.1)
    #features_test,label_test = load_Inputdata(test_input)
    
    