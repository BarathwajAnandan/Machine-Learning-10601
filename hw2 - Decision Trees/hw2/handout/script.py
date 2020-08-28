# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:55:35 2020

@author: KevinX
"""
import csv
import numpy as np
train_input = "../handout/politicians_train.tsv"
with open(train_input, "r") as csv_file:
        data_list =[]
        with open(train_input, newline = '') as data:                                                                                          
            data_reader = csv.reader(data, delimiter='\t')
            for d in data_reader:
                data_list.append(d) 
        data = np.asarray(data_list).transpose()
        
 #       labels = data[-1,1:]
        attributes = data[:-1,0] 
        
        labels = data[-1,1:]
        unique_labels = np.unique(labels)
        
        no_labels = labels.shape[0]
        a = np.asarray(np.where(labels == unique_labels[0])).shape[1]
        b = np.asarray(np.where(labels == unique_labels[1])).shape[1]
        
        
#    def gini_D (labels,unique_labels):
        
        no_labels = labels.shape[0]
        a = np.asarray(np.where(labels == unique_labels[0])).shape[1]                  
        a_prob =  a/no_labels               
        a_prime = 1 - a_prob
        b = np.asarray(np.where(labels == unique_labels[1])).shape[1]
        b_prob = b/no_labels
        b_prime = 1 - b_prob    
        marginal_impurity = a_prob*a_prime + b_prob* b_prime 


        
        
        
def gini_conditional(data):
    
     attributes = data[:-1,1:]
     output = data[-1,1:]
     label = np.unique(output)
     
     for row in attributes:
         parameters = np.unique(row)
         
         a = np.where(row==parameters[0])
         count_label1 = np.size(a)
         b = np.where(row==parameters[1])
         count_label2 = np.size(b)
         
         output1 = output[a]
         output2 = output[b]  #data split of labels to get count of + and -
         
         #count probability now !
         
         plus_a = np.size(np.where(output1==label[0]))             #+ of left split
         minus_a = np.size(np.where(output1==label[1]))            #- of left split
         
         
         plus_b = np.size(np.where(output2==label[0]))             #+ of left split
         minus_b = np.size( np.where(output2==label[1]))           #- of left split
         
         
        # print(plus_a ,minus_a,plus_b,minus_b)
         
         
         totala = plus_a + minus_a
         totalb = plus_b + minus_b
         total  = totala  + totalb
         
         gini_a = (plus_a/totala) * (minus_a/totala) + (minus_a/totala) * (plus_a/totala)
         gini_b = (plus_b/totalb) * (minus_b/totalb) + (minus_b/totalb) * (plus_b/totalb)
         
         gini_impurity =  gini_a * (totala/total)  + gini_b * (totalb/total)
         
         gini_gain = marginal_impurity - gini_impurity
         
         print(gini_gain)
         
         
         
         
         
         
         
         
         
         
         
         
         
         
    
     
     
     
    
    
        
        
        
        
        