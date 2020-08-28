import numpy as np
import csv
import math
import sys

class Node:
    def __init__(self, data, depth, max_depth, attributes, used):
        self.data = data
        self.depth = depth
        self.max_depth = max_depth
        self.attributes = attributes
        self.used = used
        self.split_feature = None
        self.split_attr_values = None
        self.branchs = []
       
        self.label_count = self.count(data)
        self.majority_vote_result =  self.majority_vote(self.label_count)
        #print(self.majority_vote_result)
        
        
    
    def majority_vote( self, stats):
        l = []
        for elem in sorted(stats.items(), reverse=True):
            l.append(elem)
            
        #print(len(stats))
        
        if len(stats)==1:
            return l[0][0]
        
        if (l[0][1] >= l[1][1]):
            return l[0][0]
        else:
            return l[1][0]
            
        
            
        
        
    
    def count(self, data):
        output = data[:, -1]
        l = np.unique(output)
        count = {}
        label2 = 0
        
        label1 = np.size(np.where(output==l[0])) 
        count[l[0]] = label1
        if len(l)==2:
            label2 = np.size(np.where(output==l[1])) 
            count[l[1]] = label2    
        return count
    
    def gini_D(self, label_count):
        b_prob = 0
        temp = []
        for k in label_count.keys():
            temp.append(label_count[k])
        a_prob =  temp[0]/sum(temp)               
        a_prime = 1 - a_prob      
        if len(temp)==2:
            b_prob = temp[1]/sum(temp)
        b_prime = 1 - b_prob    
        gini = a_prob*a_prime + b_prob* b_prime    
        return gini

    def conditional_gini_impurity(self, data,i):
        
        attr_label = data[:, [i, -1]]
        output = attr_label[:,1]
        label = np.unique(output)
        row = attr_label[:,0]
        
        parameters = np.unique(row)
        l = len(parameters)
        if l==2:
            a = np.where(row==parameters[0])
            count_label1 = np.size(a)
            b = np.where(row==parameters[1])
        #  count_label2 = np.size(b)
     
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
            
        if l==1:
            a = np.where(row==parameters[0])
            #count_label1 = np.size(a)     
            output1 = output[a]
            plus_a = np.size(np.where(output1==label[0]))
            minus_a = np.size(np.where(output1==label[1])) 
            
            totala = plus_a + minus_a
            total  = totala  
            gini_a = (plus_a/totala) * (minus_a/totala) + (minus_a/totala) * (plus_a/totala)
            gini_impurity =  gini_a * (totala/total) 

        return gini_impurity
    
    
    
    def find_split_index(self,gini_Dataset):
        gini_gain = 0.01
        split_index = -1
        for i,j in enumerate(self.attributes):
            if j not in self.used:
                gini_impurity = self.conditional_gini_impurity(self.data,i)
                curr_giniGain = gini_Dataset - gini_impurity
                if curr_giniGain > gini_gain:
                    split_index = i
                    gini_gain = curr_giniGain
                    
        return split_index
		
    def train(self):
        len(self.used)
        if self.depth == self.max_depth:            #reaches max depth 
            return
        if len(self.attributes) == len(self.used):      #exhausted the feature list
            return
        gini_Dataset = self.gini_D(self.label_count)
        if gini_Dataset == 0:
            return
        
        
        
        split_index = self.find_split_index(gini_Dataset)
# =============================================================================
#         gini_gain = 0.01
#         for i,j in enumerate(self.attributes):
#             if j not in self.used:
#                 gini_impurity = self.conditional_gini_impurity(self.data,i)
#                 curr_giniGain = gini_Dataset - gini_impurity
#                 if curr_giniGain > gini_gain:
#                     split_attr_index = i
#                     gini_gain = curr_giniGain
# =============================================================================
                    
        #find the max gini gain
        self.split_feature = self.attributes[split_index]    

        #split dataset for next level
        self.split_attr_values = np.unique(self.data[:, split_index])
        for value in self.split_attr_values:
            branch_training_data = []
            for data in self.data:	      
                if data[split_index] == value:
                    branch_training_data.append(data)
            branch_training_data = np.array(branch_training_data)
            
            #print(msg)
            self.branchs.append(Node(branch_training_data, self.depth + 1, self.max_depth, 
								  self.attributes, self.used +[self.split_feature]))
        for branch in self.branchs:
            branch.train()
            #self.msg
        return

    def predict(self, attr_dict):
        if self.split_feature == None:
            #print (self.majority_vote_result)
            return self.majority_vote_result

        #print(self.split_feature)
        attribute_value = attr_dict[self.split_feature]
        #attribute_value
        #print(attribute_value)

        for j,i in enumerate((self.split_attr_values)):
            if attribute_value == i:
                return self.branchs[j].predict(attr_dict)
        return self.majority_vote_result

		
class DecisionTree:
    def __init__(self, training_data_file, max_depth):
        #print("hello")
        data_list =[]
        with open(training_data_file, "r") as csv_file:
            reader = csv.reader(csv_file,delimiter = "\t")
            for d in reader:
                data_list.append(d) 
            data = np.asarray(data_list)
            attributes= data[0,:-1]
        self.data, self.attributes = data[1:,:],attributes
        #print(attributes)
        
        
        self.root = Node(self.data, 0, max_depth, self.attributes, [])
        
    def train(self):
        self.root.train()

    def predict(self, attr_dict):
        return self.root.predict(attr_dict)




def test_data(input):
    data_list =[]
    with open(input, newline = '') as data:                                                                                          
        data_reader = csv.reader(data, delimiter='\t')
        for d in data_reader:
            data_list.append(d) 
        data = np.asarray(data_list)
        attributes = data[0,:-1]
        label = data[1:,-1]
        data = data[1:,:]
        
        
    attr_dict_list = []
    
    for d in data:
        attr_dict = {}
        for i in range(len(attributes)):
            attr_dict[attributes[i]] = d[i]
            #print(attr_dict)
        attr_dict_list.append(attr_dict)
       
    return attr_dict_list, label


def write_to_file(data,train_out):
        
       
        with open(train_out,"w+") as file:
            for i in data:
                file.write(i + "\n")


def main():
    train_input = "../handout/politicians_train.tsv"
    test_input = "../handout/politicians_test.tsv"
    max_depth = int(4)
    train_output = "../handout/output/train.labels"
    test_output = "../handout/output/test.labels"
    metric_out = "../handout/output/metrics.txt"

    tree = DecisionTree(train_input, max_depth)
    tree.train()
    #tree.pretty_print()
    
    
    
    #test of train:
    
    attr_dict_list, label = test_data(train_input)
    
    #predicting error_rate
    predicted =[]
    for attr_dict in  attr_dict_list:
        predicted.append(tree.predict(attr_dict))
    #print("Ha")
    
    
    predicted = np.asarray(predicted)
    truth = np.asarray(label)  
    error = predicted== truth
    error_rate_train = 1 - sum(error)/np.size(predicted)  
    print(error_rate_train)
    
    
    #wrtie to file
    
    write_to_file(predicted,train_output)
    
    
    
    
    #test of test   
    attr_dict_list, label = test_data(test_input)
    
    #predicting error_rate
    predicted =[]
    for attr_dict in  attr_dict_list:
        predicted.append(tree.predict(attr_dict))
    #print("Ha")
    
    
    predicted = np.asarray(predicted)
    truth = np.asarray(label)  
    error = predicted== truth
    error_rate_test = 1 - sum(error)/np.size(predicted)  
    print(error_rate_test)
    
    
    
    write_to_file(predicted,test_output)
    
    
    
    with open(metric_out,"w+") as file:
        file.write("error(train): " + str(error_rate_train) + "\n" + "error(test): " + str(error_rate_test))
    
    
    
    
    
    
        



if __name__ == "__main__":
    main()






