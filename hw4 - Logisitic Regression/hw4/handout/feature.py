import numpy as np

import sys


def read_tsv(filepath): 
    data_testList = []
    
    file1 = open(filepath, 'r') 
    data_testList = file1.readlines()
    return np.asarray(data_testList)

def split_line(train_raw):
    label = []
    word_list = []
    a = []
    b = []
    for line in  train_raw :
        temp = line.split()
        a.append(temp[0])
        b.append(temp[1:])
        
        #splitd.append(temp)
    
    #splitting each line into words with whitespace delimiter
    for lab,storyline in zip(a,b):
        label.append(lab)
        word_list.append(storyline)
    return word_list,label

#dict writer for model 1
def dictListWriter2(word_list,dictionary):
    dict_list = []
    dict_list_thresold = []
    
    # making the sparse Dictionary . each line is a dict for eachstoryline
    for eachline in word_list:
        temp_dict = {}
        for word in eachline:                 
            if word in dictionary.keys():
                if dictionary[word] in temp_dict:
                    #print(temp_dict[dictionary[word]])
                    temp_dict[dictionary[word]] +=1
                    #print(temp_dict[dictionary[word]])
                else:
                    temp_dict[dictionary[word]] = 1               
        dict_list.append(temp_dict)
    for iDict in dict_list:
        final_temp_dict = {}
        for key,value in iDict.items():
            if (value<4):
                final_temp_dict[key]=1
        dict_list_thresold.append(final_temp_dict)
    #print(dict_list_thresold[2]) 
        #print(dict_list)    
    return dict_list_thresold

#dict writer for model 2
def dictListWriter1(word_list,dictionary):
    dict_list = []
    # making the sparse Dictionary . each line is a dict for eachstoryline
    for eachline in word_list:
        temp_dict = {}
        for word in eachline:
            if word in dictionary.keys():
                temp_dict[dictionary[word]] = 1           
        dict_list.append(temp_dict)
    #print(dict_list)
    return dict_list
        
        
def dictToTSV(dict_list,output_path,label):    
    # Write to a TSV
    with open(output_path,"w+") as file:
        for i,dicti in enumerate(dict_list):
            file.write(label[i])
            for key,value in dicti.items():
            
                file.write("\t"+str(key)+":"+str(value)) 
            file.write("\n")
    
def model1_helper(train_input,dictionary,output): 
    train_raw = read_tsv(train_input)    
    word_list,label = split_line(train_raw)    
    dict_list = dictListWriter1(word_list,dictionary)    
    dictToTSV(dict_list,output,label)   
    
    
def model2_helper(train_input,dictionary,output): 
    train_raw = read_tsv(train_input)    
    word_list,label = split_line(train_raw)    
    dict_list = dictListWriter2(word_list,dictionary)    
    dictToTSV(dict_list,output,label)


if __name__ == "__main__":  
    
    #file Paths
    train_input = '../handout/largedata/train_data.tsv'
    test_input = '../handout/largedata/test_data.tsv'
    val_input = '../handout/largedata/valid_data.tsv'
    formatted_train1 = "model_formatted_train.tsv" 
    formatted_test1 = "model_formatted_test.tsv"
    formatted_val1 = "model_formatted_valid.tsv"
    formatted_train2 = "model_formatted_train.tsv" 
    formatted_test2 = "model_formatted_test.tsv"
    formatted_val2 = "model_formatted_valid.tsv"
    dict_path =  "dict.txt"
    flag = 1
    
    
    
    file = open(dict_path,"r")
    line = file.readlines()
    dictionary = {}
    for i in line:          #reading Dict
        temp = i.split()
        dictionary[temp[0]] = temp[1]
        
#call to helper which takes care of everything :) output formatted text
    
    if flag ==0 :        
        model1_helper(train_input,dictionary,formatted_train1)
        model1_helper(test_input,dictionary,formatted_test1)
        model1_helper(val_input,dictionary,formatted_val1)
    else:
        model2_helper(train_input,dictionary,formatted_train2)
        model2_helper(test_input,dictionary,formatted_test2)
        model2_helper(val_input,dictionary,formatted_val2)
    



    #file Paths - Command line Argument
# =============================================================================
#     train_input =  sys.argv[1]
#     val_input =  sys.argv[2]
#     test_input =  sys.argv[3]
#     dict_path =   sys.argv[4]
#     formatted_train =  sys.argv[5]
#     formatted_val=  sys.argv[6]
#     formatted_test =  sys.argv[7]
#     flag =  int(sys.argv[8])
#     
#         
#     file = open(dict_path,"r")
#     line = file.readlines()
#     dictionary = {}
#     for i in line:          #reading Dict
#         temp = i.split()
#         dictionary[temp[0]] = temp[1]
#     
#     
#     if flag ==1 :        
#         model1_helper(train_input,dictionary,formatted_train)
#         model1_helper(test_input,dictionary,formatted_test)
#         model1_helper(val_input,dictionary,formatted_val)
#     else:
#         model2_helper(train_input,dictionary,formatted_train)
#         model2_helper(test_input,dictionary,formatted_test)
#         model2_helper(val_input,dictionary,formatted_val)
# =============================================================================
    
    
    
    
    
    

      
    
    
    
        
    
    
    
        
    
  
        


        

            
    

            
    
    
    


    
    
    
    

















 
    

    
    
