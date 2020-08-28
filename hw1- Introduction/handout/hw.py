import sys

import numpy as np

import csv


def

if __name__ == "__main__":     
    
    data_list = []
    
# =============================================================================
#     train_input = sys.argv[1]
#     test_input = sys.argv[2]
#     split_index = sys.argv[3]
#     train_out = sys.argv[4]
#     test_out = sys.argv[5]
#     metric_out = sys.argv[6]
#     print("The input file is: %s" % (train_input))
#     print("The input file is: %s" % (test_input))
#     print("The input file is: %s" % (split_index))
#     print("The input file is: %s" % (train_out))
#     print("The input file is: %s" % (test_out))
#     print("The input file is: %s" % (metric_out))
# =============================================================================
    
  #  print("The output file is: %s" % (output))
  
     
   
    
    train_input = "../handout/education_train.tsv"
    test_input = "../handout/education_test.tsv"
    split_index = 5
    train_out = "../handout/output/train.labels"
    test_out = "../handout/output/test.labels"
    metric_out = "../handout/output/metrics.txt"
   
  
   #found the way to open TSV Files -__-
    with open(train_input, newline = '') as data:                                                                                          
    	data_reader = csv.reader(data, delimiter='\t')
    	for d in data_reader:
    		data_list.append(d) 
            
            
            #for now. change to taking in name of feature ?
    feature_number = split_index
            
    train_data  = np.asarray(data_list).transpose()
    feature = train_data[ feature_number]
    feature = feature[1:]                    #list of y and n for the feature specified.
    
    
    
    if (train_data[len(train_data)-1][0] == 'grade'):
        yes_array = np.where(feature=='A')           #collected the Y and N element number from the feature extract 
        no_array = np.where(feature=='notA')
         
    if (train_data[len(train_data)-1][0] == ' Party '):
        yes_array = np.where(feature=='y')           #collected the Y and N element number from the feature extract 
        no_array = np.where(feature=='n')
    
    
 
    
    train_labels = train_data[np.shape(train_data)[0] -1] [1:]      #Segregated the train feature from the last column.
    
    
    #choose of labels for classification:
    
    if (train_data[len(train_data)-1][0] == ' Party '):
         label0 = 'republican'
         label1 = 'democrat'
         
    if (train_data[len(train_data)-1][0] == 'grade'):
         label0 = 'A'
         label1 = 'notA'
         
    if (train_data[len(train_data)-1][0] == ' Party '):
         label0 = 'republican'
         label1 = 'democrat'
    
   
    label_yesList = train_labels[yes_array]             #segregating the party list for YES only 
    
    count_label0_yesList = np.asarray((np.where(label_yesList ==label0))) 
    count_label0_Y = count_label0_yesList.shape[1]
    

    #count of Republican intersecting with the YES LIST
    
    count_label1_yesList = np.asarray((np.where(label_yesList ==label1 )))   #count of Democrat intersecting with the YES LIST
    count_label1_Y = count_label1_yesList.shape[1]
    
    if (count_label1_Y> count_label0_Y ):
        maj_vote_yes = label1
    else:
        maj_vote_yes = label0
    
    
    #print("Heyyyy workinggg")
    
    label_NoList = train_labels[no_array]             #segregating the party list for NO only 
    
    count_label0_NoList = np.asarray((np.where(label_NoList ==label0)))     #count of Republican intersecting with the YES LIST
    count_label0_N = count_label0_NoList.shape[1]
    
    
    
    count_label1_NoList = np.asarray((np.where(label_NoList ==label1)))      #count of Democrat intersecting with the YES LIST
    count_label1_N = count_label1_NoList.shape[1]
    
    
    
    if (count_label1_N> count_label0_N ):
        maj_vote_no = label1
    else:
        maj_vote_no = label0
    
    train_test = np.copy(train_labels)
    
    
    train_test[yes_array] =  maj_vote_yes
    
    train_test[no_array] =  maj_vote_no
 
    
    
    
    
    #training phase Initiated!!!
    
    
    data_testList = []
    
    with open(test_input, newline = '') as data:                                                                                          
    	data_reader = csv.reader(data, delimiter='\t')
    	for d in data_reader:
    		data_testList.append(d) 
            
            
            
    #test data Reading
    
    test_data  = np.asarray(data_testList).transpose()
    
    
    
    test_feature = test_data[ feature_number] [1:]
    test_labels = test_data[np.shape(test_data)[0] -1] [1:]   #label extraction for test to replace :P
    
    test_labels_OG = np.copy(test_labels)
    
    
    
    
    if (train_data[len(train_data)-1][0] == 'grade'):
        yes_array_test = np.where(test_feature=='A')           #collected the Y and N element number from the feature extract 
        no_array_test = np.where(test_feature=='notA')
         
    if (train_data[len(train_data)-1][0] == ' Party '):
        yes_array_test = np.where(test_feature=='y')           #collected the Y and N element number from the feature extract 
        no_array_test = np.where(test_feature=='n')
    

    
    test_labels[yes_array_test] =  maj_vote_yes
    
    test_labels[no_array_test] =  maj_vote_no
    
    
    
def error_calc(train_test,train_labels,test_labels,test_labels_OG):
    
    # Error:
    #trainError
    train_error= train_labels == train_test
    
    train_error = np.count_nonzero(train_error)
    
    train_error = 1 - train_error/len(train_labels)
    
    #TestError
    
    
    # import test file for vertification
# =============================================================================
#     ll = []
#     
#     with open("../handout/example_output/politicians_3_test.labels", newline = '') as data:                                                                                          
#     	data_reader = csv.reader(data, delimiter='\t')
#     	for d in data_reader:
#     		ll.append(d) 
#             
#     ll = np.asarray(ll)
# =============================================================================
            
            
    
    test_error = test_labels == test_labels_OG
    
    test_error = np.count_nonzero(test_error)
    
    test_error = 1 - test_error/len(test_labels)
    
    
    print("Train_Error: ",train_error)
    print("Test_Error: ",test_error)
    
    
    return train_error,test_error
    
    #writing files to output :)
    with open(metric_out,"w+") as file:
        file.write("error(train): " + str(train_error) + "\n" + "error(test): " + str(test_error))
    
    
    
    with open(train_out,"w+") as file:
        for i in train_test:
            file.write(i + "\n")
            
    with open(test_out,"w+") as file:
        for i in test_labels:
            file.write(i + "\n")
    
    
    
    
    
            
            



    
            
            
            
            
            
            
   
   
   
   
    
    
    
    
    
    
    
    
    
    
    




