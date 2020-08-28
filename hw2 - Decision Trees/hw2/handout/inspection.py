import csv
import sys
import numpy as np

def inspect_data(train_input):
    data_list =[]
    with open(train_input, newline = '') as data:                                                                                          
    	data_reader = csv.reader(data, delimiter='\t')
    	for d in data_reader:
    		data_list.append(d) 
    data = np.asarray(data_list)            
    labels = data[1:,-1]
    unique_labels = np.unique(labels)
    error = calc_mv_error(labels,unique_labels)
    gini = calc_gini_imp(labels,unique_labels)
    return gini, error

def calc_gini_imp(labels,unique_labels):
    no_labels = labels.shape[0]
    a = np.asarray(np.where(labels == unique_labels[0])).shape[1]                  
    a_prob =  a/no_labels               
    a_prime = 1 - a_prob
    b = np.asarray(np.where(labels == unique_labels[1])).shape[1]
    b_prob = b/no_labels
    b_prime = 1 - b_prob    
    gini = a_prob*a_prime + b_prob* b_prime   
    return gini

def calc_mv_error(labels,unique_labels):        #majority vote just based on number max labels / total labels 
    a = np.asarray(np.where(labels == unique_labels[0])).shape[1]
    b = np.asarray(np.where(labels == unique_labels[1])).shape[1]
    no_labels = labels.shape[0]   
    val = max(a,b)
    error =1 - val/no_labels
    print(error)
    return error
if __name__ == "__main__":  
	train_input = sys.argv[1]
	output_file = sys.argv[2]
	gini, error = inspect_data(train_input)
	output_file = open(output_file, "w")
	output_file.write("gini_impurity: {}\n".format(gini))
	output_file.write("error: {}\n".format(error))
	output_file.close()
    
