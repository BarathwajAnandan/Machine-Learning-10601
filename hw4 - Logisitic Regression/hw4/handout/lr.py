import csv,time,sys
import numpy as np

def load_data(input_file):
    labels,locations = [],[]  
    with open(input_file, 'r') as tsvin:
        TSVin = csv.reader(tsvin, delimiter='\t')
        for row in TSVin:
            label = int(row[0])
            feat_index = [int(i.split(":")[0]) for i in row[1:]]
            feat_index = np.append(feat_index,39176)
            labels.append(label)
            locations.append(feat_index)
            
    return locations,labels

def predict(input_file, theta):
    temp = []    
    locations,labels = load_data(input_file)
    for location in locations:
        X = np.zeros(39177)
        X[location] = 1
        final_prod = np.dot(X, theta)
        temp.append(final_prod)  
    row_list=np.zeros_like(temp)  
    yes = np.where(np.asarray(temp)>0)
    row_list[yes] =1
    labels =  np.asarray(labels)
    row_list = np.asarray(row_list)
    error_rate = np.sum(row_list != labels) / (len(labels))
    return row_list,error_rate
    
def train(input_file,  epoch):
    #J = 0
    theta = np.zeros(39177)
    locations,labels = load_data(input_file)
    for j in range(epoch):
       # print(J)
        
        for label,feat_index in zip(labels,locations): 
         #   J = - label * x + np.log(costFunction(x))
            theta_sparsed = theta[feat_index]            
            x =  np.dot(theta_sparsed,feat_index)
    #freak stuff 
            
            
            #print(J)
            theta += np.asarray(calculate_updates(label, theta_sparsed, feat_index))
    return  theta


def costFunction(x):
    
    if x < -30:
        pass
        #x = -30
    else:
         x = 30
    result = 1/(1+np.exp(-x))
    return result
    
    
     
def calculate_updates(label,theta, feat_index):
   
    theta_prime = np.zeros(39177)
    exp_value = np.math.exp(np.sum(theta))
    sgd_val = label -  exp_value / (1 + exp_value)
    theta_update = [(0.1 * sgd_val) for i in range(0, len(theta))]        
    theta_prime[feat_index] = theta_update
    return theta_prime




def write_file(file, labels):
    f = open(file, "w")
    for i in labels:
        f.write(str(i) + "\n")
    f.close()

if __name__ == '__main__':
    
    start = time.time()
    input_files = []
    train_input = "model_formatted_train.tsv" 
    val_input = 'model_formatted_valid.tsv'
    test_input = 'model_formatted_test.tsv'
    dict_input = "dict.txt"
    train_out_file = "train_out.labels"
    test_out_file = "test_out.labels"
    metrics =  "metrix.txt"
    num_epoch = 50 
    


# =============================================================================
#     train_input =  sys.argv[1]
#     val_input =  sys.argv[2]
#     test_input =  sys.argv[3]
#     dict_input = sys.argv[4]
#     train_out_file = sys.argv[5]
#     test_out_file = sys.argv[6]
#     metrics =  sys.argv[7]
#     num_epoch = int(sys.argv[8])
# =============================================================================
    
    theta = train(train_input , int(num_epoch))
    #print(theta)
    train_labels, train_error = predict(train_input , theta)
    test_labels, test_error = predict(test_input, theta)
# =============================================================================
#     theta = train(, int(num_epoch))  
#     train_labels, train_error = predict(input_files[0], theta)
#     test_labels, test_error = predict(input_files[2], theta)
# =============================================================================
    write_file(train_out_file, train_labels)
    write_file(test_out_file, test_labels)

    f = open(metrics, "w")
    f.write("error(train): " + str(train_error) + "\n" + "error(test): " + str(test_error))
    f.close() 
   # print(time.time() - start)