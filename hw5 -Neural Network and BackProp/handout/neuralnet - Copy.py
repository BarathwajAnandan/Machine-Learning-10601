import numpy as np
import sys

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
          
    for line in a:
        line = line.split(',')
        label.append(line[0])
        features.append(line[1:])
        
    return np.asarray(label),np.asarray(features)

def weight_init(feature,label,random,hidden_unit):
     
    if(random==1):
        w1_ = np.random.uniform(-0.1,0.1,(hidden_unit,feature.shape[1]))
        w2_ = np.random.random_sample((label.shape[1],hidden_unit))
    else:
        w1_ = np.zeros((hidden_unit,feature.shape[1]))
        w2_ = np.zeros((label.shape[1],hidden_unit))
    return w1_,w2_
        
def sigmoid(a):
    return 1/(1+np.exp(-1*a))
def softmax(b):
    return np.exp(b)/np.sum(np.exp(b))
def loss(y,y_hat):
    res = np.multiply(y,np.log(y_hat))
    res = np.sum(res)   
    return -res
    
def forward_prop(w1,x,w2):   
    a = np.dot(w1,x)
    #print("a:", a)
    z_  = sigmoid(a)
  #  print("z:", z_)
    z = np.hstack((1,z_))
    b = np.dot(w2,z)
    #print("	Second layer output (before softmax):", b)
    y = softmax(b)
    #print("Final output (after softmax):", y)
    return y,z_,z #return softmax
       
def back_prop(y,z_,z,y_target,x,w2_):
    dl_db =  np.asarray(y - y_target)
    #print("d(loss)/d(softmax inputs): ", dl_db)   
    dl_db= np.reshape(dl_db,(y_target.shape[0],1))
    z = np.reshape(z,(1,z.shape[0]))   
    dl_dW2 = np.dot(dl_db,z)       
    dl_dz = np.dot(dl_db.T,w2_).T  
    z_ = (z_ * (1- z_)).reshape(z_.shape[0],1)
    dl_da = np.multiply(dl_dz,z_)
    #print("dl_da:", dl_da)      
    dl_dW1 = np.dot(dl_da,x.reshape(1,x.shape[0]))
    #print(d1_   
    
    return dl_dW1,dl_dW2

def predict(x,y_target,w1,w2):   
    y,_,_ = forward_prop(w1,x,w2)
    
    #print(np.argmax(y))
    return np.argmax(y)
    
def write_label( labels, file):
    f = open(file, "w")
    for i in labels:
        f.write(str(i) + "\n")
    f.close()
    
def write_Metrics(train_error,test_error,metrics,entropy_train,entropy_test):
    f = open(metrics, "w")
    for i in range(len(entropy_train)):
        f.write("epoch=" + str(i+1) + " crossentropy(train): " + str(entropy_train[i]) + "\n")     
        f.write("epoch=" + str(i+1) + " crossentropy(test): " + str(entropy_test[i])+ "\n")
    f.write("error(train): " + str(train_error) + "\n" + "error(test): " + str(test_error))
    f.close() 
    
    
def cross_entropy_eval(w1,w2,features,y_target):
    l = 0
    for i in range(features.shape[0]):
        y_hat,z_,z = forward_prop(w1,features[i],w2)
        l += loss(y_target[i],y_hat)
        
    return l/features.shape[0]
    
if __name__ == "__main__":  
    
    plot_1_train = []
    plot_1_test = []
    
    train_input = '../handout/train.csv'
    test_input = '../handout/test.csv'               
    train_out = '../handout/model1train_out.labels'
    test_out  = '../handout/model1test_out.labels'
    metric_out = '../handout/model1_metrics_out.txt'
    
    epoch = 1
    hidden_unit = 4
    flag = 2
    lr = 0.1
    plot_1_train = []
    plot_1_test = []
       
    
# =============================================================================
#     train_input = sys.argv[1]
#     test_input = sys.argv[2]            
#     train_out = sys.argv[3]
#     test_out  = sys.argv[4]
#     metric_out = sys.argv[5]
#     epoch = int(sys.argv[6])
#     hidden_unit = int(sys.argv[7])
#     flag = int(sys.argv[8])
#     lr = float(sys.argv[9])
# =============================================================================
    
    #reading  train and test data. Splitting into features and labels.
    train_array = read_tsv(train_input)
    labels_train, features_train = split_line(train_array)
    features_train = np.hstack((np.ones((features_train.shape[0],1)),features_train))
    features_train = features_train.astype(np.float)
    labels_train = labels_train.astype(np.int)
    
    test_array = read_tsv(test_input)
    labels_test, features_test = split_line(test_array)
    features_test = np.hstack((np.ones((features_test.shape[0],1)),features_test))
    features_test = features_test.astype(np.float)
    labels_test = labels_test.astype(np.int)
    
    #oneHot 
    y_target_train = np.zeros((labels_train.size, 10))
    y_target_train[np.arange(labels_train.size),labels_train] = 1
    
    y_target_test = np.zeros((labels_test.size, 10))
    y_target_test[np.arange(labels_test.size),labels_test] = 1
    
    
    #weight Init
    w1_,w2_ = weight_init(features_train[:,1:],y_target_train,flag,hidden_unit)
    #appending 1 for bias  
    w1 = np.hstack((np.zeros((w1_.shape[0],1)),w1_))
    w2 = np.hstack((np.zeros((w2_.shape[0],1)),w2_))
    entropy_test = []
    entropy_train = []
    for j in range(epoch):   

        #iterating example
        #print(j)
        for i in range(features_train.shape[0]):
        #forward prop
            y_hat,z_,z = forward_prop(w1,features_train[i],w2)
            print(y_hat)
        #backProp
            dw1,dw2 = back_prop(y_hat,z_,z,y_target_train[i],features_train[i],w2[:,1:])
         #   print("dW2: ",dw2)
         #   print("dw1: ",dw1)
            w1 = w1 - lr*dw1
            w2 = w2 - lr*dw2
        entropy_train.append(cross_entropy_eval(w1,w2,features_train,y_target_train))
        entropy_test.append(cross_entropy_eval(w1,w2,features_test,y_target_test))
        
# =============================================================================
#         plot_1_train.append(entropy_train[-1])
#         plot_1_test.append(entropy_test[-1])
# =============================================================================
    
# =============================================================================
#     import matplotlib.pyplot as plt
#     
#     plt.plot(np.arange(0,100),entropy_train,label = 'train')
#     plt.plot(np.arange(0,100),entropy_test,label = 'test')
#     
#     plt.title("No. of epoch vs Cross Entropy loss (lr = 0.001)")
#     plt.xlabel("Epoch")
#     plt.ylabel("CrossEntropyLoss")
#     plt.legend(loc="upper right")
# =============================================================================
    
    
        
     
        
    train_predict = []
    test_predict = []
    predicted_trainLabel = []
    predicted_testLabel = []
    #prediction:
    for i in range(y_target_train.shape[0]):
        y= predict(features_train[i],labels_train[i],w1,w2)
        predicted_trainLabel.append(y)         
        print(y)#array of predicted labels                        
        p_train = y==labels_train[i]
        train_predict.append(p_train)    #train_T or F for prediction           
                 
    for i in range(y_target_test.shape[0]):

        
        y = predict(features_test[i],labels_test[i],w1,w2)
        predicted_testLabel.append(y)           #array of predicted labels
        p_test = y==labels_test[i]
        test_predict.append(p_test)    #test_T or F for prediction
            
    
    train_predict = np.asarray(train_predict) #array of true or false (train_dataset)
    test_predict = np.asarray(test_predict)     #array of true or false for (test dataset) prediction
    #print(train_predict)
    #Metrics
    
    train_error = 1- np.sum(train_predict)/train_predict.shape[0]  
    test_error = 1- np.sum(test_predict)/test_predict.shape[0]
    
    print(train_error,test_error)
    
    
    write_label(predicted_trainLabel,train_out)
    write_label(predicted_testLabel,test_out)
    write_Metrics(train_error,test_error,metric_out,entropy_train,entropy_test)
    
    
    

        
# =============================================================================
#     l_array = []
#     for i in range(features_test.shape[0]):
#         y_hat,z_,z = forward_prop(w1,features_test[i],w2)
#         
#         l = loss(y_target_test[i],y_hat)
#         l_array.append(l)
#     l_avg = np.mean(l_array)
#     print(l_avg)
#     
# =============================================================================
    
    
  
    

        
 
        
    
    
    
    
    
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    