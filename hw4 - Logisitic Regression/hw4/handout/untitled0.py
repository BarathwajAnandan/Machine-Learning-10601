import csv
import time
import numpy as np
import math
import sys






def get_xi(attribute_input):
    location_list = [int(i.split(":")[0]) for i in attribute_input]
    final_list = np.ones_like(location_list)
    final_list = final_list.tolist()
    return final_list, location_list

def predict(input_file, theta):
    labels=[]
    locations=[]
    xis=[]
    row_list=[]
    with open(input_file, 'r') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        for row in tsvin:
            label = int(row[0])
            attribute_input = row[1:]
            xi, lc = get_xi(attribute_input)
            # print (lc)
            xi.append(1)
            lc.append(l_len-1)
            #label.append(1)
            labels.append(label)

            locations.append(lc)
            # print(locations)
            xis.append(xi)

    
    for xi,location in zip(xis,locations):
        xi_new = np.zeros(l_len)
        xi_new[location] = 1
        final_prod = np.dot(xi_new, theta)

        if final_prod > 0:
            row_list.append(1)
        else:
            row_list.append(0)

    labels =  np.asarray(labels)
    row_list = np.asarray(row_list)
    g = row_list != labels
    error_rate = np.sum(g) / (len(labels))
    return row_list,error_rate


def regression(input_file,  epoch):
   
    global l_len
    l_len = 39177
    theta = np.zeros(l_len)
    labels = []
    xis = []
    locations = []
    theta_sparsed = []
    
    with open(input_file, 'r') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        for row in tsvin:
            label = int(row[0])
            attribute_input = row[1:]
            xi, lc = get_xi(attribute_input)
            xi.append(1)
            lc.append(l_len-1)

            labels.append(label)
            locations.append(lc)
            xis.append(xi)


    for j in range(epoch):
        for i in range(0, len(xis)):
            theta_sparsed = []
            label = labels[i]
            xi = xis[i]
            xi = np.asarray(xi)
            # print(xi)
            lc = locations[i]
            # print(lc)
            for k in lc:
                theta_sparsed.append(theta[k])
            # print(theta_sparsed)

            #theta_sparsed.insert(0, 0)
            theta_sparsed = np.asarray(theta_sparsed)
            theta += np.asarray(calculate_updates(label, xi, theta_sparsed, lc))
        #print (theta[-1])
    return  theta



def calculate_updates(label, xi, theta, lc):
    eta = 0.1
    sgd_val = sgd(xi, theta, label)
    theta_update = [(eta * sgd_val) for i in range(0, len(xi))]
    theta_prime = np.zeros(l_len)
    for i, k in enumerate(lc):
        theta_prime[k] = theta_update[i]
    return theta_prime


def sgd(xi, theta_sparsed, y):

    exp_value = math.exp(np.sum(theta_sparsed))
    g = ((exp_value) / (1 + exp_value))
    sgd_value = y - g
    
    return sgd_value


def write_file(file, row_list):
    f = open(file, "w")
    for i in row_list:
        f.write(str(i))
        f.write("\n")
    f.close()


if __name__ == '__main__':
    start = time.time()
    print(start)
    input_files = []
    train_input =  sys.argv[1]
    val_input =  sys.argv[2]
    test_input =  sys.argv[3]
    dict_input = sys.argv[4]
    train_out_file = sys.argv[5]
    test_out_file = sys.argv[6]
    metrics =  sys.argv[7]
    num_epoch = sys.argv[8]
    
    theta = regression(input_files[0], int(num_epoch))
    print(theta)
    train_labels, train_error = predict(input_files[0], theta)
    test_labels, test_error = predict(input_files[2], theta)


    write_file(train_out_file, train_labels)
    write_file(test_out_file, test_labels)

    train_string = "error(train): {0}".format(train_error)
    test_string = "error(test): {0}".format(test_error)
    f = open(metrics, "w")
    f.write(train_string)
    f.write("\n")
    f.write(test_string)
    f.close()
    
    print(    time.time() - start )