import sys
import numpy as np
def read_file(test_input):
    data_list = []
    with open(test_input) as file:
        for line in file:
            a=[]
            data = line.strip("\n")
            value= data.split()           
            a=' '.join(value)
            data_list.append(a)
    return data_list
def list_converter(index_to_word,input_data):
    word_list=[]
    with open (index_to_word) as file:
        for line in file:     
            data = line.strip("\n")
            word_list.append(data)
    return word_list
def writer(metric_file,average_log,accuracy):
    with open (metric_file, "w") as files:
        files.write("Average Log-Likelihood: " + str(average_log) + '\n' + "Accuracy: "+ str(accuracy))
        files.close
    print(accuracy,average_log)
if __name__ == '__main__':
    test_input = '../handout/trainwords.txt'
    ind_to_word = '../handout/index_to_word.txt'               
    ind_to_tag = '../handout/index_to_tag.txt'
    hmmprior  = '../handout/hmmprior.txt'
    hmmemit = '../handout/hmmemit.txt'
    hmmtrans = '../handout/hmmtrans.txt'
    predicted_file = '../handout/P.txt'
    metric_file = '../handout/m.txt'
# =============================================================================
#     test_input = sys.argv[1]
#     ind_to_word=sys.argv[2]
#     ind_to_tag=sys.argv[3]
#     hmmprior=sys.argv[4]
#     hmmemit=sys.argv[5]
#     hmmtrans=sys.argv[6]
#     predicted_file=sys.argv[7]
#     metric_file=sys.argv[8]
# =============================================================================
    input_data= read_file(test_input)
    word_list = list_converter(ind_to_word,input_data)
    tag_list  = list_converter(ind_to_tag, input_data)

    pi_matrix = np.genfromtxt(hmmprior)
    A  = np.genfromtxt(hmmtrans)
    B  = np.genfromtxt(hmmemit)
# Alpha matrix formation:
    length_list = []
    temp = []
    log_list = []
    correct = []
    l1 = 0
    with open(predicted_file, "w") as f:
        for idx,line in enumerate(input_data):
            line = line.split()
            word_matrix,bi_matrix,alpha_matrix,beta_matrix,predict_tag_list =[],[],[],[],[]
            #print(line)
            if idx == 0:
                length_list.append(len(line))
            else:
                length_list.append(length_list[idx - 1] + len(line))
            for i, key in enumerate(line):
                word,tag = key.split('_')
                tag_index, word_index = tag_list.index(tag), word_list.index(word)      
            #Alpha 
                if i==0: #START
                    bi=B[:, word_index] 
                    alpha = pi_matrix*bi                                      
                    alpha_matrix.append(alpha)
                    bi_matrix.append(bi)
                else:
                    bi=(B[:, word_index])
                    var=np.dot(A.T,alpha)
                    alpha= np.multiply(bi,var)
                    alpha_matrix.append(alpha)           
            #Beta
            beta = np.ones(len(tag_list))
            beta_matrix.append(beta)
            for i, key in enumerate(reversed(line)):
                word,tag = key.split('_')
                tag_index, word_index = tag_list.index(tag), word_list.index(word)
                if i < len(line)-1:
                    bi=(B[:, word_index])
                    var=bi*beta
                    beta= A @ var
                    beta_matrix.append(beta)          
            beta_matrix=list(reversed(beta_matrix))             
            #prediction
            AlphaBeta =  np.asarray(alpha_matrix) * np.asarray(beta_matrix)
            for word,AB  in zip(line,AlphaBeta):
                predict_tag_list.append(np.argmax(AB))
                word, tag = word.split('_')
                predicted_tag_value = tag_list[np.argmax(AB)]
                correct.append(predicted_tag_value == tag)                
          #printing into the file      
                temp.append((word) + '_' + str(predicted_tag_value))
            l2 = length_list[idx]
            if idx != 0:
                l1 = length_list[idx-1]
            curr_line = temp[l1:l2]
            for i,currentLine in enumerate(curr_line):   
                if i!=0:
                   f.write(' ') 
                f.write(currentLine)
            if idx < len(input_data)-1:
                f.write('\n')
            print(np.log(np.sum(alpha_matrix[-1])))
            if(np.log(np.sum(alpha_matrix[-1])) > -10000):
                log_list.append(np.log(np.sum(alpha_matrix[-1])))
        
        log_list = np.asarray(log_list)
        average_log = np.mean(log_list)
        accuracy = np.sum(correct)/len(correct)
    writer(metric_file,average_log,accuracy)