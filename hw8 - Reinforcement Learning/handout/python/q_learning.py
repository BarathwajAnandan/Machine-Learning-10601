from environment import MountainCar
import numpy as np
import sys

def dict_to_state(mode,size_state,state_dict):
    if mode == 'raw':
        state = np.array(list(state_dict.values())) 
    else:
        state = np.zeros(size_state)
        state[list(state_dict.keys())] = 1

    return state.T

def action(epsilon,q__s_a):
    
    random_ = np.random.rand()
    if random_ < epsilon:
        a = np.random.randint(mc.action_space)
    else:
        a = np.argmax(q__s_a)
    return a


if __name__ == "__main__":
    
# =============================================================================
#     mode  = 'raw'
#     weight_out = 'weight.out'
#     returns_out = 'returns.out'
#     episodes = 4 
#     max_iterations = 200
#     epsilon = 0.05 
#     gamma = 0.99
#     learning_rate = 0.01
# =============================================================================
    
    mode = str(sys.argv[1])
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])

    mc = MountainCar(mode)
    size_state = mc.state_space
    size_action = mc.action_space
    w = np.zeros((size_state,size_action))
    b = 0
    returns = [] 

    for epi in range(episodes):
        state = mc.reset()
        ret = 0
        result = 0         
        s = dict_to_state(mode,size_state,state)
        for i in range(max_iterations):
            q__s_a = np.dot(s,w) + b           
            a = action(epsilon,q__s_a)
            grad_b  = 1
            grad_w  = s            
            state_prime,r,result = mc.step(a)      
            s_prime = dict_to_state(mode,size_state,state_prime)                    
            q__sprime_aprime = np.dot(s_prime,w) + b           
            w[:,a]-= learning_rate* (q__s_a[a] - (r+(gamma*np.max(q__sprime_aprime))))*grad_w           
            b-= learning_rate* (q__s_a[a] - (r+(gamma*np.max(q__sprime_aprime))))*grad_b            
            ret+=r
            s = s_prime
            if result == True:
                #print('hey)')
                break
        returns.append(ret) 
       # mc.render()
  #  mc.close()
   
    
    w_flat = w.flatten()
    weights = np.zeros(w_flat.shape[0]+1)
    weights[0] = b
    weights[1:] = w_flat

    np.savetxt(weight_out, weights, delimiter='\n')
    np.savetxt(returns_out, returns, delimiter='\n')
    
        
    
    
        
    
    
    
    
    
  
    
    
    
    