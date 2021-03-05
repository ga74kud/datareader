'''
This is an algorithm written by Michael Hartmann, based on the desription in "Christopher Bishop, Pattern Recognition and Machine Learning
Springer 2006"
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sd

def get_new_model():
    my_model = {'pi': [.4, .6], 'mu': [10, 2.2], 'Sigma': [4.1, 9.1]}
    return my_model

'''
    plot the dataset
'''
def plot_X(X):

    plt.hist(X)
'''
    one_cycle
'''
def one_cycle(params, my_model, X, real_response):
    #####################################################
    ### Step 2: E-Step, evaluate the responsibilities ###
    #####################################################
    gamma = e_step(params, my_model, X, real_response)
    ##################################################
    ### Step 3: M-Step, re-estimate the parameters ###
    ##################################################
    my_model = m_step(params, X, gamma)
    ###########################################
    ### Step 4: Evaluate the log likelihood ###
    ###########################################
    act_log_likeli = evaluate_loglikelihood(params, my_model, X)
    return my_model, act_log_likeli
'''
    Get new sigmas (variances)
'''
def get_sigmas(params, dataset, new_mus, Nk, gamma):
    new_sigmas = [ [ 0 for i in range(params['K']) ] for j in range(params['N']) ]
    for idx, act_point in enumerate(dataset):
        gam_idx=np.argmax(gamma[idx][:])
        new_sigmas[idx][gam_idx] =  (act_point-new_mus[gam_idx])*np.transpose(act_point-new_mus[gam_idx])
    new_sigmas=np.sum(new_sigmas, axis=0)
    new_sigmas = [new_sigmas[i] / Nk[i] for i in range(params['K'])]
    return new_sigmas

'''
    Get new mean values
'''
def get_mu(params, dataset, gamma, Nk):
    new_mu = [ [ 0 for i in range(params['K']) ] for j in range(params['N']) ]
    for idx, act_point in enumerate(dataset):
        gam_idx=np.argmax(gamma[idx][:])
        new_mu[idx][gam_idx] =  act_point
    new_mu=np.sum(new_mu, axis=0)
    new_mu = [new_mu[i] / Nk[i] for i in range(params['K'])]
    return new_mu
'''
    M-Step, re-estimate the parameters
'''
def m_step(params, dataset, gamma):
    Nk=np.sum(gamma, axis=0)
    new_mus=get_mu(params, dataset, gamma, Nk)
    new_sigmas = get_sigmas(params, dataset, new_mus, Nk, gamma)
    new_pi=[Nk[i] / params['N'] for i in range(params['K'])]
    return {'pi': new_pi, 'mu': new_mus, 'Sigma': new_sigmas}

'''
    E-Step Evaluate the responsibilites using the current parameter values
'''
def e_step(params, model, dataset, real_response):
    gamma= [ [ 0 for i in range(params['K']) ] for j in range(params['N']) ]
    erg_gamma= [ [ 0 for i in range(params['K']) ] for j in range(params['N']) ]
    for idx, act_point in enumerate(dataset):
        act_real_responsibility=real_response[idx]
        modes_prob=[model['pi'][wlt]*sd.norm.pdf(act_point, model['mu'][wlt], model['Sigma'][wlt]) for wlt in range(0, params['K'])]
        act_sum=np.sum(modes_prob)
        for wlt in range(0, params['K']):
            gamma[idx][wlt]=modes_prob[wlt]/act_sum
        abc=np.argmax(gamma[idx][:])
        erg_gamma[idx][abc]=1
    return erg_gamma


'''
    Evaluate the log likelihood
'''
def evaluate_loglikelihood(params, model, dataset):
    erg=0
    for act_point in dataset:
        modes_prob=[model['pi'][wlt]*sd.norm.pdf(act_point, model['mu'][wlt], model['Sigma'][wlt]) for wlt in range(0, params['K'])]
        act_sum=np.sum(modes_prob)
        erg+=np.log(act_sum)
    return erg


##################
### Parameters ###
##################
model = {'pi': [.4, .6], 'mu': [20, 2], 'Sigma': [2, 5]}
params={'N': 1000, 'K': len(model['pi']), 'am_cycles': 50}
###############
### Dataset ###
###############
X=list()
select=np.random.choice(2, params['N'], p=model['pi'])
for idx, val in enumerate(select):
    mu=model['mu'][val]
    Sigma=model['Sigma'][val]
    pi=model['pi'][val]
    new_sample=np.float(np.random.normal(mu, Sigma, 1))
    X.append(new_sample)
X=np.array(X)


##########################################################
### Step 1: Evaluate the log likelihood the first time ###
##########################################################
my_model=get_new_model()
old_value=evaluate_loglikelihood(params, my_model, X)
###################################
### Cycles: Iterative steps 2-4 ###
###################################
for cycle in range(0, params['am_cycles']):
    my_model, conv_value=one_cycle(params, my_model, X, select)
    if(np.sum(old_value-conv_value)<10e-2):
        break
    elif(np.isnan(conv_value)):
        break
    else:
        old_value=conv_value
    print("Cycle: " + str(cycle) + " Value: " + str(conv_value))
    print(my_model)
#######################
### plot the result ###
#######################
plot_X(X)
plt.show()
