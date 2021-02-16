'''
This is an algorithm written by Michael Hartmann, based on the desription in "Christopher Bishop, Pattern Recognition and Machine Learning
Springer 2006"
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sd

'''
    plot the dataset
'''
def plot_X(X):

    plt.hist(X)
'''
    one_cycle
'''
def one_cycle(params, my_model, X):
    #####################################################
    ### Step 2: E-Step, evaluate the responsibilities ###
    #####################################################
    gamma = e_step(params, my_model, X)
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
    new_sigmas = [0 for i in range(params['K'])]
    for wlt in range(0, params['K']):
        for idx, act_point in enumerate(dataset):
            new_sigmas[wlt] += gamma[idx][wlt] * (act_point-new_mus[wlt])*np.transpose(act_point-new_mus[wlt])
    new_sigmas = [new_sigmas[i] / Nk[i] for i in range(params['K'])]
    return new_sigmas
'''
    Get new mean values
'''
def get_mu(params, dataset, gamma, Nk):
    new_mu = [0 for i in range(params['K'])]
    for wlt in range(0, params['K']):
        for idx, act_point in enumerate(dataset):
            new_mu[wlt] += gamma[idx][wlt] * act_point
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
def e_step(params, model, dataset):
    gamma= [ [ 0 for i in range(params['K']) ] for j in range(params['N']) ]
    for idx, act_point in enumerate(dataset):
        modes_prob=[model['pi'][wlt]*sd.norm.pdf(act_point, model['mu'][wlt], model['Sigma'][wlt]) for wlt in range(0, params['K'])]
        act_sum=np.sum(modes_prob)
        for wlt in range(0, params['K']):
            gamma[idx][wlt]=modes_prob[wlt]/act_sum
    return gamma


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

##########################################################
### Step 1: Evaluate the log likelihood the first time ###
##########################################################
##################
### Parameters ###
##################
model = {'pi': [.4, .6], 'mu': [20, 2], 'Sigma': [2, 5]}
my_model = {'pi': [.4, .6], 'mu': [18.1, 2.2], 'Sigma': [2.1, 6.1]}
params={'N': 1000, 'K': len(model['pi']), 'am_cycles': 5}
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
act_log_likeli=evaluate_loglikelihood(params, model, X)
###################################
### Cycles: Iterative steps 2-4 ###
###################################
for cycle in range(0, params['am_cycles']):
    my_model, act_log_likeli=one_cycle(params, my_model, X)
    print("Cycle: " + str(cycle) + " Value: " + str(act_log_likeli))
    print(my_model)
#######################
### plot the result ###
#######################
plot_X(X)
plt.show()
