'''
This is an algorithm written by Michael Hartmann, based on the desription in "Christopher Bishop, Pattern Recognition and Machine Learning
Springer 2006"
'''


import numpy as np
import scipy.stats as sd
'''
    Evaluate the log likelihood
'''
def evaluate_loglikelihood(model, dataset):
    erg=0
    for act_point in dataset:
        modes_prob=[model['pi'][wlt]*sd.norm.pdf(act_point, model['mu'][wlt], model['Sigma'][wlt]) for wlt in range(0, len(model['pi']))]
        act_sum=np.sum(modes_prob)
        erg+=np.log(act_sum)
    return erg


##################
### Parameters ###
##################
N=100
model = {'pi': [.4, .6], 'mu': [3, 2], 'Sigma': [0.1, .3]}
my_model = {'pi': [.1, .9], 'mu': [0, 1], 'Sigma': [0.1, .3]}

###############
### Dataset ###
###############
modes_X=[model['pi'][wlt]*np.random.normal(model['mu'][wlt], model['Sigma'][wlt], N) for wlt in range(0, len(model['pi']))]
X=modes_X[0]
for i in range(1, len(modes_X)):
    X+=modes_X[i]
##############
### Step 1 ###
##############
act_log_likeli=evaluate_loglikelihood(model, X)
None
