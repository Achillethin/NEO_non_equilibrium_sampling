import pickle
import os

def estimator_todict(estimator):

    config = {}
    estimator_config = vars(estimator)
    config['infine'] = {'K': estimator_config['K'], 
                        'dim': estimator_config['dim'],
                        'var_p': estimator_config['var_p'],
                        'n_samples': estimator_config['n_samples']}
    config['hamiltonian'] = vars(estimator.hamiltonian)
    config['momentum'] = vars(estimator.momentum)
    config['importance_distr'] = vars(estimator.importance_distr)

    return config
def dump_results(list_, path, exp_name):
    """
    list_: containing all the informations about the estimator
    path : where to save the experience
    exp_name : name of the experience
    """ 
    if os.path.exists(path):
        with open(path, "rb") as f:
            dico = pickle.load(f)

            while exp_name in dico:
                exp_name = exp_name + '1'
                print(exp_name)
    else:
        dico = {}
        
    dico[exp_name] = list_

    with open(path,'wb') as f:
        pickle.dump(dico, f)