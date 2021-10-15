import numpy as np
from prob import logistic_loss, l1_constraint, l2_constraint, n_supp_constraint


def fw(x_init, n_iter, feature, label, constraint_type, R, lr_type):
    """
    vanilla fw
    
    Args:
        x_init: initialized x. It is assumed to be in the constraint set.
        n_iter: number of iterations
        feature: features of dataset
        label: labels of dataset
        constraint (str): constraint set, currently only 'l1', 'l2', 'n_supp' are supported
        R: radius of constraint set, i.e., \| x \| <= R
        lr_type (str): what step size to use. It should be chosen from 'pf', 's', and 'ds',
                 which stand for parameter-free, smooth, and directionally smooth, respectively.
                 
    Returns:
        loss (list): objective value of f(x) from iteration 0 to n_iter-1
    """
    
    [n_data, dim] = feature.shape
    
    lr_type = lr_type.lower()
    if lr_type not in ['pf', 's', 'ds']:
        raise ValueError('Unsupported learning rate. Currently only pf, s, and, ds are valid choices.')
        
    constraint_type = constraint_type.lower()
    if constraint_type == 'l1':
        constraint = l1_constraint(R)
    elif constraint_type == 'l2':
        constraint = l2_constraint(R)
    elif constraint_type == 'n_supp':
        # we use n = 2 for all experiments
        constraint = n_supp_constraint(R, 2, dim)
    else:
        raise ValueError('Unsupported constraint set. Currently only l1, l2, and, n_supp norm balls are valid choices.')
    
    
    # initialization
    x = x_init
    objective = logistic_loss(feature, label)
    loss = [0 for i in range(n_iter+1)]
    loss[0] = objective.function_value(x)
    
    for k in range(n_iter):
        # step 1: gradient calculation
        grad_x = objective.grad(x)
        
        # step 2: solve the fw subproblem
        v = constraint.fw_subprob(grad_x)
        
        # step 3: determine the step size
        if lr_type == 'pf':
            lr = 2 / (k + 2)
        elif lr_type == 's':
            lr = (grad_x.T @ (x - v)) / (objective.L * (x - v).T @ (x - v))
            lr = np.amin([lr, 1])
        elif lr_type == 'ds':
            Lk = np.linalg.norm(feature @(v - x))**2 / (4 * n_data * (x - v).T @ (x-v))        
            lr = (grad_x.T @ (x - v)) / (Lk * (x - v).T @ (x - v))
            lr = np.amin([lr, 1])
            
        # step 4: update
        x = (1 - lr) * x + lr * v
        loss[k+1] = objective.function_value(x)
    
    return loss



def wfw(x_init, n_iter, feature, label, constraint_type, R, lr_type):
    """
    WFW: HFW with \delta = 2/(k+2)
    
    Args:
        x_init: initialized x. It is assumed to be in the constraint set.
        n_iter: number of iterations
        feature: features of dataset
        label: labels of dataset
        constraint (str): constraint set, currently only 'l1', 'l2', 'n_supp' are supported
        R: radius of constraint set, i.e., \| x \| <= R
        lr_type (str): what step size to use. It should be chosen from 'pf', 's', and 'ds',
                 which stand for parameter-free, smooth, and directionally smooth, respectively.
    
    Returns:
        loss (list): objective value of f(x) from iteration 0 to n_iter-1
    """
    
    [n_data, dim] = feature.shape
    
    lr_type = lr_type.lower()
    if lr_type not in ['pf', 's', 'ds']:
        raise ValueError('Unsupported learning rate. Currently only pf, s, and, ds are valid choices.')
        
    constraint_type = constraint_type.lower()
    if constraint_type == 'l1':
        constraint = l1_constraint(R)
    elif constraint_type == 'l2':
        constraint = l2_constraint(R)
    elif constraint_type == 'n_supp':
        # we use n = 2 for all experiments
        constraint = n_supp_constraint(R, 2, dim)
    else:
        raise ValueError('Unsupported constraint set. Currently only l1, l2, and, n_supp norm balls are valid choices.')
    
    # initialization
    x = x_init
    # here we use a lazy approach to initialize g, since we have \delta_0 = 1
    g = np.zeros((dim,1))   
    objective = logistic_loss(feature, label)
    loss = [0 for i in range(n_iter+1)]
    loss[0] = objective.function_value(x)
    
    for k in range(n_iter):
        # step 1: gradient calculation
        grad_x = objective.grad(x)
        
        # step 2: update heavy ball momentum 
        delta = 2/(k+2)
        g = delta * grad_x + (1 - delta) * g
        
        # step 3: solve the fw subproblem
        v = constraint.fw_subprob(g)
            
        # step 4: determine step size
        if lr_type == 'pf':
            lr = 2 / (k + 2)
        elif lr_type == 's':
            lr = (grad_x.T @ (x - v)) / (objective.L * (x - v).T @ (x - v))
            lr = np.amax([0, np.amin([lr, 1])])
        elif lr_type == 'ds':
            Lk = np.linalg.norm(feature @ (v - x))**2 / (4 * n_data * (x - v).T @ (x-v))
            lr = (grad_x.T @ (x - v)) / (Lk * (x - v).T @ (x - v))
            lr = np.amax([0, np.amin([lr, 1])]) 
        
        # step 5: update
        x = (1 - lr)*x + lr * v        
        loss[k+1] = objective.function_value(x)
    
    return loss



def ufw(x_init, n_iter, feature, label, constraint_type, R, lr_type):
    """
    UFW: HFW with \delta = 1/(k+1)
    
    Args:
        x_init: initialized x. It is assumed to be in the constraint set.
        n_iter: number of iterations
        feature: features of dataset
        label: labels of dataset
        constraint (str): constraint set, currently only 'l1', 'l2', 'n_supp' are supported
        R: radius of constraint set, i.e., \| x \| <= R
        lr_type (str): what step size to use. It should be chosen from 'pf', 's', and 'ds',
                 which stand for parameter-free, smooth, and directionally smooth, respectively.
    
    Returns:
        loss (list): objective value of f(x) from iteration 0 to n_iter-1
    """
    
    [n_data, dim] = feature.shape
    
    lr_type = lr_type.lower()
    if lr_type not in ['pf', 's', 'ds']:
        raise ValueError('Unsupported learning rate. Currently only pf, s, and, ds are valid choices.')
        
    constraint_type = constraint_type.lower()
    if constraint_type == 'l1':
        constraint = l1_constraint(R)
    elif constraint_type == 'l2':
        constraint = l2_constraint(R)
    elif constraint_type == 'n_supp':
        # we use n = 2 for all experiments
        constraint = n_supp_constraint(R, 2, dim)
    else:
        raise ValueError('Unsupported constraint set. Currently only l1, l2, and, n_supp norm balls are valid choices.')
    
    # initialization
    x = x_init
    # here we use a lazy approach to initialize g, since we have \delta_0 = 1
    g = np.zeros((dim,1))   
    objective = logistic_loss(feature, label)
    loss = [0 for i in range(n_iter+1)]
    loss[0] = objective.function_value(x)
    
    for k in range(n_iter):
        # step 1: gradient calculation
        grad_x = objective.grad(x)
        
        # step 2: update heavy ball momentum 
        delta = 1 / (k + 1)
        g = delta * grad_x + (1 - delta) * g
        
        # step 3: solve the fw subproblem
        v = constraint.fw_subprob(g)
            
        # step 4: determine step size
        if lr_type == 'pf':
            lr = 1 / (k + 1)
        elif lr_type == 's':
            lr = (grad_x.T @ (x - v)) / (objective.L * (x - v).T @ (x - v))
            lr = np.amax([0, np.amin([lr, 1])])
        elif lr_type == 'ds':
            Lk = np.linalg.norm(feature @ (v - x))**2 / (4 * n_data * (x - v).T @ (x-v))
            lr = (grad_x.T @ (x - v)) / (Lk * (x - v).T @ (x - v))
            lr = np.amax([0, np.amin([lr, 1])]) 
        
        # step 5: update
        x = (1 - lr)*x + lr * v        
        loss[k+1] = objective.function_value(x)
    
    return loss
