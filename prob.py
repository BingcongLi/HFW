import numpy as np

class logistic_loss():
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        [n_data, dim] = feature.shape
        self.n_data = n_data
        self.dim = dim
        # Lipschitz constant, which might be useful for smooth step size
        self.L = np.sum(feature.power(2))/(4*n_data) 

    def grad(self, x):
        """
        calculate gradient of logistic loss at x
        """
        tmp = - np.multiply(self.feature @ x, self.label)
        tmp = 1 - np.divide(1, 1 + np.exp(tmp))
        tmp = - np.multiply( tmp,  self.label)
        return self.feature.T @ tmp / self.n_data

    def function_value(self, x):
        """
        compute logistic loss
        """
        obj_val = np.sum(np.log(1 + np.exp(-self.label * (self.feature.dot(x))))) / self.n_data
        return obj_val.reshape(-1,)



class l1_constraint():
    def __init__(self, R):
        self.R = R

    def fw_subprob(self, grad):
        """
        fw subproblem for 
        l1 norm ball constraint
        """
        idx = np.argmax(np.abs(grad))
        v = np.zeros(grad.shape)
        v[idx] = - np.sign(grad[idx]) * self.R    
        return  v


class l2_constraint():
    def __init__(self, R):
        self.R = R
        
    def fw_subprob(self, grad):
        """
        fw subproblem for 
        l2 norm ball constraint
        """
        return - grad * self.R / np.linalg.norm(grad)


class n_supp_constraint():
    def __init__(self, R, n, dim):
        self.R = R
        self.n = n
        self.dim = dim
    
    def fw_subprob(self, grad):
        """
        fw subproblem for 
        n-support norm ball constraint
        we implicitly use n=2 in our experiements.
        """
        sorted_idx = np.argsort(np.abs(grad).reshape(-1,))
        top_n = sorted_idx[self.dim-self.n-1 : self.dim-1]
        trunc_grad = np.zeros(grad.shape)
        trunc_grad[top_n] = grad[top_n]
        return - trunc_grad * self.R / np.linalg.norm(trunc_grad)
