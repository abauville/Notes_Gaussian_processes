import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.models.gpytorch import GPyTorchModel
from IPython.display import clear_output

class ExactGPModel(GPyTorchModel, gpytorch.models.ExactGP):
    _num_outputs = 1
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())        
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
   

def train_hyper_params(model, likelihood, training_iter=20, verbose=False):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(model.train_inputs[0])
        # Calc loss and backprop gradients
        loss = -mll(output, model._train_targets[0])
        loss.backward()
        if verbose:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()

        
def get_test_x(n, mu=0.5, sigma=0.25, lower=0.0, upper=1.0):
    """ returns a n x d tensor containing random points taking from truncated 
    normal distributions of means mu, and standard deviation sigma (mu and sigma are floats or lists of length d).
    The distribution is truncated between lower and upper bounds.
    for more information on the truncated normal distribution see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
Parameters
==============
n: int
    number of points to return
mu: float or array-like, default: 0.5
    list of means of length d
sigma: float or array-like, default: 0.25
    list of standard deviations (must be the same length as mu)
lower: float, default: 0.0
    lower bound
uper: float, default: 1.0

Returns
==============
tensor of shape (n, d)

    """
    if type(mu) == int or type(mu) == float:
        mu = [mu]
    if type(sigma) == int or type(sigma) == float:
        sigma = [sigma]
    if len(mu) != len(sigma):
        raise ValueError(f"mu and sigma must have the same length, but got len(mu)={len(mu)}, len(sigma)={len(sigma)}")
        
    ndim = len(mu)
    out = []
    for i in range(ndim):
        X = truncnorm(
            (lower - mu[i]) / sigma[i], (upper - mu[i]) / sigma[i], loc=mu[i], scale=sigma[i])
        out.append(torch.from_numpy(X.rvs(n)))

    return torch.stack(out).T.reshape(-1,1,ndim).float()


def get_x_new(EI, n_test, n_iter=1, refinement_fac = 1.1, device='cpu'):
    """Performs the aquisition step n_iter times on n random samples taken from a normal distribution.
    At each iteration: 
      - the mean of the distribution is set to the location of the proposed point (argmax of aquisition function)
      - the standard deviation is decreased: std_new = std/(refinement_fac**j) where j is the iteration index
    This method allows for picking a good point to aquire while limiting the number of points tested
    """
    best_ei = -1e10
    mu = [0.5] * 6
    sigma = 0.25
    for j in range(n_iter):
        test_x = (torch.randn(int(n_test)*6).reshape(-1,1,6).float().to(device) * sigma/(refinement_fac**j))
        for ii in range(6):
            test_x[:,0,ii] += mu[ii]
        ei = EI(test_x)
        I = ei.argmax().item()
        if ei[I] > best_ei:
            best_ei = ei[I] 
            x_new = test_x[I]
    return x_new


def run_bayes_opt(Model, likelihood, train_x, train_y, gt_function, error_function=None, num_test_points=100, min_iter=2, max_iter=100):
    """
Parameters
===========
Model: class that inherits from Botorch model, e.g. GPyTorchModel, (and optionally one of gpytorch.models)
train_x: torch.tensor
    coordinates of the initial training points (i.e. observations)
train_y: torch.tensor
    target value of the training points
num_test_points: int
    number of test points (identical for every dimension)
gt_function: class
    a class that defines the test bounds, a global minimum and a method f that takes train_x, train_y as input
    See test_functions.py for examples
error_function: function, optional
    a function that computes a scalar error between the model and the ground truth
"""
    min_iter = max(min_iter, 2); 
    it = 0
    while (it<min_iter or (train_x[-1] - train_x[-2]).abs()>1e-6) and it<max_iter:
        # Run the forward model with hyperparam opt
        model = Model(train_x, train_y, likelihood)
        train_hyper_params(model, likelihood)
        
        # Run the aquisition method
        EI = ExpectedImprovement(model, best_f=0.2)
        ei = EI(test_x.reshape(-1,1,1))
        
        # Add the suggested point to the training points
        x_new = test_x[ei.argmax().item()]
        train_x = torch.cat((train_x.reshape(-1,1), x_new.reshape(-1,1))).reshape(-1)
        train_y = torch.cat((train_y.reshape(-1,1), f(x_new).reshape(-1,1))).reshape(-1)
        it += 1
        
        if error_function:
            error_function()
        
        