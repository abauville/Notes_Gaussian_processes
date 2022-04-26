import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.models.gpytorch import GPyTorchModel
from IPython.display import clear_output
from botorch.generation.gen import gen_candidates_torch, get_best_candidates
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.fit import fit_gpytorch_model
import numpy as np
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval

class ExactGPModel(GPyTorchModel, gpytorch.models.ExactGP):
    """The surrogate gaussian process model"""
    _num_outputs = 1
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())        
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
   

def train_hyper_params(model, likelihood):
    """Fit the model to the data and performs hyper parameter optimization. Returns None"""
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_model(mll)



def get_x_new(aquisition_func, model, device='cpu'):
    """Returns the point to aquire given an aquisition function and surrogate model."""
    bounds = torch.tensor([[0., 1.]] * 6).T.to(device)
    Xinit = gen_batch_initial_conditions(aquisition_func, bounds, q=1, num_restarts=25, raw_samples=500)

    batch_candidates, batch_acq_values = gen_candidates_torch(
        initial_conditions=Xinit,
        acquisition_function=aquisition_func,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
    )
    return get_best_candidates(batch_candidates, batch_acq_values)[0].detach()
        
        
def run_experiment(gt_func, n_iter=1, n_train_ini=1, print_period=1):
    """
    Parameters
    ----------
    gt_func: function
        ground truth function:
            - in: a tensor of training or test parameters (train_x, test_x) as input
            - returns: a tensor of target values
    
    n_iter: int, default=1
        number of iterations of the bayesian optimization process
        
    n_train_ini: int , default=1
        number of initial training samples drawn for a uniform distribution in the interval [0,1)
        
    print_period: int, default=1
        a summary string is printed every print_period
        
    Returns
    -------
    numpy ndarray
        array of length n_iter storing, at each iteration, the maximum discover value so far.
    """
    best_fs = np.zeros(n_iter) # maximum discovered values
    
    # Initialize the experiment
    # ============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_x = torch.rand(6*n_train_ini).reshape(-1,6).to(device)
    train_y = gt_func(train_x).to(device)
    best_f = train_y.max().item()
    
    for it in range(n_iter):
        # Run the forward model with hyperparam opt
        # ============================
        likelihood = GaussianLikelihood(noise_constraint=Interval(0.0,1e-4)).to(device)
        model = ExactGPModel(train_x, train_y, likelihood).to(device)
        train_hyper_params(model, likelihood)
        
        # New point aquisition
        # ============================
        EI = ExpectedImprovement(model, best_f=best_f, maximize=True)
        x_new = get_x_new(EI, model, device)
        y_new = gt_func(x_new).to(device)
        best_f = max(y_new.item(), best_f)

        train_x = torch.cat((train_x.reshape(-1,1), x_new.reshape(-1,1))).reshape(-1,6)
        train_y = torch.cat((train_y.reshape(-1,1), y_new.reshape(-1,1))).reshape(-1)
        
        # Print
        # ============================
        if (it+1)%print_period == 0:
            clear_output()
            print(f"{it+1:03d}/{n_iter}, best_f: {best_f:.5f}, last_f: {train_y[-1].item():.5f}")

        # Record metric
        # ============================
        best_fs[it] = best_f
    return best_fs

def baseline_model(gt_func, n_iter, n):
    """A model that attempts to maximize by picking n random sample and keeping the minimum value. 
    The process is repated over n_iter iterations
    Parameters
    ----------
    n_iter: int
        number of iterations
    n: int
        number of samples drawn at each iteration
        
    Returns
    -------
    array (float)
        contains the max of the function discovered so far at each iteration
    """
    results = -np.ones(n_iter)
    results[0] = gt_func(torch.randn(n*6).reshape(-1,6)).max()

    for i in range(2, n_iter):
        results[i] = max(results[i-1], gt_func(torch.randn(n*6).reshape(-1,6)).max())
    return results