import numpy as np
from simple_drnnlm import SimpleDRNNLM

wv_dummy = np.random.randn(10,50)
model = SimpleDRNNLM(L0 = wv_dummy, D0 = wv_dummy, U0 = wv_dummy, alpha=0.005, rseed=10, bptt=4)
model.grad_check(np.array([1,2,3]), np.array([2,3,4]), 1)
