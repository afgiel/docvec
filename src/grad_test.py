import numpy as np
from simple_drnnlm import SimpleDRNNLM

wv_dummy = np.random.randn(10,50)
dv_dummy = np.random.randn(1, 50)*.01
model = SimpleDRNNLM(L0 = wv_dummy, D0 = dv_dummy, U0 = wv_dummy, alpha=0.005, rseed=10, bptt=4)
model.grad_check(np.array([1,2,3]), np.array([2,3,4]), [0])

model.generate_docvecs([np.array([1,2,3])], [np.array([2,3,4])], [0], dv_dummy)

from drnnlm import DRNNLM

wv_dummy = np.random.randn(10,50)
dv_dummy = np.random.randn(1, 50)*.01
model = DRNNLM(L0 = wv_dummy, D0 = dv_dummy, U0 = wv_dummy, alpha=0.005, rseed=10, bptt=4)
model.grad_check(np.array([1,2,3]), np.array([2,3,4]), [0])

model.generate_docvecs([np.array([1,2,3])], [np.array([2,3,4])], [0], dv_dummy)


