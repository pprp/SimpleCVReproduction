import torch
import numpy as np
import pdb
import random

def project(meta_weights, P, Q):
  """ project meta_weights to sub_weights
  Args:
    meta_weights: a 4-D tensor [cout, cin, k, k], the meta weights for one-shot model;
    P: a 2-D tensor [cout, cout_p], projection matrix along cout;
    Q: a 2-D tensor [cin, cin_p], projection matrix along cin;
  Return:
    proj_weights: a 4-D tensor [cout_p, cin_p, k, k], the projected weights;
  """

  if meta_weights.ndimension() != 4:
    raise ValueError("shape error! meta_weights should be 4-D tensors")
  elif meta_weights.shape[0] != P.shape[0] or meta_weights.shape[1] != Q.shape[0]:
    raise ValueError("shape mismatch! The projection axises of meta weights, P and Q should be consistent.")

  proj_weights = torch.einsum('ijhw,ix,jy->xyhw', meta_weights, P, Q)
  return proj_weights
