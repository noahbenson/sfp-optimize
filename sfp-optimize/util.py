################################################################################
# sfp-optimize/util.py
# Utilities used by the sfp-optimize library.
# by Noah C. Benson

import numpy as np
import torch

# Handy utilities for dealing with either torch or numpy objects:
def safesqrt(u):
    '''
    safesqrt(u) is equivalent to torch.sqrt(u) but only operates on values that
      are greater than 0.
    '''
    if not torch.is_tensor(u):
        return np.sqrt(u)
    u = u + 0
    ii = u != 0
    u[ii] = torch.sqrt(u[ii])
    return u
def limit_param(param, min=-1, max=1):
    '''
    limit_param(q) yields the parameter q rescaled to be in the range -1 to 1.
    limit_param(q, min, max) uses the given minimum and maximum parameters.
    '''
    import torch
    return min + (max - min) * (0.5 + torch.atan(param)/np.pi)
def unlimit_param(param, min=-1, max=1):
    '''
    unlimit_param(q, ...) is the inverse of limit_param(q, ...).
    '''
    import torch
    return torch.tan(np.pi * ((param - min) / (max - min) - 0.5))
def totensor(u, **kw):
    '''
    totensor(u) yields u if u is a pytorch tensor; otherwise, converts u to a
      pytorch tensor and yields that tensor. Keyword arguments to the tensor
      function may be passed, and, if passed, a copy of the u is always made
      when u is a tensor.
    '''
    import torch
    if torch.is_tensor(u):
        if len(kw) == 0: return u
        u = u.detach().numpy()
    dtype = kw.get('dtype', None)
    if dtype is None or dtype is torch.float:
        u = np.array(u, dtype='=f')
    else:
        tmp = torch.tensor(0.0, dtype=dtype)
        tmp = tmp.detach().numpy().dtype
        u = np.array(u, dtype=tmp)
    return torch.tensor(u, **kw)
def branch(iftensor, thentensor, elsetensor=None):
    '''
    branch(q, t, e) yields, elementwise for the given tensors, t if q else e.
    branch(q, t) or branch(q, t, None) yields, elementwise, t if q else 0.
    branch(q, None, e) yields, elementwise, 0 if q else e.
    
    The output tensor will always have the same shape as q. The values for t
    and e may be constants or tensors the same shape as q.

    This function should be safe to use in optimization, i.e., for gradient
    calculatioins.
    '''
    q = totensor(iftensor)
    t = None if thentensor is None else totensor(thentensor)
    e = None if elsetensor is None else totensor(elsetensor)
    x = t if e is None else e
    if x is None: raise ValueError('branch: both then and else cannot be None')
    r = torch.zeros(q.shape, dtype=x.dtype)
    if t is not None:
        if t.shape == (): r[q] = t
        else:             r[q] = t[q]
    if e is not None:
        q = ~q
        if e.shape == (): r[q] = e
        else:             r[q] = e[q]
    return r
def zinv(x):
    '''
    zinv(x) yields 0 if x == 0 and 1/x otherwise. This is done in a way that
      is safe for torch gradients; i.e., the gradient for any element of x that
      is equal to 0 will also be 0.
    '''
    x = totensor(x)
    ii = (x != 0)
    rr = torch.zeros(x.shape, dtype=x.dtype)
    rr[ii] = 1 / x[ii]
    return rr

# The following functions are all utility functions for interacting with either
# PyTorch or NumPy arrays; since we are generally using torch tensors for the
# minimization but are using numpy arrays for visualization and testing, it's
# nice to be able to call equivalent functions on either type.
def cos(x):
    '''Elementwise cosine that works on torch tensors or numpy arrays.'''
    if torch.is_tensor(x):
        return torch.cos(x)
    else:
        return np.cos(x)
def sin(x):
    '''Elementwise sine that works on torch tensors or numpy arrays.'''
    if torch.is_tensor(x):
        return torch.sin(x)
    else:
        return np.sin(x)
def exp(x):
    '''Elementwise exponential that works on torch tensors or numpy arrays.'''
    if torch.is_tensor(x):
        return torch.exp(x)
    else:
        return np.exp(x)
def log(x):
    '''Elementwise log that works on torch tensors or numpy arrays.'''
    if torch.is_tensor(x):
        return torch.log(x)
    else:
        return np.log(x)
def log2(x):
    '''Elementwise base-2 log that works on torch tensors or numpy arrays.'''
    if torch.is_tensor(x):
        return torch.log2(x)
    else:
        return np.log2(x)
def atan2(y, x):
    '''Elementwise arctan that works on torch tensors or numpy arrays.'''
    if torch.is_tensor(y):
        return torch.atan2(y,x)
    else:
        return np.arctan2(y,x)
def sqrt(x):
    '''Elementwise square root that works on torch tensors or numpy arrays.'''
    if torch.is_tensor(x):
        return torch.sqrt(x)
    else:
        return np.sqrt(x)
def sum(x, *a, **kv):
    '''Array/tensor summation that works on torch tensors or numpy arrays.'''
    if torch.is_tensor(x):
        return torch.sum(x, *a, **kv)
    else:
        return np.sum(x, *a, **kv)
def mean(x, *a, **kv):
    '''Array/tensor mean that works on torch tensors or numpy arrays.'''
    if torch.is_tensor(x):
        return torch.mean(x, *a, **kv)
    else:
        return np.mean(x, *a, **kv)
def var(x, *a, **kv):
    '''Array/tensor variance that works on torch tensors or numpy arrays.'''
    if torch.is_tensor(x):
        return torch.var(x, *a, **kv)
    else:
        return np.var(x, *a, **kv)
def image_sfstats(phi_image):
    '''
    image_sfstats(phi_image) yields a tuple of the (theta, omega) parameters
      of the spatial frequency of the image cos(phi_image).
      
    The phi parameter is intended to represent the underlying phase of the
    actual image whose spatial frequency is being calculated. This is required
    because while the spatial frequency statistics of an image are 
    straightforward to calculate when phi is known, they are difficult to
    calculate when only cos(phi) is known.
    
    The return value theta is in radians where the positive x direction (0 deg)
    indicates increasing image columns and the negative y direction (-90 deg)
    indicates increasing image rows.
    The return value omega is in units of cycles per pixel.
    '''
    phi = phi_image
    # Start by pre-allocating some result matrices.
    if torch.is_tensor(phi):
        dphi_drow = torch.zeros(phi.shape)
        dphi_dcol = torch.zeros(phi.shape)
    else:
        if not isinstance(phi, (np.ndarray, np.generic)):
            phi = np.asarray(phi)
        dphi_drow = np.zeros(phi.shape)
        dphi_dcol = np.zeros(phi.shape)
    # Next, we make short drow and dcol matrices. For drow, we want positive to
    # point down the image, so we subtract the higher index value from the
    # lower. For dcol, we want positive to point right along the image, so we
    # subtract the lower index value from the higher.
    dphi_drow_short = phi[:-1,:] - phi[1:, :]
    dphi_dcol_short = phi[:, 1:] - phi[:,:-1]
    # The above drow matrix is one row short of the full image and the dcol
    # matrix is one column short. To return this back to full size we use
    # quadratic interpolation to estimate the derivative at each pixel. For
    # pixels along the edge, quadratic interpolation of the closest three
    # pixels is used.
    #    In quadratic interpolation, where you have three points at (-1, yn),
    # (0, y0), and (1, yp), the interpolated polynomial a*x^2 + b*x + c
    # has parameter values a = (yn + yp - 2*y0)/2, b = (yp - yn)/2, c = y0.
    # However, the derivative at x = 0 is equal to b, so that's all we need
    # for most points. For the points along the edge, we have, for x = -1 and
    # x = 1 respectively, b - 2*a and b + 2*a. We can further simplify this
    # in terms of the ys to (4*y0 - 3*yn - yp)/2 and (3*yn + yp - 4*y0)/2.
    #    We start with the center of the matrices.
    dphi_drow[1:-1,:] = 0.5*(dphi_drow_short[:-1,:] + dphi_drow_short[1:,:])
    dphi_dcol[:,1:-1] = 0.5*(dphi_dcol_short[:,:-1] + dphi_dcol_short[:,1:])
    # Next the top and bottom rows.
    dphi_drow[0, :] = -0.5*(4*phi[1, :] - 3*phi[0, :] - phi[2, :])
    dphi_drow[-1,:] =  0.5*(4*phi[-2,:] - 3*phi[-1,:] - phi[-3,:])
    # Finnally the left and right columns.
    dphi_dcol[:, 0] =  0.5*(4*phi[:, 1] - 3*phi[:, 0] - phi[:, 2])
    dphi_dcol[:,-1] = -0.5*(4*phi[:,-2] - 3*phi[:,-1] - phi[:,-3])
    # We calculate omega and theta using these values.
    omega = safesqrt(dphi_drow**2 + dphi_dcol**2)
    theta = atan2(dphi_drow, dphi_dcol)
    # Currently omega is in units of radians per pixel. The last remaining
    # thing to do is to convert this to cycles per pixel.
    return (theta, omega / (2*np.pi))
