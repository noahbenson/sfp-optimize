################################################################################
# sfp-optimize/calc.py
# Calculation code for the optimization; this uses pimms to organize the
# calculations.
# by Noah C. Benson

from .util import *
from .model import *

# For startes, we need to know the model parameters for the spatial frequency
# responses. We can not only set up these parameters but also construct a
# function that yields predictions.
@pimms.calc('model_params', 'model_fn')
def calc_model_params(sigma=2.2, a=0.12, b=0.33,
                      p=(0.06, -0.03, 0.07, 0.002),
                      A=(0.04, -0.01, 0, 0)):
    '''
    Given the parameters sigma, a, b, p (4 values), and A (4 values), 
    calculates the model_params (a dictionary) and model_fn values.
    
    Afferent parameters:
      @ sigma Must be the value of sigma to use in the spatial frequency
        preference model. Sigma is the width of the log-Gaussian.
      @ a Must be the a parameter to use in thee spatial frequency preeference
        model. The a parameter is the slope of the relationship between the
        eccentricity and the preferred spatial frequency.
      @ b Must be the a parameter to use in thee spatial frequency preeference
        model. The b parameter is the offset of the relationship between the
        eccentricity and the preferred spatial frequency.
      @ p Must be a 4-tuple of parameters to use in the spatial frequency
        preference model.
      @ A Must be a 4-tuple of parameters to use in the spatial frequency
        preference model. If this is only a 2-tuple, the last two values are
        assumed to be 0.
        
    Efferent values:
      @ model_params A dictionary of the parameters used in the spatial
        frequency preference model. The parameter names are: 'sigma', 'a', 'b',
        'p1', 'p2', 'p3', 'p4', 'A1', 'A2', 'A3', and 'A4'.
      @ model_fn A function that, when called as follows:
        `model_fn(theta_rad, eccen_deg, alpha_rad, omega_cpd)` where theta_rad
        is the polar angle, alpha_rad is the stimulus angle, and omega_cpd is
        the stimulus spatial frequency in cycles per degeree, yields the beta
        value (BOLD response) predicted by the spatial frequency preference
        model.
    '''
    from pyrsistent import pmap
    sigma = totensor(sigma)
    a = totensor(a)
    b = totensor(b)
    if len(p) != 4:
        raise ValueError("p parameter must be a 4-tuple")
    (p1,p2,p3,p4) = [totensor(x) for x in p]
    if len(A) == 2:
        A = (A[0], A[1], 0, 0)
    elif len(A) != 4:
        raise ValueError("A parrameter must be a 4-tuple or 2-tuple")
    (A1,A2,A3,A4) = [totensor(x) for x in A]
    params = {'sigma': sigma, 'a': a, 'b': b,
              'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4,
              'A1': A1, 'A2': A2, 'A3': A3, 'A4': A4}
    params = pmap(params)
    def _model_fn(angle, eccen, theta, omega):
        return beta(angle, eccen, theta, omega, params)
    return (params, _model_fn)

# For an image with size n x n where the center of the image is the fovea and
# the number of pixels per degree is n x n, the visual-field coordinates of the
# image pixel at row r column c are expressed as:
# x = (c - h)/s; y = (h - r)/s, where h = (n-1)/2.
@pimms.calc('image_x_deg', 'image_y_deg', 'image_radius_deg', 'image_center_px',
            'max_eccen')
def calc_image_coords(pixels_per_degree=16, image_size_px=384):
    '''
    Given the pixels per degree and the image size (default: 512), calculates a
    2 x N matrix whose rows are the (x,y) coordinates, in visual field degrees,
    of the pixels in the image that is to be produced.
    
    Afferent parameters:
      @ pixels_per_degree Must be the number of pixels per degree of the visual
        field use in the output image.
      @ image_size_px Must be the size (width and height) of the desired image in
        pixels.
        
    Efferent values:
      @ image_x_deg An image matrix with <image_size> rows and columns whose 
        cells contain the x coordinate of thee optimized image's pixels.
      @ image_y_deg An image matrix with <image_size> rows and columns whose 
        cells contain the y coordinate of thee optimized image's pixels.
    '''
    if image_size_px % 2 != 0:
        raise ValueError("image_size_px must be even")
    h = (image_size_px - 1) * 0.5
    maxecc = h / pixels_per_degree
    xpix = np.linspace(-h, h, image_size_px)
    xdeg = xpix / pixels_per_degree
    # x and y are the same here, but the y is reversed due to columns
    # descending in images.
    (x,y) = np.meshgrid(xdeg, -xdeg)
    x = totensor(x, requires_grad=False)
    y = totensor(y, requires_grad=False)
    rad = 0.5 * image_size_px / pixels_per_degree
    return (x, y, rad, (h,h), maxecc)

# We also want to know what the polar angle and eccentricity is at each
# point. We use standard retinotopy form here (as used in the paper; see this
# image for a visual definition of standard retinotopy:
# https://github.com/noahbenson/neuropythy/wiki/files/retinotopystyles.png
@pimms.calc('image_eccen_deg', 'image_angle_rad')
def calc_image_angles(image_x_deg, image_y_deg):
    '''
    Calculates the eccentricity and polar-angle (angle) of the image pixels.
    
    Efferent values:
      @ image_eccen_deg An image matrix containing the eccentricity of each
        pixel in the optimized image. Eccentricity is measured in degrees of
        visual angle from the fovea.
      @ image_angle_rad An image matrix containing the polar angle of each
        pixel in the optimized image. Polar angle is measured in radians
        where 0 is the right horizontal meridian and the positive direction is
        counter-clockwise.
    '''
    ecc = torch.sqrt(image_x_deg**2 + image_y_deg**2)
    ang = torch.atan2(image_y_deg, image_x_deg)
    return (ecc, ang)

# In the optimization, we optimize over an image whose pixels represent a
# quantity phi; the actual stimulus image over which the spatial frequency model
# is calculated (on which the optimization is based) is cos(phi). So, for
# example, if phi = x, then the stimulus image will have vertical stripes
# (cos(x)). Here we set up the initial phi value in the optimization.
@pimms.calc('image_phi0_deg')
def calc_image_phi0(image_x_deg,     image_y_deg,
                    image_angle_rad, image_eccen_deg,
                    max_eccen, direction='radial'):
    '''
    Calculates the initial phi value used in the optimization. The value is
    based on the direction parameter, which indicates the direction of the
    gradient stripes: radial, tangential, horizontal, or vertical.
    
    Afferent parameters:
      @ direction Must specify the direction of gradient image that is
        optimized. Valid values are "radial", "tangential", "horizontal",
        "vertical", or an image with the same dimensions as the optimization
        image.
        
    Efferent values:
      @ image_phi0_deg The initial phi value used in the optimization. Unlike
        most values, this is a numpy array and not a pytorch tensor.
    '''
    sh = image_x_deg.shape
    if pimms.is_str(direction):
        if direction in ('radial', 'rad', 'r'):
            phi = safesqrt(0.2 + 6*max_eccen*image_eccen_deg)
        elif direction in ('tangential', 'tan', 't'):
            phi = (image_angle_rad + np.pi/2)*4
        elif direction in ('horizontal', 'hrz', 'h'):
            phi = image_x_deg
        elif direction in ('vertical', 'vrt', 'v'):
            phi = image_y_deg
        else:
            raise ValueError("Could not understand direction parameter")
    else:
        phi = direction
    if phi.shape != sh:
        raise ValueError("direction shape must match image size")
    if torch.is_tensor(phi):
        phi = phi.detach().numpy()
    return (pimms.imm_array(phi),)

# We also want those values as vectors (for convenience: we will operate on the
# vectors instead of the images). Because we will operate on these, we make them
# torch tensors.
@pimms.calc('x_deg', 'y_deg', 'eccen_deg', 'angle_rad', 'phi0_deg',
            'tensor_indices')
def calc_image_tensors(image_x_deg, image_y_deg,
                       image_eccen_deg, image_angle_rad,
                       image_phi0_deg):
    '''
    Calculates the torch tensors for the x, y, eccen, and theta values.
    
    Efferent values:
      @ x_deg A torch tensor containing the x coordinates, in visual degrees,
        of the optimized image pixels (flattened).
      @ y_deg A torch tensor containing the y coordinates, in visual degrees,
        of the optimized image pixels (flattened).
      @ eccen_deg A torch tensor containing the eccentricity, in visual
        degrees, of the optimized image pixels (flattened).
      @ angle_rad A torch tensor containing the polar angle, in radians,
        of the optimized image pixels (flattened).
      @ phi0_deg A torch tensor containing the initial phi values, in visual
        degrees, of the optimization image.
      @ tensor_indices A tuple (ii,jj) such that if im is a numpy array
        the size of the optimized image, then im[ii,jj] = x_deg would
        correctly match thee x_deg values to the image pixels. This
        would work for x_deg, y_deg, eccen_deg, and theta_rad.
    '''
    ii = np.arange(image_x_deg.shape[0])
    (ii,jj) = np.meshgrid(ii, ii)
    ii = ii.flatten()
    jj = jj.flatten()
    lidcs = np.reshape(np.arange(len(ii)), image_x_deg.shape)
    a = lidcs[1:,:-1].flatten()
    b = lidcs[1:,1:].flatten()
    c = lidcs[:-1,:-1].flatten()
    d = lidcs[:-1,1:].flatten()
    for u in (ii,jj,a,b,c,d):
        u.setflags(write=False)
    (x, y, ecc, ang) = [
        totensor(im.detach().numpy().flatten(), requires_grad=False)
        for im in [image_x_deg, image_y_deg, image_eccen_deg, image_angle_rad]]
    phi0 = totensor(image_phi0_deg.flatten(), requires_grad=False)
    return (x, y, ecc, ang, phi0, (ii, jj))

# Finally, we set up the tensor that gets optimized and the instructions for
# optimization themselves.
@pimms.calc('image_phi_deg')
def calc_phi(image_phi0_deg):
    '''
    Calculates the phi tensor. This is duplicated from phi0. The phi tensor is
    the only tensor whose gradient is required and that is expected to change
    as the optimization proceeds.
    
    Efferent values:
      @ image_phi_deg The phi value tensor that is optimized by pytorch.
    '''
    return (totensor(image_phi0_deg, requires_grad=True),)
def loss_smoothness(im, circular=False):
    '''
    loss_smoothness(im) yields a loss value based on the smoothness of the
      given image im. A less smooth image yields a higher loss.
    loss_smoothness(im, circular=True) uses a circular loss function, which
      assumes a wavelength of 2 pi radians.
    
    The calculated loss value is equal to the sum of squares of the differences
    between the adjacent pixels, divided by the number of such pixel pairs. For
    a circular smoothness, the value is sum of one minus the cosines of the 
    angles between adjacent pixels divided by the number of pixel pairs.
    '''
    (rs,cs) = im.shape
    den = (rs-1)*cs + (cs-1)*rs
    if circular:
        # We assume im is in radians
        s = sin(im)
        c = cos(im)
        # The square of the cos of the angle between the directions
        # is not a bad metric of loss.
        drs = 1 - (s[1:,:] * s[:-1,:] + c[1:,:] * c[:-1,:])
        dcs = 1 - (s[:,1:] * s[:,:-1] + c[:,1:] * c[:,:-1])
        drs = sum(drs)
        dcs = sum(dcs)
    else:
        drs = sum((im[1:,:] - im[:-1,:])**2)
        dcs = sum((im[:,1:] - im[:,:-1])**2)
    return (drs + dcs) / den
def loss(phi_deg, ang_rad, ecc_deg, params, pixels_per_degree, knob=0):
    '''
    The loss function is the variance of the beta prediction.
    '''
    # Calculate the spatial frequency statistics.
    (theta, omega) = image_sfstats(phi_deg)
    # Convert to omega from cycles per pixel to to cycles per degree.
    omega *= pixels_per_degree
    # Theta needs to be fixed to be in [0, pi]
    theta = torch.remainder(theta, np.pi)
    # Calculate the predicted beta value for theta and omega.
    b = beta(ang_rad, ecc_deg, theta, omega, params)
    # The loss is the variance of this prediction.
    const = 0 if knob is None else 2.0 ** knob
    smoothloss = loss_smoothness(b)
    return var(b) + const*smoothloss
@pimms.calc('image')
def calc_optimization(image_phi_deg, image_angle_rad, image_eccen_deg,
                      model_params, pixels_per_degree,
                      steps=2000, lr=0.0004, knob=-2, tracking=None):
    '''
    Calculates the result of optimizing the image_phi_deg for a certain number
    of steps.
    
    Afferent parameters:
      @ steps Specifies the number of steps to run the optimization.
      @ lr Specifies the learning rate to use in the optimization.
      @ knob Specifies the relative weight of the angle-smoothness component of
        the objective function relative to the variance component of the
        objective function. Turn this up to prevent local high frequencies in
        the output image. If knob is None, then the weight on the smoothness
        component is 0; otherwise it is 2^knob times the weight on the variance
        component.
      @ tracking May be a mutable python list to which the loss is appended
        after each step.
    '''
    opt = torch.optim.Adagrad([image_phi_deg], lr=lr)
    for step in range(steps):
        def closure():
            opt.zero_grad()
            l = loss(image_phi_deg, image_angle_rad, image_eccen_deg,
                     model_params, pixels_per_degree, knob=knob)
            if tracking is not None:
                tracking.append(float(l))
            l.backward()
            return l
        opt.step(closure)
    # restructure into an image
    phi_im = image_phi_deg.detach().numpy()
    im = np.cos(phi_im)
    return (im,)

# The overall calculation plan:
plan_dict = {'model_params':  calc_model_params,
             'image_coords':  calc_image_coords,
             'image_angles':  calc_image_angles,
             'image_phi0':    calc_image_phi0,
             'image_tensors': calc_image_tensors,
             'phi':           calc_phi,
             'optimization':  calc_optimization}
plan = pimms.plan(plan_dict)
