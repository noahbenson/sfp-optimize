################################################################################
# sfp-optimize/model.py
# Implementation of the spatial frequency model.
# by Noah C. Benson

import pimms
from .util import *

def preferred_period(ang, ecc, theta, params):
    '''
    preferred_period(ang, ecc, theta, params) yields the preferred spatial
    period of the given coordinate given by polar angle ang (in radians) and
    eccentricity ecc (in degrees), if the stimulus spatial frequency is
    oriented in the given direction theta.The argument params must be a 
    dictionary of parameters (a, b, p1, p2, p3, p4).
    '''
    a  = params['a']
    b  = params['b']
    p1 = params['p1']
    p2 = params['p2']
    p3 = params['p3']
    p4 = params['p4']
    p = (a*ecc + b) * (1 + 
                       p1*cos(2*theta) + 
                       p2*cos(4*theta) + 
                       p3*cos(2*(theta - ang)) + 
                       p4*cos(4*(theta - ang)))
    return p
def preferred_frequency(ang, ecc, theta, params):
    '''
    preferred_frequency(ang, ecc, theta, params) yields the preferred
    spatial frequency of the given coordinate given by polar angle ang (in
    radians) and eccentricity ecc (in degrees), if the stimulus spatial
    frequency is oriented in the given direction theta. The argument params
    must be a dictionary of parameters (a, b, p1, p2, p3, p4).
    '''
    p = preferred_period(ang, ecc, theta, params)
    return 1.0/p
def max_amplitude(ang, ecc, theta, params):
    '''
    max_amplotude(ang, ecc, theta, params) yields the maximum amplitude for the
    prefered spatial frequency at the given coordinate given by polar angle ang
    (in radians) and eccentricity ecc (in degrees), if the stimulus spatial
    frequency is oriented in the given direction theta. The argument params
    must be a dictionary of parameters (A1, A2, A3, A4).
    '''
    A1 = params['A1']
    A2 = params['A2']
    A3 = params.get('A3', 0.0)
    A4 = params.get('A4', 0.0)
    A = 1 + A1*cos(2*theta) + A2*cos(4*theta)
    if A3 is not None and A3 != 0:
        A = A + A3*cos(2*(theta - ang))
    if A4 is not None and A4 != 0:
        A = A + A4*cos(4*(theta - ang))
    return A
def beta(ang, ecc, theta, omega, params):
    '''
    beta(ang, ecc, theta, omega, params) yields the predicted beta 
    value (BOLD response) at the given visual field coordinate, given by polar
    angle ang (in radians) and eccentricity ecc (in degrees), if the stimulus
    spatial frequency is oriented in the given direction theta (in radians)
    with the given spatial frequency omega (in cpd). The argument params must
    be a dictionary of parameters (sigma, a, b, p1, p2, p3, p4, A1, A2, A3,
    and A4).
    '''
    sigma = params['sigma']
    A = max_amplitude(ang, ecc, theta, params)
    p = preferred_period(ang, ecc, theta, params)
    return A * exp(-0.5 * (log2(omega * p) / sigma)**2)
def init_image(direction='radial', max_eccen=12, image_size=None):
    '''
    init_image(direction, max_eccen, image_size) initializes an image for use
    with the *_image() functions below, and yields a tuple of (angle, eccen, 
    theta) where angle is an image of the polar angle of each pixel in ccw
    radians starting at tthe RHM, eccen is the eccentricity of each pixel in
    visual degrees, and theta is the angular component to the cos() funcntion
    such that cos(theta) yields the gradient image.
    '''
    # If we're given a string for the direction, we need to construct an image
    # that goes with it; otherwise we already have the image ready.
    if pimms.is_str(dirction):
        if image_size is None: image_size = 512
    else:
        if direction.shape[0] != direction.shape[1]:
            raise ValueError("square images are required")
        image_size = direction.shape[0]
    x = np.linspace(-max_eccen, max_eccen, image_size)
    (x,y) = np.meshgrid(x, x)
    ang = np.arctan2(y, x)
    ecc = np.sqrt(x**2 + y**2)
    # We also want an image of theta values
    if direction in ('radial', 'rad', 'r'):
        theta = ang
    elif direction in ('tangential', 'tan', 't'):
        theta = ang + np.pi/2
    elif direction in ('horizontal', 'hrz', 'h'):
        theta = np.pi/2
    elif direction in ('vertical', 'vrt', 'v'):
        theta = 0
    else:
        theta = direction
    return (ang, ecc, theta)
def preferred_frequency_image(params, direction='radial',
                              max_eccen=12, image_size=None):
    '''
    preferred_frequency_image(params) yields an image of the preferred spatial
    frequency for the given parameter dictionary.
    
    The following options can be given:
      * direction (default: 'radial') may be 'radial', 'tangential',
        'horizontal', or 'vertical'. Specifies the kind of stimulus to produce
        the image for. Alternately, the angle parameter of the cosine function
        used to construct the gradient may be passed (i.e., the true image to
        make predictions for is cos(direction), not direction itself); in this
        case, the direction argument is the image, template, and so the
        image_size option is ignored.
      * max_eccen (default: 12) may be the maximum eccentricity represented in
        the produced image.
      * image_size (default: 512) may be the size of the produced image,
        assuming that the direction parameter isn't an image itself.
    '''
    # Initialize the image:
    (ang, ecc, theta) = init_image(direction, max_eccen, image_size)
    # Get the preferred spatial frequency
    p = preferred_frequency(ang, ecc, theta, params)
    return p
def beta_image(params, omega=1, direction='radial',
               max_eccen=12, image_size=None):
    '''
    beta_image(params) yields an image of the predicted beta (BOLD) response
    image for the given parameter dictionary.
    
    The following options can be given:
      * omega (default: 1) specifies the preferred spatial frequency for the
        model in question. This may be an image or an individual value.
      * direction (default: 'radial') may be 'radial', 'tangential',
        'horizontal', or 'vertical'. Specifies the kind of stimulus to produce
        the image for. Alternately, the angle parameter of the cosine function
        used to construct the gradient may be passed (i.e., the true image to
        make predictions for is cos(direction), not direction itself); in this
        case, the direction argument is the image, template, and so the
        image_size option is ignored.
      * max_eccen (default: 12) may be the maximum eccentricity represented in
        the produced image.
      * image_size (default: 512) may be the size of the produced image,
        assuming that the direction parameter isn't an image itself.
    '''
    # Initialize the image:
    (ang, ecc, theta) = init_image(direction, max_eccen, image_size)
    # Get the preferred spatial frequency
    p = beta(ang, ecc, theta, omega, params)
    return p
