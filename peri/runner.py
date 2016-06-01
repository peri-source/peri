from peri import initializers, util, models, states
from peri.comp import ilms, objs, psfs, exactpsf, comp

import peri.opt.optimize as opt
import peri.opt.addsubtract as addsub

from peri.logger import log
log = log.getChild('runner')

def locate_spheres(image, radius, dofilter=True, order=(7,7,7), invert=False):
    """
    Get an initial featuring of sphere positions in an image.

    Parameters:
    -----------
    image : `peri.util.Image` object
        Image object which defines the image file as well as the region.

    radius : float
        Radius of objects to find, in pixels

    dofilter : boolean
        Whether to remove the background before featuring. Doing so can often
        greatly increase the success of initial featuring and decrease later
        optimization time.

    zslab : float
        Location of the coverslip in z pixel distance.

    invert : boolean
        Whether to invert the image for featuring. Should be True is the image
        is bright particles on a dark background.

    Returns:
    --------
    positions : np.ndarray [N,3]
        Positions of the particles in order (z,y,x) in pixel units of the image.
    """
    # We just want a smoothed field model of the image so that the residuals
    # are simply the particles without other complications
    m = models.SmoothFieldModel()

    I = ilms.LegendrePoly2P1D(order=order, constval=image.get_image().mean())
    s = states.ImageState(image, [I], pad=0, mdl=m)

    if dofilter:
        opt.do_levmarq(s, s.params)

    return addsub.feature_guess(s, radius, invert=not invert)[0]


