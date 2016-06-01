import pylab as pl
import numpy as np

from peri import states, util, initializers
from peri.comp import psfs, objs, ilms, exactpsf, GlobalScalar, ComponentCollection
from peri.viz import interaction

import peri.opt.optimize as opt
import peri.opt.addsubtract as addsub

# Set these two variables to the image that you want to feature.  Region is the
# part of the image of interest defined by left (l) and right (r) sides of a
# box in the order (z,y,x).  All coordinates are in this format. `description`
# is a short name for use in the save files -- e.g. 1.tif-peri-description.pkl
filename = '1.tif'
description = 'featured'
region = util.Tile(left=[0,0,0], right=[64,64,64])
im = util.RawImage(filename, tile=region)

# Next, fill in some information that you already know. For example, a guess
# at the particle positions and sizes. Again, these positions must be in
# (z,y,x) order just like the image itself.
#positions = np.loadtxt("guess-positions.csv", delimiter=',')
#radii = np.loadtxt("guess-radii.csv", delimiter=',')

# If you don't have a file of initial guesses, comment the lines above and
# run these instead.
radii = 5.0
positions = runner.locate_spheres(im, radii, order=(7,7,7), invert=False)

#=============================================================================
# From here on out we are describing the model which we will attempt to fit
#=============================================================================
# The set of objects in the image (spheres and coverslips). It uses the pos
# and radii from the previous section and maybe includes a coverslip that is
# located at coverslip_zpos (in pixel units from the bottom). To include the
# slab, uncomment L35
coverslip_zpos = 10.0

P = ComponentCollection([
    objs.PlatonicSpheresCollection(positions, radii),
    objs.Slab(coverslip_zpos)
], category='obj')

# PSF to use. This is the recommended one for our confocal, don't change
H = exactpsf.FixedSSChebLinePSF()

# The light intensity variation across the image. For a large image, these are
# the recommended settings. `npts` determines how many stripes are included in
# a series of terms (stripe_i * legendre poly_i) where the number of terms for
# stripe_i is npts[i]. zorder is the number of polynomial terms in the
# z-direction
I = ilms.BarnesStreakLegPoly2P1D(
    npts=(160,80,60,40,20,10,10,10), zorder=7, category='ilm'
)

# The background light intensity of imaging a blank room. However, do to PSF
# issues, this also accounts for long tails of the PSF interacting with the
# coverslip. Therefore, a high order poly in z is needed and a small one in
# the other directions. For smaller images, turn down these orders.
B = ilms.Polynomial3D(order=(27,3,3), category='bkg', constval=0.01)

# Don't modify this one
C = GlobalScalar('offset', 0.0)

# Create the fit state, don't modify (usually)
s = states.ImageState(im, [B, I, H, P, C], pad=24)

#=============================================================================
# Now we begin the optimization
#=============================================================================
opt.burn(s, mode='burn', n_loop=6, ftol=2e-3, max_mem=1e9, desc=description)

n, inds = addsub.add_subtract(s, max_rad=1.5*radii, quiet=False)
states.save(s, desc=description)

opt.burn(s, mode='burn', n_loop=6, ftol=2e-3, max_mem=1e9, desc=description)

