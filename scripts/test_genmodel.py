
# In this script, we use several demo images to check the quality of our
# generative model. We'll start with the simplest possible image -- a blank
# ``image`` taken without a sample -- and gradually move up in complexity
# to a real microscope image.
import numpy as np
import matplotlib.pyplot as plt


from peri import states
from peri import models
from peri import util
from peri.comp import ilms, objs, exactpsf, comp
import peri.opt.optimize as opt

from peri.viz.interaction import *  # OrthoViewer & OrthoManipulator

# We start with featuring just a background image
# This image was taken with no sample, i.e. we're just measuring dark current
im_bkg = util.RawImage('./bkg_test.tif')  # located in the scripts folder

# First we try with just a constant background
bkg_const = ilms.LegendrePoly3D(order=(1,1,1))
# Since we're just fitting a blank image, we don't need a psf at first, so we
# use the simplest model for the state: a SmoothFieldModel, which has just
# returns the illumination field:
st = states.ImageState(im_bkg, [bkg_const], mdl=models.SmoothFieldModel())
opt.do_levmarq(st, st.params)

# Since there's not a whole lot to see in this image, looking at the
# OrthoViewer or OrthoManipulator doesn't provide a lot of insight. Instead,
# we look at plots of the residuals along certain axes. We'll do this several
# times so I'll make a function:
def plot_averaged_residuals(st):
    plt.figure(figsize=[15,6])
    for i in range(3):
        plt.subplot(1,3,1 + i)
        mean_ax = tuple({0,1,2} - {i})  # which 2 directions to average over
        plt.plot(st.residuals.mean(axis=mean_ax))
        plt.title('{}-averaged'.format(['$xy$', '$xz$', '$yz$'][i]),
                fontsize='large')

plot_averaged_residuals(st)

# From this we see that, while the background doesn't change much along z, it
# increases smoothly along y (the in-plane direction perpendicular to our line
# illumination; probably due to changing dwell times during the scan), and it
# changes in a bumpy manner along x (the direction of the line). This suggests
# we use a higher-order background -- perhhaps cubic in y and high-order in
# x to capture the oscillations

bkg_vry = ilms.LegendrePoly3D(order=(1,3,5))
st.set('ilm', bkg_vry)
opt.do_levmarq(st, st.params)

# Looking at the plot of the residuals again shows a significant improvement
# in the residuals:

plot_averaged_residuals(st)

# Next, let's check the illumination field. For this, we load a different
# image, one that I've taken of just dyed fluid. This image also has a
# coverslip in it, at the bottom. For now, we'll ignore this coverlip by
# setting the tile to be a specific region of z in the image. Moreover,
# since I know that our confocal has some scan issues at the edges of the
# image, I'll also crop out the image edges with the tile:
im_ilm = util.RawImage('./ilm_test.tif', tile=util.Tile([48,0,0], [49,100,100]))
# also located in the scripts folder

# Looking at the image, the illlumination is very stripey, due to the line-scan
# nature of our confocal. To account for this, we use a stripe-based ilm:
ilm = ilms.BarnesStreakLegPoly2P1D(npts=(50, 30, 20, 13, 7, 7, 7), zorder=1)
# (we only use a zorder of 1 since we've truncated to 1 pixel in z).
# Our real model will use a point-spread function that will blur out the ilm
# field slightly more. So we check the fit with a model that includes the
# type of point-spread function that we will use. A model that blur with a
# point-spread function takes considerably more time to evaluate than a
# SmoothFieldModel, so if you're not sure if your ilm is high enough order
# you should first check with a faster SmoothFieldModel.

psf = exactpsf.FixedSSChebLinePSF()
st = states.ImageState(im_ilm, [ilm, psf], mdl=models.BlurredFieldModel())
opt.do_levmarq(st, st.params)


# Plotting the residuals shows that they're good, aside from scan noise
# inherent to the line CCD camera:

plot_averaged_residuals(st)


# Next, we include the coverslip slide. To do this we first re-set the tile on
# our raw image to the full image:
im_ilm.set_tile(util.Tile([0,0,0], [60, 100, 100]))

# We then create a coverslip object:
slab = objs.Slab(zpos=35.0, category='obj')
# We also need our illumination to have a z-dependence now. Since we already
# spent time updating the ilm parameters, we update the corresponding values
# of the new ilm to the older ones:
ilm_z = ilms.BarnesStreakLegPoly2P1D(npts=(50, 30, 20, 13, 7, 7, 7), zorder=7)
ilm_z.set_values(ilm.params, ilm.values)
# Keep in mind that setting the parameters only works for this certain
# ilm classes. The BarnesStreakLegPoly2P1D (1) has the same named Barnes
# parameters regardless of the z-order, and (2) these determine the ilm field
# in the xy-plane in the same way when the number of points is the same and
# the image shape is the same. In contrast, if we had fit the in-plane
# illumination with a lower set of npts, just setting the parameters wouldn't
# work. [This is because the BarnesStreakLegPoly2P1D barnes parameters are
# labeled according to their distance from the leftmost edge of the image. So
# ilm-b0-49 would be on the rightmost side of the image if npts=(50,...), but
# it would be in the middle of the image if npts=(100,...).] We could get
# around this case by re-fitting the ilm when we start to fit the state below


# We need to add a background. In principle, we could be use the same bkg
# that worked for our blank image. However, in practice this doesn't work so
# well, leaving noticeable residuals in z (try it!). The reason for this is
# that the point-spread function has very long, power-law tails. While the psf
# describes the image of a point fairly well, when the psf is integrated over
# the entire area of the coverslip these tails become very long, too long to
# capture with a reasonably-sized numerical psf. To account for this, we do
# some minor analytical calculations and realize that the effect of the long-
# tails of the psf when convolved with a slab looks like a background that
# varies slowly in z. Thus, to account for some of the long-tails in the psf,
# we use a background which varies in z. Since this z-dependence won't couple
# with the dark-current xy dependence in our detector, we can split this out
# as bkg = f(x,y) + g(z), like so:

bkg = ilms.LegendrePoly2P1D(order=(7,3,5), category='bkg', operation='+')

# This detail is important not so much for its effect on the reconstruction
# of this blank image, but for what it illustrates -- while it is practically
# impossible to implement an exact generative model, careful thinking can allow
# for a model that is almost equivalent to the exact answer. To answer how
# much this approximation matters for measuring particle properties in an,
# we could generate an image with a more exact representation of these psf
# long tails and then fit it with our more approximate model.
# Incidentally, while the support of our psf is finite, it's quite large --
# 35 pixels in z, or 44% of the image in z! If we wanted, we could increase
# this by changing the ``support_size`` keyword when calling
# exactpsf.FixedSSChebLinePSF.


# Finally, we create an offset:
off = comp.GlobalScalar('offset', 0.0)
st = states.ImageState(im_ilm, [ilm_z, off, psf, bkg, slab])
# As an illustration, we'll optimize certain parameters first for speed.
# Since we know that our xy-ilm parameters are the same, we'll start by
# optimizing the background and the ilm-z- params.
opt.do_levmarq(st, st.get('bkg').params + ['ilm-z-{}'.format(i) for i in
                range(ilm_z.zorder)], max_iter=2)
# Looking at this with the OrthoManipulator it already looks good, but we do
# a full optimization to ensure that we're at the best fit.
opt.do_levmarq(st, st.params, exptol=1e-5, errtol=1e-3)
# (this will take some time; half an hour or so on my machine)

# Finally, plotting the average along different directions looks good:

plot_averaged_residuals(st)

# With the OrthoManipulator, we can also see that our fit looks good:
OrthoManipulator(st)
