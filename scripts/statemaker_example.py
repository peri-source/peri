from peri import states, runner
from peri.comp import ilms, objs, comp, exactpsf


def makestate(im, pos, rad, slab=None, mem_level='hi'):
    """
    Workhorse for creating & optimizing states with an initial centroid
    guess.

    This is an example function that works for a particular microscope. For
    your own microscope, you'll need to change particulars such as the psf
    type and the orders of the background and illumination.

    Parameters
    ----------
        im : :class:`~peri.util.RawImage`
            A RawImage of the data.
        pos : [N,3] element numpy.ndarray.
            The initial guess for the N particle positions.
        rad : N element numpy.ndarray.
            The initial guess for the N particle radii.

        slab : :class:`peri.comp.objs.Slab` or None, optional
            If not None, a slab corresponding to that in the image. Default
            is None.
        mem_level : {'lo', 'med-lo', 'med', 'med-hi', 'hi'}, optional
            A valid memory level for the state to control the memory overhead
            at the expense of accuracy. Default is `'hi'`

    Returns
    -------
        :class:`~peri.states.ImageState`
            An ImageState with a linked z-scale, a ConfocalImageModel, and
            all the necessary components with orders at which are useful for
            my particular test case.
    """
    if slab is not None:
        o = comp.ComponentCollection(
                [
                    objs.PlatonicSpheresCollection(pos, rad, zscale=zscale),
                    slab
                ],
                category='obj'
            )
    else:
        o = objs.PlatonicSpheresCollection(pos, rad, zscale=zscale)

    p = exactpsf.FixedSSChebLinePSF()
    npts, iorder = _calc_ilm_order(im.get_image().shape)
    i = ilms.BarnesStreakLegPoly2P1D(npts=npts, zorder=iorder)
    b = ilms.LegendrePoly2P1D(order=(9 ,3, 5), category='bkg')
    c = comp.GlobalScalar('offset', 0.0)
    s = states.ImageState(im, [o, i, b, c, p])
    runner.link_zscale(s)
    if mem_level != 'hi':
        s.set_mem_level(mem_level)

    opt.do_levmarq(s, ['ilm-scale'], max_iter=1, run_length=6, max_mem=1e4)
    return s

def _calc_ilm_order(imshape):
    """
    Calculates an ilm order based on the shape of an image. This is based on
    something that works for our particular images. Your mileage will vary.

    Parameters
    ----------
        imshape : 3-element list-like
            The shape of the image.

    Returns
    -------
        npts : tuple
            The number of points to use for the ilm.
        zorder : int
            The order of the z-polynomial.
    """
    zorder = int(imshape[0] / 6.25) + 1
    l_npts = int(imshape[1] / 42.5)+1
    npts = ()
    for a in range(l_npts):
        if a < 5:
            npts += (int(imshape[2] * [59, 39, 29, 19, 14][a]/512.) + 1,)
        else:
            npts += (int(imshape[2] * 11/512.) + 1,)
    return npts, zorder

