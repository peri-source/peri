import os
import sys
import pickle
import numpy as np
import trackpy as tp
from pandas import DataFrame

from peri.test import analyze  #bad form b/c it's in the same folder?

def track(pos, rad, maxdisp=2., threshold=None):
    """
    Accepts positions and radii as a list of frames in the shape [N, D] where
    N in the number of particles (can vary by frame) and D the number of
    dimensions.
    """
    threshold = threshold or len(pos)
    z,y,x,a,f = [],[],[],[],[]

    for i,(p,r) in enumerate(zip(pos, rad)):
        z.extend(p[:,0].tolist())
        y.extend(p[:,1].tolist())
        x.extend(p[:,2].tolist())
        a.extend(r.tolist())
        f.extend([i]*r.shape[0])

    df = DataFrame({'x': x, 'y': y, 'z': z, 'rad': a,'frame': f})
    linked = tp.filter_stubs(
        tp.link_df(df, maxdisp, pos_columns=['x','y','z']),
        threshold=threshold
    )
    return linked

def get_xyzr_t(df, particle_ind):

    ind = df['particle'] == particle_ind

    rads = np.array(df['rad'][ind].copy())
    z = np.array(df['z'][ind].copy())
    x = np.array(df['x'][ind].copy())
    y = np.array(df['y'][ind].copy())

    return x,y,z,rads

def msd(df, maxlag, pxsize=0.126, scantime=0.1):
    drift = tp.compute_drift(df, pos_columns=['x','y','z'])
    tm = tp.subtract_drift(df, drift)
    em = tp.emsd(tm, pxsize, scantime, max_lagtime=maxlag, pos_columns=['x','y','z'])
    return em

def calculate_state_radii_fluctuations(state_list, inbox=True, fullinbox=True,
         inboxrad=False, maxdisp=2., threshold=None, return_all=False):
    """
    Calculates the radii fluctuations throughout a series of images.

    Parameters
    ----------
        state_list : List
            List of peri.states objects to analyze.

        inbox : Bool
            Set to False to include all particles, not just the ones in the
            image. Default is True (only particles in image).
        fullinbox : Bool
            Set to True to only include particles which are
        inboxrad : Bool
            Set to True to include all particles which overlap at all with the
            image. Default False

        maxdisp : Float
            The maximum displacement a particle can do between frames.
        threshold : Int
            Threshold for trackpy.filter_stubs

        return_all : Bool
            Set to True to return the mean z,y,x,r for all the particles
            as well as r.std(). Default is False (just returns r.std()

    Outputs
    --------
    """
    mask_list = [analyze.good_particles(s, inbox=inbox, inboxrad=inboxrad,
            fullinbox=fullinbox) for s in state_list]
    pos = [s.obj_get_positions()[m] for s, m in zip(state_list, mask_list)]
    rad = [s.obj_get_radii()[m] for s, m in zip(state_list, mask_list)]
    df = track(pos, rad, maxdisp=maxdisp, threshold=threshold)

    zyxr_t = []
    for i in np.unique(df['particle']):
        x,y,z,r = get_xyzr_t(df, 1*i)
        if z.size > 1:
            zyxr_t.append([x.mean(), y.mean(), z.mean(), r.mean(), r.std()])
    z, y, x, rm, rs = np.transpose(zyxr_t)
    if return_all:
        return z, y, x, rm, rs
    else:
        return rs

"""
for a in xrange( int( df['particle'].max() ) ):
    x,y,z,r = get_xyzr_t(df, 1*a )
    if x.size > 1:
        r_moments.append( [x.mean(), y.mean(), z.mean(), r.mean(),r.std()] )

#Just fucking around now:
drift = tp.compute_drift(linked)
tm = tp.subtract_drift( linked, drift )
em = tp.emsd( tm, 1.0, 1.0, max_lagtime = 10, pos_columns = ['x','y','z'] )
"""
