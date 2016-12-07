import os
import sys
import pickle
import numpy as np
import trackpy as tp
from pandas import DataFrame

from peri.test import analyze  #bad form b/c it's in the same folder?

def track(pos, rad, maxdisp=2., threshold=None):
    """
    Tracks a set of positions and radii using trackpy.

    Accepts positions and radii as a list of frames in the shape [N, D] where
    N in the number of particles (can vary by frame) and D the number of
    dimensions. Returns a linked pandas.DataFrame from trackpy.link_df that
    is filtered py trackpy.filter_stubs.

    Paramters
    ---------
        pos : list
            N-element list of numpy.ndarrays of the particle positions,
            each of shape [ni, d] where ni is the number of particles in
            that frame and d the number of positional dimensions.
        rad : list
            N-element list of numpy.ndarrays of the particle radii, each
            of shape [ni]
        maxdisp : Float, optional
            The maximum displacement of a particle from frame-to-frame.
            Default is 2.
        threshold : Float or None, optional
            A threshold to filter out trajectories of particles appearing
            in very few frames. Default is None.

    Returns
    -------
        linked : `pandas.DataFrame`
            Tracked set of particle positions and radii.

    See Also
    --------
        `trackpy.locate`
        `trackpy.filter_stubs`
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
    """
    Returns the x,y,z, radii values of a dataframe for a given particle index.

    Parameters
    ----------
        df : DataFrame
        particle_ind : Int
    Returns
    -------
        x, y, z, rads : numpy.ndarray
    """
    ind = df['particle'] == particle_ind

    rads = np.array(df['rad'][ind].copy())
    z = np.array(df['z'][ind].copy())
    x = np.array(df['x'][ind].copy())
    y = np.array(df['y'][ind].copy())

    return x,y,z,rads

def msd(df, maxlag, pxsize=0.126, scantime=0.1):
    """Calculates the mean-square displacement of a dataframe using trackpy"""
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
        state_list : list
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

    Returns
    -------
        [z,] : np.ndarray
            The mean z-positions of each particle. Returned if `return_all`
            is True.
        [y,] : np.ndarray
            The mean y-positions of each particle. Returned if `return_all`
            is True.
        [x,] : np.ndarray
            The mean x-positions of each particle. Returned if `return_all`
            is True.
        [rm,] : np.ndarray
            The mean radii of each particle. Returned if `return_all` is
            True.
        rs : np.ndarray
            The standard deviation of each particle's radius across all
            states.
    """
    #FIXME This could be updated to get the mask, pos, rad all at once
    #   and make state_list a generator
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
