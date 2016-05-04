import numpy as np
from cbamf import initializers
import trackpy as tp
import cbamf.opt.optimize as opt

def feature_guess(st, rad, invert=True, minmass=None, use_tp=False):
    if minmass == None:
        #30% of the feature size mass is a good cutoff empirically for
        #initializers.local_max_featuring, less for trackpy; 
        #it's easier to remove than to add
        minmass = rad**3 * 4/3.*np.pi * 0.3 
        if use_tp:
            minmass *= 0.1 #magic #; works well
    if invert:
        im = 1 - st.get_difference_image()
    else:
        im = st.get_difference_image()
    if use_tp:
        diameter = np.ceil(2*rad)
        diameter += 1-(diameter % 2) 
        df = tp.locate(im, int(diameter), minmass = minmass)
        npart = np.array(df['mass']).size
        guess = np.zeros([npart,3])
        guess[:,0] = df['z']
        guess[:,1] = df['y']
        guess[:,2] = df['x']
        mass = df['mass']
    else:
        guess, _, mass = initializers.local_max_featuring(im, radius=rad, masscut=minmass)
        npart = guess.shape[0]
    #I want to return these sorted by mass:
    inds = np.argsort(mass)[::-1] #biggest mass first
    #Finally, we need to add the pad:
    guess += st.pad
    return guess[inds].copy(), npart

def check_add_particles(st, guess, rad='calc', im_change_frac=0.2, 
        do_opt=True, opt_box_scale=2.5, quiet=True):
    """
    comments
    st = state
    guess = list-like of poses to check to add, 
    rad = radius to add at. Default is 'calc' = np.median(st.obj.rad)
    im_change_frac : 0.2, how good the change in error needs to be relative
        to the change in the difference image. 

    """
    accepts = 0
    new_inds = []
    if rad == 'calc':
        rad = np.median(st.obj.rad[st.obj.typ==1])
    for a in xrange(guess.shape[0]):
        p = guess[a]
        if not quiet:
            old_err = opt.get_err(st)
        n = st.add_particle(p, rad)
        if do_opt:
            # db = opt_box_scale * np.ones(p.shape) * rad
            # bounds = [(p-db).tolist(), (p+db).tolist()]
            # inds = opt.find_particles_in_box(st, bounds)
            lp = opt.LMParticles(st, np.array([n],dtype='int'), damping=1.0, 
                    max_iter=1, run_length=3, particle_kwargs={'include_rad':False})
            lp.do_run_2()
            # opt.do_levmarq_particles(st, np.array([n],dtype='int'), damp=1.0, 
                    # num_iter=1, run_length=3, quiet=True, include_rad=False)
            # opt.do_levmarq_particles(st, inds, damp=1.0, 
                    # num_iter=1, run_length=3, quiet=True)
            #ok... but still doesn't get the clusters well
        did_kill = check_remove_particle(st, n, im_change_frac=im_change_frac)
        if not did_kill:
            accepts += 1
            new_inds.append(n)
            if not quiet:
                print '%d:\t%f\t%f' % (a, old_err, opt.get_err(st))
    return accepts, new_inds
    
def check_remove_particle(st, n, im_change_frac=0.2):
    present_err = opt.get_err(st); present_d = st.get_difference_image().copy()
    p, r = st.remove_particle(n)
    absent_err = opt.get_err(st); absent_d = st.get_difference_image().copy()
    
    im_change = np.sum((present_d - absent_d)**2)
    if (absent_err - present_err) >= im_change_frac * im_change:
        #If present err is sufficiently better than absent, add it back
        #if present_err - absent_err is sufficiently negative, add it back
        #if absent_err - present_err is sufficiently positive, add it back
        st.add_particle(p, r)
        killed = False
    else:
        killed = True
    return killed

def sample_n_add(st, rad='calc', tries=20, quiet=True, do_opt=True, 
        im_change_frac=0.2, opt_box_scale=3, **kwargs):
    if rad == 'calc':
        rad = np.median(st.obj.rad[st.obj.typ==1])
    
    guess, npart = feature_guess(st, rad, **kwargs)
    tries = np.min([tries, npart])
    
    accepts, new_inds = check_add_particles(st, guess[:tries], rad=rad, 
            im_change_frac=0.2, do_opt=do_opt, opt_box_scale=opt_box_scale, 
            quiet=quiet)
    return accepts, new_inds

def remove_bad_particles(s, min_rad=2.0, max_rad=12.0, min_edge_dist=2.0, 
        check_rad_cutoff=[3.5,15], check_outside_im=True, tries=100, 
        im_change_frac=0.2, quiet=True, **kwargs):
    """
    Same syntax as before, but here I'm just trying to kill the smallest particles...
    I don't think this is good because you only check the same particles each time
    Updates a single particle (labeled ind) in the state s. 
    
    Parameters
    -----------
    min_rad : Float
        All particles with radius below min_rad are automatically deleted. 
        Set to 'calc' to make it the median rad - 15* radius std. 
        Default is 2.0

    max_rad : Float
        All particles with radius above max_rad are automatically deleted. 
        Set to 'calc' to make it the median rad + 15* radius std. 
        Default is 12.0

    min_edge_dist : Float
        All particles within min_edge_dist of the (padded) image 
        edges are automatically deleted. Default is 2.0

    check_rad_cutoff : 2-element list of floats
        Particles with radii < check_rad_cutoff[0] or > check_rad_cutoff[1]
        are checked if they should be deleted. Set to 'calc' to make it the
        median rad +- 3.5 * radius std. Default is [3.5, 15]. 

    check_outside_im : Bool
        If True, checks if particles located outside the unpadded image
        should be deleted. Default is True. 

    tries : Int
        The maximum number of particles with radii < check_rad_cutoff
        to try to remove. Checks in increasing order of radius size. 
        Default is 100. 

    im_change_frac : Float, between 0 and 1.
        If removing a particle decreases the error less than im_change_frac*
        the change in the image, the particle is deleted. Default is 0.2. 

    quiet : Bool
        Set to False to print out details about the particles that are 
        checked to remove. 
        
    Returns
    -----------
    removed: Int
        The cumulative number of particles removed. 
    
    """
    if s.varyn == False:
        raise ValueError('s.varyn must be True')

    is_near_im_edge = lambda pos, pad: ((pos < pad) | (pos > 
            np.array(s.image.shape) - pad)).any(axis=1)
    removed = 0
    attempts = 0
    typ = s.obj.typ.astype('bool')
    
    q10 = int(0.1 * typ.sum())#10% quartile
    r_sig = np.sort(s.obj.rad[typ])[q10:-q10].std()
    r_med = np.median(s.obj.rad[typ])
    if max_rad == 'calc':
        max_rad = r_med + 15*r_sig
    if min_rad == 'calc':
        # min_rad = r_med - 5*r_sig
        min_rad = r_med - 25*r_sig
    if check_rad_cutoff == 'calc':
        check_rad_cutoff = [r_med - 7.6*r_sig, r_med + 7.5*r_sig]
    
    #1. Automatic deletion:
    rad_wrong_size = np.nonzero(((s.obj.rad < min_rad) | 
            (s.obj.rad > max_rad)) & typ)[0]
    near_im_edge = np.nonzero(is_near_im_edge(s.obj.pos, min_edge_dist) & typ)[0]
    delete_inds = np.unique(np.append(rad_wrong_size, near_im_edge))
    if not quiet:
        print '  R\t  Z\t  Y\t  X\t|\t  ERR0\t\t  ERR1'

    for ind in delete_inds:
        er0 = opt.get_err(s)
        p, r = s.remove_particle(ind)
        er1 = opt.get_err(s)
        if not quiet:
            print '%2.2f\t%3.2f\t%3.2f\t%3.2f\t|\t%5.3f\t%5.3f' % (
                    s.obj.rad[ind], s.obj.pos[ind,0], s.obj.pos[ind,1], 
                    s.obj.pos[ind,2], er0, er1)
        removed += 1
    
    #2. Conditional deletion:
    typ = s.obj.typ.astype('bool') #since we've updated particles
    check_rad_inds = np.nonzero(((s.obj.rad < check_rad_cutoff[0]) | 
            (s.obj.rad > check_rad_cutoff[1])) & typ)[0]
    if check_outside_im:
        check_edge_inds= np.nonzero(is_near_im_edge(s.obj.pos, s.pad) & typ)[0]
        check_inds = np.unique(np.append(check_rad_inds, check_edge_inds))
    else:
        check_inds = check_rad_inds
    
    
    check_inds = check_inds[np.argsort(s.obj.rad[check_inds])]
    tries = np.max([tries, check_inds.size])
    for ind in check_inds[:tries]:
        if s.obj.typ[ind] == 0:
            raise RuntimeError('you messed up coding this')
        er0 = opt.get_err(s)
        killed = check_remove_particle(s, ind, im_change_frac=im_change_frac)
        if killed:
            removed += 1
            er1 = opt.get_err(s)
            if not quiet:
                print '%2.2f\t%3.2f\t%3.2f\t%3.2f\t|\t%5.3f\t%5.3f' % (
                        s.obj.rad[ind], s.obj.pos[ind,0], s.obj.pos[ind,1], 
                        s.obj.pos[ind,2], er0, er1)
    return removed