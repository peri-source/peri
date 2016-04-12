import os, sys, time, warnings
import numpy as np
from numpy.random import randint
from scipy.optimize import newton, minimize_scalar
from cbamf.util import Tile

"""
To add: 
1. Engine instantiation for groups of particles. I'd like it to be a list
    of individual engine instantiations for particle groups, so they
    can keep their own damping etc.... but that means it won't work
    because you'll run out of memory. 
2. Augmented state
3. Engine instantiation for augmented state. 
6. With opt using big regions for particles, globals, it makes sense to
    put stuff back on the card again....

To fix:
1. In the engine, make do_run_1() and do_run_2() play nicer with each other. 
2. Maybe make an augmented state class that lets you update things like
    rscale(z)? (e.g. instead of updating all r's equally, you could update
    the r's in proportion to LegPoly(x,y,z) etc. Similar for their x,y,z pos's?
    
Algorithm is:
1. Evaluate J_ia = df(xi,mu)/dmu_a
2. Solve the for delta:
    (J^T*J + l*Diag(J^T*J))*delta = J^T(y-f(xi,mu))     (1)
3. Update mu -> mu + delta

To solve eq. (1), we need to:
1. Construct the matrix JTJ = J^T*J
2. Construct the matrix A=JTJ + l*Diag(JTJ)
3. Construct err= y-f(x,beta)
4. np.linalg.solve(A,err)

My only change to this is, instead of calculating J_ia, we calculate
J_ia for a small subset (say 1%) of the pixels in the image randomly selected,
rather than for all the pixels

You need a way to deal with flat / near flat components in the JTJ. 
If the eigenvalue is 0 then the matrix inverse is super ill-conditioned
and you get crap for the suggested step. 

"""

def calculate_J_approx(s, blocks, inds, **kwargs):
    """
    Calculates an approximage J for levenberg-marquardt
    Inputs:
        - s: state to minimize
        - blocks: List of the blocks to optimize over
        - inds: 3-element list; the indices of the _inner_ image to compare
            with. Recommended to be selected randomly. 
    """
    to_return = []
    for b in blocks:
        a_der = eval_deriv(s, b, **kwargs)
        to_return.append(a_der[inds[0], inds[1], inds[2]].ravel().copy())
    return np.array(to_return)
    
def calculate_J_exact(s, blocks, **kwargs): #delete me
    """
    Calculates an exact J for levenberg-marquardt
    Inputs:
        - s: state to minimize
        - blocks: List of the blocks to optimize over
    """
    to_return = []
    for b in blocks:
        a_der = eval_deriv(s, b, **kwargs)
        to_return.append(a_der.ravel())
    return np.array(to_return)

def calculate_err_approx(s, inds):
    return s.get_difference_image()[inds[0], inds[1], inds[2]].ravel().copy()
    
def eval_deriv(s, block, dl=1e-8, be_nice=False, threept=False, **kwargs):
    """
    Using a centered difference / 3pt stencil approximation:
    """
    if not threept:
        i0 = s.get_difference_image().copy()
    
    p0 = s.state[block]
    s.update(block, p0+dl)
    i1 = s.get_difference_image().copy()
    if threept:
        s.update(block, p0-dl)
        i2 = s.get_difference_image().copy()
        deriv = 0.5*(i1-i2)/dl
    else:
        deriv = (i1-i0)/dl
    if be_nice:
       s.update(block, p0) 
    return deriv
    
def calculate_JTJ_grad_approx(s, blocks, num_inds=1000, **kwargs):
    if num_inds < s.image[s.inner].size:
        inds = [randint(v, size=num_inds) for v in s.image[s.inner].shape]
    else:
        inds = [slice(0,None), slice(0,None), slice(0,None)]
    J = calculate_J_approx(s, blocks, inds, **kwargs)
    JTJ = np.dot(J, J.T)
    err = calculate_err_approx(s, inds)
    return JTJ, np.dot(J,err)
    
def get_rand_Japprox(s, blocks, num_inds=1000, quiet=False, **kwargs):
    """
    """
    if not quiet:
        start_time = time.time()
    #You need a better way to select indices to minimize over.
    tot_pix = s.image[s.inner].size
    if num_inds < tot_pix:
        # inds = [randint(v, size=num_inds) for v in s.image[s.inner].shape]
        inds = list(np.unravel_index(np.random.choice(tot_pix, size=num_inds, 
                replace=False), s.image[s.inner].shape))
    else:
        inds = [slice(0,None), slice(0,None), slice(0,None)]
    J = calculate_J_approx(s, blocks, inds, **kwargs)
    if not quiet:
        print 'JTJ:\t%f' % (time.time()-start_time)
    return J, inds
        
def j_to_jtj(J):
    return np.dot(J, J.T)

def calc_im_grad(s, J, inds):
    err = calculate_err_approx(s, inds)
    return np.dot(J, err)

def find_LM_updates(JTJ, grad, damp=1.0, min_eigval=1e-12, quiet=True, **kwargs):
    diag = np.diagflat(np.diag(JTJ))
    
    A0 = JTJ + damp*diag
    delta0, res, rank, s = np.linalg.lstsq(A0, -grad, rcond=min_eigval)
    if not quiet:
        print '%d degenerate of %d total directions' % (delta0.size-rank, delta0.size)
    
    return delta0
    
def update_state_global(s, block, data, keep_time=False, **kwargs):
    """
    """
    #We need to update:
    #obj, psf, ilm, bkg, off, slab, zscale, rscale, sigma
    if keep_time:
        start_time = time.time()
    
    old_state = s.state
    new_state = old_state.copy(); new_state[block] = data.copy()

    updated = {'pos', 'rad', 'typ', 'psf', 'ilm', 'bkg', 'off', 'slab', \
        'zscale', 'sigma', 'rscale'}
    if updated != set(s.param_order):
        raise RuntimeError('This state has parameters that arent supported!')
    
    #Since zscale affects obj, it needs to be first:
    #zscale:
    bz = s.create_block('zscale')
    if (bz & block).sum() > 0:
        new_zscale = new_state[bz].copy()
        s.update(bz, new_zscale)
    
    #obj:
    #pos:
    bpos = s.create_block('pos')
    if (bpos & block).sum() > 0:
        new_pos_params = new_state[bpos].copy().reshape(-1,3)
        s.obj.pos = new_pos_params
    #rad:
    brad = s.create_block('rad')
    if (brad & block).sum() > 0:
        new_rad_params = new_state[brad].copy()
        s.obj.rad = new_rad_params
    #typ:
    btyp = s.create_block('typ')
    if (btyp & block).sum() > 0:
        new_typ_params = new_state[btyp].copy()
        s.obj.typ = new_typ_params
    if np.any(bpos | brad | btyp):
        s.obj.initialize(s.zscale)
    
    #psf:
    bp = s.create_block('psf')
    if (bp & block).sum() > 0:
        new_psf_params = new_state[bp].copy()
        s.psf.update(new_psf_params)
        #And for some reason I need to do this:
        # s.psf._setup_ffts()
        
    #ilm:
    bi = s.create_block('ilm')
    if (bi & block).sum() > 0:
        new_ilm_params = new_state[bi].copy()
        s.ilm.update(s.ilm.block, new_ilm_params)
        
    #bkg:
    bb = s.create_block('bkg')
    if (bb & block).sum() > 0:
        new_bkg_params = new_state[bb].copy()
        s.bkg.update(s.bkg.block, new_bkg_params)
    
    #slab:
    bs = s.create_block('slab')
    if (bs & block).sum() > 0:
        new_slab_params = new_state[bs].copy()
        s.slab.update(new_slab_params)
    
    #and I think the psf support size, ffts aren't updated after updating psf
    #which is why....
    
    #off:
    bo = s.create_block('off')
    if (bo & block).sum() > 0:
        new_off = new_state[bo].copy()
        s.update(bo, new_off)
    
    #rscale:
    brscl = s.create_block('rscale')
    if (brscl & block).sum() > 0:
        new_rscale = new_state[brscl].copy()
        s.update(brscl, new_rscale)
    
    #sigma:
    bsig = s.create_block('sigma')
    if (bsig & block).sum() > 0:
        new_sig = new_state[bsig].copy()
        s.update(bsig, new_sig)
    
    #Now we need to reset the state and return:
    s._build_state()
    s._update_global()
    if keep_time:
        print 'update_state_global:\t%f' % (time.time()-start_time) 

def get_err(s):
    d = s.get_difference_image()
    return np.sum(d*d)

def get_num_px_jtj(s, nparams, decimate=1, max_mem=2e9, min_redundant=20, **kwargs):
    #1. Max for a given max_mem:
    px_mem = int( max_mem /8/ nparams) #1 float = 8 bytes
    #2. # for a given
    px_red = min_redundant*nparams
    #3. And # desired for decimation
    px_dec = s.image[s.inner].size/decimate
    
    if px_red > px_mem:
        raise RuntimeError('Insufficient max_mem for desired redundancy.')
    num_px = np.clip(px_dec, px_red, px_mem)
    return num_px
    
def do_levmarq(s, block, damp=0.1, ddamp=0.1, num_iter=5, do_run=True, \
         run_length=5, **kwargs):
    """
    Runs Levenberg-Marquardt minimization on a cbamf state, based on a
    random approximant of J. Updates the state and returns a bool
    based on a guess at success. 
    Parameters:
    -----------
    s : State
        The cbamf state to optimize. 
    block: Boolean numpy.array
        The desired blocks of the state s to minimize over. Do NOT explode it. 
    damp: Float scalar, >=0. 
        The damping parameter in the Levenberg-Marquardt optimization.
        Big damping means gradient descent with a small step, damp=0
        means Hessian inversion. Default is 0.1. 
    ddamp: Float scalar, >0. 
        Multiplicative value to update damp by. Internally decides when to
        update and when to use damp or 1/damp. Default is 0.1. 
    num_iter: Int. 
        Number of Levenberg-Marquardt iterations before returning. 
    do_run: Bool
        Set to True to attempt multiple minimzations by using the old
        (expensive) JTJ to re-calculate a next step. Default is True. 
    run_length: Int. 
        Maximum number of attempted iterations with a fixed JTJ. Only 
        matters if do_run=True. Default is 5. 
    decimate: Float scalar, >1
        The desired amount to decimate the pixels by for a random image (e.g.
        decimate of 10 takes  1/10 of the pixels). However, the actual amount
        of pixels is determined by max_mem and min_redundant as well. If < 1, 
        the attempts to use all the pixels in the image. Default is 1, i.e.
        uses the max amount of memory allowed. 
    max_mem: Float scalar. 
        The maximum memory (in bytes) that J should occupy. Default is 2GB. 
    min_redundant: Float scalar. 
        Enforces a minimum amount of pixels to include in J, such that the 
        min # of pixels is at least min_redundant * number of parameters. 
        If max_mem and min_redundant result in an incompatible size an
        error is raised. Default is 20. 
    min_eigval: Float scalar, <<1. 
        The minimum eigenvalue to use in inverting the JTJ matrix, to avoid
        degeneracies in the parameter space (i.e. 'rcond' in np.linalg.lstsq).
        Default is 1e-12. 
    keep_time: Bool
        Set to True to print messages about how long each step of the
        algorithm took. Default is False. 
    be_nice: Bool
        If True, spends extra updates placing the parameters back to
        their original values after evaluating the derivatives. Default
        is False, i.e. plays fast (2x with threept=False) and dangerous. 
    dl: Float scalar.
        The amount to update parameters by when evaluating derivatives. 
        Default is 1e-8, which is fine for everything but rscale. 
    threept: Bool
        Set to True to use a 3-point stencil instead of a 2-point. More 
        accurate but 20% slower. Default is False. 
    
    Returns
    --------
    no_troube: Bool
        Boolean of whether the algorithm terminated at a good minimum or
        had trouble. True = Good, False = Bad. 
    
    See Also
    --------
    do_linear_fit: Does a linear fit on linear subblocks of the state,
        using a random subset of the image pixels. 
    do_levmarq_particles: Exact (not random-block) Levenberg-Marquardt 
        specially designed for optimizing particle positions. 
    do_levmarq_all_particle_groups: Wrapper for do_levmarq_particles 
        which blocks particles by distance and iterates over all 
        the particles in the state. 
    do_conj_grad_jtj: Conjugate gradient minimization with a random-block
        approximation to J. 
    
    Comments
    --------
    The sampling of pixels for JTJ is stochastic, but the evaluations 
    of the data are not. Since the error is checked against the global 
    image the log-likelihood will never increase during minimization 
    (there are internal checks for this). 
    """
    blocks = s.explode(block)
    #First I find out how many pixels I use:
    num_px = get_num_px_jtj(s, block.sum(), **kwargs)    
    
    def do_internal_run(J, inds, damp):
        """
        Uses the local vars s, damp, run_length, block
        """
        print 'Running....'
        for rn in xrange(run_length):
            new_grad = calc_im_grad(s, J, inds)
            p0 = s.state[block].copy()
            dnew = find_LM_updates(JTJ, new_grad, damp=damp)
            old_err = get_err(s)
            update_state_global(s, block, p0+dnew)
            new_err = get_err(s)
            print '%f\t%f' % (old_err, new_err)
            if new_err > old_err:
                #done running
                update_state_global(s, block, p0)
                break
    
    recalc_J = True
    counter = 0
    # for _ in xrange(num_iter):
    while counter < num_iter:
        p0 = s.state[block].copy()
        err_start = get_err(s)
        
        #1. Get J, JTJ
        if recalc_J:
            J, inds = get_rand_Japprox(s, blocks, num_inds=num_px, **kwargs)
            JTJ = j_to_jtj(J)
            has_inverted = False
            no_trouble = True
        
        #1b. Get grad, outside recalc_J because it's fast and changes quickly
        grad = calc_im_grad(s, J, inds)
            
        #2. Calculate and implement first guess for updates
        d0 = find_LM_updates(JTJ, grad, damp=damp)
        d1 = find_LM_updates(JTJ, grad, damp=damp*ddamp)
        update_state_global(s, block, p0+d0)
        err0 = get_err(s)
        update_state_global(s, block, p0+d1)
        err1 = get_err(s)
        
        if np.min([err0, err1]) > err_start: #Bad step...
            print 'Bad step!\t%f\t%f\t%f' % (err_start, err0, err1)
            update_state_global(s, block, p0)
            recalc_J = False
            #Avoiding infinite loops by adding a small amount to counter:
            counter += 0.2
            if err0 < err1: 
                #d_damp is the wrong "sign", so we invert:
                if has_inverted:
                    print 'Stuck, drastically increasing damping'
                    ddamp = np.max([ddamp, 1/ddamp])
                    damp = np.max([100.0, damp * ddamp])
                    no_trouble = False
                else:
                    print 'Inverting ddamp:\t%f\t%f' % (ddamp, 1.0/ddamp)
                    ddamp = 1.0/ddamp
                    has_inverted = True
                #You need a better way to avoid infinite loops here
            else: 
                #ddamp is the correct sign but damp is too big:
                damp *= (ddamp*ddamp)
        
        else: #at least 1 good step:
            if err0 < err1: #good step and damp, re-update:
                update_state_global(s, block, p0+d0)
                print 'Good step:\t%f\t%f\t%f' % (err_start, err0, err1)
            else: #err1 < err0 < err_start, good step but decrease damp:
                damp *= ddamp
                print 'Changing damping:\t%f\t%f\t%f' % (err_start, err0, err1)
            if do_run:
                do_internal_run(J, inds, damp)
            recalc_J = True
            counter += 1
    return no_trouble        

def do_conj_grad_jtj(s, block, min_eigval=1e-12, num_sweeps=2, **kwargs):
    """
    Runs conjugate-gradient descent on a cbamf state, based on a random
    approximation of the Hessian matrix from JTJ. Does not 
    return anything, only updates the state. 
    Parameters:
    -----------
    s : State
        The cbamf state to optimize. 
    block: Boolean numpy.array
        The desired blocks of the state s to minimize over. Do NOT explode it. 
    min_eigval: Float scalar, <<1. 
        The minimum eigenvalue to use in inverting the JTJ matrix, to avoid
        degeneracies in the parameter space (i.e. 'rcond' in np.linalg.lstsq).
        Default is 1e-12. 
    num_sweeps: Int. 
        Number of sweeps to do over the eigenvectors of the parameter
        block, using the same Hessian approximation. Default is 2. 
    decimate: Float scalar, >1
        The desired amount to decimate the pixels by for a random image (e.g.
        decimate of 10 takes  1/10 of the pixels). However, the actual amount
        of pixels is determined by max_mem and min_redundant as well. If < 1, 
        the attempts to use all the pixels in the image. Default is 1, i.e.
        uses the max amount of memory allowed. 
    max_mem: Float scalar. 
        The maximum memory (in bytes) that J should occupy. Default is 2GB. 
    min_redundant: Float scalar. 
        Enforces a minimum amount of pixels to include in J, such that the 
        min # of pixels is at least min_redundant * number of parameters. 
        If max_mem and min_redundant result in an incompatible size an
        error is raised. Default is 20. 
    keep_time: Bool
        Set to True to print messages about how long each step of the
        algorithm took. Default is False. 
    be_nice: Bool. 
        If True, evaluating the derivative doesn't change the state. If 
        False, when the derivative is evaluated the state isn't changed 
        back to its original value, which saves time (33%) but may wreak 
        havoc. Default is True (i.e. slower, no havoc). 
    dl: Float scalar.
        The amount to update parameters by when evaluating derivatives. 
        Default is 2e-5. 
    threept: Bool
        Set to True to use a 3-point stencil instead of a 2-point. More 
        accurate but 20% slower. Default is False. 
    See Also
    --------
    do_levmarq: Levenberg-Marquardt minimization with a random-block
        approximation to J. 
    
    Comments
    --------
    The sampling of pixels for J, JTJ is stochastic, but the evaluations 
    of the data are not. Since the error is checked against the global 
    image the log-likelihood will never increase during minimization 
    (there are internal checks for this). 
    """
    #First find out how many pixels I use:
    num_px = get_num_px_jtj(s, block.sum(), **kwargs)
    blocks = s.explode(block)
    
    #Then I get JTJ etc:
    J, inds = get_rand_Japprox(s, blocks, num_inds=num_px, **kwargs)
    JTJ = j_to_jtj(J)
    grad = calc_im_grad(s, J, inds) #internal because s gets updated
    
    #Now I get the conj. grad. directions:
    evals, evecs = np.linalg.eigh(JTJ)
    
    #And we line minimize over each direction, starting with the smallest
    print 'Starting:\t%f' % get_err(s)
    for n in xrange(num_sweeps):
        for a in xrange(evals.size-1,-1, -1): #high eigenvalues first
            if evals[a] < min_eigval*evals.max():
                print "Degenerate Direction\t%d" % a
                continue #degenerate direction, ignore

            #Picking a good direction/magnitude to move in:
            cur_vec = evecs[:,a] * 0.5*np.dot(-grad, evecs[:,a])/evals[a]
            
            do_line_min(s, block, cur_vec, **kwargs)
            print 'Direction min %d \t%f' % (a, get_err(s))

def do_line_min(s, block, vec, maxiter=10, **kwargs):
    """
    Does a line minimization over the state
    Make vec something that update_state_global(block, p0+vec) is the "correct"
    step size (i.e. takes you near the minimum)
    """
    start_err = get_err(s)
    p0 = s.state[block].copy()
    
    def for_min(x):
        update_state_global(s, block, p0 + x*vec, keep_time=False)
        return get_err(s)
    
    #Need to check for a bracket:
    brk_vals = [for_min(-2.), for_min(2.)]
    cnt_val = for_min(0)
    if cnt_val > np.min(brk_vals):
        #the min is outside the interval, need a new bracket
        run = True
        if brk_vals[0] < brk_vals[1]:
            bracket = [-2,0]
            start = -2.
            old = brk_vals[0]
            while run:
                start *= 3
                new = for_min(start)
                run = new < old; print '....'
            bracket = [start] + bracket
        else:
            bracket = [0,2]
            start = 2
            old = brk_vals[1]
            while run:
                start *= 3
                new = for_min(start)
                run = new < old; print '....'
            bracket.append(start)
    else: #bracket is fine
        bracket = [-2,0,2]
    
    res = minimize_scalar(for_min, bracket=bracket, method='Brent', \
            options={'maxiter':maxiter})
    if res.fun <= start_err:
        _ = for_min(res.x)
    else:
        #wtf
        raise RuntimeError('wtf')
        
#=============================================================================#
#               ~~~~~        Single particle stuff    ~~~~~
#=============================================================================#
def update_one_particle(s, particle, pos, rad, typ=None, relative=False, 
        do_update_tile=True, fix_errors=False, **kwargs):
    """
    Updates a single particle (labeled ind) in the state s. 
    
    Parameters
    -----------
    s : State
        The cbamf state to optimize. 
    particle: 1-element int numpy.ndarray
        Index of the particle to update.
    pos: 3-element numpy.ndarray
        New position of the particle. 
    rad: 1-element numpy.ndarray
        New radius of the particle. 
    typ: 1-element numpy.ndarray.
        New typ of the particle; defaults to the previous typ.
    relative: Bool
        Set to true to make pos, rad updates relative to the previous 
        position (i.e. p1 = p0+pos instead of p1 = pos, etc for rad). 
        Default is False. 
    do_update_tile: Bool
        If False, only updates s.object and not the actual model image. 
        Set to False only if you're going to update manually later. 
        Default is True
    fix_errors: Bool
        Set to True to fix errors for placing particles outside the 
        image or with negative radii (otherwise a ValueError is raised). 
        Default is False
    
    Returns
    --------
    tiles: 3-element list. 
        cbamf.util.Tile's of the region of the image affected by the 
        update. Returns what s._tile_from_particle_change returns 
        (outer, inner, slice)
    """
    #the closest a particle can get to an edge, > any reasonable dl 
    MIN_DIST=3e-3
    
    if type(particle) != np.ndarray:
        particle = np.array([particle])
    
    prev = s.state.copy()
    p0 = prev[s.b_pos].copy().reshape(-1,3)[particle]
    r0 = prev[s.b_rad][particle]
    if s.varyn:
        t0 = prev[s.b_typ][particle]
    else:
        t0 = np.ones(particle.size)
        
    is_bad_update = lambda p, r: np.any(p < 0) or np.any(p > \
            np.array(s.image.shape)) or np.any(r < 0)
    if fix_errors:
        #Instead of ignoring errors, we modify pos, rad in place
        #so that they are the best possible updates. 
        if relative:
            if is_bad_update(p0+pos, r0+rad):
                pos[:] = np.clip(pos, MIN_DIST-p0, np.array(s.image.shape)-
                        MIN_DIST-p0)
                rad[:] = np.clip(rad, MIN_DIST-r0, np.inf)
        else:
            if is_bad_update(pos, rad):
                pos[:] = np.clip(pos, MIN_DIST, np.array(s.image.shape)-MIN_DIST)
                rad[:] = np.clip(rad, MIN_DIST, np.inf)
    
    if typ is None:
        t1 = t0.copy()
    else:
        t1 = typ
    
    if relative:
        p1 = p0 + pos
        r1 = r0 + rad
    else:
        p1 = pos.copy()
        r1 = rad.copy()
        
    if is_bad_update(p1, r1):
        raise ValueError('Particle outside image / negative radius!')
    
    tiles = s._tile_from_particle_change(p0, r0, t0, p1, r1, t1) #312 us
    s.obj.update(particle, p1.reshape(-1,3), r1, t1, s.zscale, difference=s.difference) #4.93 ms
    if do_update_tile:
        s._update_tile(*tiles, difference=s.difference) #66.5 ms
    s._build_state()
    return tiles
    
def eval_one_particle_grad(s, particle, dl=1e-6, threept=False, slicer=None, 
        be_nice=False, **kwargs):
    """
    Evaluates the gradient of the image for 1 particle, for all of its 
    parameters (x,y,z,R). Used for J for LM. 
    
    Parameters
    -----------
    s : State
        The cbamf state to evaluate derivatives on. 
    particle: 1-element int numpy.ndarray
        Index of the particle to evaluate derivatives for.
    slicer: slice
        Slice object of the regions of the image at which to evaluate 
        the derivative. Default is None, which finds the tile of the 
        pixels changed by updating the particle and returns that. 
    dl: Float
        The amount to change values by when evaluating derivatives. 
        Default is 1e-6. 
    threept: Bool
        If True, uses a 3-point finite difference instead of 2-point. 
        Default is False (using two-point for 1 less function call).
    be_nice: Bool
        If True, spends 1x4 extra updates placing the (x,y,z,R) back to
        their original values after evaluating the derivatives. Default
        is False, i.e. plays fast and dangerous. 
    
    Returns
    --------
    grads: numpy.ndarray 
        [4,N] element numpy.ndarray; grads[i,j] is the [x,y,z,R][i] 
        derivative evaluated at the j'th pixel in the affected region. 
    slicer: slice
        The slice containing which points in the image the derivate was 
        evaluated at. If the input parameter slicer is not None, then it
        is the same as the slice passed in. 
    """
    p0 = s.state[s.block_particle_pos(particle)].copy()
        
    grads = []
    if slicer is not None:
        mask = s.image_mask[slicer] == 1
    
    #xyz pos:
    for a in xrange(3):
        if not threept:
            i0 = get_slicered_difference(s, slicer, mask)
        dx = np.zeros(p0.size)
        dx[a] = 1*dl
        tiles = update_one_particle(s, particle, dx, 0, relative=True)
        if slicer == None:
            slicer = tiles[0].slicer
            mask = s.image_mask[slicer] == 1
        
        i1 = get_slicered_difference(s, slicer, mask)
        if threept:
            toss = update_one_particle(s, particle, -2*dx, 0, relative=True)
            #we want to compare the same slice. Also -2 to go backwards
            i2 = get_slicered_difference(s, slicer, mask)
            if be_nice:
                toss = update_one_particle(s, particle, dx, 0, relative=True)
            grads.append((i1-i2)*0.5/dl)
        else:
            if be_nice:
                toss = update_one_particle(s, particle, -dx, 0, relative=True)
            grads.append( (i1-i0)/dl )
    #rad:
    if not threept:
        i0 = get_slicered_difference(s, slicer, mask)
    dr = np.array([dl])
    toss = update_one_particle(s, particle, 0, 1*dr, relative=True)
    i1 = get_slicered_difference(s, slicer, mask)
    if threept:
        toss = update_one_particle(s, particle, 0, -2*dr, relative=True)
        i2 = get_slicered_difference(s, slicer, mask)
        if be_nice:
            toss = update_one_particle(s, particle, 0, dr, relative=True)
        grads.append((i1-i2)*0.5/dl)
    else:
        if be_nice:
            toss = update_one_particle(s, particle, 0, -dr, relative=True)
        grads.append( (i1-i0)/dl )
        
    return np.array(grads), slicer

def eval_many_particle_grad(s, particles, **kwargs):
    """Wrapper for eval_one_particle_grad. Particles is an iterable"""
    grad = []
    for p in particles:
        #throwing out the slicer info right now...
        grad.extend(eval_one_particle_grad(s, p, **kwargs)[0].tolist())
    return np.array(grad)

def find_particles_in_box(s, bounds):
    """
    Finds the particles in a box.
    
    Parameters
    -----------
    s : State
        The cbamf state to find the particle in. 
    bounds: 2-element list-like of lists. 
        bounds[0] is the lower left corner of the box, bounds[1] the upper
        right corner. Each of those are 3-element list-likes of the coords. 
    
    Returns
    --------
    inds: numpy.ndarray
        The indices of the particles in the box. 
    
    """
    is_in_xi = lambda i: (s.obj.pos[:,i] > bounds[0][i]) & (s.obj.pos[:,i] <= bounds[1][i])
    in_region = is_in_xi(0) & is_in_xi(1) & is_in_xi(2) & (s.obj.typ==1)
    return np.nonzero(in_region)[0]

def get_slicered_difference(s, slicer, mask):
    return np.copy((s.image[slicer] - s.get_model_image()[slicer])[mask])

def get_tile_from_multiple_particle_change(s, inds):
    """
    Finds the tile from changing multiple particles by taking the maximum
    enclosing tile
    """
    all_left = []
    all_right = []
    for i in inds:
        p0 = np.array(s.obj.pos[i])
        r0 = np.array([s.obj.rad[i]])
        t0 = np.array([s.obj.typ[i]])
        tile, _, __ = s._tile_from_particle_change(p0, r0, t0, p0, r0, t0)
        all_left.append(tile.l)
        all_right.append(tile.r)
    
    left = np.min(all_left, axis=0)
    right= np.max(all_right,axis=0)
    
    return Tile(left, right=right)

def update_particles(s, particles, params, **kwargs):
    #eval_particle_grad returns parameters in order(p0,r0,p1,r1,p2,r2...)
    #so that determines the order for updating particles:
    all_part_tiles = []
    
    #FIXME 
    #You have a function get_tile_from_particle change. You should update
    #it to use the new tile version below, and call it here. 
    #Right now it's being used to get the LM comparison region for do_levmarq_particles
    #but you could also use it in update_particles and in 
    #do_levmarq_all_particle_groups.calc_mem_usage
    #Plus, it would be nice to give it to Matt
    
    #We update the object but only update the field at the end
    for a in xrange(particles.size):
        # pos = params[4*a:4*a+3]
        # rad = params[4*a+3:4*a+4]
        
        #To play nice with relative, we get p0, r0 first, then p1, r1
        p0 = s.obj.pos[particles[a]].copy()
        r0 = np.array([s.obj.rad[particles[a]]])
        t0 = np.array([s.obj.typ[particles[a]]])

        toss = update_one_particle(s, particles[a], params[4*a:4*a+3], 
                params[4*a+3:4*a+4], do_update_tile=False, **kwargs)
        
        p1 = s.obj.pos[particles[a]].copy()
        r1 = np.array([s.obj.rad[particles[a]]])
        t1 = np.array([s.obj.typ[particles[a]]])
        #We reconstruct the tiles separately to deal with edge-particle
        #overhang pads:
        left, right = s.obj.get_support_size(p0, r0, t0, p1, r1, t1, s.zscale)
        all_part_tiles.append(Tile(left, right, mins=0, maxs=s.image.shape))
        
    particle_tile = all_part_tiles[0].boundingtile(all_part_tiles)
    
    #From here down is basically copied from states._tile_from_particle_change
    psf_pad_l = s.psf.get_support_size(particle_tile.l[0])
    psf_pad_r = s.psf.get_support_size(particle_tile.r[0])
    psftile = Tile.boundingtile([Tile(np.ceil(i)) for i in [psf_pad_l, psf_pad_r]])
    img = Tile(s.image.shape)
    outer_tile = particle_tile.pad(psftile.shape/2+1)
    inner_tile, outer_tile = outer_tile.reflect_overhang(img)
    iotile = inner_tile.translate(-outer_tile.l)
    ioslice = iotile.slicer
    #end copy

    s._update_tile(outer_tile, inner_tile, ioslice, difference=s.difference)
    return outer_tile, inner_tile, ioslice
    
def do_levmarq_particles(s, particles, damp=0.1, ddamp=0.2, num_iter=3,
        do_run=True, run_length=5, quiet=False, **kwargs):
    """
    Runs an exact (i.e. not stochastic number of pixels) Levenberg-Marquardt 
    minimization on a set of particles in a cbamf state. Doesn't return 
    anything, only updates the state. 
    Parameters:
    -----------
    s : State
        The cbamf state to optimize. 
    particles: Int numpy.array
        The desired indices of the particles to minimize over.
    damp: Float scalar, >=0. 
        The damping parameter in the Levenberg-Marquardt optimization.
        Big damping means gradient descent with a small step, damp=0
        means Hessian inversion. Default is 0.1. 
    ddamp: Float scalar, >0. 
        Multiplicative value to update damp by. Internally decides when
        to update and when to use damp or 1/damp. Default is 0.2. 
    num_iter: Int. 
        Number of Levenberg-Marquardt iterations to execute. Default is 3.
    do_run: Bool
        Set to True to attempt multiple minimzations by using the old
        (expensive) JTJ to re-calculate a next step. Default is True. 
    run_length: Int. 
        Maximum number of attempted iterations with a fixed JTJ. Only 
        matters if do_run=True. Default is 5. 
    quiet: Bool
        Set to True to avoid printing out messages on how the fit is going. 
        Default is False, no messages. 
    min_eigval: Float scalar, <<1. 
        The minimum eigenvalue to use in inverting the JTJ matrix, to 
        avoid degeneracies in the parameter space (i.e. 'rcond' in 
        np.linalg.lstsq). Default is 1e-12. 
    dl: Float scalar.
        The amount to update parameters by when evaluating derivatives. 
        Default is 1e-6. 
    threept: Bool
        Set to True to use a 3-point stencil instead of a 2-point. More 
        accurate but 20% slower. Default is False. 
    be_nice: Bool
        If True, spends extra updates placing the parameters back to
        their original values after evaluating the derivatives. Default
        is False, i.e. plays fast and dangerous. 

    Returns
    --------
    no_troube: Bool
        Boolean of whether the algorithm terminated at a good minimum or
        had trouble. True = Good, False = Bad.

    See Also
    --------
    do_levmarq: Runs Levenberg-Marquardt minimization using a random
        subset of the image pixels. Works for any fit blocks. 
    do_linear_fit: Does a linear fit on linear subblocks of the state,
        using a random subset of the image pixels. 
    do_levmarq_all_particle_groups: Convenience wrapper that splits all
        the particles in the state into separate groups, then calls 
        do_levmarq_particles on each group. 
    find_particles_in_box: Given a bounding region finds returns an array
        of the indices of the particles in that box. 

    Comments
    --------
    """    
    recalc_J = True
    counter = 0
    
    def do_internal_run():
        """
        Uses the local vars s, damp, run_length, dif_tile, J, JTJ
        """
        if not quiet:
            print 'Running....'
        for rn in xrange(run_length):
            new_grad = np.dot(J, get_slicered_difference(s, dif_tile.slicer, 
                s.image_mask[dif_tile.slicer].astype('bool')))

            dnew = find_LM_updates(JTJ, new_grad, damp=damp)
            old_err = get_err(s)
            update_particles(s, particles, dnew, relative=True, \
                    fix_errors=True)
            new_err = get_err(s)
            if not quiet:
                print '%f\t%f' % (old_err, new_err)
            if new_err > old_err:
                #done running, put back old params
                update_particles(s, particles, -dnew, relative=True, \
                        fix_errors=True)
                break
    
    while counter < num_iter:
        # p0 = s.state[block].copy() -- actually I don't know if I want to use this
        err_start = get_err(s)
        
        #1. Get J, JJT, 
        if recalc_J:
            dif_tile = get_tile_from_multiple_particle_change(s, particles)
            J = eval_many_particle_grad(s, particles, slicer=dif_tile.slicer, 
                    **kwargs)
            JTJ = j_to_jtj(J)
            has_inverted = False
            no_trouble = True

        #2. Get -grad from the tiles:
        grad = np.dot(J, get_slicered_difference(s, dif_tile.slicer, 
                s.image_mask[dif_tile.slicer].astype('bool')))
        
        #3. get LM updates: -- here down is practically copied; could be moved into an engine
        d0 = find_LM_updates(JTJ, grad, damp=damp)
        d1 = find_LM_updates(JTJ, grad, damp=damp*ddamp)
        update_particles(s, particles, d0, relative=True, fix_errors=True)
        err0 = get_err(s)
        d1md0 = d1-d0 #we need update_particles to modify this in place
        # update_particles(s, particles, d1-d0, relative=True, fix_errors=True)
        update_particles(s, particles, d1md0, relative=True, fix_errors=True)
        err1 = get_err(s)
        
        #4. Pick the best value, continue
        if np.min([err0, err1]) > err_start: #Bad step...
            if not quiet:
                print 'Bad step!\t%f\t%f\t%f' % (err_start, err0, err1)
            # update_particles(s, particles, -d1, relative=True)
            update_particles(s, particles, -d1md0-d0, relative=True)
            if get_err(s) > (err_start + 1e-3):#0.1):
                err_bad = get_err(s)
                warnings.warn('Problem with putting back d1; resetting', RuntimeWarning)
                s.reset()
                # err_rst = get_err(s)
                # raise RuntimeError('Problem with putting back d1')
            recalc_J = False
            #Avoiding infinite loops by adding a small amount to counter:
            counter += 0.1 
            if err0 < err1: 
                if has_inverted:
                    if not quiet:
                        print 'Stuck, drastically increasing damping'
                    ddamp = np.max([ddamp, 1/ddamp])
                    damp = np.max([100.0, damp * ddamp])
                    no_trouble = False
                else:
                    if not quiet:
                        print 'Inverting ddamp:\t%f\t%f' % (ddamp, 1.0/ddamp)
                    ddamp = 1.0/ddamp
                    has_inverted = True
            else: 
                #ddamp is the correct sign but damp is too big:
                damp *= (ddamp*ddamp)
        
        else: #at least 1 good step:
            if err0 < err1: #good step and damp, re-update:
                update_particles(s, particles, -d1md0, relative=True)
                if not quiet:
                    print 'Good step:\t%f\t%f\t%f' % (err_start, err0, err1)
            else: #err1 < err0 < err_start, good step but decrease damp:
                damp *= ddamp
                if not quiet:
                    print 'Changing damping:\t%f\t%f\t%f' % (err_start,err0,err1)
            if do_run:
                do_internal_run()
            recalc_J = True
            counter += 1
    return no_trouble

def separate_particles_into_groups(s, region_size=40, bounds=None, **kwargs):
    """
    Given a state, returns a list of groups of particles. Each group of 
    particles are located near each other in the image. Every particle 
    located in the desired region is contained in exactly 1 group. 
    
    Parameters:
    -----------
    s : State
        The cbamf state to find particles in. 
    region_size: Int or 3-element list-like of ints. 
        The size of the box. Groups particles into boxes of shape
        (region_size[0], region_size[1], region_size[2]). If region_size
        is a scalar, the box is a cube of length region_size. 
        Default is 40.
    bounds: 2-element list-like of 3-element lists. 
        The sub-region of the image over which to look for particles. 
            bounds[0]: The lower-left  corner of the image region. 
            bounds[1]: The upper-right corner of the image region. 
        Default (None -> ([0,0,0], s.image.shape)) is a box of the entire
        image size, i.e. the default places every particle in the image
        somewhere in the groups.
        
    Returns:
    -----------
    particle_groups: List
        Each element of particle_groups is an int numpy.ndarray of the
        group of nearby particles. Only contains groups with a nonzero
        number of particles, so the elements don't necessarily correspond
        to a given image region. 
    """
    if bounds is None:
        bounds = ([0,0,0], s.image.shape)
    if type(region_size) == int:
        rs = [region_size, region_size, region_size]
    else:
        rs = region_size
        
    pts = [range(bounds[0][i], bounds[1][i], rs[i]) for i in xrange(3)]

    all_boxes = []
    for start_0 in pts[0]:
        for start_1 in pts[1]:
            for start_2 in pts[2]:
                all_boxes.append([[start_0, start_1, start_2], 
                        [start_0+rs[0], start_1+rs[1], start_2+rs[2]]])
    
    particle_groups = []
    for box in all_boxes:
        cur_group = find_particles_in_box(s, box)
        if cur_group.size > 0:
            particle_groups.append(cur_group)
        
    return particle_groups 

def do_levmarq_all_particle_groups(s, region_size=40, calc_region_size=True, 
        max_mem=2e9, **kwargs):
    """
    Runs an exact Levenberg-Marquardt minimization on all the particles
    in the state, by splitting them up into groups of nearby particles
    and running do_levmarq_particles on each group. 
    This is a wrapper for do_levmarq_particles, so see its documentation for
    further details. 
    
    Parameters:
    -----------
    s : State
        The cbamf state to optimize. 
    region_size: Int or 3-element numpy.ndarray
        The size of the box for particle grouping, through 
        separate_particles_into_groups. This groups particles into 
        boxes of shape (region_size, region_size, region_size). Default
        is 40. 
    calc_region_size: Bool
        If True, calculates an approximate region size based on a maximum
        allowed memory for J. Set to False to use the input region size.
        Default is True. 
    max_mem: Numeric
        The approximate maximum memory for J. Default is 2e9. 

    **kwargs parameters of interest:
    ------------
    damp: Float scalar, >=0. 
        The damping parameter in the Levenberg-Marquardt optimization.
        Big damping means gradient descent with a small step, damp=0
        means Hessian inversion. Default is 0.1. 
    ddamp: Float scalar, >0. 
        Multiplicative value to update damp by. Internally decides when
        to update and when to use damp or 1/damp. Default is 0.2. 
    num_iter: Int. 
        Number of Levenberg-Marquardt iterations to execute. Default is 2.
        
    Returns
    --------
    no_troube: List of Bools
        List of bools of whether the algorithm terminated without problems
        for each particle group. 
    
    See Also
    --------    
    do_levmarq: Runs Levenberg-Marquardt minimization using a random
        subset of the image pixels. Works for any fit blocks. 
    do_linear_fit: Does a linear fit on linear subblocks of the state,
        using a random subset of the image pixels. 
    do_levmarq_particles: Exact (not random-block) Levenberg-Marquardt 
        specially designed for optimizing particle positions. 
    do_conj_grad_jtj: Conjugate gradient minimization with a random-block
        approximation to J.
    """
    
    def calc_mem_usage(region_size):
        rs = np.array(region_size)
        particle_groups = separate_particles_into_groups(s, region_size=
                rs.tolist(), **kwargs)
        num_particles = np.max(map(np.size, particle_groups))
        mem_per_part = 32 * np.prod(rs + 2*s.pad*np.ones(3))
        return num_particles * mem_per_part
    
    region_size = np.array(region_size)
    if calc_region_size:
        if calc_mem_usage(region_size) > max_mem:
            while ((calc_mem_usage(region_size) > max_mem) and 
                    np.all(region_size > 2)):
                region_size -= 1
        else:
            while ((calc_mem_usage(region_size) < max_mem) and 
                    np.all(region_size < np.array(s.image.shape))):
                region_size += 1
            region_size -= 1
    
    particle_groups = separate_particles_into_groups(s, region_size=region_size,
            **kwargs)
    no_trouble = []
    for group in particle_groups: 
        group_ok = do_levmarq_particles(s, group, **kwargs)
        no_trouble.append(group_ok)
    return no_trouble
    
#=============================================================================#
#               ~~~~~        Linear fit stuff    ~~~~~
#=============================================================================#

def do_linear_fit(s, block, min_eigval=1e-11, relative=False, num_iter=5, **kwargs):
    """
    Linear-least-squares fit minimization on linear sub-blocks of the 
    state. Iterates the solution several times to fix float roundoff 
    errors and improve the answer, and uses a random subset of pixels in
    the image to reduce memory. Doesn't return anything, only updates
    the state. 

    Parameters:
    -----------
    s : State
        The cbamf state to optimize. 
    block: Boolean numpy.array
        The desired blocks of the state s to minimize over. Do NOT explode it. 
        The blocks must correspond to a _linear_ fit in the state. 
    min_eigval: Float scalar, <<1. 
        The minimum eigenvalue to use in solving the matrix equation, to
        avoid degeneracies in the parameter space (i.e. 'rcond' in 
        np.linalg.lstsq). Default is 1e-11. 
    relative: Bool
        Set to True to do linear least-squares with the updated parameters
        relative to the current values. Useful if the state is already
        highly optimized. Default is False. 
    num_iter: Int
        The number of times the matrix solution is iterated, to reduce
        float errors. The more the merrier, but slower by num_iter 
        update_state_global calls. Default is 5. 
    decimate: Float scalar, >1
        The desired amount to decimate the pixels by for a random image (e.g.
        decimate of 10 takes  1/10 of the pixels). However, the actual amount
        of pixels is determined by max_mem and min_redundant as well. If < 1, 
        the attempts to use all the pixels in the image. Default is 1, i.e.
        uses the max amount of memory allowed.  
    max_mem: Float scalar. 
        The maximum memory (in bytes) that the fit matrix should occupy. 
        Default is 2GB. 
    min_redundant: Float scalar. 
        Enforces a minimum amount of pixels to include in the fit matrix, 
        such that the min # of pixels is at least min_redundant * number 
        of parameters. If max_mem and min_redundant result in an 
        incompatible size an error is raised. Default is 20. 
    
    See Also
    --------
    do_levmarq: Runs Levenberg-Marquardt minimization using a random
        subset of the image pixels. Works for any fit blocks. 
    do_levmarq_particles: Exact (not random-block) Levenberg-Marquardt 
        specially designed for optimizing particle positions. 
    do_levmarq_all_particle_groups: Wrapper for do_levmarq_particles 
        which blocks particles by distance and iterates over all 
        the particles in the state. 
    do_conj_grad_jtj: Conjugate gradient minimization with a random-block
        approximation to J. 
        
    Comments
    --------
    This is sort of pointless, since it's (almost) the exact same as 
    do_levmarq with damp=0. There are a few differences in how its 
    implemented though...
    """
    #1. Getting the raveled indices for comparison:
    tot_pix = s.image[s.inner].size
    num_params = block.sum()
    num_px = np.min([get_num_px_jtj(s, num_params, **kwargs), tot_pix])
    inds = np.random.choice(tot_pix, size=num_px, replace=False)
    
    #2. Getting the old state, model, error for comparison:
    old_err = get_err(s)
    old_params = s.state[block].copy()
    mdl_0 = s.get_model_image()[s.inner].ravel()[inds].copy()
    
    #3. Getting the fitting matrix:
    fit_matrix = np.zeros([num_px, num_params])
    cur_state = np.zeros([num_params])
    delta_params = np.zeros([num_params])
    for a in xrange(num_params):
        delta_params *= 0
        delta_params[a] = 1.0
        update_state_global(s, block, delta_params)
        fit_matrix[:,a] = s.get_model_image()[s.inner].ravel()[inds].copy()
        
    #4. Getting the 'b' vector in Ax=b for leastsq, and the '0' vector:
    if relative:
        b_vec = s.image[s.inner].ravel()[inds] - mdl_0
    else:
        b_vec = s.image[s.inner].ravel()[inds].copy()
    
    #5. Solving:
    stuff = np.linalg.lstsq(fit_matrix, b_vec, rcond=min_eigval)
    if relative:
        best_params = old_params + stuff[0]
    else:
        best_params = stuff[0]
    update_state_global(s, block, best_params)
    new_err = get_err(s)
    
    print old_err, '\t', new_err
    
    #6. Running to fix float errors:
    for _ in xrange(num_iter):
        mdl_1 = s.get_model_image()[s.inner].ravel()[inds].copy()
        b_vec1 = s.image[s.inner].ravel()[inds] - mdl_1
        mo_stuff = np.linalg.lstsq(fit_matrix, b_vec1, rcond=min_eigval)
        best_params += mo_stuff[0]
        update_state_global(s, block, best_params)
        new_err = get_err(s)
        print old_err, '\t', new_err
        
    if new_err > old_err:
        print 'No improvement via linear fit, reverting.'
        update_state_global(s, block, old_params)
    
#=============================================================================#
#         ~~~~~        Class/Engine LM minimization Stuff     ~~~~~
#=============================================================================#
class LMEngine(object):
    def __init__(self, damping=1., increase_damp_factor=3., decrease_damp_factor=8., 
                min_eigval=1e-12, marquardt_damping=True, transtrum_damping=None,
                use_accel=False, max_accel_correction=1., ptol=1e-6, 
                errtol=1e-5, costol=None, max_iter=5, run_length=5, 
                update_J_frequency=5, broyden_update=False, eig_update=False, 
                partial_update_frequency=3, num_eig_dirs=4, quiet=True):
        """
        Levenberg-Marquardt engine with all the options from the 
        M. Transtrum J. Sethna 2012 ArXiV paper. 

        Inputs:
        -------
            damping: Float
                The initial damping factor for Levenberg-Marquardt. Adjusted 
                internally. Default is 1. 
            increase_damp_factor: Float
                The amount to increase damping by when an attempted step
                has failed. Default is 3. 
            decrease_damp_factor: Float
                The amount to decrease damping by after a successful step.
                Default is 8. increase_damp_factor and decrease_damp_factor
                must not have all the same factors.
            
            min_eigval: Float scalar, <<1. 
                The minimum eigenvalue to use in inverting the JTJ matrix,
                to avoid degeneracies in the parameter space (i.e. 'rcond' 
                in np.linalg.lstsq). Default is 1e-12. 
            marquardt_damping: Bool
                Set to False to use Levenberg damping (damping matrix 
                proportional to the identiy) instead of Marquardt damping
                (damping matrix proportional to the diagonal terms of JTJ).
                Default is True. 
            transtrum_damping: Float or None
                If not None, then clips the Marquardt damping diagonal
                entries to be at least transtrum_damping. Default is None.
                
            use_accel: Bool
                Set to True to incorporate the geodesic acceleration term
                from M. Transtrum J. Sethna 2012. Default is False. 
            max_accel_correction: Float
                Acceleration corrections bigger than max_accel_correction*
                the normal LM step are viewed as bad steps, causing a 
                decrease in damping. Default is 1.0. Only applies to the 
                do_run_1 method. 
            
            ptol: Float
                Algorithm has converged when the none of the parameters 
                have changed by more than ptol. Default is 1e-6.
            errtol: Float
                Algorithm has converged when the error has changed
                by less than ptol after 1 step. Default is 1e-6. 
            costol: Float
                Algorithm has converged when the cosine of the angle 
                between (residuals projected onto the model manifold)
                and (the residuals) is < costol. Default is None, i.e.
                doesn't check the cosine (since it takes a bit of time).
            max_iter: Int
                The maximum number of iterations before the algorithm
                stops iterating. Default is 5. 
                
            update_J_frequency: Int
                The frequency to re-calculate the full Jacobian matrix. 
                Default is 5. Only matters for do_run_v1
            broyden_update: Bool
                Set to True to do a Broyden partial update on J after 
                each step, updating the projection of J along the 
                parameter change direction. Cheap in time cost, but not
                always accurate. Default is False. 
            eig_update: Bool
                Set to True to update the projection of J along the most
                stiff eigendirections of JTJ. Slower than broyden but 
                more accurate & useful. Default is False. 
            num_eig_dirs: Int
                If eig_update == True, the number of eigendirections to
                update when doing the eigen update. Default is 4. 
            partial_update_frequency: Int
                If broyden_update or eig_update, the frequency to do 
                either/both of those partial updates. Default is 3. 
                
            quiet: Bool
                Set to False to print messages about convergence. 
        
        Relevant attributes
        -------------------
            do_run_1: Function
                ...what you should set when you use run_1 v run_2 etc
                For instance run_2 might stop prematurely since its
                internal runs update last_error, last_params, and it 
                usually just runs until it takes a bad step == small
                param update. 
            do_run_2: Function
            
        """
        self.damping = float(damping)
        self.increase_damp_factor = float(increase_damp_factor)
        self.decrease_damp_factor = float(decrease_damp_factor)
        self.min_eigval = min_eigval
        self.quiet = quiet #maybe multiple quiets? one for algorithm speed, one for option speed/efficacy?
        self.marquardt_damping = marquardt_damping
        self.transtrum_damping = transtrum_damping
        
        self.use_accel = use_accel
        self.max_accel_correction = max_accel_correction
        
        self.ptol = ptol
        self.errtol = errtol
        self.costol = costol
        self.max_iter = max_iter
        
        self.update_J_frequency = update_J_frequency
        self.broyden_update = broyden_update
        self.eig_update = eig_update
        self.num_eig_dirs = num_eig_dirs
        self.run_length = run_length
        self._inner_run_counter = 0
        self.partial_update_frequency = partial_update_frequency
        
        self._num_iter = 0
        
        #We want to start updating JTJ
        self._J_update_counter = update_J_frequency
        self._fresh_JTJ = False
        
        #the max # of times trying to decrease damping before giving up
        self._max_inner_loop = 10 
        
        #Finally we set the error and parameters
        self._set_err_params()
        self._has_run = False
    
    def reset(self, new_damping=None):
        """
        Keeps all user supplied options the same, but resets counters etc. 
        """
        self._num_iter = 0
        self._inner_run_counter = 0
        self._J_update_counter = self.update_J_frequency
        self._fresh_JTJ = False
        self._has_run = False
        if new_damping is not None:
            self.damping = new_damping
        
    def _set_err_params(self):
        """
        Must update:
            self.error, self._last_error, self.params, self._last_params 
        """
        raise NotImplementedError('implement in subclass')
    
    def calc_J(self):
        """Updates self.J, returns nothing"""
        raise NotImplementedError('implement in subclass')
        
    def calc_residuals(self):
        raise NotImplementedError('implement in subclass')
        
    def calc_model_cosine(self):
        """
        Calculates the cosine of the fittable residuals with the actual
        residuals, cos(phi) = |P^T r| / |r| where P^T is the projection
        operator onto the model manifold and r the residuals. 
        """
        #1. Calculate projection term
        u, sig, v = np.linalg.svd(self.J, full_matrices=False)
        # p = np.dot(v.T, v) - memory error, so term-by-term
        r = self.calc_residuals()
        v_r = np.dot(v,r)
        projected = np.dot(v.T, v_r)
        
        #2. Calculate indiviual norms:
        abs_r = np.sqrt((r*r).sum())
        abs_prj = np.sqrt((projected*projected).sum())
        
        return abs_prj / abs_r
    
    def do_run_1(self):
        """
        LM run evaluating 1 step at a time. Broyden or eigendirection
        updates replace full-J updates. No internal runs. 
        """
        while not self.check_terminate():
            self._has_run = True
            if self.check_update_J():
                self.update_J()
            else:
                if self.check_Broyden_J():
                    self.update_Broyden_J()
                if self.check_update_eig_J():
                    self.update_eig_J()

            #1. Assuming that J starts updated:
            delta_params = self.find_LM_updates(self.calc_grad())
            
            #2. Increase damping until we get a good step:
            good_step = self.eval_n_check_step(self.params + delta_params, 
                    put_back=False)
            _try = 0
            while (_try < self._max_inner_loop) and (not good_step):
                _try += 1
                self.increase_damping()
                delta_params = self.find_LM_updates(self.calc_grad())
                good_step = self.eval_n_check_step(self.params + delta_params, 
                        put_back=False)
            if _try == (self._max_inner_loop-1):
                warnings.warn('Stuck!', RuntimeWarning)
            
            #state is updated, now params:
            self.update_params(delta_params, incremental=True)
            self.decrease_damping()
            self._num_iter += 1; self._inner_run_counter += 1
    
    def do_run_2(self):
        """
        LM run evaluating 2 steps (damped and not) and choosing the best. 
        Runs with that damping + Broyden or eigendirection updates, until
        deciding to do a full-J update. Only changes damping after full-J
        updates. 
        """
        while not self.check_terminate():
            self._has_run = True
            if self.check_update_J():
                self.update_J()
            else:
                if self.check_Broyden_J():
                    self.update_Broyden_J()
                if self.check_update_eig_J():
                    self.update_eig_J()

            #1. Calculate 2 possible steps
            delta_params_1 = self.find_LM_updates(self.calc_grad(), 
                    do_correct_damping=False)
            self.decrease_damping()
            delta_params_2 = self.find_LM_updates(self.calc_grad(), 
                    do_correct_damping=False)
            self.decrease_damping(undo_decrease=True)
            
            #2. Check which step is best:
            er1 = self.update_function(self.params + delta_params_1)
            er2 = self.update_function(self.params + delta_params_2)
            #ARG I need to put it back... this costs 1-2 extra function
            #evaluations per loop. Try to clean up eval_n_check_step FIXME
            _ = self.update_function(self.params)
            
            triplet = (self.error, er1, er2)
            if self.error < min([er1, er2]):
                #Both bad steps, increase damping:
                _try = 0
                good_step = False
                if not self.quiet:
                    print 'Bad step, increasing damping\t%f\t%f\t%f' % triplet
                while (_try < self._max_inner_loop) and (not good_step):
                    self.increase_damping()
                    delta_params = self.find_LM_updates(self.calc_grad())
                    good_step = self.eval_n_check_step(self.params + 
                            delta_params, put_back=False)
                    _try += 1
                if _try == (self._max_inner_loop-1):
                    warnings.warn('Stuck!', RuntimeWarning)
                elif good_step:
                    print 'Sufficiently decreased damping\t%f\t%f' % (triplet[0], self.error)
            elif er1 <= er2:
                if not self.quiet:
                    print 'Good step, same damping\t%f\t%f\t%f' % triplet
                good_step = self.eval_n_check_step(self.params + delta_params_1)
                self.update_params(delta_params_1, incremental=True)

                if not good_step:
                    raise NotImplementedError('You messed up coding this') #FIXME obv
            else: #Good steps
                #er2 < er1
                if not self.quiet:
                    print 'Good step, decreasing damping\t%f\t%f\t%f' % triplet
                good_step = self.eval_n_check_step(self.params + delta_params_2)
                self.update_params(delta_params_2, incremental=True)
                self.decrease_damping()
                if not good_step:
                    raise NotImplementedError('You messed up coding this') #FIXME obv
                    
            #3. Run with current J, damping:
            if good_step:
                self.do_internal_run()
            #1 loop
            self._num_iter += 1
    
    def do_internal_run(self):
        """
        Given a fixed damping, J, JTJ, iterates calculating steps, with
        optional Broyden or eigendirection updates. 
        Called internally by do_run_2() but might also be useful on its own. 
        """
        self._inner_run_counter = 0; good_step = True
        if not self.quiet:
            print 'Running...'
        while ((self._inner_run_counter < self.run_length) & good_step &
                (not self.check_terminate())):
            if self.check_Broyden_J() and self._inner_run_counter != 0:
                self.update_Broyden_J()
            if self.check_update_eig_J() and self._inner_run_counter != 0:
                self.update_eig_J()
            delta_params = self.find_LM_updates(self.calc_grad(), 
                do_correct_damping=False)
            good_step = self.eval_n_check_step(self.params + delta_params,
                    put_back=False)
            if good_step:
                if not self.quiet:
                    print '%f\t%f' % (self._last_error, self.error)
                self.update_params(delta_params, incremental=True)
                #^I don't like that syntax FIXME
            elif not self.quiet:
                print 'Bad step!' #right now thru eval_n_check doesn't give bad step
            self._inner_run_counter += 1
            
    def eval_n_check_step(self, new_params, put_back=False):
        """
        Doesn't update back to the old params ONLY IF the step is good
        If the step is bad, you HAVE to go back to the original params
        """        
        _last_error = self.error*1
        _last_residuals = self.calc_residuals()
        self.error = self.update_function(new_params)
        good_step = self.error <= _last_error
        
        #something for cosine options...
        
        if (not good_step) and not(self.quiet) and (not put_back):
            pass
            # print 'Bad step:\t%f\t->\t%f' % (self._last_error, self.error)
        if (not good_step) or put_back:
            self.error = self.update_function(self.params)
        else:
            if not self.quiet:
                pass
                # print 'Good step:\t%f\t->\t%f' % (self._last_error, self.error)
            self._last_error = _last_error
            self._last_residuals = _last_residuals
        return good_step
        
    def update_function(self, params):
        """Takes an array params, updates function, returns the new error"""
        raise NotImplementedError('implement in subclass')
    
    def _calc_damped_jtj(self):
        diag = 0*self.JTJ
        mid = np.arange(diag.shape[0], dtype='int')

        if self.marquardt_damping:
            diag[mid, mid] = self.JTJ[mid, mid].copy()
        elif self.transtrum_damping is not None:
            diag[mid, mid] = np.clip(self.JTJ[mid, mid].copy(),
                    self.transtrum_damping, np.inf)
        else:
            diag[mid, mid] = 1.0
        
        damped_JTJ = self.JTJ + self.damping*diag
        return damped_JTJ
        
    def find_LM_updates(self, grad, do_correct_damping=True):
        """
        Calculates LM updates, with or without the acceleration correction. 
        """
        damped_JTJ = self._calc_damped_jtj()
        delta0, res, rank, s = np.linalg.lstsq(damped_JTJ, -grad, rcond=self.min_eigval)
        if (not self.quiet) & self._fresh_JTJ:
            print '%d degenerate of %d total directions' % (delta0.size-rank, delta0.size)
        
        if self.use_accel:
            accel_correction = self.calc_accel_correction(damped_JTJ, delta0)
            nrm_d0 = np.sqrt(np.sum(delta0**2))
            nrm_corr = np.sqrt(np.sum(accel_correction**2))
            if not self.quiet:
                print '|correction term| / |initial vector|\t%e' % (nrm_corr/nrm_d0)
            if nrm_corr/nrm_d0 < self.max_accel_correction:
                delta0 += accel_correction
            elif do_correct_damping:
                if not self.quiet:
                    print 'Untrustworthy step! Increasing damping...'
                    self.increase_damping()
                    damped_JTJ = self._calc_damped_jtj()
                    delta0, res, rank, s = np.linalg.lstsq(damped_JTJ, -grad, \
                            rcond=self.min_eigval)
            
        return delta0
    
    def increase_damping(self):
        self.damping *= self.increase_damp_factor
        
    def decrease_damping(self, undo_decrease=False):
        if undo_decrease:
            self.damping *= self.decrease_damp_factor
        else:
            self.damping /= self.decrease_damp_factor
        
    def update_params(self, new_params, incremental=False):
        self._last_params = self.params.copy()
        if incremental:
            self.params += new_params
        else:
            self.params = new_params.copy()
        #And we've updated, so JTJ is no longer valid:
        self._fresh_JTJ = False
            
    def check_terminate(self):
        """
        Termination if ftol, ptol, costol are < a certain amount
        Currently costol is not used / supported
        """
        
        if not self._has_run:
            return False
        else:
            terminate = False
            
            #1. change in params small enough?
            delta_params = self._last_params - self.params
            terminate |= np.all(np.abs(delta_params) < self.ptol)
            
            #2. change in err small enough?
            delta_err = self._last_error - self.error
            terminate |= (delta_err < self.errtol)
            
            #3. change in cosine small enough? 
            if self.costol is not None:
                curcos = self.calc_model_cosine()
                terminate |= (curcos < self.costol)
            
            #4. too many iterations??
            terminate |= (self._num_iter > self.max_iter)
            return terminate
        
    def check_update_J(self):
        """
        Checks if the full J should be updated. Right now, just updates if 
        we've done update_J_frequency loops
        """
        self._J_update_counter += 1
        update = self._J_update_counter > self.update_J_frequency
        return update & (not self._fresh_JTJ)
    
    def update_J(self):
        self.calc_J()
        self.JTJ = np.dot(self.J, self.J.T)
        self._fresh_JTJ = True
        self._J_update_counter = 0
        
    def calc_grad(self):
        residuals = self.calc_residuals()
        return np.dot(self.J, residuals)
    
    def check_Broyden_J(self):
        do_update = (self.broyden_update & (not self._fresh_JTJ) & 
                (self._inner_run_counter % self.partial_update_frequency) == 0)
        return do_update
        
    def update_Broyden_J(self):
        """
        Broyden update of jacobian. 
        """
        delta_params = self.params - self._last_params
        delta_residuals = self.calc_residuals() - self._last_residuals
        broyden_update = np.outer(delta_params, (delta_residuals -\
                np.dot(self.J.T, delta_params))) / np.sum(delta_params**2)
        self.J += broyden_update
        self.JTJ = np.dot(self.J, self.J.T)
    
    def check_update_eig_J(self):
        do_update = (self.eig_update & (not self._fresh_JTJ) & 
                (self._inner_run_counter % self.partial_update_frequency) == 0)
        return do_update
        
    def update_eig_J(self):
        vls, vcs = np.linalg.eigh(self.JTJ)
        res0 = self.calc_residuals()
        for a in xrange(min([self.num_eig_dirs, vls.size])):
            #1. Finding stiff directions
            stif_dir = vcs[-(a+1)] #already normalized
            
            #2. Evaluating derivative along that direction, we'll use dl=5e-4:
            dl = 5e-4
            _ = self.update_function(self.params+dl*stif_dir)
            res1 = self.calc_residuals()
            
            #3. Updating
            grad_stif = (res1-res0)/dl
            update = np.outer(stif_dir, grad_stif - np.dot(self.J.T, stif_dir))
            self.J += update
            self.JTJ = np.dot(self.J, self.J.T)
        
        #Putting the parameters back:
        _ = self.update_function(self.params)
    
    def calc_accel_correction(self, damped_JTJ, delta0):
        dh = 0.2 #paper recommends 0.1, but I think it'll be better to do
        #slightly higher to get more of a secant approximation
        rm0 = self.calc_residuals()
        _ = self.update_function(self.params + delta0*dh)
        rm1 = self.calc_residuals()
        term1 = (rm1 - rm0) / dh
        #and putting back the parameters:
        _ = self.update_function(self.params)
        
        term2 = np.dot(self.J.T, delta0)
        der2 = 2./dh*(term1 - term2)
        
        damped_JTJ = self._calc_damped_jtj()
        corr, res, rank, s = np.linalg.lstsq(damped_JTJ, np.dot(self.J, der2),
                rcond=self.min_eigval)
        corr *= -0.5
        return corr
    
class LMGlobals(LMEngine):
    def __init__(self, state, block, max_mem=3e9, opt_kwargs={}, **kwargs):
        """
        Levenberg-Marquardt engine for state globals with all the options 
        from the M. Transtrum J. Sethna 2012 ArXiV paper. See LMGlobals
        for documentation. 

        Inputs:
        -------
        state: cbamf.states.ConfocalImagePython instance
            The state to optimize
        block: np.ndarray of bools
            The (un-exploded) blocks to optimize over. 
        max_mem: Int
            The maximum memory to use for the optimization; controls block
            decimation. Default is 3e9. 
        opt_kwargs: Dict
            Dict of **kwargs for opt implementation. Right now only for
            opt.get_num_px_jtj, i.e. keys of 'decimate', min_redundant'.
        """
        self.state = state
        self.kwargs = opt_kwargs
        self.num_pix = get_num_px_jtj(state, block.sum(), **self.kwargs)
        self.blocks = state.explode(block)
        self.block = block
        super(LMGlobals, self).__init__(**kwargs)

    def _set_err_params(self):
        self.error = get_err(self.state)
        self._last_error = get_err(self.state)
        self.params = self.state.state[self.block].copy()
        self._last_params = self.params.copy()

    def calc_J(self):
        J, inds = get_rand_Japprox(self.state, self.blocks, 
                num_inds=self.num_pix, **self.kwargs)
        self._inds = inds
        self.J = J
    
    def calc_residuals(self):
        return self.state.get_difference_image()[self._inds].ravel()
        
    def update_function(self, params):
        update_state_global(self.state, self.block, params)
        return get_err(self.state)
    
    def set_block(self, new_block, new_damping=None):
        self.block = new_block
        self.blocks = self.state.explode(new_block)
        self._set_err_params()
        self.reset(new_damping=new_damping)

class LMParticles(LMEngine):
    def __init__(self, state, particles, opt_kwargs={}, **kwargs):
        self.state = state
        self.kwargs = opt_kwargs
        self.particles = particles
        self.error = get_err(self.state)
        self._dif_tile = get_tile_from_multiple_particle_change(state, particles)
        super(LMParticles, self).__init__(**kwargs)
        
    def _set_err_params(self):
        self.error = get_err(self.state)
        self._last_error = get_err(self.state)
        params = []
        for p in self.particles:
            params.extend(self.state.obj.pos[p].tolist() + [self.state.obj.rad[p]])
        self.params = np.array(params)
        self._last_params = self.params.copy()
    
    def calc_J(self):
        self._dif_tile = get_tile_from_multiple_particle_change(self.state, 
                self.particles)
        self.J = eval_many_particle_grad(self.state, self.particles, 
                slicer=self._dif_tile.slicer, **self.kwargs)
                
    def calc_residuals(self):
        return get_slicered_difference(self.state, self._dif_tile.slicer, 
                self.state.image_mask[self._dif_tile.slicer].astype('bool'))
                
    def update_function(self, params):
        update_particles(self.state, self.particles, params, 
                relative=False, fix_errors=True)
        return get_err(self.state)
        
    def set_particles(self, new_particles, new_damping=None):
        self.particles = new_particles
        self._dif_tile = get_tile_from_multiple_particle_change(
                self.state, particles)
        self._set_err_params()
        self.reset(new_damping=new_damping)
        
class LMParticleGroupCollection(object):
    """
    ... I don't think this will work since you'll run out of memory keeping
    J for every particle in the image
    """
    def __init__(self, state, region_size, do_calc_size=True, **kwargs):
        """**kwargs for the individual lm collections"""
        raise NotImplementedError
    
    def do_run_1(self):
        for lm in self.lm_collections:
            lm.do_run_1()
        
    def do_run_2(self):
        for lm in self.lm_collections:
            lm.do_run_2()
            
    def reset(self, new_damping=None):
        for lm in self.lm_collections:
            lm.reset(new_damping=new_damping)
    
    def create_particle_lm_groups(self):
        pass
        
    def full_reset():
        pass #also recalculates the groups, sets everything to the original values
        #basically calls __init__ again except without the setup parts. 
        #Or __init__ basically calls this

class AugmentedState(object):
    """
    A state that, in addition to having normal state update options, 
    allows for updating all the particle R, xyz's depending on their 
    positions -- basically rscale(x) for everything. 
    Right now I'm just doing this with R(z)
    """
    def __init__(self, state, block, rz_order=3):
        """
        block can be an array of False, that's OK
        """
        self.state = state
        self.block = block

        #You need some way of controlling the params, so:
        globals_mask = np.zeros(block.sum() + rz_order, dtype='bool')
        globals_mask[:blocks.sum()] = True
        rscale_mask = -globals_mask
        self.globals_mask = globals_mask
        self.rscale_mask = rscale_mask
        
        params = np.zeros(globals_mask.size, dtype='float')
        params[:block.sum()] = self.state.state[block].copy()
        self.params = params

    def set_block(self, new_block):
        """
        I don't think there is a point to this since the rscale(z) aren't
        actual parameters
        """
        raise NotImplementedError
        
        
    def rad_func(self, pos):
        """Right now exp(legval(z))"""
        return np.exp(self._poly(pos[:,2]))
    
    def _poly(self, z):
        """Right now legval(z)"""
        shp = self.state.image.shape
        zmax = float(shp[0])
        zmin = 0.0
        zmid = zmax / 2
        
        ans = np.polynomial.legendre.legval((z-zmid)/zmid, 
                self.params[self.rscale_mask])
        return ans
    
    def update(self, params):
        """Updates all the parameters of the state + rscale(z)"""
        self.update_rscl_x_params(params[self.rscale_mask], do_reset=False)
        update_state_global(self.state, self.block, self.params[self.globals_mask])
        self.params[:] = params
    
    def update_rscl_x_params(self, new_rscl_params, do_reset=True):
        s = self.state
        #1. What to change:
        m = s.obj.typ == 1
        p = s.obj.pos[m]
        rold = s.obj.rad[m]
        
        #2. New, old values:
        old_scale = self.rad_func(p)
        self.params[self.rscale_mask] = new_rscl_params
        new_scale = self.rad_func(p)
        
        rnew = rold * (new_scale / old_scale)
        s.obj.rad[m] = rnew
        s.obj.initialize(zscale=s.zscale)
        if do_reset:
            s.reset()
        
class LMAugmentedState(LMEngine):
    def __init__(self, aug_state, rz_order=3, max_mem=3e9,
            opt_kwargs={}, **kwargs):
        """
        Levenberg-Marquardt engine for state globals with all the options 
        from the M. Transtrum J. Sethna 2012 ArXiV paper. See LMGlobals
        for documentation. 

        Inputs:
        -------
        state: cbamf.states.ConfocalImagePython instance
            The state to optimize
        max_mem: Int
            The maximum memory to use for the optimization; controls block
            decimation. Default is 3e9. 
        opt_kwargs: Dict
            Dict of **kwargs for opt implementation. Right now only for
            opt.get_num_px_jtj, i.e. keys of 'decimate', min_redundant'.
        """
        self.aug_state = aug_state
        self.kwargs = opt_kwargs
        self.num_pix = get_num_px_jtj(state, block.sum()+rz_order, 
                **self.kwargs)
        super(LMGlobals, self).__init__(**kwargs)
        raise NotImplementedError
        
    def _set_err_params(self):
        self.error = get_err(self.aug_state.state)
        self._last_error = get_err(self.aug_state.state)
        self.params = self.aug_state.params
        self._last_params = self.params.copy()

    def calc_J(self):
        #1. J for the state:
        s = self.aug_state.state
        sa = self.aug_state
        blocks = s.explode(self.aug_state.block)
        J_st, inds = get_rand_Japprox(s, blocks, num_inds=self.num_pix,
                **self.kwargs)
        self._inds = inds
        
        #2. J for the augmented portion:
        old_aug_params = sa.params[sa.rscale_mask].copy()
        dl = 1e-6
        J_aug = []
        i0 = s.get_difference_image()
        for a in xrange(old_aug_params.size):
            dx = np.zeros(old_aug_params.size)
            dx[a] = dl
            sa.update_rscl_x_params(old_aug_params + dl, do_reset=True)
            i1 = s.get_difference_image()
            der = (i1-i0)/dl
            J_aug.apppend(der[self._inds].ravel())
        
        self.J = np.append(J_st, np.array(J_aug), axis=0)

    def calc_residuals(self):
        return self.state.get_difference_image()[self._inds].ravel()

    def update_function(self, params):
        self.aug_state.update(params)
        return get_err(self.aug_state.state)
