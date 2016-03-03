import os,sys, time
import numpy as np
from numpy.random import randint


"""
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
        to_return.append(a_der[inds[0], inds[1], inds[2]].copy())
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
    return s.get_difference_image()[inds[0], inds[1], inds[2]].copy()
    
def eval_deriv(s, block, dl=2e-5, be_nice=True, **kwargs):
    """
    Using a centered difference / 3pt stencil approximation:
    """
    p0 = s.state[block]
    s.update(block, p0+dl)
    i1 = s.get_difference_image().copy()
    s.update(block, p0-dl)
    i2 = s.get_difference_image().copy()
    if be_nice:
       s.update(block, p0) 
    return 0.5*(i1-i2)/dl
    
def calculate_JTJ_grad_approx(s, blocks, num_inds=1000, **kwargs):
    if num_inds < s.image[s.inner].size:
        inds = [randint(v, size=num_inds) for v in s.image[s.inner].shape]
    else:
        inds = [slice(0,None), slice(0,None), slice(0,None)]
    J = calculate_J_approx(s, blocks, inds, **kwargs)
    JTJ = np.dot(J, J.T)
    err = calculate_err_approx(s, inds)
    return JTJ, np.dot(J,err)
    
def get_rand_Japprox(s, blocks, num_inds=1000, keep_time=True, **kwargs):
    """
    """
    if keep_time:
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
    if keep_time:
        print 'JTJ:\t%f' % (time.time()-start_time)
    return J, inds
        
def j_to_jtj(J):
    return np.dot(J, J.T)

def calc_im_grad(s, J, inds):
    err = calculate_err_approx(s, inds)
    return np.dot(J, err)

def find_LM_updates(JTJ, grad, accel=1.0, min_eigval=1e-12, quiet=True, **kwargs):
    diag = 0*JTJ
    mid = np.arange(diag.shape[0], dtype='int')
    diag[mid, mid] = JTJ[mid, mid].copy()
    
    A0 = JTJ + accel*diag
    # delta0 = np.linalg.solve(A0, -grad)
    delta0, res, rank, s = np.linalg.lstsq(A0, -grad, rcond=min_eigval)
    if not quiet:
        print '%d degenerate of %d total directions' % (delta0.size-rank, delta0.size)
    
    return delta0
    
def update_state_global(s, block, data, keep_time=True, **kwargs):
    """
    """
    #We need to update:
    #obj, psf, ilm, bkg, off, slab, zscale, sigma
    #We dont' need to update:
    #rscale
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
        raise NotImplementedError("updating the particle positions is hard")
        #Mostly because I don't know wtf ns is in s.obj.update()
        #ns = "n's" = numbers. Do s.obj.rad.size to get n, then update rad, type
    #rad:
    brad = s.create_block('rad')
    if (brad & block).sum() > 0:
        new_rad_params = new_state[brad].copy()
        s.obj.rad = new_rad_params
        s.obj.initialize(s.zscale)
    #typ:
    btyp = s.create_block('typ')
    if (brad & btyp).sum() > 0:
        new_typ_params = new_state[btyp].copy()
        s.obj.typ = new_typ_params
        s.obj.initialize(s.zscale)
    
    #psf:
    bp = s.create_block('psf')
    if (bp & block).sum() > 0:
        new_psf_params = new_state[bp].copy()
        s.psf.update(new_psf_params)
        
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
    
    #off:
    bo = s.create_block('off')
    if (bo & block).sum() > 0:
        new_off = new_state[bo].copy()
        s.update(bo, new_off)
    
    #slab:
    bs = s.create_block('slab')
    if (bs & block).sum() > 0:
        new_slab_params = new_state[bs].copy()
        s.slab.update(new_slab_params)
        
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

def get_num_px_jtj(s, nparams, decimate=400, max_mem=2e9, min_redundant=20, **kwargs):
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
    
def do_levmarq(s, block, accel=0.1, daccel=0.1, num_iter=5, do_run=True, \
         run_length=3, **kwargs):
    """
    Runs Levenberg-Marquardt minimization on a cbamf state, based on a
    random approximant of J. Doesn't return anything, only updates the state. 
    Parameters:
    -----------
    s : State
        The cbamf state to optimize. 
    block: Boolean numpy.array
        The desired blocks of the state s to minimize over. Do NOT explode it. 
    accel: Float scalar, >=0. 
        The acceleration parameter in the Levenberg-Marquardt optimization.
        Big acceleration means gradient descent with a small step, accel=0
        means Hessian inversion. Default is 0.1. 
    daccel: Float scalar, >0. 
        Multiplicative value to update accel by. Internally decides when to
        update and when to use accel or 1/accel. Default is 0.1. 
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
        the attempts to use all the pixels in the image. Default is 400. 
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
        Set to True to print a bunch of irritating messages about how long
        each step of the algorithm took. Default is False. 
    be_nice: Bool. 
        If True, evaluating the derivative doesn't change the state. If 
        False, when the derivative is evaluated the state isn't changed 
        back to its original value, which saves time (33%) but may wreak 
        havoc. Default is True (i.e. slower, no havoc). 
    dl: Float scalar.
        The amount to update parameters by when evaluating derivatives. 
        Default is 2e-5. 
    See Also
    --------
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
    
    def do_internal_run(J, inds, accel):
        """
        Uses the local vars s, accel, run_length, block
        """
        print 'Running....'
        for rn in xrange(run_length):
            new_grad = calc_im_grad(s, J, inds)
            p0 = s.state[block].copy()
            dnew = find_LM_updates(JTJ, new_grad, accel=accel)
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
        
        #1. Get J, JTJ, grad
        if recalc_J:
            J, inds = get_rand_Japprox(s, blocks, num_inds=num_px, **kwargs)
            JTJ = j_to_jtj(J)
            grad = calc_im_grad(s, J, inds) #internal because s gets updated
        
        #2. Calculate and implement first guess for updates
        d0 = find_LM_updates(JTJ, grad, accel=accel)
        d1 = find_LM_updates(JTJ, grad, accel=accel*daccel)
        update_state_global(s, block, p0+d0)
        err0 = get_err(s)
        update_state_global(s, block, p0+d1)
        err1 = get_err(s)
        
        
        if np.min([err0, err1]) > err_start: #Bad step...
            print 'Bad step!\t%f\t%f\t%f' % (err_start, err0, err1)
            update_state_global(s, block, p0)
            recalc_J = False
            #Avoiding infinite loops by adding a small amount to counter:
            counter += 0.1 
            if err0 < err1: 
                #d_accel is the wrong "sign", so we invert:
                print 'Changing daccel:\t%f\t%f' % (daccel, 1.0/daccel)
                daccel = 1.0/daccel
            else: 
                #daccel is the correct sign but accel is too big:
                accel *= (daccel*daccel)
        
        else: #at least 1 good step:
            if err0 < err1: #good step and accel, re-update:
                update_state_global(s, block, p0+d0)
                print 'Good step:\t%f\t%f\t%f' % (err_start, err0, err1)
            else: #err1 < err0 < err_start, good step but decrease accel:
                accel *= daccel
                print 'Decreasing acceleration:\t%f\t%f' % (err_start, err1)
            if do_run:
                do_internal_run(J, inds, accel)
            recalc_J = True
            counter += 1

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
        the attempts to use all the pixels in the image. Default is 400. 
    max_mem: Float scalar. 
        The maximum memory (in bytes) that J should occupy. Default is 2GB. 
    min_redundant: Float scalar. 
        Enforces a minimum amount of pixels to include in J, such that the 
        min # of pixels is at least min_redundant * number of parameters. 
        If max_mem and min_redundant result in an incompatible size an
        error is raised. Default is 20. 
    keep_time: Bool
        Set to True to print a bunch of irritating messages about how long
        each step of the algorithm took. Default is False. 
    be_nice: Bool. 
        If True, evaluating the derivative doesn't change the state. If 
        False, when the derivative is evaluated the state isn't changed 
        back to its original value, which saves time (33%) but may wreak 
        havoc. Default is True (i.e. slower, no havoc). 
    dl: Float scalar.
        The amount to update parameters by when evaluating derivatives. 
        Default is 2e-5. 
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

from scipy.optimize import minimize_scalar
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
    

    
