from . import models, samplers, engines

def sample_state(state, blocks, slicing=True, N=1, doprint=False):
    m = models.PositionsRadiiPSF(imsig=GS)

    eng = engines.SequentialBlockEngine(m, state)
    opsay = observers.Printer()
    ohist = observers.HistogramObserver(block=blocks[0])
    eng.add_samplers([samplers.SliceSampler(RADIUS/1e1, block=b) for b in blocks])

    eng.add_likelihood_observers(opsay) if doprint else None
    eng.add_state_observers(ohist)

    eng.dosteps(N)
    return ohist

def sample_ll(state, element, size=0.1, N=1000):
    m = models.PositionsRadiiPSF(imsig=GS)
    start = state.state[element]

    ll = []
    vals = np.linspace(start-size, start+size, N)
    for val in vals:
        state.update(element, val)
        l = m.loglikelihood(state)
        ll.append(l)
    return vals, np.array(ll)

def scan_noise(image, state, element, size=0.01, N=1000):
    start = state.state[element]

    xs, ys = [], []
    for i in xrange(N):
        print i
        test = image + np.random.normal(0, GS, image.shape)
        x,y = sample_ll(test, state, element, size=size, N=300)
        state.update(element, start)
        xs.append(x)
        ys.append(y)

    return xs, ys

def sample_particles(state):
    for particle in xrange(s.N):
        print particle
        sys.stdout.flush()

        renorm(state)

        if s.set_current_particle(particle):
            blocks = s.blocks_particle()
            sample_state(s, blocks)

