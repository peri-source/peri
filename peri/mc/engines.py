from builtins import range, object

class SequentialBlockEngine(object):
    def __init__(self, state):
        self.state = state
        self.loglike = None
        self.samplers = []

        # likelihood and state observers (treated differently)
        self.like_obs = []
        self.state_obs = []

    def add_samplers(self, samplers):
        if hasattr(samplers, '__iter__'):
            self.samplers.extend(samplers)
        else:
            self.samplers.append(samplers)

    def add_likelihood_observers(self, obs):
        if hasattr(obs, '__iter__'):
            self.like_obs.extend(obs)
        else:
            self.like_obs.append(obs)

    def add_state_observers(self, obs):
        if hasattr(obs, '__iter__'):
            self.state_obs.extend(obs)
        else:
            self.state_obs.append(obs)

    def reset_observers(self):
        for ob in self.state_obs:
            ob.reset()
        for ob in self.like_obs:
            ob.reset()

    def remove_samplers(self):
        self.samplers = []

    def remove_likelihood_observers(self):
        self.like_obs = []

    def remove_state_observers(self):
        self.state_obs = []

    def dosteps(self, nsteps=1, burnin=False):
        if len(self.samplers) == 0:
            raise AttributeError("Engine does not have samplers, add samplers to continue")

        ll, s = self.loglike, self.state
        for i in range(nsteps):
            for sampler in self.samplers:
                if ll is not None and ll < -1e50:
                    pass
                ll, s = sampler.sample(s, ll)
                if ll < -1e50:
                    pass

            if not burnin:
                for ob in self.state_obs:
                    ob.update(s)
                for ob in self.like_obs:
                    ob.update(ll)

        self.loglike, self.state = ll, s
