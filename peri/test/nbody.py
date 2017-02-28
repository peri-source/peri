from builtins import range, object

from peri import util

import numpy as np
import scipy as sp

def create_configuration(N, tile, radius=5.0, **kwargs):
    """
    Quick configuration creation with `N` particles in a tile of size `tile`.
    """
    pos, rad, tile = initialize_particles(N, tile, radius=radius, **kwargs)
    sim = BrownianHardSphereSimulation(pos, rad, tile)
    sim.relax(1000)
    sim.step(1000)
    sim.relax(1000)
    return sim.pos, sim.rad, tile

def initialize_particles(N=None, tile=None, phi=None, radius=5., polydispersity=0.):
    """
    Creates an initial (overlapping) particle configuration by 3 different ways:
        (N, phi) : creates an appropriate box
        (N, tile) : uses a box and determines phi
        (tile, phi): uses both to determine N
    """
    vparticle = 4./3*np.pi*radius**3

    if phi and tile is not None:
        vbox = np.prod(tile.shape)
        N = int(phi * vbox / vparticle)

    elif N and phi:
        vbox = N * vparticle / phi
        side = vbox**(1./3)
        tile = util.Tile(side)

    elif N and tile is not None:
        pass

    else:
        raise AttributeError("Any pair of (phi, tile, N) must be specified")

    pos = np.random.rand(N, 3)*tile.shape + tile.l
    rad = np.abs(np.random.normal(
        loc=radius, scale=np.abs(polydispersity*radius)+1e-20, size=N
    ))
    return pos, rad, tile

class BrownianHardSphereSimulation(object):
    def __init__(self, pos, rad, tile, D=0.2, epsilon=10.0, dt=1e-1):
        """
        Creates a hard sphere brownian dynamics simulation in ND as specified
        by the pos, rad and tile supplied to initializer.

        Parameters:
        -----------
        pos : ndarray [N, dim]
            Positions of particles in the simulation.

        rad : ndarray [N]
            Radii of particles in the simulation.

        tile : `peri.util.Tile`
            The simulation box defined by a tile with left and right bounds.

        D : float
            diffusion constant in proper units

        epsilon : float
            Force constant for the soft-sphere potential f = \epsilon (1-d/d_0)^{3/2}.
            This parameter is really a ratio of epsilon / eta (the damping parameter)

        dt : float
            timestep for the integrator
        """
        self.pos = pos
        self.rad = rad
        self.tile = tile

        self.N = self.pos.shape[0]
        self.dim = self.pos.shape[1]
        self.forces = 0*self.pos

        self.D = D
        self.dt = dt
        self.epsilon = epsilon

    def force_noise(self):
        """ Calculate the effective force of the Langevin dynamics """
        coeff = np.sqrt(2*self.D)
        return coeff * np.random.randn(*self.pos.shape)
    
    def boundary_condition(self):
        """ Apply hard reflective boundary conditions to particles """
        for i in range(self.dim):
            mask = (self.pos[:,i] < self.tile.l[i])
            self.pos[mask,i] = 2*self.tile.l[i]-self.pos[mask,i]
    
            mask = (self.pos[:,i] > self.tile.r[i])
            self.pos[mask,i] = 2*self.tile.r[i]-self.pos[mask,i]
    
    def integrate(self, forces):
        """
        Integrate the equations of motion. For this simple integrator, we are
        using the simplest sympletic integrator, NSV where

            v_{n+1} = v_n + f*dt
            x_{n+1} = x_n + v_{n+1}*dt

        Parameters:
        -----------
        forces : ndarray[N,2]
            the forces on each particle
        """
        self.pos += forces*self.dt
    
    def step(self, steps=100, mask=None):
        """
        Perform a set of integration / BC steps and update plot

        Parameters:
        -----------
        steps : int
            number of time steps of size self.dt to perform

        mask : boolean ndarray[N]
            if provided, only simulate the motion of certain particles given
            by the boolean array mask
        """
        if mask is None:
            mask = np.ones_like(self.rad).astype('bool')

        for step in range(steps):
            self.forces = self.force_hardsphere(mask) + self.force_noise()
            self.forces[~mask, :] = 0.
            self.integrate(self.forces)
            self.boundary_condition()

    def relax(self, steps=1000):
        """ Relax the current configuration using just pair wise forces (no noise) """
        for step in range(steps):
            self.forces = self.force_hardsphere()
            self.integrate(self.forces)
            self.boundary_condition()

    def force_hardsphere(self, mask=None):
        """
        Calculate 'hard shere' forces between all particles. In this case,
        hard sphere means Hertzian potential

            V(r_{ij}) = \epsilon (1 - r_{ij} / 2a)^{5/2}
        """
        N, pos, rad = self.N, self.pos, self.rad

        if mask is None:
            mask = np.ones_like(self.rad).astype('bool')
    
        f = np.zeros_like(pos)
        for i in np.arange(N)[mask]:
            rij = pos[i] - pos
            dist = np.sqrt((rij**2).sum(axis=-1))

            dia = rad + rad[i]
            mask = (dist > 1e-5)&(dist < dia)
            dia = dia[mask]
            rij = rij[mask]
            dist = dist[mask][:,None]
    
            if len(rij) > 0:
                forces = self.epsilon*(1-dist/dia[:,None])**2 * rij/dist
                f[i] += forces.sum(axis=0)
    
        return f

    def neighbors(self, cutoff, indices=None):
        """ Returns neighbors within a cutoff for certain particles """
        indices = indices if indices is not None else np.arange(self.N)
        indices = util.listify(indices)
        neighs = []

        for i in indices:
            rij = self.pos[i] - self.pos
            dist = np.sqrt((rij**2).sum(axis=-1))
            neighs.append(np.arange(self.N)[dist <= cutoff])

        return neighs
