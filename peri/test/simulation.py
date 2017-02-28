from builtins import range, object

import numpy as np
import scipy as sp

class SoftSphereSimulation(object):
    def __init__(self, N=500, phi=0.65, radius=5.0, polydispersity=0.0, beta=1,
            epsilon=120, T=1, dt=1e-2, dim=3, box_side_ratio=1.0):
        """
        Creates a simulation of soft sphere particles. By default, creates
        particles in a random uniform (perhaps overlapping) distribution of
        particles inside a box dimension `dim` given a packing fraction

        Parameters:
        -----------
        N : integer [default: 500]
            the number of particles in the simulation

        phi : float 
            packing fraction to use while initializing the particles

        radius : float [default: 5]
            mean radius of the particles, see polydispersity

        polydispersity : float [default: 0]
            relative polydispersity goal sqrt(<(a-<a>)^2>)/<a>

        beta : float
            damping parameter f = -beta v

        epsilon : float
            force constant for the soft-sphere potential f = \epsilon (1-d/d_0)^{3/2}

        T : float
            temperature of the system

        dt : float
            timestep for the integrator

        dim : integer [default: 3]
            number of dimensions for the simulation

        box_side_ratio : float [default: 1]
            ratio of the box's z height to original z height before scaling.
            therefore, you can squish the box by half (while elongating the
            sides) by setting box_side_ratio=0.5

        """
        self.N = N
        self.phi = phi
        self.beta = beta
        self.epsilon = epsilon
        self.T = T
        self.dt = dt
        self.radius = radius
        self.polydispersity = polydispersity
        self.box_side_ratio = box_side_ratio
        self.dim = int(dim)

        # find the box size based on the number of particles and packing fraction
        if self.dim == 2:
            self.box_side = (self.N*np.pi*self.radius**2 / self.phi)**(1./2)
            self.box = np.array([self.box_side]*self.dim)
        if self.dim == 3:
            self.box_side = (self.N*4./3*np.pi*self.radius**3 / self.phi)**(1./3)
            sxy = self.box_side/np.sqrt(self.box_side_ratio)
            sz  = self.box_side*self.box_side_ratio
            self.box = np.array([sz, sxy, sxy])

        self.init_random()

    def init_random(self):
        """
        Place the current particles into a stationary, random distribution
        and create a polydispersity for the radii
        """
        self.pos = self.box[None,:]*np.random.rand(self.N,self.dim)
        self.rad = self.radius*(1 + self.polydispersity*np.random.randn(self.N))
        self.rad = np.clip(self.rad, 1e-8, 1e8)
        self.vel = 0*self.pos

    def force_damp(self):
        """ Calculate the damping force -beta v """
        return - self.beta * self.vel
    
    def force_noise(self):
        """ Calculate the effective force of the Langevin dynamics """
        coeff = np.sqrt(2*self.T*self.beta/self.dt)
        return coeff * np.random.randn(*self.pos.shape)
    
    def boundary_condition(self):
        """ Apply hard reflective boundary conditions to particles """
        for i in range(self.dim):
            mask = (self.pos[:,i] < 0)
            self.pos[mask,i] = 2*0-self.pos[mask,i]
            self.vel[mask,i] *= -1
    
            mask = (self.pos[:,i] > self.box[i])
            self.pos[mask,i] = 2*self.box[i]-self.pos[mask,i]
            self.vel[mask,i] *= -1
    
    def integrate(self, forces):
        """
        Integrate the equations of motion. For this simple integrator, we are
        using the simplest sympletic integrator, NSV where

            v_{n+1} = v_n + f*dt
            x_{n+1} = x_n + v_{n+1}*dt

        Parameters:
        -----------
        forces : ndarray[N,dim]
            the forces on each particle
        """
        self.vel += forces*self.dt
        self.pos += self.vel*self.dt
    
    def step(self, steps=1):
        """
        Perform a set of integration / BC steps including finite temperature

        Parameters:
        -----------
        steps : int
            number of time steps of size self.dt to perform
        """
        for step in range(steps):
            self.forces = self.force_hertzian() + self.force_damp() + self.force_noise()
            self.integrate(self.forces)
            self.boundary_condition()
    
    def relax(self, steps=1000):
        """ Relax the current configuration using just pair wise forces (no noise) """
        for step in range(steps):
            self.forces = self.force_hertzian() + self.force_damp()
            self.integrate(self.forces)
            self.boundary_condition()

    def force_hertzian(self):
        """
        Return the force on all spheres, do naive all-all interaction, so as
        to not introduce new dependencies in the code (like anaconda jit)
        """
        N = self.N
        pos, rad = self.pos, self.rad
    
        f = np.zeros_like(pos)
        for i in range(N):
            rij = pos[i] - pos
            dij = rad[i] + rad
            dist = np.sqrt((rij**2).sum(axis=-1))
    
            mask = (dist > 0)&(dist < dij)
            rij = rij[mask]
            dij = dij[mask][:,None]
            dist = dist[mask][:,None]
   
            if len(rij) > 0:
                forces = self.epsilon*(1-dist/dij)**2 * rij/dist
                f[i] += forces.sum(axis=0)
    
        return f

