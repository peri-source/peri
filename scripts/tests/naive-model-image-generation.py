from colloids.bamf import sampler, engine, observers, model, initialize
import numpy as np

itrue,p = initialize.local_max_points("./scripts/data/colloid-2d-slice.png", offset=32, zlevel=16, radius=0)
r = p[:,0]*0 + 8
s = np.hstack([p.flatten(), r, [3.02,100.]])
n = model.NaiveColloids(itrue, p.shape[0], Lz=32)
it = n.docalculate(s) 
print it.max(), it.min()

h = sampler.MHSampler(0.1)
h = sampler.SliceSampler(0.1)
e = engine.SequentialBlockEngine(n, s)
e.add_samplers(h)
e.dosteps(100)

