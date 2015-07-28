import numpy as np
import pylab as pl

from cbamf.test import init

s = init.create_state_random_packing(imsize=36, radius=5.0, sigma=0.05, seed=10)
bl_pos = s.explode(s.create_block('pos'))
bl_rad = s.explode(s.create_block('rad'))

f = s.fisher_information(bl_pos + bl_rad)

inv = np.linalg.inv(f)
crb = np.sqrt(np.diag(inv))

bp = 3*s.N
br = 4*s.N
pl.figure()
pl.imshow(inv)
pl.colorbar()
pl.hlines(bp, 0, br, lw=1)
pl.vlines(bp, 0, br, lw=1)
pl.xlim(0, br-1)
pl.ylim(0, br-1)
pl.title("Dense suspension CRB")

pl.figure()
pl.semilogy(crb, 'o')
pl.xlabel("Parameter index")
pl.ylabel("CRB")
pl.title("Dense suspension CRB")
pl.xlim(0, br)
pl.show()
