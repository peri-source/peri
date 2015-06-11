from ase import data
from ase.lattice.cubic import FaceCenteredCubic
atoms = FaceCenteredCubic(directions=[[1,-1,0], [1,1,-2], [1,1,1]],
        size=(3,2,2), symbol='Cu', pbc=(1,1,1),
        latticeconstant=data.covalent_radii[data.atomic_numbers['Cu']]*4)
l = data.covalent_radii[data.atomic_numbers['Cu']]
print atoms.get_cell()/l

