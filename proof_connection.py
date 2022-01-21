import sys
import os
from pathlib import Path
import numpy as np
import argparse
from fenics import *
import matplotlib.pyplot as plt
import math
from pdb import set_trace
from progress.bar import Bar
from dolfin.cpp.fem import GenericDofMap
import itertools

p1 = Point(0,0)
p2 = Point(2,2)

mesh = RectangleMesh(p1, p2, 2, 2, diagonal="right")
print("Plotting a RectangleMesh")

VelocityElement = VectorElement("CG", mesh.ufl_cell(), 2)
PressureElement = FiniteElement("CG", mesh.ufl_cell(), 1)
Space = FunctionSpace(mesh, VelocityElement * PressureElement)


f0 = Expression(('x[0]', 'x[1]', '0'), degree=2)
w = interpolate(f0, Space)

u, p = w.split(deepcopy=True)
u.set_allow_extrapolation(True)

#### GNN connectivity #####
collapsed_space = Space.sub(0).collapse()
dofmap = collapsed_space.dofmap()

C = [] ##connection list
D = [] ##distances between connection

for i, j in enumerate(cells(mesh)):
    c0 = Cell(mesh, i)
    dofs_list = dofmap.cell_dofs(c0.index())
    coord_dofs = collapsed_space.element().tabulate_dof_coordinates(c0)
    c1 = [int((dofs_list[3]) / 2), int((dofs_list[1]) / 2)]
    c2 = [int((dofs_list[3]) / 2), int((dofs_list[2]) / 2)]
    c3 = [int((dofs_list[4]) / 2), int((dofs_list[0]) / 2)]
    c4 = [int((dofs_list[4]) / 2), int((dofs_list[2]) / 2)]
    c5 = [int((dofs_list[5]) / 2), int((dofs_list[0]) / 2)]
    c6 = [int((dofs_list[5]) / 2), int((dofs_list[1]) / 2)]

    dist_c1_x = coord_dofs[3][0] - coord_dofs[1][0]
    dist_c1_y = coord_dofs[3][1] - coord_dofs[1][1]

    dist_c2_x = coord_dofs[3][0] - coord_dofs[2][0]
    dist_c2_y = coord_dofs[3][1] - coord_dofs[2][1]

    dist_c3_x = coord_dofs[4][0] - coord_dofs[4][0]
    dist_c3_y = coord_dofs[4][1] - coord_dofs[0][1]

    dist_c4_x = coord_dofs[4][0] - coord_dofs[4][0]
    dist_c4_y = coord_dofs[2][1] - coord_dofs[2][1]

    dist_c5_x = coord_dofs[5][0] - coord_dofs[5][0]
    dist_c5_y = coord_dofs[0][1] - coord_dofs[0][1]

    dist_c6_x = coord_dofs[5][0] - coord_dofs[5][0]
    dist_c6_y = coord_dofs[1][1] - coord_dofs[1][1]

    C = list(itertools.chain(C, [c1], [c2], [c3], [c4], [c5], [c6]))
    D = list(itertools.chain(D, [dist_c1_x, dist_c1_y], [dist_c2_x, dist_c2_y], [dist_c3_x, dist_c3_y],
                             [dist_c4_x, dist_c4_y], [dist_c5_x, dist_c5_y], [dist_c6_x, dist_c6_y]))

# for y in C:
#     if y in C:
#         C.remove(y)

####definisco le node features della gnn #####
coord = list(collapsed_space.tabulate_dof_coordinates().tolist())
coord = list(k for k,_ in itertools.groupby(coord))

U = []

for x in coord:
  U.append(list(u(np.array(x))))

######################### Fine creazione elementi############################
set_trace()
# plt.figure()
# plot(mesh)
# plt.show()