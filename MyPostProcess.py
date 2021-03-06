import numpy as np
from fenics import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import argparse
import traceback
from pdb import set_trace
import numpy as np
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from collections import OrderedDict

cmaps = OrderedDict()

cmaps['Perceptually Uniform Sequential'] = ['plasma_r']

def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def plot(obj):
    # plt.gca().set_aspect('equal')
    mesh = obj.function_space().mesh()

    if isinstance(obj, Function):
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().array()
            plt.tripcolor(mesh2triang(mesh), C, cmap=cmaps, norm='Normalize')

        else:
            x = mesh.coordinates()[:, 0]
            y = mesh.coordinates()[:, 1]
            t = mesh.cells()
            v = obj.compute_vertex_values(mesh)
            vmin = v.min()
            vmax = v.max()
            v[v < vmin] = vmin + 1e-12
            v[v > vmax] = vmax - 1e-12
            from matplotlib.ticker import ScalarFormatter
            cmap = 'viridis'
            levels = np.linspace(vmin, vmax, 100)
            formatter = ScalarFormatter()
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            c = ax.tricontourf(x, y, t, v, levels=levels, norm=norm,
                               cmap=plt.get_cmap(cmap))
            plt.axis('equal')
            p = ax.triplot(x, y, t, '-', lw=0.5, alpha=0.0)
            ax.set_xlim([x.min(), x.max()])
            ax.set_ylim([y.min(), y.max()])
            ax.set_xlabel(' $\it{Coordinata\:x}$')
            ax.set_ylabel(' $\it{Coordinata\:y}$')
            # tit = plt.title('Componente x del tensore di Reynolds')
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes('right', "4%", pad="2%")
            colorbar_format = '% 1.1f'
            cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, format=colorbar_format)



    elif isinstance(obj, Mesh):
        plt.triplot(mesh2triang(obj), color='k')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--case_num', help='case number', type=str, default="40")
parser.add_argument('-n', '--n_epoch', help='num epoch', type=str, default="")
args = parser.parse_args()

case_num = args.case_num
n_epoch = args.n_epoch

directory = "./processed/train/" + case_num + "/"

####### loading mesh ########
mesh = Mesh()
mesh_file = "../Dataset/" + case_num + "/Mesh.h5"
with HDF5File(MPI.comm_world, mesh_file, "r") as h5file:
    h5file.read(mesh, "mesh", False)
    facet = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    h5file.read(facet, "facet")

####### initializing holder ########
VelocityElement = VectorElement("CG", mesh.ufl_cell(), 2)
PressureElement = FiniteElement("CG", mesh.ufl_cell(), 1)
Space = FunctionSpace(mesh, VelocityElement * PressureElement)

F = Function(Space)

f, _ = F.split(deepcopy=True)

# ####### loading forcing from GNN ################
F_gnn = np.load('./Results/' + 'results.npy').flatten()
# set_trace()

collapsed_space = Space.sub(0).collapse()
coord = list(collapsed_space.tabulate_dof_coordinates().tolist())
# for i, x in enumerate(coord):
#     f(x) = F_gnn[i]

# set_trace()

# for i, x in enumerate(F_gnn):
#     f.vector()[i] = x
#     # print(i,x)

f.vector().set_local(F_gnn)

plot(f.sub(0))
plt.show()




