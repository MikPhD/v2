import ast
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from fenics import *
import argparse

class Plot:
    def __init__(self):
        pass

    def plot_loss(self):
        ### Open log files
        with open('Stats/loss_train_log.txt', 'r') as f_train:
            mydata_train = ast.literal_eval(f_train.read())
        with open('Stats/loss_val_log.txt', 'r') as f_val:
            mydata_val = ast.literal_eval(f_val.read())

        ### define axis and data ###
        dt = 1
        x = np.arange(0, len(mydata_train), dt)
        y_train = mydata_train
        y_val = mydata_val

        plt.plot(x, y_train, x, y_val)
        plt.savefig("Stats/plot_loss.jpg")

        ### Close Files ###
        f_train.close()
        f_val.close()

    def mesh2triang(self, mesh):
        xy = mesh.coordinates()
        return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

    def plot(self, obj):
        plt.gca().set_aspect('equal')
        if isinstance(obj, Function):
            mesh = obj.function_space().mesh()
            if obj.vector().size() == mesh.num_cells():
                C = obj.vector().array()
                plt.tripcolor(self.mesh2triang(mesh), C)
            else:
                C = obj.compute_vertex_values(mesh)
                plt.tripcolor(self.mesh2triang(mesh), C, shading='gouraud')
        elif isinstance(obj, Mesh):
            plt.triplot(self.mesh2triang(obj), color='k')

    def plot_results(self, n_epoch = ""):
        directory = "./Results/"

        ####### loading mesh ########
        mesh = Mesh()
        mesh_file = directory + "Mesh.h5"
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

        f.vector().set_local(F_gnn)

        # ####### plot ########
        plt.figure()
        self.plot(f.sub(0))
        plt.savefig("Stats/plot_results" + n_epoch + ".jpg")
