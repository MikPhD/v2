from fenics import *
from progress.bar import Bar
import numpy as np
import matplotlib.pyplot as plt




class CreateData:
    def __init__(self):
        self.comm = MPI.comm_world
        set_log_level(40)
        parameters["form_compiler"]["optimize"] = True
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["std_out_all_processes"] = False

    def transform(self, cases, mode):
        for h in cases:
            #####initializing###
            print("Elaborazione dataset case: ", h, " in modalità: ", mode)

            ################# inizio lettura file ##########################
            ######### lettura mesh #########
            mesh = Mesh()
            mesh_file = "../Dataset/" + str(h) + "/Mesh.h5"
            with HDF5File(self.comm, mesh_file, "r") as h5file:
                h5file.read(mesh, "mesh", False)
                facet = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
                h5file.read(facet, "facet")


            ###### lettura risultati ########
            VelocityElement = VectorElement("CG", mesh.ufl_cell(), 2)
            PressureElement = FiniteElement("CG", mesh.ufl_cell(), 1)
            Space = FunctionSpace(mesh, VelocityElement * PressureElement)

            w = Function(Space)
            f = Function(Space)

            with HDF5File(self.comm, "../Dataset/" + str(h) + "/Results.h5", "r") as h5file:
                h5file.read(w, "mean")
                h5file.read(f, "forcing")

            u, p = w.split(deepcopy='true')
            u.set_allow_extrapolation(True)
            forc, _ = f.split(deepcopy='true')
            forc.set_allow_extrapolation(True)
            ############### End lettura file ####################################

            ################### Creazione elementi dataset ##########################

            ### analizzo ogni edge della mesh ###
            mesh.init()
            mesh_topology = mesh.topology()
            mesh_connectivity = mesh_topology(1, 0)

            mesh_point = []
            suppl_point = []
            with Bar("Creazione supplementary...", max=mesh.num_edges()) as bar:
                for i in range(mesh.num_edges()):
                    bar.next()
                    connection = np.array(mesh_connectivity(i)).astype(int)
                    coord_vert1 = (mesh.coordinates()[connection[0]]).tolist()
                    coord_vert2 = (mesh.coordinates()[connection[1]]).tolist()

                    coord_vert3_x = (coord_vert1[0] + coord_vert2[0]) / 2
                    coord_vert3_y = (coord_vert1[1] + coord_vert2[1]) / 2
                    coord_vert3 = [coord_vert3_x, coord_vert3_y]

                    if coord_vert1 not in mesh_point:
                        mesh_point.append(coord_vert1)
                    if coord_vert2 not in mesh_point:
                        mesh_point.append(coord_vert2)
                    if coord_vert3 not in suppl_point:
                        suppl_point.append(coord_vert3)

            plt.figure()
            plot(mesh)
            for x, y in mesh_point:
                plot(x, y, marker="o", color="blue", markeredgecolor="black")
            # plt.plot(bmesh_x, bmesh_y, 'r*')
            # plt.plot(suppl_point_x, suppl_point_y, 'g*')
            plt.show()
            #### problema al plot! riprendere da qua|!!!!
