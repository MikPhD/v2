from fenics import *
from progress.bar import Bar
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
import matplotlib.tri as tri
import os

class CreateData:
    def __init__(self):
        self.comm = MPI.comm_world
        set_log_level(40)
        parameters["form_compiler"]["optimize"] = True
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["std_out_all_processes"] = False

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
            #### lista delle connessioni della mesh ###
            mesh_points = mesh.coordinates().tolist()
            bmesh = BoundaryMesh(mesh, "exterior", True).coordinates().tolist()

            ### analizzo ogni edge della mesh ###
            mesh.init()
            mesh_topology = mesh.topology()
            mesh_connectivity = mesh_topology(1, 0)

            suppl_points = []

            C = [] #list of connection
            D = [] #lenght of connection in coordinates
            # with Bar("Creazione connection, distances, value of function...", max=mesh.num_edges()) as bar: #does not work on slurm
            for i in range(mesh.num_edges()):
                if i % (mesh.num_edges()/2) == 0:
                    print("Elaborazione in corso: {} su {}".format(i, mesh.num_edges()))
                # bar.next()
                connection_mesh_index = np.array(mesh_connectivity(i)).astype(int)
                coord_vert1 = (mesh.coordinates()[connection_mesh_index[0]]).tolist()
                coord_vert2 = (mesh.coordinates()[connection_mesh_index[1]]).tolist()

                ### find point of boundary ###
                if coord_vert1 and coord_vert2 in bmesh:
                        pass
                else:
                    #### create intermediate points ###
                    coord_vert3_x = (coord_vert1[0] + coord_vert2[0]) / 2
                    coord_vert3_y = (coord_vert1[1] + coord_vert2[1]) / 2
                    coord_vert3 = [coord_vert3_x, coord_vert3_y]

                    suppl_points.append(coord_vert3)  ##useless
                    mesh_points.append(coord_vert3)
                    ### create connection ####
                    index1 = mesh_points.index(coord_vert1)
                    index2 = mesh_points.index(coord_vert2)
                    index3 = mesh_points.index(coord_vert3)
                    connection1 = [index1, index3]
                    connection2 = [index3, index2]
                    C.append(connection1)
                    C.append(connection2)

                    ### compute lenght of connection ###
                    dist_c1_x = abs(coord_vert1[0] - coord_vert3[0])
                    dist_c1_y = abs(coord_vert1[1] - coord_vert3[1])
                    dist_c1 = [dist_c1_x, dist_c1_y]

                    dist_c2_x = abs(coord_vert3[0] - coord_vert2[0])
                    dist_c2_y = abs(coord_vert3[1] - coord_vert2[1])
                    dist_c2 = [dist_c2_x, dist_c2_y]
                    D.append(dist_c1)
                    D.append(dist_c2)

                    if coord_vert1 in bmesh:
                        connection3 = [index2, index3]
                        C.append(connection3)
                        D.append(dist_c2)
                    elif coord_vert2 in bmesh:
                        connection3 = [index3, index1]
                        C.append(connection3)
                        D.append(dist_c1)
                    else:
                        connection3 = [index3, index1]
                        C.append(connection3)
                        D.append(dist_c1)
                        connection4 = [index2, index3]
                        C.append(connection4)
                        D.append(dist_c2)


            U = []
            F = []
            for x in mesh_points:
                U.append(list(u(np.array(x))))
                F.append(list(forc(np.array(x))))

            ###################### Salvataggio file Numpy ##############################
            os.makedirs("./dataset/raw/" + mode + "/" + h, exist_ok=True)
            specific_dir = "./dataset/raw/" + mode + "/" + h
            np.save(specific_dir + "/C.npy", C)
            np.save(specific_dir + "/D.npy", D)
            np.save(specific_dir + "/U.npy", U)
            np.save(specific_dir + "/F.npy", F)
            # np.save(specific_dir + "/coord.npy", coord)
            np.save(specific_dir + "/re.npy", int(h))
            if mode == 'val':
                np.save("./Results/mesh_points.npy", mesh_points)
            ################# Fine salvataggio file ##################################

        ################# Print interface ########################################
        print("Trasformazione file di " + mode + " completata!")

            # set_trace()

        # plt.figure()
        # plot(mesh)
        # for x, y in mesh_points:
        #     plt.plot(x, y, 'r*')
        # plt.show()
        # plot(mesh)
        # for x, y in suppl_points:
        #     plt.plot(x, y, 'g*')
        # plt.show()
        # plot(mesh)
        # for x, y in bmesh:
        #     plt.plot(x, y, 'b*')
        # plt.show()
        # x = []
        # y = []
        # for i, (index1, index2) in enumerate(C[0:1000]):
        #     xv1 = mesh_points[index1][0]
        #     yv1 = mesh_points[index1][1]
        #     xv2 = mesh_points[index2][0]
        #     yv2 = mesh_points[index2][1]
        #     xmean = (xv1+xv2)/2
        #     ymean = (yv1+yv2)/2
        #
        #     u = xv2-xv1
        #     v = yv2-yv1
        #
        #     plt.plot([xv1,xv2], [yv1, yv2], 'ro-', label=i)
        #     plt.annotate(i, xy=(xmean, ymean), xycoords='data')
        #     plt.quiver(xmean, ymean, u, v)
        # #
        # plot(mesh)
        # plt.show()
