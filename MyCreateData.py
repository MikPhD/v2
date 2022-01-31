from fenics import *
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import os
from pathlib import Path
import argparse
from pdb import set_trace
from progress.bar import Bar
import itertools
import time

class CreateData:
    def __init__(self):
        self.comm = MPI.comm_world
        set_log_level(40)
        parameters["form_compiler"]["optimize"] = True
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["std_out_all_processes"] = False

    def bound_index_list(self, coord_dofs, bmesh, bound_index):
        for n, x in enumerate(coord_dofs):
            if x in bmesh:
                bound_index.append(n)
        return bound_index

    def create_connection(self, dofs_list, coord_dofs, bound_index, node1, node2):
        if node1 in bound_index:
            if node2 in bound_index:
                # print("entrambi sul bordo!", node1, node2)
                return None, None
        else:
            c = [dofs_list[node1]/2, dofs_list[node2]/2]

            dist_c_x = abs(coord_dofs[node1][0] - coord_dofs[node2][0])
            dist_c_y = abs(coord_dofs[node1][1] - coord_dofs[node2][1])
            dist_c = [dist_c_x, dist_c_y]

            return c, dist_c

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
            #### iterative tool used ####
            collapsed_space = Space.sub(0).collapse()
            dofmap = collapsed_space.dofmap()

            ####definisco le node features della gnn #####
            coord = []
            coord_temp = list(collapsed_space.tabulate_dof_coordinates().tolist())
            for i in coord_temp:
                if i not in coord:
                    coord.append(i)

            U = []
            F = []
            for x in coord:
                U.append(list(u(np.array(x))))
                F.append(list(forc(np.array(x))))

            #### GNN connectivity #####
            C = [] ##connection list
            D = [] ##distances between connection

            bmesh = BoundaryMesh(mesh, "exterior", True).coordinates()

            with Bar("Creazione connessioni...", max=mesh.num_cells()) as bar:
                for i, j in enumerate(cells(mesh)):
                    bar.next()

                    c0 = Cell(mesh, i)

                    #### dofs list and coordinates - remove odds ####
                    dofs_list_tot = dofmap.cell_dofs(c0.index())
                    coord_dofs_tot = collapsed_space.element().tabulate_dof_coordinates(c0)
                    dofs_list = []
                    coord_dofs = []

                    for i, j in enumerate(dofs_list_tot):
                        if j % 2 == 0:
                            dofs_list.append(j)
                            coord_dofs.append(list(coord_dofs_tot[i]))

                    ###### Vertex on boundary - dofs index on boundary ######
                    bound_index = []
                    bound_index = CreateData.bound_index_list(self, coord_dofs, bmesh, bound_index)

                    #### create unidirectional connection ####
                    c, d = CreateData.create_connection(self, dofs_list, coord_dofs, bound_index, int(3), int(1))
                    if (c != None) and c not in C:
                        C.append(c)
                        D.append(d)

                    c, d = CreateData.create_connection(self, dofs_list, coord_dofs, bound_index, int(3), int(2))
                    if (c != None) and c not in C:
                        C.append(c)
                        D.append(d)

                    c, d = CreateData.create_connection(self, dofs_list, coord_dofs, bound_index, int(4), int(0))
                    if (c != None) and c not in C:
                        C.append(c)
                        D.append(d)

                    c, d = CreateData.create_connection(self, dofs_list, coord_dofs, bound_index, int(4), int(2))
                    if (c != None) and c not in C:
                        C.append(c)
                        D.append(d)

                    c, d = CreateData.create_connection(self, dofs_list, coord_dofs, bound_index, int(5), int(0))
                    if (c != None) and c not in C:
                        C.append(c)
                        D.append(d)

                    c, d = CreateData.create_connection(self, dofs_list, coord_dofs, bound_index, int(5), int(1))
                    if (c != None) and c not in C:
                        C.append(c)
                        D.append(d)

                    # c1 = [dofs_list[3], dofs_list[1]]
                    # c2 = [dofs_list[3], dofs_list[2]]
                    #
                    # dist_c1_x = abs(coord_dofs[3][0] - coord_dofs[1][0])
                    # dist_c1_y = abs(coord_dofs[3][1] - coord_dofs[1][1])
                    # dist_c1 = [dist_c1_x, dist_c1_y]
                    #
                    # dist_c2_x = abs(coord_dofs[3][0] - coord_dofs[2][0])
                    # dist_c2_y = abs(coord_dofs[3][1] - coord_dofs[2][1])
                    # dist_c2 = [dist_c2_x, dist_c2_y]
                    #
                    # c3 = [dofs_list[4], dofs_list[0]]
                    # c4 = [dofs_list[4], dofs_list[2]]
                    #
                    # dist_c3_x = abs(coord_dofs[4][0] - coord_dofs[0][0])
                    # dist_c3_y = abs(coord_dofs[4][1] - coord_dofs[0][1])
                    # dist_c3 = [dist_c3_x, dist_c3_y]
                    #
                    # dist_c4_x = abs(coord_dofs[4][0] - coord_dofs[2][0])
                    # dist_c4_y = abs(coord_dofs[4][1] - coord_dofs[2][1])
                    # dist_c4 = [dist_c4_x, dist_c4_y]
                    #
                    # c5 = [dofs_list[5], dofs_list[0]]
                    # c6 = [dofs_list[5], dofs_list[1]]
                    #
                    # dist_c5_x = abs(coord_dofs[5][0] - coord_dofs[0][0])
                    # dist_c5_y = abs(coord_dofs[5][1] - coord_dofs[0][1])
                    # dist_c5 = [dist_c5_x, dist_c5_y]
                    #
                    # dist_c6_x = abs(coord_dofs[5][0] - coord_dofs[1][0])
                    # dist_c6_y = abs(coord_dofs[5][1] - coord_dofs[1][1])
                    # dist_c6 = [dist_c6_x, dist_c6_y]
                    #
                    # if c1 not in C:
                    #     C.append(c1)
                    #     D.append(dist_c1)
                    # if c2 not in C:
                    #     C.append(c2)
                    #     D.append(dist_c2)
                    # if c3 not in C:
                    #     C.append(c3)
                    #     D.append(dist_c3)
                    # if c4 not in C:
                    #     C.append(c4)
                    #     D.append(dist_c4)
                    # if c5 not in C:
                    #     C.append(c5)
                    #     D.append(dist_c5)
                    # if c6 not in C:
                    #     C.append(c6)
                    #     D.append(dist_c6)
                    #
                    # C_temp = list(itertools.chain(C_temp, [c1], [c2], [c3], [c4], [c5], [c6]))
                    # # D = list(itertools.chain(D, [dist_c1_x, dist_c1_y], [dist_c2_x, dist_c2_y], [dist_c3_x, dist_c3_y],
                    # #                          [dist_c4_x, dist_c4_y], [dist_c5_x, dist_c5_y], [dist_c6_x, dist_c6_y]))
                    #
                    # D = list(itertools.chain(D, [dist_c1], [dist_c2], [dist_c3], [dist_c4], [dist_c5], [dist_c6]))

            # ############## remove duplicate ####################
            # C = []
            # remove = []
            # with Bar("Rimozione duplicati connessione...", max=len(C_temp)) as bar2:
            #     for i, j in enumerate(C_temp, start=0):
            #         bar2.next()
            #         if j not in C:
            #             C.append(j)
            #         else:
            #             remove.append(i)

            # with Bar("Rimozione duplicati distanze...", max=len(remove)) as bar3:
            #     for index in sorted(remove, reverse=True):
            #         del D[index]

            C_rev = []
            with Bar("Creazione doppia direzionalità...", max=len(C)) as bar4:
                for i in range(len(C)):
                    bar4.next()
                    C_rev.append(C[i][::-1])

            C += C_rev
            D += D

            # set_trace()
            ######################### Fine creazione elementi############################

            ###################### Salvataggio file Numpy ##############################
            os.makedirs("./dataset/raw/" + mode + "/" + h, exist_ok=True)
            specific_dir = "./dataset/raw/" + mode + "/" + h
            np.save(specific_dir + "/C.npy", C)
            np.save(specific_dir + "/D.npy", D)
            np.save(specific_dir + "/U.npy", U)
            np.save(specific_dir + "/F.npy", F)
            # np.save(specific_dir + "/coord.npy", coord)
            np.save(specific_dir + "/re.npy", int(h))
            ################# Fine salvataggio file ##################################

        ################# Print interface ########################################
        print("Trasformazione file di " + mode + " completata!")#
