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

    # def bound_index_list(self, coord_dofs, bmesh, bound_index):
    #     for n, x in enumerate(coord_dofs):
    #         if x in bmesh:
    #             bound_index.append(n)
    #     return bound_index

    # def create_connection(self, dofs_list, coord_dofs, bound_index_cell, node1, node2):
    #     if node1 in bound_index_cell:
    #         if node2 in bound_index_cell:
    #             # print("entrambi sul bordo!", node1, node2)
    #             return None, None
    #     else:
    #         c = [dofs_list[node1]/2, dofs_list[node2]/2]
    #
    #         dist_c_x = abs(coord_dofs[node1][0] - coord_dofs[node2][0])
    #         dist_c_y = abs(coord_dofs[node1][1] - coord_dofs[node2][1])
    #         dist_c = [dist_c_x, dist_c_y]
    #
    #         return c, dist_c
    def create_connection(self, dofs_list_cell, coord_dofs_cell, bound_index_cell, node1, node2):
        if node1 in bound_index_cell:
            if node2 in bound_index_cell:
                        # print("entrambi sul boundary!", node1, node2)
                        return None, None, None
        else:
            c = [int(dofs_list_cell[node1] / 2), int(dofs_list_cell[node2] / 2)]
            c_rev = c[::-1]

            dist_c_x = abs(coord_dofs_cell[node1][0] - coord_dofs_cell[node2][0])
            dist_c_y = abs(coord_dofs_cell[node1][1] - coord_dofs_cell[node2][1])
            dist_c = [dist_c_x, dist_c_y]

            return c, c_rev, dist_c

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

            # bmesh = BoundaryMesh(mesh, "exterior", True).coordinates()

            # dofs_on_bound =[]

            with Bar("Creazione connessioni...", max=mesh.num_cells()) as bar:
                for i, j in enumerate(cells(mesh)):
                    bar.next()

                    c0 = Cell(mesh, i)

                    #### dofs list and coordinates for each cell - remove odds ####
                    dofs_list_cell_tot = dofmap.cell_dofs(c0.index())
                    coord_dofs_cell_tot = collapsed_space.element().tabulate_dof_coordinates(c0)
                    dofs_list_cell = []
                    coord_dofs_cell = []

                    for i, j in enumerate(dofs_list_cell_tot):
                        if j % 2 == 0:
                            dofs_list_cell.append(j)
                            coord_dofs_cell.append(list(coord_dofs_cell_tot[i]))

                    # ###### dofs index on boundary ######
                    # bound_index_cell = []
                    # for n, x in enumerate(coord_dofs_cell):
                    #     if x in bmesh:
                    #         bound_index_cell.append(n)

                    #### create monodirectional connection ####
                    c, c_rev, d = CreateData.create_connection(self, dofs_list_cell, coord_dofs_cell, bound_index_cell, int(3), int(1))
                    if (c != None) and c not in C:
                        C.append(c)
                        C.append(c_rev)
                        D.append(d)
                        D.append(d)

                    c, c_rev, d = CreateData.create_connection(self, dofs_list_cell, coord_dofs_cell, bound_index_cell, int(3), int(2))
                    if (c != None) and c not in C:
                        C.append(c)
                        C.append(c_rev)
                        D.append(d)
                        D.append(d)

                    c, c_rev, d = CreateData.create_connection(self, dofs_list_cell, coord_dofs_cell, bound_index_cell, int(4), int(0))
                    if (c != None) and c not in C:
                        C.append(c)
                        C.append(c_rev)
                        D.append(d)
                        D.append(d)

                    c, c_rev, d = CreateData.create_connection(self, dofs_list_cell, coord_dofs_cell, bound_index_cell, int(4), int(2))
                    if (c != None) and c not in C:
                        C.append(c)
                        C.append(c_rev)
                        D.append(d)
                        D.append(d)

                    c, c_rev, d = CreateData.create_connection(self, dofs_list_cell, coord_dofs_cell, bound_index_cell, int(5), int(0))
                    if (c != None) and c not in C:
                        C.append(c)
                        C.append(c_rev)
                        D.append(d)
                        D.append(d)

                    c, c_rev, d = CreateData.create_connection(self, dofs_list_cell, coord_dofs_cell, bound_index_cell, int(5), int(1))
                    if (c != None) and c not in C:
                        C.append(c)
                        C.append(c_rev)
                        D.append(d)
                        D.append(d)

                    # for x in coord_dofs_cell:
                    #     if x[0] == 0:
                    #         if x not in dofs_on_bound:
                    #             dofs_on_bound.append(x)


            # C_prov = []
            # for x in C:
            #     if x not in C_prov:
            #         C_prov.append(x)
            #
            # D_prov = []
            # for x in D:
            #     if x not in D_prov:
            #         D_prov.append(x)
            #
            # bmesh_onbound = []
            # for x in bmesh:
            #     if x[0] == 0:
            #         bmesh_onbound.append(x)

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
