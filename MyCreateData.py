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
            #### GNN connectivity #####
            collapsed_space = Space.sub(0).collapse()
            dofmap = collapsed_space.dofmap()

            C = [] ##connection list
            D = [] ##distances between connection
            with Bar("Creazione connessioni...", max=mesh.num_cells()) as bar:
                for i, j in enumerate(cells(mesh)):
                    bar.next()

                    c0 = Cell(mesh, i)
                    dofs_list = dofmap.cell_dofs(c0.index())
                    coord_dofs = collapsed_space.element().tabulate_dof_coordinates(c0)
                    c1 = [int((dofs_list[3])/2), int((dofs_list[1])/2)]
                    c2 = [int((dofs_list[3])/2), int((dofs_list[2])/2)]
                    c3 = [int((dofs_list[4])/2), int((dofs_list[0])/2)]
                    c4 = [int((dofs_list[4])/2), int((dofs_list[2])/2)]
                    c5 = [int((dofs_list[5])/2), int((dofs_list[0])/2)]
                    c6 = [int((dofs_list[5])/2), int((dofs_list[1])/2)]

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
                for y in C:
                    bar.next()
                    if y in C:
                        C.remove(y)


            ####definisco le node features della gnn #####
            coord = list(collapsed_space.tabulate_dof_coordinates().tolist())
            coord = list(k for k,_ in itertools.groupby(coord))

            U = []
            F = []

            for x in coord:
              U.append(list(u(np.array(x))))
              F.append(list(forc(np.array(x))))

            ######################### Fine creazione elementi############################

            ###################### Salvataggio file Numpy ##############################
            os.makedirs("./dataset/raw/" + mode + "/" + h, exist_ok=True)
            specific_dir = "./dataset/raw/" + mode + "/" + h
            np.save(specific_dir + "/C.npy", C)
            np.save(specific_dir + "/D.npy", D)
            np.save(specific_dir + "/U_P.npy", U)
            np.save(specific_dir + "/F.npy", F)
            # np.save(specific_dir + "/coord.npy", coord)
            np.save(specific_dir + "/re.npy", int(h))
            ################# Fine salvataggio file ##################################

        ################# Print interface ########################################
        print("Trasformazione file di " + mode + " completata!")#
