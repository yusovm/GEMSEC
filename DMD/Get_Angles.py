#import easygui as eg
from MD_Analysis import Angle_Calc

#pdb = eg.fileopenbox(msg = "WT_295K_500ns_50ps_1_run.pdb")
pdb = "WT_295K_500ns_50ps_1_run.pdb"

AC = Angle_Calc(pdb)
Angle_DF = AC.get_phi_psi()