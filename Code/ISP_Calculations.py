from rocketcea.cea_obj import CEA_Obj
import numpy as np
ispObj = CEA_Obj( oxName='N2O', fuelName='ETHANOL')
s = ispObj.get_full_cea_output( Pc=30, MR=np.arange(4.6, 4.7, 0.02), eps=4.59, short_output=1, pc_units='bar')
#Optimal ISP at O/F = 4.6
print( s )
