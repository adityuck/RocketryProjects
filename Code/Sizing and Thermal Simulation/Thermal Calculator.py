from rocketcea.cea_obj import CEA_Obj
import numpy as np
import math
from pyfluids import Fluid, FluidsList, Input

ox = 'N2O'
fuel = 'Ethanol'
ispObj = CEA_Obj(oxName=ox, fuelName=fuel)

Radius= 44.416 * 1e-3
Area= math.pi * Radius**2
Rt = 21 * 1e-3
At = math.pi * Rt**2
Dt = 2*Rt
tw = 0.7 * 1e-3
P0 = 30e+5

T0 = ispObj.get_Tcomb(Pc=435.114, MR=4.7)
cp, mu, _ , Pr = ispObj.get_Chamber_Transport(Pc=435.114, MR=4.7, eps=4.836, frozen=0)
c_star = ispObj.get_Cstar(Pc=435.114, MR=4.7)
M = ispObj.get_Chamber_MachNumber(Pc=435.114, MR=4.7, fac_CR=Area/At)
_, gamma = ispObj.get_Chamber_MolWt_gamma(Pc=435.114, MR=4.7, eps=4.836)
cp, mu, c_star, T0 = cp * 4186.8 , mu * 1e-4, c_star/3.28, T0 * 5/9 #Convert into SI Units

#Stuff to fix later lmao
Tc = 300
fuel = Fluid(FluidsList.Ethanol).with_state(Input.pressure(P0 + 10e+5), Input.temperature(Tc - 273.15))
sigma = 1
Re = 90000
lambda_c = fuel.conductivity
de = 0.002
lambda_w = 140
alpha_T = 0.026 * Dt**-0.2 * mu **0.2 * cp * Pr **0.6 *  (P0/c_star)**0.8 * (Dt/Radius)**0.1 * (At/Area) **0.9 * sigma


Te = T0 * ( 1 + (gamma-1)*M**2*Pr**0.33/2)  /  ( 1 + (gamma-1)*M**2/2)
Twg_0 = Te - 50
qg = alpha_T * (Te - Twg_0)
Nu = 0.023* Re**0.8 * Pr ** 0.4
alpha_c = Nu * lambda_c / de
Twc = qg / alpha_c + Tc
Twg_1 = qg * tw /lambda_w  + Twc

while abs(Twg_1 - Twg_0)/Twg_0 > 0.05:
    Twg_0 = Twg_1
    qg = alpha_T * (Te - Twg_0)
    Nu = 0.023* Re**0.8 * Pr ** 0.4
    alpha_c = Nu * lambda_c / de
    Twc = qg / alpha_c + Tc
    Twg_1 = qg * tw /lambda_w  + Twc


print(Twg_1)