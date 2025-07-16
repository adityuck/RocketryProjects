import math
import numpy as np
import os
import pandas as pd
from pyfluids import Fluid, FluidsList, Input
from rocketcea.cea_obj_w_units import CEA_Obj


#Initialise Values according to design
channel_height = 1.5e-3 #Channel Height
arc_angle = 3 #Arc Angle of Channel
n = 48 #Number of Channels
t_w = 1e-3 #Firewall Thickness
feed_press = 40e+5 #In Pa
chamber_press = 30e+5 #In Pa
ffr = 0.462 #Fuel mass flow rate (kg/s)
fcp = 0.15 #Filmcooling percentage (kg/s)
fuelp_dist = 125e-3 #Distance of fuel pipe from chamber centre
fuelp_d = 10e-3 #Diameter of fuel pipe


#Loads Contour from RPA file
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, 'Contour.txt')
contour = np.loadtxt(file_path, skiprows=14, usecols=1)
segment_heights = np.loadtxt(file_path, skiprows=14, usecols=0)
segment_heights = segment_heights*10**-3
contour = contour*10**-3 #Convert to meters

#Calculate Channel Width
channel_width = (contour+t_w+0.5*channel_height)*math.sin(math.radians(arc_angle))

#Loads fuel properties
fuel = Fluid(FluidsList.Ethanol).with_state(Input.pressure(feed_press), Input.temperature(27))
rho = fuel.density #Density of Fuel
mu = fuel.dynamic_viscosity #Dynamic Viscosity of Fuel


#Accounting for Roughness effects
v_roughness = 53e-6 #Vertical Resolution of Additive Manufacturing (for ABD-900)
roughness = np.zeros_like(contour)
for i in range(len(contour)):
    if i == 0: angle = 0
    elif i == len(contour): angle = 0
    else: angle = math.atan((contour[i]-contour[i-1])/(segment_heights[i] - segment_heights[i-1]))
    h_roughness = v_roughness*math.tan(angle)/2
    roughness[i] = v_roughness + h_roughness


#Solving for regenerative Cooling Channels First
#Print Headers
print('\n')
print("ANALYSIS ACROSS REGENERATIVE COOLING CHANNELS:\n")
header = f"{'Segment':>12}{'Reynold Number':>20}{'fD':>10}{'dP':>10}{'Fuel Velocity' :>20}{'Segment Length':>20}{'Hydraulic Diameter':>20}{'Dynamic Head':>20}"
print('=' * len(header))
print(header)
print('=' * len(header))

#Iterate over each contour segment
dP = 0 #Initialise sum for dP
L = 0 #Initialise sum for channel length
for i in range(len(contour)-1):
    w = channel_width[i] 
    h = channel_height
    mass_flow_rate = ffr*(1+fcp)/n #Total Mass Flow Rate per Channel
    u1 = mass_flow_rate/(rho*h*w) #Velocity of flow

    Dh = 2*h*w/(h+w) #Hydraulic Diameter
    Re = rho*u1*Dh/mu #Reynolds Number 
    fD = (-1.8*math.log10(roughness[i]/(3.7*Dh) + (6.9/Re)**1.1))**-2 #Friction Factor for frictional effects through rectangular channel
    dyn_head = 0.5*rho*u1**2 #Dynamic head/pressure

    #Discretized contour lengths
    delx = (contour[i+1]-contour[i]) 
    dely = segment_heights[i+1] -segment_heights[i]
    segment_length = math.sqrt(delx**2 + dely**2)
  
    L += segment_length
    dP += fD * segment_length/Dh*dyn_head #Darcy-Weisbach Equation

    if i == 0 or i ==len(contour-1):
        dP += 0.8*dyn_head #Accounts for losses at edges of channel entrance/exit
    
    #Print properties for each segment
    state = f"{i:>12}{Re:>20.2f}{fD:>10.2f}{fD * segment_length/Dh*dyn_head:>10.2f}{u1:>20.2f}{segment_length:>20.2e}{Dh:>20.2e}{dyn_head:>20.2e}"
    print(state)

print('=' * len(header))
print('\n'*1)



#Now, solving for drop across Fuel Pipe

#Getting fuel-pipe contours:
offset = fuelp_dist - contour[0] 
fuelp_contour = np.zeros_like(contour)
for i in range(len(contour)): fuelp_contour[i] = contour[i] + offset
dP_f = 0
#Print Headers
print('\n')
print("ANALYSIS ACROSS FUEL INLET PIPE:\n")
header = f"{'Segment':>12}{'Reynold Number':>20}{'fD':>10}{'dP':>10}{'Fuel Velocity' :>20}{'Segment Length':>20}{'Hydraulic Diameter':>20}{'Dynamic Head':>20}"
print('=' * len(header))
print(header)
print('=' * len(header))
#Iterate over each contour segment

for i in range(len(fuelp_contour)-1):
    mass_flow_rate = ffr*(1+fcp) #Total Mass Flow Rate
    u1 = mass_flow_rate/(rho*(math.pi*fuelp_d**2/4)) #Velocity of flow
    Re = rho*u1*fuelp_d/mu #Reynolds Number
    fD = 0.25*(math.log10(roughness[i]/(3.7*fuelp_d) + (5.74/Re**0.9)))**-2 #Friction Factor for frictional effects through cylindrical channel
    dyn_head = 0.5*rho*u1**2 #Dynamic head/pressure

    #Discretized contour lengths
    delx = (fuelp_contour[i+1]-fuelp_contour[i])
    dely = segment_heights[i+1] -segment_heights[i]
    segment_length = math.sqrt(delx**2 + dely**2)

    dP_f += fD * segment_length/fuelp_d*dyn_head #Darcy-Weisbach Equation

    if i == 0 or i ==len(contour-1):
        dP_f += 0.8*dyn_head #Accounts for losses at edges of channel entrance/exit
    state = f"{i:>12}{Re:>20.2f}{fD:>10.2f}{fD * segment_length/fuelp_d*dyn_head:>10.2f}{u1:>20.2f}{segment_length:>20.2e}{Dh:>20.2e}{dyn_head:>20.2e}"
    print(state)


print('=' * len(header))
print('\n'*1)
print(f"{'Total Pressure Drop Across Regen Channel:':<45} {dP:>10.2f} Pa = {dP * 1e-5:0.2f} Bar")
print(f"{'Total Pressure Drop Across Fuel Inlet Pipe:':<45} {dP_f:>10.2f} Pa = {dP_f * 1e-5:0.2f} Bar")
print(f"{'Total Pressure Drop:':<45} {dP_f +dP:>10.2f} Pa = {(dP_f+dP) * 1e-5:>0.2f} Bar")
print(f"{'Total Channel Length:':<45} {L * 1000:>10.2f} mm\n")
