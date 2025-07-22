import math
import numpy as np
import os
import pandas as pd
from pyfluids import Fluid, FluidsList, Input
from rocketcea.cea_obj_w_units import CEA_Obj
import matplotlib.pyplot as plt

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'  # This works well with LaTeX
plt.rcParams['text.latex.preamble'] = r''  # Empty or no override


#Initialise Values according to design
Materials = ['AlSi10Mg','Inconel718','ABD900'] #'Inconel', 'AlSi10Mg', 'ABD900'
dPdict1 = {}
dPdict2 = {}
feed_press = 40e+5 #In Pa
ffr = 0.5 #Fuel mass flow rate (kg/s)
fcp = 0.15 #Filmcooling percentage 
fuelp_dist = 70e-3 #Distance of fuel pipe from chamber centre
fuelp_d = 10e-3 #Diameter of fuel pipe


#Loads Contour from RPA file
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, 'Contour.txt')
contour = np.loadtxt(file_path, skiprows=14, usecols=1)
segment_heights = np.loadtxt(file_path, skiprows=14, usecols=0)
segment_heights = segment_heights*10**-3
contour = contour*10**-3 #Convert to meters
contour = contour[::-1] #Go from Nozzle to Injector


#Loads fuel properties
fuel = Fluid(FluidsList.Ethanol).with_state(Input.pressure(feed_press), Input.temperature(27))
rho = fuel.density #Density of Fuel
mu = fuel.dynamic_viscosity #Dynamic Viscosity of Fuel

for a in range(len(Materials)):
    Material = Materials[a]
    #Accounting for Roughness effects
    match Material:
        case 'Inconel718':
            Surf_roughness = np.array([54, 38, 12, 6, 5]) * 1e-6 *10
            manufacturing_angle = np.array([30, 45, 60, 75, 90]) * math.pi/180 
            channel_height = 1.5e-3 #Channel Height
            arc_angle = 3.8 #Arc Angle of Channel
            n = 50 #Number of Channels
            t_w = 0.4e-3 #Firewall Thickness
        case 'AlSi10Mg': 
            Surf_roughness = np.array([66, 56, 21, 12, 8]) * 1e-6 *10
            manufacturing_angle = np.array([30, 45, 60, 75, 90]) * math.pi/180 
            channel_height = 1.5e-3 #Channel Height
            arc_angle = 3.8 #Arc Angle of Channel
            n = 50 #Number of Channels
            t_w = 0.8e-3 #Firewall Thickness
        case 'ABD900': 
            Surf_roughness = np.array([54, 38, 12, 6, 5]) * 1e-6 *0.7 *10#ABD900 ~30% less rough than inconel. 
            manufacturing_angle = np.array([30, 45, 60, 75, 90]) * math.pi/180
            channel_height = 1.5e-3 #Channel Height
            arc_angle = 3.8 #Arc Angle of Channel
            n = 50 #Number of Channels
            t_w = 0.3e-3 #Firewall Thickness
        case _: print("Material does not match any of the cases.")

    #Calculate Channel Width
    channel_width = (contour+t_w+0.5*channel_height)*math.sin(math.radians(arc_angle))

    angles = np.zeros_like(contour)
    for i in range(len(contour)):
        if i == 0: angle = 0
        elif i == len(contour): angle = 0
        else: angle = math.atan((contour[i]-contour[i-1])/(segment_heights[i] - segment_heights[i-1]))
    roughness = np.interp(angles, manufacturing_angle, Surf_roughness)
    roughness = roughness*2 #Our prints seemed much rougher lmao

    #Solving for regenerative Cooling Channels First
    #Print Headers

    # print('\n')
    # print("ANALYSIS ACROSS REGENERATIVE COOLING CHANNELS:\n")
    # header = f"{'Segment':>12}{'Reynold Number':>20}{'fD':>10}{'dP':>10}{'Fuel Velocity' :>20}{'Segment Length':>20}{'Hydraulic Diameter':>20}{'Dynamic Head':>20}"
    # print('=' * len(header))
    # print(header)
    # print('=' * len(header))

    #Iterate over each contour segment
    dP = 0 #Initialise sum for dP
    dP_array = np.zeros_like(contour)
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
        dP_array[i] = dP
        if i == 0 or i ==len(contour-1):
            dP += 0.8*dyn_head #Accounts for losses at edges of channel entrance/exit
        
        #Print properties for each segment
        # state = f"{i:>12}{Re:>20.2f}{fD:>10.2f}{fD * segment_length/Dh*dyn_head:>10.2f}{u1:>20.2f}{segment_length:>20.2e}{Dh:>20.2e}{dyn_head:>20.2e}"
        # print(state)
    
    plot_array = dP_array
    plot_array = plot_array[-1:0:-1] #Arranges it from Injector to Nozzle for Plotting

    plt.plot(segment_heights[1:-1]*100, plot_array[1:]*1e-5, label=Material)
    plt.title("Pressure Drop in Regenerative Cooling Channel", fontsize=16)
    plt.xlabel("Axial Distance Along Chamber (mm)", fontsize=14)
    plt.ylabel("Pressure Drop (bar)", fontsize=14)
    plt.legend(prop={'size': 12})
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12)  # Increase tick label font size


plt.plot(segment_heights * 100, contour[::-1] * 25, linestyle="--", color="dimgray")
plt.show()