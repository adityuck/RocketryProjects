import math
import numpy as np
import os
import pandas as pd
import re
from pyfluids import Fluid, FluidsList, Input
from rocketcea.cea_obj_w_units import CEA_Obj
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D projection

#=========================== [1] Import Data from RPA Thermals =============================================
columns = ['axial_pos', 'radius', 'conv_hf_coeff', 'q_conv', 'q_rad', 'q_total', 'tbc_temp', 
           'firewall_temp', 'coolant_wall_temp', 'coolant_temp', 'channel_pressure', 
           'coolant_velocity', 'coolant_density']

base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, 'RPA_Thermals.txt')
with open(file_path, 'r', encoding="utf8") as input_file:
    lines = input_file.readlines()

lines = lines[8:-1]
data = []

for line in lines:
    values = []
    for val in line.strip().split():
        if re.match(r'^[-+]?\d*\.?\d+([eE][-+]?\d+)?$', val):
            values.append(float(val))
    data.append(values)

df = pd.DataFrame(data, columns=columns)
del base_path, file_path, lines, line, data, val, values, input_file, columns

# Make Arrays of all the required parameters
rpa = {col: df[col].values for col in df.columns}
del df
    
#=========================== [2] Import Data from User Defined Parameters ====================================
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, 'User_Defined_Params.csv')
udp = pd.read_csv(file_path, names=['Parameter','Value','Notes'], skiprows=1)
udp_array = udp.to_numpy()
udp = {}
for i in range (np.size(udp_array,0)):
    try:
        val = float(udp_array[i, 1])
    except ValueError:
        val = udp_array[i, 1]
    udp[udp_array[i, 0]] = val
    
del base_path, file_path, i, udp_array, val

#=========================== [3] Calculate Pressure Values along Regen Channels ===============================
# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'  # This works well with LaTeX
plt.rcParams['text.latex.preamble'] = r''  # Empty or no override

#Accounting for Roughness effects
Material = udp["Material"]
match Material:
    case 'Inconel718':
        Surf_roughness = np.array([54, 38, 12, 6, 5]) * 1e-6 *10
        manufacturing_angle = np.array([30, 45, 60, 75, 90]) * math.pi/180 
        v = 0.28 # Poisson's ratio
        conductivity = 12 # thermal conductivity (W/mK)
        cte = 16e-6 # coefficient of thermal expansion (1/K)
        modulus_temps = np.array([21, 93, 204, 316, 427, 538, 649, 760, 871, 954]) + 273.15 # E Temps (K)
        modulus = np.array([208, 205, 202, 194, 186, 179, 172, 162, 127, 78]) * 1e9 # E (Pa)
        yield_temps = np.array([93, 204, 316, 427, 538, 649, 760]) + 273.15 # Ys Temps (K)
        yield_stress = np.array([1172, 1124, 1096, 1076, 1069, 1027, 758]) * 1e6 # Ys (Pa)
        
    case 'AlSi10Mg': 
        Surf_roughness = np.array([66, 56, 21, 12, 8]) * 1e-6 *10
        manufacturing_angle = np.array([30, 45, 60, 75, 90]) * math.pi/180 
        modulus_temps = np.array([25, 50, 100, 150, 200, 250, 300, 350, 400]) + 273.15 # E Temps (K)
        modulus = np.array([77.6, 75.5, 72.8, 63.2, 60, 55, 45, 37, 28]) * 1e9 # E (Pa)
        yield_temps = np.array([298, 323, 373, 423, 473, 523, 573, 623, 673]) # Ys Temps (K)
        yield_stress = np.array([204, 198, 181, 182, 158, 132, 70, 30, 12]) * 1e6 # Ys (Pa)
        conductivity = 130 # thermal conductivity (W/mK)
        cte = 27e-6 # coefficient of thermal expansion (1/K)
        v = 0.33 # Poisson's ratio
        
    case 'ABD900': 
        Surf_roughness = np.array([54, 38, 12, 6, 5]) * 1e-6 *0.7 *10#ABD900 ~30% less rough than inconel. 
        manufacturing_angle = np.array([30, 45, 60, 75, 90]) * math.pi/180
        modulus_temps = np.array([25, 700, 800, 900]) + 273.15 # E Temps (K)
        modulus = np.array([192, 157, 131, 103]) * 1e9 # E (Pa)
        yield_temps = np.array([25, 427, 538, 649, 732, 760, 788, 816, 843, 871, 927]) + 273.15 # Ys Temps (K)
        yield_stress = np.array([978, 931, 903, 903, 758, 680, 600, 503, 434, 352, 221]) * 1e6 # Ys (Pa)
        cte = np.array([11.4, 12.9, 13.7, 14.4, 15.5, 17.5, 19.2])*1e-6
        cte_temps = np.array([50, 200, 400, 600, 800, 1000, 1200]) +273.15 
        conductivity = np.array([11, 12.6, 15.7, 18.8, 23.2, 26.4, 30.1]) 
        conductivity_temps = np.array([25, 200, 400, 600, 800, 1000, 1200]) + 273.15
        v = 0.28
    case _: print("Material does not match any of the cases.")


#Calculate Channel Width
channel_width = 2 * (rpa["radius"] + udp["Firewall Thickness"]) * np.sin(np.radians(udp["Arc Angle"] / 2))


contour = rpa["radius"]
ax_pos = rpa["axial_pos"]
ax_pos = ax_pos[::-1] # Reverse axial position to match contour direction
angles = np.zeros_like(contour)
for i in range(len(contour)):
    if i == 0: angle = 0
    elif i == len(contour): angle = 0
    else: angle = math.atan((contour[i]-contour[i-1])/(ax_pos[i] - ax_pos[i-1]))
roughness = np.interp(angles, manufacturing_angle, Surf_roughness)


#Iterate over each contour segment
dP = 0 #Initialise sum for dP
dP_array = np.zeros_like(contour)
L = 0 #Initialise sum for channel length

#Shifting data to go from nozzle to injector (Direction of regen flow) 
contour = contour[::-1]
channel_width = channel_width[::-1]
ax_pos = ax_pos[::-1]
roughness = roughness[::-1]
fuel_temp = rpa['coolant_temp']
fuel_temp = fuel_temp[::-1]

for i in range(len(contour)-1):
    #Loads fuel properties
    fuel = Fluid(FluidsList.Ethanol).with_state(Input.pressure(udp['Feed Pressure']), Input.temperature(fuel_temp[i]-273.15))
    rho = fuel.density #Density of Fuel
    mu = fuel.dynamic_viscosity #Dynamic Viscosity of Fuel
    
    w = channel_width[i] 
    h = udp['Channel Height']
    mass_flow_rate = udp['Fuel Mass Flow Rate ']*(1+udp['Film Cooling'])/udp['Number of Channels'] #Total Mass Flow Rate per Channel
    u1 = udp['Fuel Mass Flow Rate ']/(rho*h*w) #Velocity of flow

    Dh = 2*h*w/(h+w) #Hydraulic Diameter
    Re = rho*u1*Dh/mu #Reynolds Number 
    fD = (-1.8*math.log10(roughness[i]/(3.7*Dh) + (6.9/Re)**1.1))**-2 #Friction Factor for frictional effects through rectangular channel
    dyn_head = 0.5*rho*u1**2 #Dynamic head/pressure

    #Discretized contour lengths
    delx = (contour[i+1]-contour[i]) 
    dely = ax_pos[i+1] -ax_pos[i]
    segment_length = math.sqrt(delx**2 + dely**2)
    L += segment_length
    dP += fD * segment_length/Dh*dyn_head #Darcy-Weisbach Equation
    dP_array[i] = dP
    if i == 0 or i ==len(contour-1):
        dP += 0.8*dyn_head #Accounts for losses at edges of channel entrance/exit
    

dP_array[-1] = dP_array[-2] #Approximation for last segment

#Shifting data back so that its from injector nozzle
contour = contour[::-1]
channel_width = channel_width[::-1]
ax_pos = ax_pos[::-1]
roughness = roughness[::-1]
fuel_temp = rpa['coolant_temp']
fuel_temp = fuel_temp[::-1]
dP_array = dP_array[::-1]
regen_press = udp['Feed Pressure'] - dP_array

del angle, angles, contour, delx, dely, Dh, dP, dyn_head, fD, h, i, L, manufacturing_angle, Re, segment_length, Surf_roughness, u1
#=========================== [4] Conduct Temperature and Pressure Calculations ==============================================
stagnation_pressure = udp["Chamber Pressure"]
radii = rpa["radius"]
throat_radius = np.min(radii)
area_ratios = (radii/throat_radius)**2
mach_numbers = np.zeros_like(area_ratios)
for i in range(len(area_ratios)):
    gamma = 1.1674
    area_ratio = area_ratios[i]
    
    if i==0:
        machfunc = lambda M: (1/area_ratio)*(1 + (gamma-1)*M**2/2)**((gamma+1)/(2*(gamma-1))) - M * (1 + (gamma - 1)/2)**((gamma+1)/(2*(gamma-1)))
        sol = sp.optimize.root_scalar(machfunc, method = 'brentq',bracket=[0,1])
        mach_numbers[i] = sol.root if sol.converged else np.nan
    
    elif radii[i] <= radii[i-1]:
        machfunc = lambda M: (1/area_ratio)*(1 + (gamma-1)*M**2/2)**((gamma+1)/(2*(gamma-1))) - M * (1 + (gamma - 1)/2)**((gamma+1)/(2*(gamma-1)))
        sol = sp.optimize.root_scalar(machfunc, method = 'brentq',bracket=[0,1])
        mach_numbers[i] = sol.root if sol.converged else np.nan
    
    elif radii[i] >= radii[i-1]:
        machfunc = lambda M: (1/area_ratio)*(1 + (gamma-1)*M**2/2)**((gamma+1)/(2*(gamma-1))) - M * (1 + (gamma - 1)/2)**((gamma+1)/(2*(gamma-1)))
        sol = sp.optimize.root_scalar(machfunc, method = 'brentq',bracket=[1.00001,20])
        mach_numbers[i] = sol.root if sol.converged else np.nan

stagnation_pressure = stagnation_pressure*np.ones_like(mach_numbers)
pressures = stagnation_pressure*(1+(gamma-1)/2 * mach_numbers**2)**((gamma-1)/gamma)
pressures = np.divide(stagnation_pressure, (1+(gamma-1)*mach_numbers**2/2)**(gamma/(gamma-1)))

del area_ratio, area_ratios, i, sol, stagnation_pressure

#=========================== [5] Conduct Stress Calculations =================================================================
longitudinal_thermal_stress = np.zeros_like(pressures)
avg_innerwall_temp = (rpa["firewall_temp"] + rpa["coolant_wall_temp"])/2
E = np.interp(avg_innerwall_temp, modulus_temps, modulus)
yield_strength = np.interp(avg_innerwall_temp, yield_temps, yield_stress)
dT = (rpa["firewall_temp"] - rpa["coolant_wall_temp"])
channel_dP = regen_press - pressures
if isinstance(modulus, np.ndarray):
    E = np.interp(avg_innerwall_temp, modulus_temps, modulus)
else:
    E = modulus
if isinstance(yield_stress, np.ndarray):
    yield_strength = np.interp(avg_innerwall_temp, yield_temps, yield_stress)
else:   
    yield_strength = yield_stress
if isinstance(cte, np.ndarray):
    coeff_thermal_expansion = np.interp(avg_innerwall_temp, cte_temps, cte)
else:
    coeff_thermal_expansion = cte

if isinstance(conductivity, np.ndarray):
    thermal_conductivity = np.interp(avg_innerwall_temp, conductivity_temps, conductivity)
else:
    thermal_conductivity = conductivity
    
    
longitudinal_thermal_stress = np.multiply(E,coeff_thermal_expansion,dT)
tangential_thermal_stress = (E*coeff_thermal_expansion*rpa["q_total"]*1000*udp["Firewall Thickness"])/(2*(1-v)*thermal_conductivity)
tangential_pressure_stress = channel_dP*(channel_width*10**-3/udp["Firewall Thickness"])**2 * 0.5
crit_longitudinal_buckling_stress = E*udp["Firewall Thickness"]/(np.sqrt(3*(1-v**2))*(radii/1000))
tot_tangential_stress = -tangential_pressure_stress + tangential_thermal_stress
von_mises_stress = np.sqrt(((tot_tangential_stress-longitudinal_thermal_stress)**2 + 6*(tot_tangential_stress**2 + longitudinal_thermal_stress**2))/2)
yield_sf = yield_strength/von_mises_stress

# 3D Polar Surface Plot: Z = ax_pos, theta = [0, 3.8 deg], r = radius, color = yield_sf

theta_min = 0
theta_max = np.deg2rad(3.8)
n_theta = 30  # Number of points along the arc

thetas = np.linspace(theta_min, theta_max, n_theta)
radii = np.array(radii)
ax_pos = np.array(ax_pos)
yield_sf = np.array(yield_sf)

# Create meshgrid for surface
Theta, Radius = np.meshgrid(thetas, radii)
Z, _ = np.meshgrid(ax_pos, thetas, indexing='ij')
Yield_SF, _ = np.meshgrid(yield_sf, thetas, indexing='ij')

# Ensure Z and Yield_SF have the same shape as X and Y
# If ax_pos and yield_sf are 1D arrays of length N, and thetas is length M,
# Z and Yield_SF should be shape (N, M), matching Radius and Theta

X = Radius * np.cos(Theta)
Y = Radius * np.sin(Theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot surface, coloring by yield_sf
norm = plt.Normalize(vmin=np.nanmin(Yield_SF), vmax=np.nanmax(Yield_SF))
facecolors = plt.cm.viridis(norm(Yield_SF))
surf = ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

# Discretize yield_sf into bins 0-12, with >12 set to 12
Yield_SF_capped = np.clip(Yield_SF, 0, 12)

# Use reversed jet colormap so red is lowest
cmap = plt.cm.get_cmap('jet_r')
norm = plt.Normalize(vmin=0, vmax=12)
facecolors = cmap(norm(Yield_SF_capped))
surf = ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array(Yield_SF_capped)

fig.colorbar(mappable, ax=ax, label='Yield Safety Factor (Capped at 12)')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Axial Position (mm)')
ax.set_title('3D Channel Stress Surface (Color: Yield SF, Jet Colormap)')

# Set equal scaling for all axes
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
mid_x = (X.max() + X.min()) * 0.5
mid_y = (Y.max() + Y.min()) * 0.5
mid_z = (Z.max() + Z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Firewall thickness for wall surfaces
firewall_thickness = udp["Firewall Thickness"]

# Inner wall surface
X_inner = Radius * np.cos(Theta)
Y_inner = Radius * np.sin(Theta)
Z_inner = Z

# Outer wall surface
Radius_outer = Radius + firewall_thickness
X_outer = Radius_outer * np.cos(Theta)
Y_outer = Radius_outer * np.sin(Theta)
Z_outer = Z  # Same axial positions




# Add label at (50, -50, 0)
label_x, label_y, label_z = 50, -50, 25
ax.text(
    label_x, label_y, label_z,
    "Minimum SF",
    color='black', fontsize=12, fontweight='bold'
)
# Store all rotated surfaces
all_X = []
all_Y = []
all_Z = []
num_channels = int(udp["Number of Channels"])
channel_angles = np.linspace(0, 2 * np.pi, num_channels, endpoint=False)
for phi in channel_angles:
    # Rotate the surface about the central axis by angle phi
    X_rot = X_inner * np.cos(phi) - Y_inner * np.sin(phi)
    Y_rot = X_inner * np.sin(phi) + Y_inner * np.cos(phi)
    Z_rot = Z_inner

    surf = ax.plot_surface(
        X_rot, Y_rot, Z_rot,
        facecolors=facecolors,
        rstride=1, cstride=1,
        linewidth=0, antialiased=False, shade=False, alpha=1.0
    )
    all_X.append(X_rot)
    all_Y.append(Y_rot)
    all_Z.append(Z_rot)
# Find the contour whose minimum SF point is closest to (50, -50, 0)
min_dist = np.inf
closest_contour_idx = None
for idx in range(len(all_X)):
    # Find minimum SF location on this contour
    min_idx = np.unravel_index(np.nanargmin(Yield_SF), Yield_SF.shape)
    x_sf = all_X[idx][min_idx]
    y_sf = all_Y[idx][min_idx]
    z_sf = all_Z[idx][min_idx]
    dist = np.sqrt((x_sf - label_x)**2 + (y_sf - label_y)**2 + (z_sf - label_z)**2)
    if dist < min_dist:
        min_dist = dist
        closest_contour_idx = idx

# Get the minimum SF location on the closest contour
min_idx = np.unravel_index(np.nanargmin(Yield_SF), Yield_SF.shape)
x_target = all_X[closest_contour_idx][min_idx]
y_target = all_Y[closest_contour_idx][min_idx]
z_target = all_Z[closest_contour_idx][min_idx]
min_sf_value = Yield_SF[min_idx]

# Draw arrow from label to minimum SF location on closest contour
ax.quiver(
    label_x, label_y, label_z,
    x_target - label_x, y_target - label_y, z_target - label_z,
    color='black', linewidth=2, arrow_length_ratio=0.1
)

# Update label to include min SF value
ax.text(
    label_x, label_y, label_z,
    f"Minimum SF: {min_sf_value:.2f}",
    color='black', fontsize=12, fontweight='bold'
)

# Set default view (azim=300, elev=20 matches your screenshot)
ax.view_init(elev=10, azim=210)

plt.show()


