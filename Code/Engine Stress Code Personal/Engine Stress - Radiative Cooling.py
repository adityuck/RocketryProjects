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
    
del base_path, file_path, i, val

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

dP = pressures - 1.013e5
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
    

longitudinal_thermal_stress = E*coeff_thermal_expansion*dT
tangential_thermal_stress = (E*coeff_thermal_expansion*rpa["q_total"]*1000*udp["Firewall Thickness"])/(2*(1-v)*thermal_conductivity)
tangential_pressure_stress = dP*2*radii/(2*udp["Firewall Thickness"])
crit_longitudinal_buckling_stress = E*udp["Firewall Thickness"]/(np.sqrt(3*(1-v**2))*(radii/1000))
tot_tangential_stress = tangential_pressure_stress + tangential_thermal_stress
von_mises_stress = np.sqrt(tot_tangential_stress**2 + longitudinal_thermal_stress**2  - tot_tangential_stress*longitudinal_thermal_stress)
yield_sf = yield_strength/von_mises_stress
#=========================== [6] Plotting =================================================================
# ============================= [Helpers for Plotting] ===================================

def setup_3d_subplot(fig, position, title, xlabel='X (mm)', ylabel='Y (mm)', zlabel='Axial Position (mm)'):
    ax = fig.add_subplot(position, projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    return ax

def set_equal_axes(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def rotate_and_plot_surfaces(ax, X, Y, Z, facecolors, num_channels):
    all_X, all_Y, all_Z = [], [], []
    channel_angles = np.linspace(0, 2 * np.pi, num_channels, endpoint=False)
    for phi in channel_angles:
        X_rot = X * np.cos(phi) - Y * np.sin(phi)
        Y_rot = X * np.sin(phi) + Y * np.cos(phi)
        surf = ax.plot_surface(X_rot, Y_rot, Z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=1.0)
        all_X.append(X_rot)
        all_Y.append(Y_rot)
        all_Z.append(Z)
    return all_X, all_Y, all_Z

def annotate_extreme_point(ax, all_X, all_Y, all_Z, target_array, label_coords, text_prefix="Value", is_max=True):
    idx_func = np.nanargmax if is_max else np.nanargmin
    label_x, label_y, label_z = label_coords
    min_dist = np.inf
    closest_contour_idx = None
    extreme_idx = np.unravel_index(idx_func(target_array), target_array.shape)

    for idx in range(len(all_X)):
        x_val = all_X[idx][extreme_idx]
        y_val = all_Y[idx][extreme_idx]
        z_val = all_Z[idx][extreme_idx]
        dist = np.sqrt((x_val - label_x)**2 + (y_val - label_y)**2 + (z_val - label_z)**2)
        if dist < min_dist:
            min_dist = dist
            closest_contour_idx = idx

    x_target = all_X[closest_contour_idx][extreme_idx]
    y_target = all_Y[closest_contour_idx][extreme_idx]
    z_target = all_Z[closest_contour_idx][extreme_idx]
    value = target_array[extreme_idx]

    ax.quiver(label_x, label_y, label_z, x_target - label_x, y_target - label_y, z_target - label_z, color='black', linewidth=2, arrow_length_ratio=0.1)
    ax.text(label_x, label_y, label_z, f"{text_prefix}: {value:.2f}", color='black', fontsize=12, fontweight='bold')

# ============================= [Common Mesh Setup] ======================================

theta_min = 0
theta_max = np.deg2rad(3.8)
n_theta = 30
thetas = np.linspace(theta_min, theta_max, n_theta)
ax_pos = np.array(rpa["axial_pos"])
radii = np.array(radii)
Theta, Radius = np.meshgrid(thetas, radii)
Z, _ = np.meshgrid(ax_pos, thetas, indexing='ij')
X = Radius * np.cos(Theta)
Y = Radius * np.sin(Theta)
label_coords = (50, -50, 25)

# ============================= [Prepare Data Arrays] ====================================
Yield_SF, _ = np.meshgrid(yield_sf, thetas, indexing='ij')
Yield_SF_capped = np.clip(Yield_SF, 0, 12)
cmap_sf = plt.cm.get_cmap('jet_r')
norm_sf = plt.Normalize(vmin=0, vmax=12)
facecolors_sf = cmap_sf(norm_sf(Yield_SF_capped))

firewall_temp = np.array(rpa["firewall_temp"])
Firewall_Temp, _ = np.meshgrid(firewall_temp, thetas, indexing='ij')
norm_temp = plt.Normalize(vmin=np.nanmin(Firewall_Temp), vmax=np.nanmax(Firewall_Temp))
cmap_temp = plt.cm.get_cmap('hot_r')
facecolors_temp = cmap_temp(norm_temp(Firewall_Temp))

# ============================= [Plot Combined Figure] ====================================
fig = plt.figure(figsize=(16, 8))

# Plot 1: Yield Safety Factor
ax1 = setup_3d_subplot(fig, 121, "Yield Safety Factor Contours")
mappable_sf = plt.cm.ScalarMappable(cmap=cmap_sf, norm=norm_sf)
mappable_sf.set_array(Yield_SF_capped)
fig.colorbar(mappable_sf, ax=ax1, shrink=0.6, label='Yield Safety Factor (Capped at 12)')
set_equal_axes(ax1, X, Y, Z)
all_X_sf, all_Y_sf, all_Z_sf = rotate_and_plot_surfaces(ax1, X, Y, Z, facecolors_sf, int(udp["Number of Channels"]))
annotate_extreme_point(ax1, all_X_sf, all_Y_sf, all_Z_sf, Yield_SF, label_coords, "Minimum SF", is_max=False)
ax1.view_init(elev=10, azim=210)

# Plot 2: Firewall Temperature
ax2 = setup_3d_subplot(fig, 122, "Firewall Temperature Contours")
mappable_temp = plt.cm.ScalarMappable(cmap=cmap_temp, norm=norm_temp)
mappable_temp.set_array(Firewall_Temp)
fig.colorbar(mappable_temp, ax=ax2, shrink=0.6, label='Firewall Temperature (K)')
set_equal_axes(ax2, X, Y, Z)
all_X_temp, all_Y_temp, all_Z_temp = rotate_and_plot_surfaces(ax2, X, Y, Z, facecolors_temp, int(udp["Number of Channels"]))
annotate_extreme_point(ax2, all_X_temp, all_Y_temp, all_Z_temp, Firewall_Temp, label_coords, "Max Temp. (K)", is_max=True)
ax2.view_init(elev=10, azim=210)

plt.tight_layout()
plt.show()

# ============================= [Other Plots] =============================================
plt.plot(ax_pos, tangential_pressure_stress, label='Tangential Pressure Stress')
plt.plot(ax_pos, tangential_thermal_stress, label = 'Tangential Thermal Stress')
plt.plot(ax_pos, radii*0.25e7, label = 'Contour')
plt.legend()
plt.show()