import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Inconel 718
inconel_modulus_temps = np.array([21, 93, 204, 316, 427, 538, 649, 760, 871, 954]) # E temps (deg C)
inconel_modulus = np.array([208, 205, 202, 194, 186, 179, 172, 162, 127, 78]) * 1e9 # E values (Pa)
inconel_yield_temps = np.array([93, 204, 316, 427, 538, 649, 760]) # Yield stress temps (deg C)
inconel_yield_stress = np.array([1172, 1124, 1096, 1076, 1069, 1027, 758]) * 1e6 # Yield stress values (Pa)

# Aluminum AlSi10Mg
aluminum_modulus_temps = np.array([25, 50, 100, 150, 200, 250, 300, 350, 400]) # E temps (deg C)
aluminum_modulus = np.array([77.6, 75.5, 72.8, 63.2, 60, 55, 45, 37, 28]) * 1e9 # E values (Pa)
aluminum_yield_temps = np.array([298, 323, 373, 423, 473, 523, 573, 623, 673]) - 273.15 # Yield stress temps (deg C)
aluminum_yield_stress = np.array([204, 198, 181, 182, 158, 132, 70, 30, 12]) * 1e6 # Yield stress values (Pa)

# Aluminum 6082-T6 (from Eurocode 9)
ys_0_6082 = 260e6  # Yield stress at room temp (for under 6mm thickness)
aluminum_6082_modulus_temps = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 550])  # E Temps (deg C)
aluminum_6082_modulus = np.array([70, 69.3, 67.9, 65.1, 60.2, 54.6, 47.6, 37.8, 28.0, 0]) * 1e9  # E values (Pa)
aluminum_6082_yield_temps = np.array([20, 100, 150, 200, 250, 300, 350, 550])  # Yield stress temps (deg C)
aluminum_6082_yield_stress = np.array([1, 0.90, 0.79, 0.65, 0.38, 0.20, 0.11, 0]) * ys_0_6082  # Yield stress values (Pa)

# GRCop-42
grcop_modulus_temps = np.array([25, 50])  # E Temps (deg C)
grcop_modulus = np.array([78.9, 78.9]) * 1e9  # E values (Pa) - assumed constant
grcop_yield_temps = np.array([300, 400, 500, 600, 700, 800, 900, 1000]) - 273.15  # Ys Temps (deg C)
grcop_yield_stress = np.array([175, 170, 160, 150, 135, 120, 95, 70]) * 1e6  # Ys values (Pa)

# ABD900
abd900_modulus_temps = np.array([21, 93, 204, 316, 427, 538, 649, 760, 871, 954])  # E temps (deg C)
abd900_modulus = np.array([208, 205, 202, 194, 186, 179, 172, 162, 127, 78]) * 1e9  # E values (Pa) - same as Inconel
abd900_yield_temps = np.array([29, 225, 440, 599, 755, 843, 873, 917])  # Ys temps (deg C)
abd900_yield_stress = np.array([1090, 1028, 976, 937, 897, 883, 836, 711]) * 1e6  # Ys values (Pa)

# Create interpolation functions to calculate yield stress to elastic modulus ratio at common temperature points
# For Inconel 718
inconel_modulus_func = interp1d(inconel_modulus_temps, inconel_modulus, bounds_error=False, fill_value="extrapolate")
inconel_yield_func = interp1d(inconel_yield_temps, inconel_yield_stress, bounds_error=False, fill_value="extrapolate")
inconel_temps_common = np.linspace(100, 760, 50)  # Common temperature range where both data exist
inconel_ys_e = inconel_yield_func(inconel_temps_common) / inconel_modulus_func(inconel_temps_common)

# For AlSi10Mg
aluminum_modulus_func = interp1d(aluminum_modulus_temps, aluminum_modulus, bounds_error=False, fill_value="extrapolate")
aluminum_yield_func = interp1d(aluminum_yield_temps, aluminum_yield_stress, bounds_error=False, fill_value="extrapolate")
aluminum_temps_common = np.linspace(25, 400, 50)  # Common temperature range where both data exist
aluminum_ys_e = aluminum_yield_func(aluminum_temps_common) / aluminum_modulus_func(aluminum_temps_common)

# For 6082-T6
aluminum_6082_modulus_func = interp1d(aluminum_6082_modulus_temps, aluminum_6082_modulus, bounds_error=False, fill_value="extrapolate")
aluminum_6082_yield_func = interp1d(aluminum_6082_yield_temps, aluminum_6082_yield_stress, bounds_error=False, fill_value="extrapolate")
aluminum_6082_temps_common = np.linspace(20, 400, 50)  # Common temperature range where both data exist
aluminum_6082_ys_e = aluminum_6082_yield_func(aluminum_6082_temps_common) / aluminum_6082_modulus_func(aluminum_6082_temps_common)

# For GRCop-42
grcop_modulus_func = interp1d(grcop_modulus_temps, grcop_modulus, bounds_error=False, fill_value="extrapolate")
grcop_yield_func = interp1d(grcop_yield_temps, grcop_yield_stress, bounds_error=False, fill_value="extrapolate")
grcop_temps_common = np.linspace(25, 725, 50)  # Common temperature range where both data exist
grcop_ys_e = grcop_yield_func(grcop_temps_common) / grcop_modulus_func(grcop_temps_common)

# For ABD900
abd900_modulus_func = interp1d(abd900_modulus_temps, abd900_modulus, bounds_error=False, fill_value="extrapolate")
abd900_yield_func = interp1d(abd900_yield_temps, abd900_yield_stress, bounds_error=False, fill_value="extrapolate")
abd900_temps_common = np.linspace(100, 900, 50)  # Common temperature range where both data exist
abd900_ys_e = abd900_yield_func(abd900_temps_common) / abd900_modulus_func(abd900_temps_common)

# Create figure 1: Yield Strength
plt.figure(figsize=(10, 6))
plt.plot(inconel_yield_temps, inconel_yield_stress/1e6, 'r', label='Inconel 718')
plt.plot(aluminum_yield_temps, aluminum_yield_stress/1e6, 'b', label='AlSi10Mg')
plt.plot(aluminum_6082_yield_temps, aluminum_6082_yield_stress/1e6, 'g', label='Al 6082-T6')
plt.plot(grcop_yield_temps, grcop_yield_stress/1e6, 'c', label='GRCop-42')
plt.plot(abd900_yield_temps, abd900_yield_stress/1e6, 'm', label='ABD900')
plt.xlabel('Temperature (deg C)')
plt.ylabel('Yield Strength (MPa)')
plt.grid()
plt.legend(loc='upper right')
plt.title('Yield Strength vs Temperature')
plt.ylim(0, 1300)
plt.xlim(0, None)

# Create figure 2: Elastic Modulus
plt.figure(figsize=(10, 6))
plt.plot(inconel_modulus_temps, inconel_modulus/1e9, 'r', label='Inconel 718')
plt.plot(aluminum_modulus_temps, aluminum_modulus/1e9, 'b', label='AlSi10Mg')
plt.plot(aluminum_6082_modulus_temps, aluminum_6082_modulus/1e9, 'g', label='Al 6082-T6')
plt.plot(grcop_modulus_temps, grcop_modulus/1e9, 'c', label='GRCop-42')
plt.plot(abd900_modulus_temps, abd900_modulus/1e9, 'm', label='ABD900')
plt.xlabel('Temperature (deg C)')
plt.ylabel('Elastic Modulus (GPa)')
plt.grid()
plt.legend(loc='upper right')
plt.title('Elastic Modulus vs Temperature')
plt.ylim(0, 220)
plt.xlim(0, None)

# Create figure 3: Yield Strength / Elastic Modulus Ratio
plt.figure(figsize=(10, 6))
plt.plot(inconel_temps_common, inconel_ys_e * 1000, 'r', label='Inconel 718')
plt.plot(aluminum_temps_common, aluminum_ys_e * 1000, 'b', label='AlSi10Mg')
plt.plot(aluminum_6082_temps_common, aluminum_6082_ys_e * 1000, 'g', label='Al 6082-T6')
plt.plot(grcop_temps_common, grcop_ys_e * 1000, 'c', label='GRCop-42')
plt.plot(abd900_temps_common, abd900_ys_e * 1000, 'm', label='ABD900')
plt.xlabel('Temperature (deg C)')
plt.ylabel('Yield Strength / Elastic Modulus (‰)')  # Using permille for better readability
plt.grid()
plt.legend(loc='upper right')
plt.title('Yield Strength to Elastic Modulus Ratio vs Temperature')
plt.ylim(0, None)
plt.xlim(0, None)

plt.show()