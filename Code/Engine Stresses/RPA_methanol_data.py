from pyfluids import Fluid, FluidsList, Input

print("\n")
pressures = [1, 20, 30, 40]  # bar
temps = [250, 300, 355, 400, 450, 500]  # K
fluid_name = FluidsList.Methanol

# Define column widths for consistent formatting
col_widths = [18, 12, 15, 20, 20, 20]

# Print header with column names
headers = ["Pressure (bar)", "Temp (K)", "Cp (kJ/kg·K)", "Density (kg/m³)", "Viscosity (μPa·s)", "Thermal Cond. (mW/m·K)"]
header_line = ""
for i, header in enumerate(headers):
    header_line += f"{header:<{col_widths[i]}}"
print(header_line)
print("-" * sum(col_widths))

for p in pressures:
    # Convert bar to Pa for saturation calculations
    p_pa = p * 1e5
    
    # Create fluid instance for saturation properties
    fluid_sat = Fluid(fluid_name)
    fluid_sat.update(Input.pressure(p_pa), Input.quality(0))
    sat_temp = fluid_sat.temperature
    
    # Calculate heat of vaporization
    fluid_sat_vapor = Fluid(fluid_name)
    fluid_sat_vapor.update(Input.pressure(p_pa), Input.quality(100))
    heat_vap = (fluid_sat_vapor.enthalpy - fluid_sat.enthalpy) / 1000  # kJ/kg
    
    for T in temps:
        fluid = Fluid(fluid_name)
        fluid.update(Input.pressure(p_pa), Input.temperature(T-273.15))
        
        cp = fluid.specific_heat / 1000  # kJ/kg·K
        density = fluid.density  # kg/m³
        viscosity = fluid.dynamic_viscosity * 1e6  # μPa·s
        conductivity = fluid.conductivity * 1000  # mW/m·K
        
        data_line = f"{p:<{col_widths[0]}}{T:<{col_widths[1]}}{cp:<{col_widths[2]}.2f}{density:<{col_widths[3]}.2f}{viscosity:<{col_widths[4]}.2f}{conductivity:<{col_widths[5]}.2f}"
        print(data_line)

    print(f"Saturation temperature at {p} bar: {sat_temp+273.15:.1f} K   Heat of vaporization: {heat_vap:.1f} kJ/kg")
    print("-" * sum(col_widths))