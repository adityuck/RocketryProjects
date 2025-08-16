import numpy as np
import math
from rocketcea.cea_obj import CEA_Obj
import Chamber_Geometry
import matplotlib.pyplot as plt
from pyfluids import Fluid, FluidsList, Input

# =========================[USER INPUTS]=========================
ox = 'N2O'
fuel = 'Ethanol'
Material = 'AlSi10Mg' #'Inconel', 'AlSi10Mg', 'ABD900'

Chamber_Pressure = 30.0   # bar
Thrust_req       = 6000.0 # N (sea-level sizing target)
OF               = 4.7
CR               = 4.5    # contraction ratio Ac/At
Lstar_target_mm  = 600.0  # target L* in mm (chamber only)
tw               = 0.7 * 1e-3
channel_height   = 1.5e-3 # Channel Height
arc_angle        = 3.8    # Arc Angle of Channel
n                = 50     # Number of Channels
fcp              = 0.2    # Film Cooling Percentage

# Efficiency knobs (tune with hot-fire data / conservatism)
eta_cstar = 0.97   # c* efficiency (0.96–0.99)
eta_Cf    = 0.98   # Cf efficiency (0.96–0.995)

# =========================[CEA / IDEAL PERFORMANCE]=========================
ispObj = CEA_Obj(oxName=ox, fuelName=fuel)

Pc_psia = Chamber_Pressure * 14.5038
Pc_SI   = Chamber_Pressure * 1e5         # Pa
g0      = 9.80665

# Expansion ratio for Pe ~ 1 atm (Pc/Pe in bar units)
Area_Ratio = ispObj.get_eps_at_PcOvPe(Pc=Pc_psia, MR=OF, PcOvPe=Chamber_Pressure/1.013,
                                      frozen=0, frozenAtThroat=0)

# Ideal Isp and c*
ISP_vac_ideal = ispObj.get_Isp(Pc=Pc_psia, MR=OF, eps=Area_Ratio, frozen=0, frozenAtThroat=0)     # s
cstar_ideal   = ispObj.get_Cstar(Pc=Pc_psia, MR=OF) / 3.28                                        # m/s

# Ideal Cf at sea level (note: returns (Pamb_list, Cf_list))
Cf_list_SL = ispObj.get_PambCf(Pamb=14.7, Pc=Pc_psia, MR=OF, eps=Area_Ratio)
Cf_SL_ideal = float(Cf_list_SL[0])


# Ideal sea-level Isp from Cf*c*
ISP_SL_ideal_from_Cf = Cf_SL_ideal * cstar_ideal / g0

# =========================[APPLY EFFICIENCIES]=========================
cstar_act = eta_cstar * cstar_ideal
Cf_SL_act = eta_Cf    * Cf_SL_ideal

# --- Size throat area from thrust target at sea level, using actual Cf ---
At_SI = Thrust_req / (Cf_SL_act * Pc_SI)   # m²
Rt_SI = math.sqrt(At_SI / math.pi)         # m

# Mass flow and Isp (actual, sea-level)
mdot_act = Pc_SI * At_SI / cstar_act                 # kg/s
ISP_SL_act = Cf_SL_act * cstar_act / g0              # s
F_check_SL = Cf_SL_act * Pc_SI * At_SI               # N (should match Thrust_req)


# =========================[GEOMETRY IN mm]=========================
Rt_mm = Rt_SI * 1e3
Re_mm = math.sqrt(Area_Ratio) * Rt_mm
Rc_mm = math.sqrt(CR) * Rt_mm

# Bell nozzle (Rao)
xn, yn, vn_mm3, Rn_mm, theta_n, theta_e = Chamber_Geometry.rao_nozzle(Rt_mm, Area_Ratio)

# Convergent (arc–cone–arc with your internal defaults)
xc, yc, vc_mm3, R1_mm, R2_mm, b_rad = Chamber_Geometry.convergent_sizing(CR, Rt_mm)

# Cylinder length to hit L* target (uses convergent + divergent volumes as "other")
# NOTE: L* includes chamber (cylinder + convergent). You can pass vn_mm3 or not.
xcyl, ycyl, Lcyl_mm = Chamber_Geometry.cylinder_sizing(Lstar_target_mm, vc_mm3, Rt_mm, CR)

# Stitch sections axially
xc = xc + np.max(xcyl)
xn = xn + np.max(xc)

x_full = np.concatenate((xcyl, xc, xn))
y_full = np.concatenate((ycyl, yc, yn))

# =========================[DERIVED LENGTHS / VOLUMES / L*]=========================
L_cyl_mm  = np.max(xcyl) - np.min(xcyl)
L_conv_mm = np.max(xc)   - np.min(xc)
L_noz_mm  = np.max(xn)   - np.min(xn)
L_chamber_mm = L_cyl_mm + L_conv_mm
L_total_mm   = L_chamber_mm + L_noz_mm

At_mm2 = math.pi * Rt_mm**2
Ac_mm2 = CR * At_mm2

V_cyl_mm3 = Ac_mm2 * L_cyl_mm

# L* (chamber only) achieved
Lstar_actual_mm = (vc_mm3 + V_cyl_mm3) / At_mm2
Lstar_actual_m  = Lstar_actual_mm / 1000.0

# R2/R2max for reference
R2max_mm = (Rc_mm - Rt_mm) / (1 - math.cos(b_rad)) - R1_mm
R2_frac  = R2_mm / R2max_mm if R2max_mm > 0 else float('nan')

# =========================[CONSISTENCY CHECKS]=========================
# Ideal pathway (from Isp_vac_ideal with simple sea-level correction)
ISP_SL_ideal_simple = ISP_vac_ideal - (1.013/Chamber_Pressure) * Area_Ratio * cstar_ideal / g0

# Mass flow from ideal c* (for comparison only)
mdot_ideal = Pc_SI * At_SI / cstar_ideal

# Thrust via ideal & actual Cf (sea level)
F_SL_ideal = Cf_SL_ideal * Pc_SI * At_SI
F_SL_actual = Cf_SL_act * Pc_SI * At_SI

# =========================[FORMATTED OUTPUT]=========================
# print("\n===== Engine & Geometry Parameters =====")
# print(f"{'Ox / Fuel':<28}: {ox} / {fuel}")
# print(f"{'Chamber Pressure':<28}: {Chamber_Pressure:9.3f} bar  ({Pc_psia:9.3f} psia)")
# print(f"{'O/F Ratio':<28}: {OF:9.3f}")
# print(f"{'Contraction Ratio (CR)':<28}: {CR:9.3f}")
# print(f"{'Expansion Ratio (Ae/At)':<28}: {Area_Ratio:9.3f}")

# print("\n===== Performance (Ideal vs Actual) =====")
# print(f"{'c* (ideal)':<28}: {cstar_ideal:9.3f} m/s")
# print(f"{'c* (actual) = ηc*·c*':<28}: {cstar_act:9.3f} m/s   (ηc*={eta_cstar:.3f})")
# print(f"{'Cf SL (ideal)':<28}: {Cf_SL_ideal:9.4f}")
# print(f"{'Cf SL (actual)=ηCf·Cf':<28}: {Cf_SL_act:9.4f}       (ηCf={eta_Cf:.3f})")
# print(f"{'Isp SL (ideal from Cf)':<28}: {ISP_SL_ideal_from_Cf:9.3f} s")
# print(f"{'Isp SL (ideal simple)':<28}: {ISP_SL_ideal_simple:9.3f} s")
# print(f"{'Isp SL (actual)':<28}: {ISP_SL_act:9.3f} s")
# print(f"{'Mass Flow (actual)':<28}: {mdot_act:9.4f} kg/s")

# print("\n===== Throat / Key Radii (mm) =====")
# print(f"{'Rt (throat radius)':<28}: {Rt_mm:9.3f} mm")
# print(f"{'Re (exit radius)':<28}: {Re_mm:9.3f} mm")
# print(f"{'Rc (chamber radius)':<28}: {Rc_mm:9.3f} mm")

# print("\n===== Convergent Details =====")
# print(f"{'R1 (throat fillet)':<28}: {R1_mm:9.3f} mm")
# print(f"{'R2 (chamber fillet)':<28}: {R2_mm:9.3f} mm")
# print(f"{'R2/R2max':<28}: {R2_frac:9.3f}")
# print(f"{'Convergent angle b':<28}: {math.degrees(b_rad):9.3f} deg")

# print("\n===== Lengths (mm) =====")
# print(f"{'L_cyl (cyl length)':<28}: {L_cyl_mm:9.3f} mm")
# print(f"{'L_conv (convergent)':<28}: {L_conv_mm:9.3f} mm")
# print(f"{'L_noz (bell nozzle)':<28}: {L_noz_mm:9.3f} mm")
# print(f"{'L_chamber = L_cyl+L_conv':<28}: {L_chamber_mm:9.3f} mm")
# print(f"{'L_total (chamber+noz)':<28}: {L_total_mm:9.3f} mm")

# print("\n===== L* (Chamber Only) =====")
# print(f"{'L* target':<28}: {Lstar_target_mm:9.3f} mm  ({Lstar_target_mm/1000:9.4f} m)")
# print(f"{'L* actual':<28}: {Lstar_actual_mm:9.3f} mm  ({Lstar_actual_m:9.4f} m)")

# print("\n===== Consistency Checks =====")
# print(f"{'At (sized from F, actual Cf)':<28}: {At_SI:9.6e} m^2")
# print(f"{'mdot (actual) = Pc*At/c*act':<28}: {mdot_act:9.6f} kg/s")
# print(f"{'mdot (ideal)  = Pc*At/c*ideal':<28}: {mdot_ideal:9.6f} kg/s   (Δ {mdot_act - mdot_ideal:+.6f})")
# print(f"{'F SL (ideal) = Cf_ideal*Pc*At':<28}: {F_SL_ideal:9.3f} N")
# print(f"{'F SL (actual)= Cf_act *Pc*At':<28}: {F_SL_actual:9.3f} N   (Δ {F_SL_actual - Thrust_req:+.3f})")


# # =========================[PLOT]=========================
# plt.figure()
# plt.plot(x_full, y_full, linewidth=2)
# plt.xlabel("Axial Distance [mm]")
# plt.ylabel("Radius [mm]")
# plt.title("Chamber + Convergent + Bell Nozzle Contour")
# plt.ylim(0,200)
# plt.grid(True)
# plt.axis('equal')
# plt.show()

# =========================[MATERIAL PROPERTIES]=========================
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
        conductivity = 140 # thermal conductivity (W/mK)
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





# =========================[REGEN CALCULATIONS]=========================
fuel_mdot = mdot_act * 1 * (1 +fcp)/(OF + 1)
contour = y_full
channel_width = (contour+tw+0.5*channel_height)*math.sin(math.radians(arc_angle)) * 1e-3
Re = np.zeros_like((contour))
w = channel_width 
h = channel_height
Dh = 2*h*w/(h+w) #Hydraulic Diameter

# =========================[THERMAL CALCULATIONS]=========================
qg_array  = np.zeros((len(contour) -1 ,1))
twg_array = np.zeros_like(qg_array)
twc_array = np.zeros_like(qg_array)
tc_array  = np.zeros((len(contour) ,1))
tc_array[0] = 300
rev_contour = {'ax_pos' : x_full[::-1], 'radii' : y_full[::-1]}
rev_Dh = Dh[::-1]
rev_w = w[::-1]
throat_idx = np.argmin(rev_contour['radii'])

#Nozzle Exit Calculations
T0 = ispObj.get_Tcomb(Pc=Pc_psia, MR=OF)
cp, mu, _ , Pr = ispObj.get_Exit_Transport(Pc=Pc_psia, MR=OF, frozen=0)
c_star = ispObj.get_Cstar(Pc=Pc_psia, MR=OF)

_, gamma = ispObj.get_exit_MolWt_gamma(Pc=Pc_psia, MR=OF, eps=Area_Ratio)
cp, mu, c_star, T0 = cp * 4186.8 , mu * 1e-4, c_star/3.28, T0 * 5/9 #Convert into SI Units

    
for i in range(throat_idx):
    Radius= rev_contour['radii'][i] *1e-3
    delx = abs(rev_contour['ax_pos'][i+1] - rev_contour['ax_pos'][i])*1e-3
    Area= math.pi * Radius**2
    Rt = Rt_mm * 1e-3
    Dt = 2*Rt 
    
    Tc = tc_array[i]
    fuel = Fluid(FluidsList.Ethanol).with_state(Input.pressure(Pc_SI + 10e+5), Input.temperature(Tc - 273.15)) # Fix Later to Use Pressure from Pressure drops
    sigma = 1 # Fix Later too
    rho = fuel.density #Density of Fuel
    mu_f = fuel.dynamic_viscosity #Dynamic Viscosity of Fuel
    u1 = fuel_mdot/(n*rho*h*rev_w[i]) #Velocity of flow
    dh = rev_Dh[i]
    Re = rho*u1*dh/mu_f #Reynolds Number 
    lambda_c = fuel.conductivity
    
    alpha_T = 0.026 * Dt**-0.2 * mu **0.2 * cp * Pr **0.6 *  (Pc_SI/c_star)**0.8 * (Dt/Radius)**0.1 * (At_SI/Area) **0.9 * sigma
    M = M = ispObj.get_MachNumber(Pc=Pc_psia, MR=OF, eps=Area/At_SI)
    Te = T0 * ( 1 + (gamma-1)*M**2*Pr**0.33/2)  /  ( 1 + (gamma-1)*M**2/2)
    Twg_0 = Te - 50
    qg = alpha_T * (Te - Twg_0)
    cp_c = fuel.specific_heat
    Pr_c = cp_c * mu_f / lambda_c
    Nu   = 0.023 * (Re**0.8) * (Pr_c**0.4)     
    alpha_c = Nu * lambda_c / dh
    
    Twc = qg / alpha_c + Tc
    if isinstance(conductivity, np.ndarray):
        lambda_w = np.interp((Twg_0+Twc)/2, conductivity_temps, conductivity)
    else:
        lambda_w = conductivity
    
    Twg_1 = qg * tw /lambda_w  + Twc

    while abs(Twg_1 - Twg_0)/Twg_0 > 0.05:
        Twg_0 = Twg_1
        qg = alpha_T * (Te - Twg_0)
        Nu = 0.023* Re**0.8 * Pr ** 0.4
        alpha_c = Nu * lambda_c / dh
        Twc = qg / alpha_c + Tc
        if isinstance(conductivity, np.ndarray):
            lambda_w = np.interp((Twg_0+Twc)/2, conductivity_temps, conductivity)
        else:
            lambda_w = conductivity
        Twg_1 = qg * tw /lambda_w  + Twc

    twg_array[i]  = Twg_1
    twc_array[i]  = Twc
    qg_array[i]   = qg
    pwet = 2*(h+rev_w[i])
    tc_array[i+1] = tc_array[i] + qg * pwet * delx / (fuel_mdot/n * fuel.specific_heat)
   

#Chamber Calculations
T0 = ispObj.get_Tcomb(Pc=Pc_psia, MR=OF)
cp, mu, _ , Pr = ispObj.get_Chamber_Transport(Pc=Pc_psia, MR=OF, frozen=0)
c_star = ispObj.get_Cstar(Pc=Pc_psia, MR=OF)

_, gamma = ispObj.get_Chamber_MolWt_gamma(Pc=Pc_psia, MR=OF, eps=Area_Ratio)
cp, mu, c_star, T0 = cp * 4186.8 , mu * 1e-4, c_star/3.28, T0 * 5/9 #Convert into SI Units

    
for i in range(throat_idx, len(rev_contour['radii']) - 1):
    Radius= rev_contour['radii'][i] *1e-3
    delx = abs(rev_contour['ax_pos'][i+1] - rev_contour['ax_pos'][i])*1e-3
    Area= math.pi * Radius**2
    Rt = Rt_mm * 1e-3
    Dt = 2*Rt 
    
    Tc = tc_array[i]
    fuel = Fluid(FluidsList.Ethanol).with_state(Input.pressure(Pc_SI + 10e+5), Input.temperature(Tc - 273.15)) # Fix Later to Use Pressure from Pressure drops
    sigma = 1 # Fix Later too
    rho = fuel.density #Density of Fuel
    mu_f = fuel.dynamic_viscosity #Dynamic Viscosity of Fuel
    u1 = fuel_mdot/(n*rho*h*rev_w[i]) #Velocity of flow
    dh = rev_Dh[i]
    Re = rho*u1*dh/mu_f #Reynolds Number 
    lambda_c = fuel.conductivity
    
    alpha_T = 0.026 * Dt**-0.2 * mu **0.2 * cp * Pr **0.6 *  (Pc_SI/c_star)**0.8 * (Dt/Radius)**0.1 * (At_SI/Area) **0.9 * sigma
    M = ispObj.get_Chamber_MachNumber(Pc=Pc_psia, MR=OF, fac_CR=Area/At_SI)
    Te = T0 * ( 1 + (gamma-1)*M**2*Pr**0.33/2)  /  ( 1 + (gamma-1)*M**2/2)
    Twg_0 = Te - 50
    qg = alpha_T * (Te - Twg_0)
    cp_c = fuel.specific_heat
    Pr_c = cp_c * mu_f / lambda_c
    Nu   = 0.023 * (Re**0.8) * (Pr_c**0.4)       # ✅
    alpha_c = Nu * lambda_c / dh
    alpha_c = Nu * lambda_c / dh
    Twc = qg / alpha_c + Tc
    if isinstance(conductivity, np.ndarray):
        lambda_w = np.interp((Twg_0+Twc)/2, conductivity_temps, conductivity)
    else:
        lambda_w = conductivity
    
    Twg_1 = qg * tw /lambda_w  + Twc

    while abs(Twg_1 - Twg_0)/Twg_0 > 0.05:
        Twg_0 = Twg_1
        qg = alpha_T * (Te - Twg_0)
        Nu = 0.023* Re**0.8 * Pr ** 0.4
        alpha_c = Nu * lambda_c / dh
        Twc = qg / alpha_c + Tc
        if isinstance(conductivity, np.ndarray):
            lambda_w = np.interp((Twg_0+Twc)/2, conductivity_temps, conductivity)
        else:
            lambda_w = conductivity
        Twg_1 = qg * tw /lambda_w  + Twc

    twg_array[i] = Twg_1
    twc_array[i] = Twc
    qg_array[i]  = qg
    pwet = 2*(h+rev_w[i])
    tc_array[i+1] = tc_array[i] + qg * pwet * delx / (fuel_mdot/n * fuel.specific_heat)
    

twg_array, twc_array, qg_array = twg_array[::-1], twc_array[::-1], qg_array[::-1]