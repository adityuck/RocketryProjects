def rao_nozzle(Rt, epsilon):
    """
    Generate a Rao bell nozzle contour and compute the divergent section volume.

    The profile is generated as:
    1. A short circular arc downstream of the throat (0.382*Rt radius).
    2. A parabolic bell section fitted between the arc end and the nozzle exit.

    Parameters:
    ----------
    Rt : float
        Throat radius [m]
    epsilon : float
        Nozzle expansion ratio (Ae / At)

    Returns:
    -------
    x_array : ndarray
        Axial coordinates of nozzle contour [m]
    y_array : ndarray
        Radius coordinates of nozzle contour [m]
    volume : float
        Internal volume of divergent section [m³]
    """
    import numpy as np
    import math

    # Approximate Rao bell length as 80% of equivalent 15° conical nozzle
    nozzle_length = 0.8 * ((math.sqrt(epsilon) - 1) * Rt / math.tan(math.radians(15)))

    # Wall angles (in radians) at:
    # - theta_n: start of parabolic bell (end of throat arc)
    # - theta_e: nozzle exit
    theta_n = math.radians(24)
    theta_e = math.radians(12)
    Rn = 0.382*Rt

    # -------------------
    # 1) Circular arc at throat
    # -------------------
    x_change = 0.382 * Rt * math.sin(theta_n)  # axial length of circular arc
    x_circ = np.linspace(0, x_change, 5)       # axial coordinate points along arc
    theta_circ = np.arcsin(x_circ / (0.382 * Rt))  # corresponding arc angles
    y_circ = Rt * (1.382 - 0.382 * np.cos(theta_circ))  # radial positions along arc

    # -------------------
    # 2) Parabolic section
    # -------------------
    # Start point (N) of parabola
    Nx = 0.382 * Rt * math.cos(theta_n - math.radians(90))
    Ny = 0.382 * Rt * math.sin(theta_n - math.radians(90)) + 0.382 * Rt + Rt

    # End point (E) of parabola (exit)
    Ex = nozzle_length
    Ey = np.sqrt(epsilon) * Rt

    # Slopes at start (m1) and end (m2)
    m1, m2 = math.tan(theta_n), math.tan(theta_e)

    # Intersection point (Q) of tangents from N and E
    C1, C2 = Ny - m1 * Nx, Ey - m2 * Ex
    Qx = (C2 - C1) / (m1 - m2)
    Qy = (m1 * C2 - m2 * C1) / (m1 - m2)

    # Quadratic Bezier curve parameter t
    t = np.linspace(0, 1, 96)
    x_para = (1 - t)**2 * Nx + 2 * (1 - t) * t * Qx + t**2 * Ex
    y_para = (1 - t)**2 * Ny + 2 * (1 - t) * t * Qy + t**2 * Ey

    # -------------------
    # 3) Combine arc + parabola
    # -------------------
    x_array = np.concatenate((x_circ, x_para[1:]))
    y_array = np.concatenate((y_circ, y_para[1:]))

    # -------------------
    # 4) Volume integration (solid of revolution)
    # -------------------
    volume = 0
    for i in range(len(x_array) - 1):
        volume += math.pi * y_array[i]**2 * (x_array[i+1] - x_array[i])

    return x_array, y_array, volume, Rn, theta_n, theta_e


def convergent_sizing(CR, Rt):
    """
    Generate a convergent section contour with:
    - Upstream chamber fillet R2 (fraction of maximum possible R2)
    - Straight conical section at chosen half-angle b
    - Upstream throat fillet R1

    Parameters:
    ----------
    CR : float
        Contraction ratio (Ac / At)
    Rt : float
        Throat radius [m]

    Returns:
    -------
    x_array : ndarray
        Axial coordinates of convergent contour [m]
    y_array : ndarray
        Radius coordinates of convergent contour [m]
    volume : float
        Internal volume of convergent section [m³]
    """
    import math
    import numpy as np

    # Upstream throat fillet (empirical ~1.5 * Rt)
    R1 = 1.5 * Rt

    # Convergent half-angle b (deg → rad)
    b = math.radians(30)

    # Chamber radius from contraction ratio
    Rc = np.sqrt(CR) * Rt

    # Maximum possible R2 before it touches R1 (two-arc blend case)
    R2max = (Rc - Rt) / (1 - math.cos(b)) - R1

    # Select fraction of maximum R2 (empirical choice: 0.35)
    R2 = R2max * 0.5

    # -------------------
    # 1) R2 arc from chamber wall to straight cone
    # -------------------
    theta2 = np.linspace(0, b, 40)
    x2_array = R2 * np.sin(theta2)
    y2_array = Rc - R2 + R2 * np.cos(theta2)

    # -------------------
    # 2) R1 arc from straight cone into throat
    # -------------------
    theta1 = np.linspace(b, 0, 40)
    x1_array = R1 * math.sin(b) - R1 * np.sin(theta1)
    y1_array = Rt + R1 - R1 * np.cos(theta1)

    # -------------------
    # 3) Straight cone segment between arcs
    # -------------------
    dely = y2_array[-1] - y1_array[0]       # radial drop between arcs
    delx = dely / math.tan(b)                # axial length of straight section
    xl_array = np.linspace(np.max(x2_array), np.max(x2_array) + delx, 22)
    yl_array = np.linspace(np.min(y2_array), np.min(y2_array) - dely, 22)

    # Shift R1 arc forward in x so it starts after straight section
    x1_array += np.max(xl_array)

    # -------------------
    # 4) Combine arcs + straight
    # -------------------
    x_array = np.concatenate((x2_array, xl_array[1:-1], x1_array))
    y_array = np.concatenate((y2_array, yl_array[1:-1], y1_array))

    # -------------------
    # 5) Volume integration (solid of revolution)
    # -------------------
    volume = 0
    for i in range(len(x_array) - 1):
        volume += math.pi * y_array[i]**2 * (x_array[i+1] - x_array[i])

    return x_array, y_array, volume, R1, R2, b


def cylinder_sizing(Lstar, other_volume, Rt, CR):
    """
    Size the cylindrical chamber section so total L* requirement is met.

    L* is the characteristic length: total chamber volume (including convergent)
    divided by throat area.

    Parameters:
    ----------
    Lstar : float
        Target characteristic length [m]
    other_volume : float
        Volume already present (convergent + divergent) [m³]
    Rt : float
        Throat radius [m]
    CR : float
        Contraction ratio (Ac / At)

    Returns:
    -------
    x_array : ndarray
        Axial coordinates of cylinder [m]
    y_array : ndarray
        Radius coordinates of cylinder [m] (constant)
    """
    import math
    import numpy as np

    # Required total chamber volume from L* target
    required_volume = Lstar * math.pi * Rt**2

    # Remaining volume to be filled by cylinder
    Volume = required_volume - other_volume

    # Length of cylinder = volume / chamber cross-section
    Length = Volume / (CR * math.pi * Rt**2)

    # Coordinates of straight cylinder
    x_array = np.linspace(0, Length, 100)
    y_array = np.ones_like(x_array) * Rt * math.sqrt(CR)

    return x_array, y_array, Length
