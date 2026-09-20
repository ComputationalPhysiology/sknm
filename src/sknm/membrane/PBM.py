"""Generated from `PBM.ode` -- do not edit by hand.

Regenerate with `python3 tools/generate_membrane_models.py`.

Singularity guarding is on (gotranx's default) but adds no guard here: every
denominator in this model's right hand side is a parameter, never a state.

The scheme is `generalized_rush_larsen`; the backend is numpy, so this module imports numpy
and nothing else, which is what keeps `sknm`'s runtime dependencies to numpy + scipy.
"""

import numpy

parameter = {
    "Cm": 0,
    "Dz": 1,
    "G": 2,
    "alpha": 3,
    "fca": 4,
    "gca": 5,
    "gk": 6,
    "gkatpbar": 7,
    "gkca": 8,
    "kd": 9,
    "kpmca": 10,
    "kr": 11,
    "kserca": 12,
    "pleak": 13,
    "pr": 14,
    "sa": 15,
    "sigmaer": 16,
    "sm": 17,
    "sn": 18,
    "taua": 19,
    "taun": 20,
    "vca": 21,
    "vk": 22,
    "vm": 23,
    "vn": 24,
}


def parameter_index(name: str) -> int:
    """Return the index of the parameter with the given name

    Arguments
    ---------
    name : str
        The name of the parameter

    Returns
    -------
    int
        The index of the parameter

    Raises
    ------
    KeyError
        If the name is not a valid parameter
    """

    return parameter[name]


state = {"n": 0, "cer": 1, "v": 2, "a": 3, "c": 4}


def state_index(name: str) -> int:
    """Return the index of the state with the given name

    Arguments
    ---------
    name : str
        The name of the state

    Returns
    -------
    int
        The index of the state

    Raises
    ------
    KeyError
        If the name is not a valid state
    """

    return state[name]


monitor = {
    "ik": 0,
    "ikatp": 1,
    "jerleak": 2,
    "jpmca": 3,
    "jserca": 4,
    "minf": 5,
    "ninf": 6,
    "omega": 7,
    "r": 8,
    "jer": 9,
    "ica": 10,
    "dn_dt": 11,
    "ikca": 12,
    "ainf": 13,
    "dcer_dt": 14,
    "jmem": 15,
    "dv_dt": 16,
    "da_dt": 17,
    "dc_dt": 18,
}


def monitor_index(name: str) -> int:
    """Return the index of the monitor with the given name

    Arguments
    ---------
    name : str
        The name of the monitor

    Returns
    -------
    int
        The index of the monitor

    Raises
    ------
    KeyError
        If the name is not a valid monitor
    """

    return monitor[name]


def init_parameter_values(**values):
    """Initialize parameter values"""
    # Cm=5300, Dz=1, G=11, alpha=4.5e-06, fca=0.01, gca=1200
    # gk=3000, gkatpbar=500, gkca=10, kd=0.3, kpmca=0.2, kr=58
    # kserca=0.4, pleak=0.0005, pr=1.75, sa=0.1, sigmaer=5, sm=12
    # sn=5, taua=300000, taun=16, vca=25, vk=-75, vm=-20, vn=-16

    parameters = numpy.array(
        [
            5300,
            1,
            11,
            4.5e-06,
            0.01,
            1200,
            3000,
            500,
            10,
            0.3,
            0.2,
            58,
            0.4,
            0.0005,
            1.75,
            0.1,
            5,
            12,
            5,
            300000,
            16,
            25,
            -75,
            -20,
            -16,
        ],
        dtype=numpy.float64,
    )

    for key, value in values.items():
        parameters[parameter_index(key)] = value

    return parameters


def init_state_values(**values):
    """Initialize state values"""
    # n=0, cer=110, v=-66, a=0.52, c=0.1

    states = numpy.array([0, 110, -66, 0.52, 0.1], dtype=numpy.float64)

    for key, value in values.items():
        states[state_index(key)] = value

    return states


def rhs(t, states, parameters):

    # Assign states
    n = states[0]
    cer = states[1]
    v = states[2]
    a = states[3]
    c = states[4]

    # Assign parameters
    Cm = parameters[0]
    Dz = parameters[1]
    G = parameters[2]
    alpha = parameters[3]
    fca = parameters[4]
    gca = parameters[5]
    gk = parameters[6]
    gkatpbar = parameters[7]
    gkca = parameters[8]
    kd = parameters[9]
    kpmca = parameters[10]
    kr = parameters[11]
    kserca = parameters[12]
    pleak = parameters[13]
    pr = parameters[14]
    sa = parameters[15]
    sigmaer = parameters[16]
    sm = parameters[17]
    sn = parameters[18]
    taua = parameters[19]
    taun = parameters[20]
    vca = parameters[21]
    vk = parameters[22]
    vm = parameters[23]
    vn = parameters[24]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    ik = n * (gk * (v - vk))
    ikatp = a * ((Dz * gkatpbar) * (v - vk))
    jerleak = pleak * (-c + cer)
    jpmca = c * kpmca
    jserca = c * kserca
    minf = 1.0 / (numpy.exp((-v + vm) / sm) + 1.0)
    ninf = 1.0 / (numpy.exp((-v + vn) / sn) + 1.0)
    omega = c**5.0 / (c**5.0 + kd**5.0)
    r = (G - pr) / kr
    jer = jerleak - jserca
    ica = minf * (gca * (v - vca))
    dn_dt = (-n + ninf) / taun
    values[0] = dn_dt
    ikca = omega * (gkca * (v - vk))
    ainf = 1.0 / (numpy.exp((-c + r) / sa) + 1.0)
    dcer_dt = jer * ((-fca) * sigmaer)
    values[1] = dcer_dt
    jmem = -alpha * ica - jpmca
    dv_dt = (-ikca + (-ikatp + (-ica - ik))) / Cm
    values[2] = dv_dt
    da_dt = (-a + ainf) / taua
    values[3] = da_dt
    dc_dt = fca * (jer + jmem)
    values[4] = dc_dt

    return values


def monitor_values(t, states, parameters):

    # Assign states
    n = states[0]
    cer = states[1]
    v = states[2]
    a = states[3]
    c = states[4]

    # Assign parameters
    Cm = parameters[0]
    Dz = parameters[1]
    G = parameters[2]
    alpha = parameters[3]
    fca = parameters[4]
    gca = parameters[5]
    gk = parameters[6]
    gkatpbar = parameters[7]
    gkca = parameters[8]
    kd = parameters[9]
    kpmca = parameters[10]
    kr = parameters[11]
    kserca = parameters[12]
    pleak = parameters[13]
    pr = parameters[14]
    sa = parameters[15]
    sigmaer = parameters[16]
    sm = parameters[17]
    sn = parameters[18]
    taua = parameters[19]
    taun = parameters[20]
    vca = parameters[21]
    vk = parameters[22]
    vm = parameters[23]
    vn = parameters[24]

    # Assign expressions
    shape = 19 if len(states.shape) == 1 else (19, states.shape[1])
    values = numpy.zeros(shape)
    ik = n * (gk * (v - vk))
    values[0] = ik
    ikatp = a * ((Dz * gkatpbar) * (v - vk))
    values[1] = ikatp
    jerleak = pleak * (-c + cer)
    values[2] = jerleak
    jpmca = c * kpmca
    values[3] = jpmca
    jserca = c * kserca
    values[4] = jserca
    minf = 1.0 / (numpy.exp((-v + vm) / sm) + 1.0)
    values[5] = minf
    ninf = 1.0 / (numpy.exp((-v + vn) / sn) + 1.0)
    values[6] = ninf
    omega = c**5.0 / (c**5.0 + kd**5.0)
    values[7] = omega
    r = (G - pr) / kr
    values[8] = r
    jer = jerleak - jserca
    values[9] = jer
    ica = minf * (gca * (v - vca))
    values[10] = ica
    dn_dt = (-n + ninf) / taun
    values[11] = dn_dt
    ikca = omega * (gkca * (v - vk))
    values[12] = ikca
    ainf = 1.0 / (numpy.exp((-c + r) / sa) + 1.0)
    values[13] = ainf
    dcer_dt = jer * ((-fca) * sigmaer)
    values[14] = dcer_dt
    jmem = -alpha * ica - jpmca
    values[15] = jmem
    dv_dt = (-ikca + (-ikatp + (-ica - ik))) / Cm
    values[16] = dv_dt
    da_dt = (-a + ainf) / taua
    values[17] = da_dt
    dc_dt = fca * (jer + jmem)
    values[18] = dc_dt

    return values


def generalized_rush_larsen(states, t, dt, parameters):

    # Assign states
    n = states[0]
    cer = states[1]
    v = states[2]
    a = states[3]
    c = states[4]

    # Assign parameters
    Cm = parameters[0]
    Dz = parameters[1]
    G = parameters[2]
    alpha = parameters[3]
    fca = parameters[4]
    gca = parameters[5]
    gk = parameters[6]
    gkatpbar = parameters[7]
    gkca = parameters[8]
    kd = parameters[9]
    kpmca = parameters[10]
    kr = parameters[11]
    kserca = parameters[12]
    pleak = parameters[13]
    pr = parameters[14]
    sa = parameters[15]
    sigmaer = parameters[16]
    sm = parameters[17]
    sn = parameters[18]
    taua = parameters[19]
    taun = parameters[20]
    vca = parameters[21]
    vk = parameters[22]
    vm = parameters[23]
    vn = parameters[24]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    ik = n * (gk * (v - vk))
    ikatp = a * ((Dz * gkatpbar) * (v - vk))
    jerleak = pleak * (-c + cer)
    jpmca = c * kpmca
    jserca = c * kserca
    minf = 1.0 / (numpy.exp((-v + vm) / sm) + 1.0)
    ninf = 1.0 / (numpy.exp((-v + vn) / sn) + 1.0)
    omega = c**5.0 / (c**5.0 + kd**5.0)
    r = (G - pr) / kr
    jer = jerleak - jserca
    ica = minf * (gca * (v - vca))
    dn_dt = (-n + ninf) / taun
    dn_dt_linearized = -1 / taun
    values[0] = dn_dt * (numpy.exp(dn_dt_linearized * dt) - 1) / dn_dt_linearized + n
    ikca = omega * (gkca * (v - vk))
    ainf = 1.0 / (numpy.exp((-c + r) / sa) + 1.0)
    dcer_dt = jer * ((-fca) * sigmaer)
    dcer_dt_linearized = -fca * pleak * sigmaer
    values[1] = cer + numpy.where(
        numpy.logical_or((dcer_dt_linearized > 1e-08), (dcer_dt_linearized < -1e-08)),
        dcer_dt * (numpy.exp(dcer_dt_linearized * dt) - 1) / dcer_dt_linearized,
        dcer_dt * dt,
    )
    jmem = -alpha * ica - jpmca
    dv_dt = (-ikca + (-ikatp + (-ica - ik))) / Cm
    _linearization_temp_0 = 1 / sm
    _linearization_temp_1 = numpy.exp(-_linearization_temp_0 * (v - vm))
    dv_dt_linearized = (
        -(
            Dz * a * gkatpbar
            + gca
            * (
                _linearization_temp_0
                * _linearization_temp_1
                * (1.0 * v - 1.0 * vca)
                / (_linearization_temp_1 + 1.0) ** 2
                + minf
            )
            + gk * n
            + gkca * omega
        )
        / Cm
    )
    values[2] = v + numpy.where(
        numpy.logical_or((dv_dt_linearized > 1e-08), (dv_dt_linearized < -1e-08)),
        dv_dt * (numpy.exp(dt * dv_dt_linearized) - 1) / dv_dt_linearized,
        dt * dv_dt,
    )
    da_dt = (-a + ainf) / taua
    da_dt_linearized = -1 / taua
    values[3] = a + da_dt * (numpy.exp(da_dt_linearized * dt) - 1) / da_dt_linearized
    dc_dt = fca * (jer + jmem)
    dc_dt_linearized = -fca * (kpmca + kserca + pleak)
    values[4] = c + numpy.where(
        numpy.logical_or((dc_dt_linearized > 1e-08), (dc_dt_linearized < -1e-08)),
        dc_dt * (numpy.exp(dc_dt_linearized * dt) - 1) / dc_dt_linearized,
        dc_dt * dt,
    )

    return values
