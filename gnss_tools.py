import numpy as np
import xarray as xr
import pandas
import warnings
import gzip
import datetime
import re
from pathlib import Path 

###############################################################################


def nav2orbit(
    nav_in: xr.Dataset,
    time_start: np.datetime64 = None,
    time_end: np.datetime64 = None,
    time_interval: np.timedelta64 = np.timedelta64(30, "s"),
) -> xr.Dataset:
    """
    Calculates satellite positions in ECEF coords from navigation data
    Positions are returned in their true position at time of transmission (approximately!)
    This is not accurate enough for geodetic use, but is more than sufficient for TEC

    Parameters
    ----------
    nav_in : xarray Dataset
        Navigation file data from georinex
    time_start : np.datetime64, optional
        The default is None.
    time_end : np.datetime64, optional
        The default is None.
    time_interval : np.timedelta64, optional
        The default is np.timedelta64(30, 's').

    Returns
    -------
    orbit : xarray Dataset
        Satellite orbit positions

    """
    mu = 3.986005e14
    omegaE = 7.2921151467e-5

    if time_start is None:
        time_start = nav_in["time"].min()

    if time_end is None:
        time_end = nav_in["time"].max()

    if isinstance(time_start, xr.core.dataarray.DataArray):
        time_start = time_start.values
    if isinstance(time_end, xr.core.dataarray.DataArray):
        time_end = time_end.values

    tperiod = 1 + int((time_end - time_start) / time_interval)
    timesv = pandas.date_range(time_start, time_end, tperiod)

    nav = nav_in.ffill(dim="time")
    nav = nav.bfill(dim="time")  # better to have values even if stale

    with warnings.catch_warnings():  # catch futurewarning from xarray
        warnings.simplefilter("ignore")
        nav = nav.interp(
            time=timesv, method="nearest", kwargs={"fill_value": "extrapolate"}
        )

    time = np.mod(
        (
            nav.time
            - np.timedelta64(70000000, "ns")
            - np.datetime64("1980-01-06T00:00:00")
        )
        / np.timedelta64(1, "s"),
        7 * 24 * 60 * 60,
    )

    tk = time - nav.Toe

    with warnings.catch_warnings():  # catch np.mod(nan) warning
        warnings.simplefilter("ignore")
        tk = np.mod(tk + 302400, 604800) - 302400

    Mk = nav.M0 + (np.sqrt(mu) / nav.sqrtA**3 + nav.DeltaN) * tk

    Ek0 = Mk
    for i in range(10):
        Ek = Mk + nav.Eccentricity * np.sin(Ek0)
        if np.max(Ek - Ek0) < 1.0e16:
            break
        Ek0 = Ek

    vk = np.arctan2(
        (np.sqrt(1 - nav.Eccentricity**2) * np.sin(Ek)),
        (np.cos(Ek) - nav.Eccentricity),
    )

    uk = (
        nav.omega
        + vk
        + nav.Cuc * np.cos(2 * (nav.omega + vk))
        + nav.Cus * np.sin(2 * (nav.omega + vk))
    )

    rk = (
        nav.sqrtA**2 * (1 - nav.Eccentricity * np.cos(Ek))
        + nav.Crc * np.cos(2 * (nav.omega + vk))
        + nav.Crs * np.sin(2 * (nav.omega + vk))
    )

    ik = (
        nav.Io
        + nav.IDOT * tk
        + nav.Cic * np.cos(2 * (nav.omega + vk))
        + nav.Cis * np.sin(2 * (nav.omega + vk))
    )

    lk = nav.Omega0 + (nav.OmegaDot - omegaE) * tk - omegaE * nav.Toe

    nav["X"] = rk * (np.cos(uk) * np.cos(lk) - np.sin(uk)
                     * np.cos(ik) * np.sin(lk))
    nav["Y"] = rk * (np.cos(uk) * np.sin(lk) + np.sin(uk)
                     * np.cos(ik) * np.cos(lk))
    nav["Z"] = rk * (np.sin(uk) * np.sin(ik))
    nav["svdt"] = nav.SVclockBias + tk * \
        nav.SVclockDrift + tk**2 * nav.SVclockDriftRate

    nav = nav.dropna(dim="sv", how="all", subset=["X", "Y", "Z"])

    if not [sv for sv in nav["sv"].data if "E" in sv]:
        print("Navigation file missing GALILEO data!")

    if not [sv for sv in nav["sv"].data if "G" in sv]:
        print("Navigation file missing GPS data!")

    return _xrTimeFix(
        nav[
            [
                v
                for v in ["X", "Y", "Z", "svdt", "TGD", "BGDe5a", "BGDe5b"]
                if v in nav.data_vars
            ]
        ]
    )

###############################################################################


def _xrTimeFix(data: xr.Dataset) -> xr.Dataset:
    """
    helper function to fix xarray times to the nearest second
    """
    data["time"] = pandas.DatetimeIndex(data["time"]).round(freq="1s")

    (uNew, iNew) = np.unique(data["time"].data, return_index=True)
    if uNew.size != data["time"].size:
        data = data.isel(time=iNew)

    return data

###############################################################################


def simple_phase_level(TEC_code: xr.Dataset, TEC_phase: xr.Dataset,
                        max_time_skip: np.timedelta64 = np.timedelta64(30, "s"),
                        max_tec_diff: float = 1,
                        max_tec_deriv: float = 10,
                        min_arc_time: np.timedelta64 = np.timedelta64(5, 'm'),
                        max_error: float = None) -> xr.Dataset:
    """
    simple_phase_level performs simple phase levelling of TEC data

    Parameters
    ----------
    TEC_code : xr.Dataset
        dataset of pseudorange-derived sTEC data (in TECU)
    TEC_phase : xr.Dataset
        dataset of phase-derived sTEC data (in TECU)
    max_time_skip : np.timedelta64, optional
        maximum length of missing epochs before all arcs are considered to have a skip, by default np.timedelta64(30, "s")
    max_tec_diff : float, optional
        TEC threshold (in TECU) where a cycle slip is considered, by default 1 
    max_tec_deriv : float, optional
        TEC derivative threshold (in TECU) where a cycle slip is considered, by default 10 
    min_arc_time : np.timedelta64, optional
        minimum arc length to be considered for phase levelling, by default np.timedelta64(5, 'm')
    max_error : float, optional
        if not None, any arcs with an error greater than max_error (in TECU) will be discarded

    Returns
    -------
    xr.Dataset
        dataset containing phase-levelled sTEC, sTEC errors, and arc index
    """

    # detect if any missing times in the data
    time_skip = np.atleast_2d(
        np.hstack((TEC_code.time.diff("time") > max_time_skip, [False]))
    ).T

    # detect if any nan data in the phase
    nan_tec = np.isnan(TEC_code + TEC_phase)

    # detect if any TEC derivatives are too big
    deriv_tec = np.abs((TEC_phase - TEC_code).differentiate(coord='time', datetime_unit='s')) > max_tec_deriv # TECU

    # detect if any jumps in TEC
    jump_tec = np.abs(np.diff(TEC_phase, prepend=TEC_phase.isel(time=slice(0,1)), axis=0)) > max_tec_diff

    # everything that is True is a place with bad TEC (or a cycle slip)
    bad_tec = (time_skip + nan_tec + jump_tec + deriv_tec).astype(bool).T

    # find all the start points and end poinbts of good data
    (ss_list, ts_list) = np.nonzero(np.diff(bad_tec, prepend=1, axis=1) < 0)
    (se_list, te_list) = np.nonzero(np.diff(bad_tec, append=1, axis=1) > 0)

    # only keep arcs longer than 5 minutes
    igood = (
        TEC_code.time[te_list].data -
        TEC_phase.time[ts_list].data) > min_arc_time

    ts_list = ts_list[igood]
    te_list = te_list[igood]
    ss_list = ss_list[igood]
    se_list = se_list[igood]

    stec = np.full_like(TEC_code, np.nan)
    stec_error = np.full_like(TEC_code, np.nan)
    arc_index = np.full(TEC_code.shape, fill_value=-1, dtype=int)

    n = 0
    for s, ts, te in zip(ss_list, ts_list, te_list):
 
        plo = np.nanmean(TEC_code[ts: te + 1, s] - TEC_phase[ts: te + 1, s])

        assert not np.isnan(plo)

        stec[ts: te + 1, s] = TEC_phase[ts: te + 1, s] + plo

        # residuals to estimate error
        nrms = np.sqrt(np.nanmean((stec[ts: te + 1, s] - TEC_code[ts: te + 1, s])**2))/np.sqrt(1 + te - ts)

        if max_error is not None and nrms > max_error:
            # skip arc if error too large
            stec[ts: te + 1, s] = np.nan
            continue

        stec_error[ts: te + 1, s] = nrms

        n += 1
        arc_index[ts: te + 1, s] = np.short(n)

    data = xr.Dataset({
        'sTEC': (('time', 'sv'), stec),
        'sTEC_error': (('time', 'sv'), stec_error),
        'arc_index': (('time', 'sv'), arc_index),
    },
        coords={
        "sv": TEC_code.sv,
        "time": TEC_code.time
    }
    )
    return data

###############################################################################


def elevation(rX, rY, rZ, tX, tY, tZ):
    """
    rX, rY, rZ: receiver position in ECEF coordinates
    tX, tY, tZ: transmitter (satellite) position in ECEF coordinates

    Returns
    -------
    Elevation (radians)
    """
    px = tX - rX
    py = tY - rY
    pz = tZ - rZ

    pn = 1.0 / np.sqrt(px**2 + py**2 + pz**2)
    un = 1.0 / np.sqrt(rX**2 + rY**2 + rZ**2)

    return np.arcsin(pn * un * (rX * px + rY * py + rZ * pz))


###############################################################################


def enu2xyz(E, N, U, lat, lon):
    """
    East-North-Up to X-Y-Z
    """
    rlat = np.deg2rad(lat)
    rlon = np.deg2rad(lon)
    sa = np.sin(rlat)
    ca = np.cos(rlat)
    so = np.sin(rlon)
    co = np.cos(rlon)
    X = -so * E - co * sa * N + co * ca * U
    Y = co * E - so * sa * N + so * ca * U
    Z = ca * N + sa * U

    return X, Y, Z


###############################################################################


def xyz2enu(X, Y, Z, lat, lon):
    """
    X-Y-Z to East-North-Up
    """
    rlat = np.deg2rad(lat)
    rlon = np.deg2rad(lon)
    sa = np.sin(rlat)
    ca = np.cos(rlat)
    so = np.sin(rlon)
    co = np.cos(rlon)
    E = -so * X + co * Y
    N = -co * sa * X - so * sa * Y + ca * Z
    U = co * ca * X + so * ca * Y + sa * Z

    return E, N, U


###############################################################################


def wgs2ecef(lat, lon, alt):
    """
    function to convert from WGS ellipsoidal coords to ECEF cartesian coords
    ECEF = [x, y, z], [m, m, m]
    WGS = [lat, lon, alt], [deg, deg, km]
    """

    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = (2.0 * f) - (f**2)

    rlat = np.deg2rad(lat)
    rlon = np.deg2rad(lon)
    ralt = 1.0e3 * alt
    sa = np.sin(rlat)
    ca = np.cos(rlat)

    N = a / (np.sqrt(1.0 - e2 * (sa**2)))

    x = (N + ralt) * ca * np.cos(rlon)
    y = (N + ralt) * ca * np.sin(rlon)
    z = (((1.0 - e2) * N) + ralt) * sa

    return x, y, z


###############################################################################


def ecef2wgs(x, y, z, tol=1.0e-6):
    """
    function to convert from ECEF cartesian coordinates to WGS ellipsoidal
    coords
    ECEF = [x, y, z], [m, m, m]
    WGS = [lat, lon, alt], [deg, deg, km]
    """

    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = (2 * f) - (f**2)

    rad, rlat, rlon = cartsph(x, y, z)

    p = np.sqrt((x**2) + (y**2))
    zp = z / p
    la = np.arctan2(zp, 1.0 - (e2))
    la0 = rlat + 1.0e3
    h = rad

    # fixes problem at poles
    zn = np.abs(z) - (1.0 - e2) * a / np.sqrt(1.0 - e2)

    for i in range(100):
        la0 = la + 0.0
        N = a / (np.sqrt(1.0 - e2 * (np.sin(la) ** 2)))
        h = np.fmax((p / np.cos(la)) - N, zn)
        la = np.arctan2(zp, 1.0 - ((N / (N + h)) * e2))

        if np.all(np.abs(la0 - la) < tol):
            break

    lat = np.rad2deg(la)
    lon = np.rad2deg(rlon)
    alt = 1.0e-3 * h
    return lat, lon, alt


###############################################################################


def cartsph(x, y, z):
    """
    function to convert from ECEF cartesian coordinates to spherical coords
    ECEF = [x, y, z], [m, m, m]
    Sph = [radius, lat, lon], [m, rad, rad]
    """

    rad = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    lon = np.arctan2(y, x)

    return rad, lat, lon


###############################################################################


def sphcart(rad, lat, lon):
    """
    function to convert from spherical coords to ECEF cartesian coordinates
    Sph = [radius, lat, lon], [m, rad, rad]
    ECEF = [x, y, z], [m, m, m]
    """
    x = rad * np.cos(lat) * np.cos(lon)
    y = rad * np.cos(lat) * np.sin(lon)
    z = rad * np.sin(lat)

    return x, y, z


###############################################################################


def loadSinex(file):
    """


    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    if not isinstance(file, SinexFile):
        file = SinexFile(file)

    if not file.is_valid:
        raise ValueError(file)

    if file.compression == ".gz":
        with gzip.open(file.path.expanduser(), "rb") as f:
            td = f.readlines()

    elif file.compression == "":
        with open(file.path.expanduser(), "rb") as f:
            td = f.readlines()

    bs = np.flatnonzero(np.char.startswith(td, b"+BIAS/SOLUTION")) + 1
    be = np.flatnonzero(np.char.startswith(td, b"-BIAS/SOLUTION"))

    g = np.genfromtxt(
        td[bs[0]: be[0]],
        comments="*",
        autostrip=True,
        delimiter=[5, 5, 4, 10, 5, 5, 15, 15, 5, 22, 12, 22, 12],
        dtype=[
            "U5",
            "U5",
            "U4",
            "U10",
            "U5",
            "U5",
            "U15",
            "U15",
            "U5",
            "float",
            "float",
            "float",
            "float",
        ],
        names=[
            "BIAS",
            "SVN",
            "PRN",
            "STATION",
            "OBS1",
            "OBS2",
            "BIAS_START",
            "BIAS_END",
            "UNIT",
            "VALUE",
            "VALUE_STD",
            "SLOPE",
            "SLOPE_STD",
        ],
    )

    sv = np.unique(g["PRN"])
    stn = np.unique(g["STATION"])

    combination = np.char.strip(
        np.char.add(np.char.add(g["OBS1"], "_"), g["OBS2"]), chars="_"
    )
    cmb = np.unique(combination)

    time_start = np.array(
        [_replaceTimeWithDefault(t, file.time_start) for t in g["BIAS_START"]]
    )

    time_end = np.array(
        [_replaceTimeWithDefault(t, file.time_end) for t in g["BIAS_END"]]
    )

    u_time = np.unique(time_start)

    sv = np.array([s for s in sv if len(s) == 3])
    stn = np.array([s for s in stn if len(s) == 4])

    satDCB = np.nan * np.ones((sv.size, cmb.size, u_time.size))
    satRMS = np.nan * np.ones((sv.size, cmb.size, u_time.size))
    stnDCB = np.nan * np.ones((stn.size, cmb.size, u_time.size))
    stnRMS = np.nan * np.ones((stn.size, cmb.size, u_time.size))

    for i, ln in enumerate(g):
        lcmb = combination[i]
        if ln["PRN"] in sv:
            satDCB[sv == ln["PRN"], cmb == lcmb, u_time == time_start[i]] = ln["VALUE"]
            satRMS[sv == ln["PRN"], cmb == lcmb, u_time == time_start[i]] = ln[
                "VALUE_STD"
            ]

        elif ln["STATION"] in stn:
            stnDCB[stn == ln["STATION"], cmb == lcmb, u_time == time_start[i]] = ln[
                "VALUE"
            ]
            stnRMS[stn == ln["STATION"], cmb == lcmb, u_time == time_start[i]] = ln[
                "VALUE_STD"
            ]

    data = xr.Dataset(
        coords={"sv": sv, "stn": stn, "combination": cmb, "date": u_time}
    )

    data["satDCB"] = xr.DataArray(satDCB, dims=["sv", "combination", "date"])
    data["satRMS"] = xr.DataArray(satRMS, dims=["sv", "combination", "date"])
    data["stnDCB"] = xr.DataArray(stnDCB, dims=["stn", "combination", "date"])
    data["stnRMS"] = xr.DataArray(stnRMS, dims=["stn", "combination", "date"])

    # assumes the whole file has the same start and end times
    data["bias_start"] = xr.DataArray(np.unique(time_start), dims=["date"])
    data["bias_end"] = xr.DataArray(np.unique(time_end), dims=["date"])

    g = np.genfromtxt(
        td[:1],
        comments="*",
        autostrip=True,
        delimiter=[5, 5, 4, 15, 4, 15, 15, 2, 9],
        dtype=["U5", "U5", "U4", "U15", "U4", "U15", "U15", "U2", "U9"],
        names=[
            "FILE_ID",
            "FORMAT VERSION",
            "FILE AGENCY CODE",
            "FILE CREATION TIME",
            "DATA AGENCY CODE",
            "TIME START",
            "TIME END",
            "BIAS MODE",
            "BIAS COUNT",
        ],
    )

    for i in g.dtype.fields:
        data.attrs[i] = g[i]

    bs = np.flatnonzero(np.char.startswith(td, b"+FILE/REFERENCE")) + 1
    be = np.flatnonzero(np.char.startswith(td, b"-FILE/REFERENCE"))

    g = np.genfromtxt(
        td[bs[0]: be[0]],
        comments="*",
        autostrip=True,
        delimiter=[19, 61],
        dtype=["U19", "U61"],
        names=["Label", "Info"],
    )

    for i, ln in enumerate(g):
        data.attrs[ln["Label"]] = ln["Info"]

    bs = np.flatnonzero(np.char.startswith(td, b"+BIAS/DESCRIPTION")) + 1
    be = np.flatnonzero(np.char.startswith(td, b"-BIAS/DESCRIPTION"))

    g = np.genfromtxt(
        td[bs[0]: be[0]],
        comments="*",
        autostrip=True,
        delimiter=[40, 40],
        dtype=["U40", "U40"],
        names=["Label", "Info"],
    )

    for i, ln in enumerate(g):
        data.attrs[ln["Label"]] = ln["Info"]

    return data.dropna(dim="sv", how="all").dropna(dim="combination", how="all")


#################################################################################


def processDCB(data):
    """


    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    if not isinstance(data, xr.Dataset):
        data = loadSinex(data)

    data = data.isel(
        sv=np.char.startswith(data["sv"], "G") + np.char.startswith(data["sv"], "E")
    )

    data = (
        data.drop_isel(
            combination=np.char.startswith(data["combination"], "L")
        )  # Code only
        .dropna(dim="sv", how="all")
        .dropna(dim="combination", how="all", subset=["satDCB"])
    )

    # add dummy combo for OSB
    data = data.assign_coords(
        {"combination": [_add_dummy(obs) for obs in data["combination"].data]}
    )

    # regular SINEX combinations
    combo_0 = list(data.combination.values)

    # reversed SINEX combinations
    combo_1 = [f"{k[1]}_{k[0]}" for k in np.char.split(combo_0, "_")]

    # each permutation of combinations
    combo_00 = [
        f"{c1[0]}_{c2[1]}"
        for c1 in np.char.split(combo_0, "_")
        for c2 in np.char.split(combo_0, "_")
        if c1[1] == c2[0] and int(c1[0][1]) < int(c2[1][1])
    ]

    combo_01 = [
        f"{c1[0]}_{c2[1]}"
        for c1 in np.char.split(combo_0, "_")
        for c2 in np.char.split(combo_1, "_")
        if c1[1] == c2[0] and int(c1[0][1]) < int(c2[1][1])
    ]

    combo_10 = [
        f"{c1[0]}_{c2[1]}"
        for c1 in np.char.split(combo_1, "_")
        for c2 in np.char.split(combo_0, "_")
        if c1[1] == c2[0] and int(c1[0][1]) < int(c2[1][1])
    ]

    combo_11 = [
        f"{c1[0]}_{c2[1]}"
        for c1 in np.char.split(combo_1, "_")
        for c2 in np.char.split(combo_1, "_")
        if c1[1] == c2[0] and int(c1[0][1]) < int(c2[1][1])
    ]

    combo = [
        f"{c1[0]}_{c1[1]}"
        for c1 in np.char.split(
            combo_0 + combo_1 + combo_00 + combo_01 + combo_10 + combo_11, "_"
        )
        if c1[0][1] <= c1[1][1] or c1[1][1] == "0"  # OSB Dummy has 0
    ]
    combo = np.unique(combo)

    stnDCB = np.full(
        (data["stn"].size, combo.size, data["date"].size), fill_value=np.nan
    )
    stnRMS = np.full(
        (data["stn"].size, combo.size, data["date"].size), fill_value=np.nan
    )
    satDCB = np.full(
        (data["sv"].size, combo.size, data["date"].size), fill_value=np.nan
    )
    satRMS = np.full(
        (data["sv"].size, combo.size, data["date"].size), fill_value=np.nan
    )

    # first populate dcb with values from SINEX
    for i, cmb in enumerate(combo):
        if cmb not in data.combination:
            continue

        stnDCB[:, i, ...] = data["stnDCB"].sel(combination=cmb)
        stnRMS[:, i, ...] = data["stnRMS"].sel(combination=cmb)
        satDCB[:, i, ...] = data["satDCB"].sel(combination=cmb)
        satRMS[:, i, ...] = data["satRMS"].sel(combination=cmb)

    for i, cmb in enumerate(combo):
        if not np.all(np.isnan(satDCB[:, i, :])):
            continue

        # combination not in SINEX Data
        if cmb in combo_00:
            tcmb = [
                [f"{c1[0]}_{c1[1]}", f"{c2[0]}_{c2[1]}"]
                for c1 in np.char.split(combo_0, "_")
                for c2 in np.char.split(combo_0, "_")
                if c1[1] == c2[0] and f"{c1[0]}_{c2[1]}" == cmb
            ]
            s1 = 1.0
            s2 = 1.0
        elif cmb in combo_01:
            tcmb = [
                [f"{c1[0]}_{c1[1]}", f"{c2[1]}_{c2[0]}"]
                for c1 in np.char.split(combo_0, "_")
                for c2 in np.char.split(combo_1, "_")
                if c1[1] == c2[0] and f"{c1[0]}_{c2[1]}" == cmb
            ]
            s1 = 1.0
            s2 = -1.0
        elif cmb in combo_10:
            tcmb = [
                [f"{c1[1]}_{c1[0]}", f"{c2[0]}_{c2[1]}"]
                for c1 in np.char.split(combo_1, "_")
                for c2 in np.char.split(combo_0, "_")
                if c1[1] == c2[0] and f"{c1[0]}_{c2[1]}" == cmb
            ]
            s1 = -1.0
            s2 = 1.0

        elif cmb in combo_11:
            tcmb = [
                [f"{c1[1]}_{c1[0]}", f"{c2[1]}_{c2[0]}"]
                for c1 in np.char.split(combo_1, "_")
                for c2 in np.char.split(combo_1, "_")
                if c1[1] == c2[0] and f"{c1[0]}_{c2[1]}" == cmb
            ]
            s1 = -1.0
            s2 = -1.0
        elif cmb in combo_1:
            # not a real combo, just the negative
            continue
        else:
            print("Something has gone very wrong with combination" + f" {cmb}")

        for tcb in tcmb:

            # need to draw out one element for numpy reasons
            ic1 = np.flatnonzero(combo == tcb[0])[0]
            ic2 = np.flatnonzero(combo == tcb[1])[0]
            if np.all(np.isnan(satDCB[:, ic1, ...] + satDCB[:, ic2, ...])):
                continue

            stnDCB[:, i, ...] = s1 * stnDCB[:, ic1, ...] + s2 * stnDCB[:, ic2, ...]
            satDCB[:, i, ...] = s1 * satDCB[:, ic1, ...] + s2 * satDCB[:, ic2, ...]

            stnRMS[:, i, ...] = np.sqrt(
                stnRMS[:, ic1, ...] ** 2 + stnRMS[:, ic2, ...] ** 2
            )
            satRMS[:, i, ...] = np.sqrt(
                satRMS[:, ic1, ...] ** 2 + satRMS[:, ic2, ...] ** 2
            )
            break
        else:
            print(f" No valid DCB combination to produce {cmb}")
            continue

    for i, cmb in enumerate(combo):
        scaleTEC = ns2tec(cmb)
        stnDCB[:, i, ...] = scaleTEC * stnDCB[:, i, ...]
        stnRMS[:, i, ...] = scaleTEC * stnRMS[:, i, ...]
        satDCB[:, i, ...] = scaleTEC * satDCB[:, i, ...]
        satRMS[:, i, ...] = scaleTEC * satRMS[:, i, ...]

    dcb = xr.Dataset(
        coords={
            "combination": combo,
            "stn": data["stn"].data,
            "sv": data["sv"].data,
            "date": data["date"].data,
        }
    )

    dcb["stnDCB"] = (("stn", "combination", "date"), stnDCB)
    dcb["stnRMS"] = (("stn", "combination", "date"), stnRMS)
    dcb["satDCB"] = (("sv", "combination", "date"), satDCB)
    dcb["satRMS"] = (("sv", "combination", "date"), satRMS)

    if "C1C_C2W" in dcb["combination"].data:
        dcb = xr.concat(
            (
                dcb,
                dcb.sel(combination="C1C_C2W").assign_coords(
                    combination=np.array("C1_P2", dtype="<U7")
                ),
            ),
            dim="combination",
        )

    if "C1C_C2C" in dcb["combination"].data:
        dcb = xr.concat(
            (
                dcb,
                dcb.sel(combination="C1C_C2C").assign_coords(
                    combination=np.array("C1_C2", dtype="<U7")
                ),
            ),
            dim="combination",
        )

    for d in dcb.data_vars:
        dcb[d].attrs["method"] = "sinex"

    # drop dummy OSB combinations
    dcb = dcb.drop_isel(combination=np.char.find(dcb["combination"], "00") >= 0)

    return dcb.dropna(dim="combination", how="all")


###############################################################################


def ns2tec(combo):
    c = 299792458.0

    freq = np.zeros(2)
    for j, Obs in enumerate(combo.split("_")):
        if Obs[1] == "1":
            freq[j] = 1.57542e9
        elif Obs[1] == "2":
            freq[j] = 1.22760e9
        elif Obs[1] == "5":
            freq[j] = 1.17645e9
        elif Obs[1] == "7":
            freq[j] = 1.207140e9
        elif Obs[1] == "8":
            freq[j] = 1.191795e9
        elif Obs[1] == "6":
            freq[j] = 1.27875e9
        elif Obs[1] == "0":
            # dummy value for OSB
            freq[j] = 1.27875e9

    if np.isclose(freq[0], freq[1]):
        return np.nan

    return (
        (freq[0] ** 2)
        * (freq[1] ** 2)
        * c
        / (40.3 * 1e9 * (freq[0] ** 2 - freq[1] ** 2))
    )


###############################################################################


def _add_dummy(obs: str) -> str:
    """
    _add_dummy adds dummy OBS2

    Parameters
    ----------
    obs : str
        OBS1

    Returns
    -------
    str
        OBS1_Dummy
    """
    if "_" in obs:
        return obs
    else:
        return f"{obs}_{obs[0]}00"


def _replaceTimeWithDefault(input_time_str, default_time):

    try:
        return dt2npdt(datetime.datetime.strptime(input_time_str[:8], "%Y:%j"))
    except ValueError:
        return default_time

###############################################################################


class DataFile:
    """
    Base data class for ionospheric filetypes
    """

    def __init__(self, file_path):
        """


        Parameters
        ----------
        file_path : String or pathlib.Path
            Path to the RINEX File to be analyzed

        Returns
        -------
        None.

        """
        if isinstance(file_path, DataFile):
            file_path = file_path.path.expanduser()

        self.path = Path(file_path).expanduser()
        self.filename = self.path.name
        self.is_valid = False
        self.time = np.datetime64("NaT")

    def __repr__(self):
        out = f"{self.__class__.__name__}:"
        for attr in ["path", "is_valid"]:
            out += f" {attr}: {getattr(self, attr)}"
        return out


###############################################################################


class RinexFile(DataFile):
    """
    class to interpret the names of RINEX v2 and v3 files and store the
    attributes. Assumes files follow the IGS naming schema, otherwise it will
    not populate the fields. Does not need to be a local file, can be used to
    sort files to download on remote servers.

    See RinexFile.__init__() for details.
    """

    interpretV2 = re.compile(
        r"(?P<site>^\S{4})"
        r"(?P<doy>\d{3})"
        r"(?P<time>(?:0|[a-x])|(?:[a-x]\d{2}))"
        r"\."
        r"(?P<year>\d{2})"
        r"(?P<type>.)"
        r"(?P<compression>.*$)"
    )

    interpretV3 = re.compile(
        r"(?P<site>^\S{4})"
        r"(?:("
        r"(?P<monument>\d)"
        r"(?P<receiver>\d)"
        r"(?P<country>\S{3})"
        r")|("
        r"_(?P<stream>[^_]*)"
        r"))"
        "_"
        r"(?P<source>[RSU])"
        "_"
        r"(?P<year>\d{4})"
        r"(?P<doy>\d{3})"
        r"(?P<hour>\d{2})"
        r"(?P<minute>\d{2})"
        "_"
        r"(?P<period>\d{2})"
        r"(?P<periodunits>[MHDYU])"
        "_?"
        r"(?:(?:(?P<frequency>\d{2})"
        r"(?P<frequencyunits>[CXSMHDU]))|)"
        "_?"
        r"(?P<type>\D{2})"
        "[^.]*"
        r"\.(?P<format>\w{3})"
        r"(?P<compression>.*$)"
    )

    def __init__(self, file_path):
        """


        Parameters
        ----------
        file_path : String or pathlib.Path
            Path to the RINEX File to be analyzed

        Returns
        -------
        None.

        """
        super().__init__(file_path)

        self.truename = None  # ..... true filename (name with no compression)
        self.is_valid = False  # .... is a valid RINEX file name?
        self.is_netcdf = False  # ... is NetCDF compressed?
        self.is_hatanaka = False  # . hatanaka compressed?
        self.version = None

        self.site = None  # ........ four char station name
        self.type = None  # ........ file type (obs, nav, met)
        self.constellation = None  # constellation (None for v2.11 obs)
        self.time = None  # ........
        self.time_start = None
        self.time_end = None

        self.network = None
        self.ID = None

        self.year = None
        self.doy = None
        self.hour = None
        self.minute = None

        self.monument = None
        self.receiver = None
        self.country = None
        self.source = None
        self.period = None
        self.periodunits = None
        self.frequency = None
        self.frequencyunits = None
        self.format = None
        self.compression = None

        v2 = RinexFile.interpretV2.search(self.filename.lower())

        v3 = RinexFile.interpretV3.search(self.filename)

        try:
            if v2 is not None:
                fdata = v2.groupdict()
                self.version = 2

                self.truename = (
                    re.search(r"(?P<true>.*\.\d{2}[odnglhbmcODNGLHBMC])", self.filename)
                    .groupdict()["true"]
                    .upper()
                )

                if self.truename[-1].upper() == "D":
                    self.truename = self.truename[:-1] + "O"

                self.site = fdata["site"].upper()

                if fdata["type"].lower() == "o":
                    self.type = "observation"
                    self.is_hatanaka = False
                elif fdata["type"].lower() == "d":
                    self.type = "observation"
                    self.is_hatanaka = True
                elif fdata["type"].lower() == "n":
                    self.type = "navigation"
                    self.constellation = "GPS"
                elif fdata["type"].lower() == "g":
                    self.type = "navigation"
                    self.constellation = "GLONASS"
                elif fdata["type"].lower() == "l":
                    self.type = "navigation"
                    self.constellation = "Galileo"
                elif fdata["type"].lower() == "h":
                    self.type = "navigation"
                    self.constellation = "GEO"
                elif fdata["type"].lower() == "b":
                    self.type = "sbas broadcast"
                    self.constellation = "GEO"
                elif fdata["type"].lower() == "m":
                    self.type = "meteorological"
                elif fdata["type"].lower() == "c":
                    self.type = "clock"
                else:
                    print(
                        f"error interpreting {self.filename}: unknown v2 type {fdata['type']}"
                    )

                self.year = int(fdata["year"])
                self.year = self.year + 1900 if self.year > 60 else self.year + 2000

                self.doy = int(fdata["doy"])

                if fdata["time"] == "0":
                    self.hour = 0
                    self.minute = 0
                    self.period = 1
                    self.periodunits = "D"
                elif len(fdata["time"]) == 1:
                    self.minute = 0
                    h = re.search(fdata["time"].lower(), "abcdefghijklmnopqrstuvwx")
                    self.hour = h.start()
                    self.period = 1
                    self.periodunits = "H"
                else:
                    self.minute = int(fdata["time"][1:3])
                    h = re.search(fdata["time"][0].lower(), "abcdefghijklmnopqrstuvwx")
                    self.hour = h.start()
                    self.period = 15
                    self.periodunits = "M"

                self.compression = fdata["compression"]

                self.is_valid = True
            elif v3 is not None:
                self.version = 3
                self.truename = (
                    re.search(r"(?P<true>.*(?=\.crx|\.rnx))", self.filename)
                    .groupdict()["true"]
                    .upper()
                )

                self.truename = self.truename + ".rnx"

                fdata = v3.groupdict()

                self.site = fdata["site"].upper()
                self.year = int(fdata["year"])
                self.doy = int(fdata["doy"])
                self.hour = int(fdata["hour"])
                self.minute = int(fdata["minute"])

                self.monument = fdata["monument"]
                self.receiver = fdata["receiver"]
                self.country = fdata["country"]
                self.source = fdata["source"]
                self.period = int(fdata["period"])
                self.periodunits = fdata["periodunits"]

                if fdata["frequency"] is not None:
                    self.frequency = int(fdata["frequency"])
                    self.frequencyunits = fdata["frequencyunits"]

                if fdata["type"][1].upper() == "O":
                    self.type = "observation"
                elif fdata["type"][1].upper() == "N":
                    self.type = "navigation"
                elif fdata["type"][1].upper() == "M":
                    self.type = "meteorological"
                else:
                    print(
                        "error interpreting "
                        + self.filename
                        + ": unknown v3 type "
                        + fdata["type"][1]
                    )

                if fdata["type"][0].upper() == "G":
                    self.constellation = "GPS"
                elif fdata["type"][0].upper() == "R":
                    self.constellation = "GLONASS"
                elif fdata["type"][0].upper() == "E":
                    self.constellation = "Galileo"
                elif fdata["type"][0].upper() == "J":
                    self.constellation = "QZSS"
                elif fdata["type"][0].upper() == "C":
                    self.constellation = "BDS"
                elif fdata["type"][0].upper() == "I":
                    self.constellation = "IRNSS"
                elif fdata["type"][0].upper() == "S":
                    self.constellation = "SBAS"
                elif (
                    fdata["type"][0].upper() == "M"
                    and not self.type == "meteorological"
                ):
                    self.constellation = "mixed"
                else:
                    print(
                        "error interpreting "
                        + self.filename
                        + ": unknown v3 constellation "
                        + fdata["type"][0]
                    )

                self.format = fdata["format"]

                if self.format == "crx":
                    self.is_hatanaka = True
                elif self.format == "rnx":
                    self.is_hatanaka = False
                else:
                    print(
                        "error interpreting "
                        + self.filename
                        + ": unknown v3 format "
                        + fdata["format"]
                    )

                self.compression = fdata["compression"]

                self.is_valid = True
            else:
                print(f" error interpreting {self.filename} as RINEX")
                return

            self.ID = f"{self.network}-{self.site}"

            if self.compression is not None:
                self.is_netcdf = self.compression.__contains__(".nc")
            else:
                self.is_netcdf = False

            self.time = np.datetime64(str(self.year) + "-01-01") + np.timedelta64(
                24 * 60 * (self.doy - 1) + 60 * self.hour + self.minute, "m"
            )

            self.time_start = self.time

            if self.periodunits == "D":
                dtime = np.timedelta64(self.period, "D")
            elif self.periodunits != "U":
                dtime = np.timedelta64(self.period, self.periodunits.lower())
            else:
                dtime = np.timedelta64(0, "D")

            self.time_end = self.time_start + dtime
        except ValueError as error:
            print(f"{self.path}{error}")
            self.is_valid = False
        except AttributeError as error:
            print(f"{self.path}{error}")
            self.is_valid = False


###############################################################################


class SinexFile(DataFile):
    """
    class to interpret the names of SINEX files and store the attributes
    """

    def __init__(self, file_path, verbose=False):
        """


        Parameters
        ----------
        name : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        super().__init__(file_path)

        self.is_valid = False  # ..... is a valid RINEX file name?
        self.is_netcdf = False

        self.analysis = None  # ........ three char analysis centre name
        self.time = None  # ........
        self.time_start = None
        self.time_end = None

        self.year = None
        self.doy = None
        self.hour = None
        self.minute = None

        self.version = None
        self.project = None
        self.type = None
        self.content = None
        self.relative = None

        self.period = None
        self.periodunits = None
        self.frequency = None
        self.frequencyunits = None
        self.compression = None

        v2 = re.search(
            r"(?P<centre>^\S{3})"
            r"(?P<gpsweek>\d{4})"
            r"(?P<day>\d)" + r"\."
            r"(?P<content>\S{3})" + r"\."
            r"(?P<compression>.*$)",
            self.filename.lower(),
        )

        v3 = re.search(
            r"(?P<centre>^\S{3})"
            r"(?P<version>\S)"
            r"(?P<project>\S{3})"
            r"(?P<type>\S{3})" + "_"
            r"(?P<year>\d{4})"
            r"(?P<doy>\d{3})"
            r"(?P<hour>\d{2})"
            r"(?P<minute>\d{2})" + "_"
            r"(?P<period>\d{2})"
            r"(?P<periodunits>[MHDYUL])" + "_?"
            r"(?:(?:(?P<frequency>\d{2})"
            r"(?P<frequencyunits>[CXSMHDU]))|)" + "_?"
            r"(?P<relative>\S{3})\.(?P<content>\S{3})"
            r"(?P<compression>.*$)",
            self.filename,
        )

        if v2 is not None:
            fdata = v2.groupdict()

            self.analysis = fdata["centre"].upper()

            self.time = (
                np.datetime64("1980-01-06T00:00:00")
                + int(fdata["gpsweek"]) * np.timedelta64(7, "D")
                + np.mod(int(fdata["day"]), 7) * np.timedelta64(1, "D")
            )

            self.doy = int(
                (self.time - np.datetime64(self.time, "Y")) / np.timedelta64(1, "D")
            )

            dtime = self.time.tolist()
            self.year = dtime.year
            self.hour = dtime.hour
            self.minute = dtime.minute

            if fdata["day"] == "7":
                self.period = 7
                self.periodunits = "D"
            else:
                self.period = 1
                self.periodunits = "D"

            self.compression = fdata["compression"]

            self.is_valid = True
        elif v3 is not None:
            fdata = v3.groupdict()

            self.compression = fdata["compression"]

            content = fdata["content"]

            self.truename = (
                re.search(
                    r"(?P<true>.*"
                    + f"{content}"
                    + "(?="
                    + f"{self.compression}"
                    + r"))",
                    self.filename,
                )
                .groupdict()["true"]
                .upper()
            )

            self.analysis = fdata["centre"].upper()
            self.version = fdata["version"]
            self.project = fdata["project"]
            self.type = fdata["type"]

            self.year = int(fdata["year"])
            self.doy = int(fdata["doy"])
            self.hour = int(fdata["hour"])
            self.minute = int(fdata["minute"])

            self.time = np.datetime64(str(self.year) + "-01-01") + np.timedelta64(
                24 * 60 * (self.doy - 1) + 60 * self.hour + self.minute, "m"
            )

            self.period = int(fdata["period"])
            self.periodunits = fdata["periodunits"]

            if fdata["frequency"] is not None:
                self.frequency = int(fdata["frequency"])
                self.frequencyunits = fdata["frequencyunits"]

            self.is_valid = True
        else:
            print(f" error interpreting {self.filename} as SINEX")
            return

        self.is_netcdf = self.compression.__contains__(".nc")

        self.time_start = self.time

        if self.periodunits == "D":
            dtime = np.timedelta64(self.period, "D")
        elif self.periodunits == "L":
            # this probably isn't perfect, but it's a dumb system
            dtime = np.timedelta64(int(self.period / 12 * 365), "D")
        elif self.periodunits != "U":
            dtime = np.timedelta64(self.period, self.periodunits.lower())
        else:
            dtime = np.timedelta64(0, "D")

        self.time_end = self.time_start + dtime

###############################################################################

###############################################################################


def dt2epoch(time):
    """
    datetime to epoch
    """
    epoch = (
        time.replace(tzinfo=datetime.timezone.utc)
        - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    ).total_seconds()
    epoch = epoch + time.microsecond / 1e6
    return epoch


###############################################################################


def epoch2dt(epoch):
    """
    epoch to datetime
    """
    time = datetime.datetime(
        1970, 1, 1, tzinfo=datetime.timezone.utc
    ) + datetime.timedelta(seconds=epoch)
    return time


###############################################################################


def npdt2epoch(time):
    """
    np.datetime64 to epoch
    """
    epoch = (time - np.datetime64("1970-01-01")) / np.timedelta64(1, "ns")
    return epoch * 1e-9


###############################################################################


def epoch2npdt(epoch):
    """
    epoch to np.datetime64
    """
    time = epoch * np.timedelta64(1, "s") + np.datetime64("1970-01-01", "ns")
    return time


###############################################################################


def dt2npdt(time):
    """
    datetime to np.datetime64
    """
    return np.datetime64(time.isoformat()).astype("datetime64[ns]")


###############################################################################


def npdt2dt(time):
    """
    np.datetime64 to datetime
    """
    return datetime.datetime.fromisoformat(time.astype("datetime64[ms]").astype("str"))

