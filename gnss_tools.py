import numpy as np
import xarray as xr
import pandas
import warnings



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

    nav["X"] = rk * (np.cos(uk) * np.cos(lk) - np.sin(uk) * np.cos(ik) * np.sin(lk))
    nav["Y"] = rk * (np.cos(uk) * np.sin(lk) + np.sin(uk) * np.cos(ik) * np.cos(lk))
    nav["Z"] = rk * (np.sin(uk) * np.sin(ik))
    nav["svdt"] = nav.SVclockBias + tk * nav.SVclockDrift + tk**2 * nav.SVclockDriftRate

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
