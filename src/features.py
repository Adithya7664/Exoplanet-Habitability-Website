import numpy as np
import pandas as pd
from typing import Tuple, List

#reference values(perfect earth)
EARTH_RADIUS_RE=1.0
EARTH_MASS_ME=1.0
EARTH_EQT_K=255.0
EARTH_SURFACE_K=288.0
EARTH_DENSITY_GCM3=5.51
EARTH_ESCAPE_KMS=11.2
EARTH_INSOL=1.0

#ESI exponent weights
W_RADIUS=0.57
W_DENSITY=1.07
W_ESCAPE=0.70
W_TEMP=5.58

#ML model input features
FEATURE_COLS: List[str] = [
    "log_mass",
    "log_radius",
    "log_period",
    "insol_clipped",
    "stellar_temp_ratio",
    "stellar_lum_log",
    "temp_ratio",
    "density_ratio",
    "eccentricity",
    "st_logg_filled",
    "st_met_filled",
    "st_age_filled",
    "in_hz",
]

#ML model output features
FEATURE_LABELS: dict = {
    "log_mass": "Planet Mass (log)",
    "log_radius": "Planet Radius (log)",
    "log_period": "Orbital Period (log days)",
    "insol_clipped": "Insolation Flux",
    "stellar_temp_ratio": "Stellar Temp / Sun",
    "stellar_lum_log": "Stellar Luminosity (log Solar)",
    "temp_ratio": "Eq. Temp / Earth",
    "density_ratio": "Bulk Density / Earth",
    "eccentricity": "Orbital Eccentricity",
    "st_logg_filled": "Stellar Surface Gravity",
    "st_met_filled": "Stellar Metallicity [dex]",
    "st_age_filled": "Stellar Age [Gyr]",
    "in_hz": "In Habitable Zone",
}

def _esi_component(values: pd.Series, earth_val: float, weight: float) -> pd.Series:
    ratio=(values-earth_val).abs()/(values+earth_val)
    return (1-ratio)**weight

def compute_esi(df: pd.DataFrame)->pd.Series:
    radius = df["pl_rade"].fillna(EARTH_RADIUS_RE).clip(0.01, 100)
    mass = df["pl_bmasse"].fillna(EARTH_MASS_ME).clip(0.01, 50000)
    density = df["pl_dens"].fillna(
        (mass / (radius ** 3)) * EARTH_DENSITY_GCM3
    ).clip(0.01, 30)
    escape_vel = (mass / radius).apply(np.sqrt) * EARTH_ESCAPE_KMS
    temp = df["pl_eqt"].fillna(EARTH_EQT_K).clip(1, 10000)
    
    r_esi=_esi_component(radius, EARTH_RADIUS_RE, W_RADIUS)
    d_esi = _esi_component(density, EARTH_DENSITY_GCM3, W_DENSITY)
    e_esi = _esi_component(escape_vel, EARTH_ESCAPE_KMS, W_ESCAPE)
    t_esi = _esi_component(temp, EARTH_EQT_K, W_TEMP)
    esi=((r_esi * d_esi * e_esi * t_esi) ** 0.25).clip(0, 1)
    has_data = (
        df["pl_rade"].notna() | df["pl_bmasse"].notna() | df["pl_eqt"].notna()
    )
    return esi.where(has_data, np.nan)


def compute_esi_components(row: dict) -> dict:
    radius = float(row.get("pl_rade") or EARTH_RADIUS_RE)
    mass = float(row.get("pl_bmasse") or EARTH_MASS_ME)
    density = float(row.get("pl_dens") or (mass / max(radius, 0.01) ** 3) * EARTH_DENSITY_GCM3)
    escape_vel = (mass / max(radius, 0.01)) ** 0.5 * EARTH_ESCAPE_KMS
    temp= float(row.get("pl_eqt") or EARTH_EQT_K)
    
    def comp(v, ev, w):
        ratio=(v-ev)/(v+ev)
        return float((1-ratio)**w)
    
    return {
        "Radius": comp(max(radius, 0.01), EARTH_RADIUS_RE, W_RADIUS),
        "Density": comp(max(density, 0.01), EARTH_DENSITY_GCM3, W_DENSITY),
        "Escape Vel.": comp(max(escape_vel, 0.01), EARTH_ESCAPE_KMS, W_ESCAPE),
        "Temperature": comp(max(temp, 1), EARTH_EQT_K, W_TEMP),
        "Habitable Zone": float(bool(row.get("in_hz", 0))),
    }
    
def compute_hz(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    lum = (10 ** df["st_lum"].fillna(0.0)).clip(1e-6, 1e6)
    sma = df["pl_orbsmax"].fillna(1.0).clip(0.001, 1000)
    derived = lum / (sma ** 2)
    insol = df["pl_insol"].fillna(derived)
    in_hz = ((insol >= 0.36) & (insol <= 1.11)).astype(float)
    return in_hz, insol

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["esi"] = compute_esi(out)
    out["in_hz"], out["insol_flux"] = compute_hz(out)

    mass = out["pl_bmasse"].fillna(EARTH_MASS_ME).clip(0.01, 50000)
    radius = out["pl_rade"].fillna(EARTH_RADIUS_RE).clip(0.01, 100)
    density = out["pl_dens"].fillna(
        (mass / (radius ** 3)) * EARTH_DENSITY_GCM3
    ).clip(0.01, 30)

    out["log_mass"] = np.log1p(mass)
    out["log_radius"] = np.log1p(radius)
    out["log_period"] = np.log1p(out["pl_orbper"].fillna(365.0).clip(0.01, 1e6))
    out["insol_clipped"] = out["insol_flux"].clip(0, 200)
    out["stellar_temp_ratio"] = out["st_teff"].fillna(5778.0) / 5778.0
    out["stellar_lum_log"] = out["st_lum"].fillna(0.0)
    out["temp_ratio"] = out["pl_eqt"].fillna(EARTH_EQT_K) / EARTH_EQT_K
    out["density_ratio"] = density / EARTH_DENSITY_GCM3
    out["eccentricity"] = out["pl_orbeccen"].fillna(0.0)
    out["st_logg_filled"] = out["st_logg"].fillna(4.44)
    out["st_met_filled"] = out["st_met"].fillna(0.0)
    out["st_age_filled"] = out["st_age"].fillna(4.6)
    
    return out

