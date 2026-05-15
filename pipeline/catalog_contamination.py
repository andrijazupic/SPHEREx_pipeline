"""
catalog_contamination.py
════════════════════════
Multi-survey photometric contamination pipeline for SPHEREx target validation.

Detection strategy — Omni-Band Threat Detection:
  Each survey's catalog is searched within the SPHEREx pixel search radius.
  A neighbouring source is flagged as a contaminant if it exceeds the per-survey
  SNR threshold in ANY band that falls within red wavelength range
  (stored in CATALOG_CONFIGS['threat_bands']). Bluer bands (g, u) are fetched
  for target-identification bookkeeping but are explicitly excluded from the
  contamination decision — a bright UV/optical source that is invisible in the
  infrared cannot contaminate SPHEREx pixels.

"""

import time
import warnings
import logging
from astroquery.gaia import Gaia
from pyvo.dal import TAPService
from astroquery.ipac.irsa import Irsa
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np


# ── Catalog Registry ───────────────────────────────────────────────────────────
# all_bands    : every band fetched from the catalog query. Used for SNR
#                computation and for determining the target's best detection band.
# threat_bands: the physics-motivated subset of all_bands that fall within
#                SPHEREx's 0.75–5 µm range. Contamination is declared if a
#                neighbour clears min_snr in ANY of these bands (union logic).
CATALOG_CONFIGS = {
    'DESI': {
        'centering_tolerance':   2.0,
        'min_separation_arcsec': 1.0,
        'min_snr':               10.0,
        'all_bands':             ['z', 'r', 'g'],
        'mission_threat_bands': {
            'SPHEREx': ['z'],           # SPHEREx is strictly > 750nm
            'TESS':    ['z', 'r'],      # TESS spans 600 - 1000nm
            'PLATO':   ['z', 'r', 'g']  # PLATO spans 500 - 1000nm (needs g-band too!)
        }
    },
    'PanSTARRS': {
        'centering_tolerance':   2.0,
        'min_separation_arcsec': 1.0,
        'min_snr':               6.5,
        'all_bands':             ['y', 'z', 'i', 'r', 'g'],
        'mission_threat_bands': {
            'SPHEREx': ['y', 'z', 'i'], 
            'TESS':    ['y', 'z', 'i', 'r'],
            'PLATO':   ['y', 'z', 'i', 'r', 'g']
        }
    },
    'SDSS': {
        'centering_tolerance':   2.5,
        'min_separation_arcsec': 1.5,
        'min_snr':               7.0,
        'all_bands':             ['z', 'i', 'r', 'g', 'u'],
        'mission_threat_bands': {
            'SPHEREx': ['z', 'i'], 
            'TESS':    ['z', 'i', 'r'],
            'PLATO':   ['z', 'i', 'r', 'g'] # u-band (~350nm) is excluded for all
        }
    },
    'unWISE': {
        'centering_tolerance':   3.0,
        'min_separation_arcsec': 3.5,
        'min_snr':               9.0,
        'all_bands':             ['w2', 'w1'],
        'mission_threat_bands': {
            'SPHEREx': ['w2', 'w1'],    # Core Mid-IR SPHEREx bands
            'TESS':    [],              # Silicon is blind to Mid-IR
            'PLATO':   []               # Silicon is blind to Mid-IR
        }
    },
    'VHS': {
        'centering_tolerance':   2.0,
        'min_separation_arcsec': 1.0,
        'min_snr':               7.0,
        'all_bands':             ['Ks', 'H', 'J'],
        'mission_threat_bands': {
            'SPHEREx': ['Ks', 'H', 'J'], # Core Near-IR SPHEREx bands
            'TESS':    [],               # Silicon cutoff is ~1.0 µm. J is 1.25 µm.
            'PLATO':   []                # Silicon cutoff is ~1.0 µm. J is 1.25 µm.
        }
    },
    'DES': {
        'centering_tolerance':   2.0,
        'min_separation_arcsec': 1.0,
        'min_snr':               8.0,
        'all_bands':             ['y', 'z', 'i', 'r', 'g'],
        'mission_threat_bands': {
            'SPHEREx': ['y', 'z', 'i'],  
            'TESS':    ['y', 'z', 'i', 'r'],
            'PLATO':   ['y', 'z', 'i', 'r', 'g']
        }
    },
}

# Silence the "INFO: Query finished" spam from astroquery
logging.getLogger('astroquery').setLevel(logging.WARNING)


# ── Shared helper ──────────────────────────────────────────────────────────────

def _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr):
    """
    Build the per-contaminant description string for verbose log messages.
    For each contaminant, reports:
      • morphological type (Star / Galaxy / Unknown / IR_Source)
      • angular separation from the target (arcsec)
      • every SPHEREx-relevant band whose SNR cleared the threshold, with its
        individual SNR value — e.g.  'Galaxy (4.21", [z:12.3, i:9.8])'
    """
    result = []
    for _, row in contaminants.iterrows():
        triggered = ', '.join(
            f"{b}:{row[f'snr_{b}']:.1f}"
            for b in threat_bands
            if row[f'snr_{b}'] >= min_snr
        )
        result.append(f"{row['type_clean']} ({row[dist_col]:.2f}\", [{triggered}])")
    return result


# ── Gaia DR3 ───────────────────────────────────────────────────────────────────

def remove_gaia_blends(df, search_radius_arcsec=9.3, max_retries=3, verbose=False):
    """
    Flags candidates with neighbouring sources in Gaia DR3.
    Uses a direct ADQL cone count: any extra Gaia source within the search
    radius is treated as a potential SPHEREx contaminant regardless of its
    flux or colour, since Gaia is our most complete all-sky astrometric reference.
    """
    total_sources = len(df)

    if verbose:
        print(f"Checking {total_sources} candidates against Gaia DR3 (ADQL cone count)...\n")

    is_contaminated = []
    notes           = []
    radius_deg = search_radius_arcsec / 3600.0

    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sid = row['source_id']
        ra  = row['ra']
        dec = row['dec']

        query = f"""
            SELECT source_id
            FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
        """

        success  = False
        attempts = 0

        while not success and attempts < max_retries:
            try:
                job           = Gaia.launch_job(query)
                cone_data     = job.get_results()
                total_objects = len(cone_data)

                # Safe State Assignment
                if total_objects == 0:
                    temp_is_contam = False
                    msg = f"⚠️ Source {sid}: No Gaia data found (outside footprint). Kept as clean."
                elif total_objects == 1:
                    temp_is_contam = False
                    msg = f"🟢 Source {sid}: Clean isolated target. Found 0 extra neighbours."
                else:
                    temp_is_contam = True
                    msg = f"🔴 Flagging Source {sid}: Found {total_objects - 1} extra Gaia neighbour(s)."

                if verbose: print(f"[{i}/{total_sources}] {msg}")
                
                # Append strictly at the end of the try block
                is_contaminated.append(temp_is_contam)
                notes.append(msg)
                success = True

            except Exception as e:
                attempts += 1
                if attempts < max_retries:
                    time.sleep(2)
                else:
                    msg = f"❌ Gave up on Source {sid} after {max_retries} attempts. Kept as clean. Error: {e}"
                    print(msg)
                    is_contaminated.append(False)
                    notes.append(msg)

        time.sleep(0.5)

    df = df.copy()
    df['is_contaminated_gaia'] = is_contaminated
    df['note_gaia']            = notes

    clean_count   = (~df['is_contaminated_gaia']).sum()
    flagged_count =   df['is_contaminated_gaia'].sum()

    if verbose:
        print(f"\n✅ Flagging complete!")
        print(f"Started with: {len(df)} sources")
        print(f"Clean:        {clean_count} isolated sources")
        print(f"Flagged:      {flagged_count} blended/error sources")

    return df


# ── DESI DR10 ──────────────────────────────────────────────────────────────────

def remove_desi_blends(df, target_mission='SPHEREx', search_radius_arcsec=9.3, max_retries=3,
                       ignore_psf_contaminants=False, verbose=False):
    """
    Flags candidates with neighbouring sources in the DESI DR10 Tractor catalog.
    Omni-Band Threat Detection: a neighbour is flagged if it exceeds min_snr in
    ANY of the SPHEREx-relevant bands. 
    """
    centering_tolerance   = CATALOG_CONFIGS['DESI']['centering_tolerance']
    min_separation_arcsec = CATALOG_CONFIGS['DESI']['min_separation_arcsec']
    min_snr               = CATALOG_CONFIGS['DESI']['min_snr']
    all_bands             = CATALOG_CONFIGS['DESI']['all_bands']
    threat_bands          = CATALOG_CONFIGS['DESI']['mission_threat_bands'][target_mission]

    tap_service          = TAPService("https://datalab.noirlab.edu/tap")
    total_sources        = len(df)
    query_radius_arcsec  = search_radius_arcsec + centering_tolerance
    radius_deg           = query_radius_arcsec / 3600.0

    if verbose:
        print(f"Checking {total_sources} candidates against DESI DR10 Tractor "
              f"(checking {search_radius_arcsec}\" around target)...\n")
        if ignore_psf_contaminants:
            print("-> Ignoring PSF contaminants (Only flagging extended sources: REX, EXP, DEV, SER).")
        print(f"-> SPHEREx-relevant bands checked for contamination: {', '.join(threat_bands)}")
        print(f"-> Flagging neighbours detected in ANY band above SNR = {min_snr}")
        print(f"-> Ignoring ultra-close de-blending artefacts at distance < {min_separation_arcsec}\"")

    is_contaminated = []
    notes           = []

    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sid = row['source_id']
        ra  = row['ra']
        dec = row['dec']

        query = f"""
            SELECT objid, ra, dec, type,
                   flux_z, flux_ivar_z, flux_r, flux_ivar_r, flux_g, flux_ivar_g,
                   Q3C_DIST(ra, dec, {ra}, {dec}) * 3600.0 AS dist_arcsec
            FROM   ls_dr10.tractor
            WHERE  't' = Q3C_RADIAL_QUERY(ra, dec, {ra}, {dec}, {radius_deg})
        """

        success      = False
        attempts     = 0
        target_found = False

        while not success and attempts < max_retries:
            try:
                result    = tap_service.search(query)
                cone_data = result.to_table().to_pandas()
                n_total   = len(cone_data)

                if n_total > 0:
                    cone_data['dist_arcsec'] = pd.to_numeric(cone_data['dist_arcsec'], errors='coerce')
                    cone_data['type_clean']  = cone_data['type'].apply(
                        lambda t: t.decode('utf-8').strip() if isinstance(t, bytes) else str(t).strip()
                    )

                    # ── SNR PER BAND (all fetched bands) ──
                    for b in all_bands:
                        f    = pd.to_numeric(cone_data[f'flux_{b}'],      errors='coerce').fillna(0)
                        ivar = pd.to_numeric(cone_data[f'flux_ivar_{b}'], errors='coerce').fillna(0)
                        cone_data[f'snr_{b}'] = f * np.sqrt(np.clip(ivar, 0, None))

                    # best_snr/best_band: reddest band with any detection — TARGET LOGGING ONLY
                    conditions             = [cone_data[f'snr_{b}'] > 0 for b in all_bands]
                    cone_data['best_snr']  = np.select(conditions, [cone_data[f'snr_{b}'] for b in all_bands], default=0)
                    cone_data['best_band'] = np.select(conditions, all_bands, default='none')

                    # ── NEAREST NEIGHBOUR LOGIC ──
                    min_idx      = cone_data['dist_arcsec'].idxmin()
                    min_dist     = cone_data.loc[min_idx, 'dist_arcsec']
                    target_found = min_dist <= centering_tolerance

                    if target_found:
                        target_coord = SkyCoord(ra=cone_data.loc[min_idx, 'ra'],
                                                dec=cone_data.loc[min_idx, 'dec'],
                                                unit=(u.deg, u.deg), frame='icrs')
                        cat_coords   = SkyCoord(ra=cone_data['ra'].values,
                                                dec=cone_data['dec'].values,
                                                unit=(u.deg, u.deg), frame='icrs')
                        cone_data['sep_from_target'] = target_coord.separation(cat_coords).arcsec
                        contaminants = cone_data[
                            (cone_data['objid'] != cone_data.loc[min_idx, 'objid']) &
                            (cone_data['sep_from_target'] <= search_radius_arcsec)
                        ]
                        dist_col                = 'sep_from_target'
                        target_snr, target_band = (cone_data.loc[min_idx, 'best_snr'],
                                                   cone_data.loc[min_idx, 'best_band'])
                    else:
                        contaminants = cone_data[cone_data['dist_arcsec'] <= search_radius_arcsec]
                        dist_col     = 'dist_arcsec'

                    # ── THE FILTERS ──
                    if ignore_psf_contaminants:
                        contaminants = contaminants[contaminants['type_clean'] != 'PSF']

                    # ── OMNI-BAND SPHEREx CONTAMINATION DETECTION ──
                    contaminants   = contaminants[
                        contaminants[[f'snr_{b}' for b in threat_bands]].ge(min_snr).any(axis=1)
                    ]
                    contaminants   = contaminants[contaminants[dist_col] >= min_separation_arcsec]
                    n_contaminants = len(contaminants)

                else:
                    n_contaminants = 0
                    contaminants   = pd.DataFrame()

                # ── SAFE DECISION BLOCK ──
                if n_contaminants == 0:
                    temp_is_contam = False
                    if n_total == 0:
                        msg = f"⚠️ Source {sid}: No data found (likely outside footprint). Kept as clean."
                    elif not target_found:
                        msg = f"⚠️ Source {sid}: Target not found in field. No other sources found nearby. Kept as clean."
                    else:
                        msg = f"🟢 Source {sid}: Clean. (Target SNR_{target_band}: {target_snr:.1f})"
                else:
                    if not target_found:
                        if n_contaminants == 1:
                            temp_is_contam = False
                            msg = f"⚠️ Source {sid}: Target missing AND {n_contaminants} DESI neighbour found. Kept as clean."
                        else:
                            temp_is_contam = True
                            t_list = _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr)
                            msg = f"🔴 Flagging Source {sid}: Target missing AND {n_contaminants} DESI neighbour(s) found {t_list}."
                    else:
                        temp_is_contam = True
                        t_list = _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr)
                        msg = f"🔴 Flagging Source {sid}: {n_contaminants} DESI neighbour(s) found {t_list}."
                        
                if verbose: print(f"[{i}/{total_sources}] {msg}")
                
                # Append securely
                is_contaminated.append(temp_is_contam)
                notes.append(msg)
                success = True

            except Exception as e:
                attempts += 1
                if attempts < max_retries:
                    time.sleep(2)
                else:
                    msg = f"❌ Gave up on Source {sid} after {max_retries} attempts. Kept as clean. Error: {e}"
                    print(msg)
                    is_contaminated.append(False)
                    notes.append(msg)

        time.sleep(0.5)

    df = df.copy()
    df['is_contaminated_desi'] = is_contaminated
    df['note_desi']            = notes

    clean_count   = (~df['is_contaminated_desi']).sum()
    flagged_count =   df['is_contaminated_desi'].sum()

    if verbose:
        print(f"\n✅ Flagging complete!")
        print(f"Started with: {len(df)} sources")
        print(f"Clean:        {clean_count} isolated sources")
        print(f"Flagged:      {flagged_count} blended/error sources")

    return df


# ── Pan-STARRS DR1 ─────────────────────────────────────────────────────────────

def remove_panstarrs_blends(df, target_mission='SPHEREx', search_radius_arcsec=9.3, max_retries=3,
                            ignore_psf_contaminants=False, verbose=False):
    """
    Flags candidates with neighbouring sources in Pan-STARRS DR1.
    Omni-Band Threat Detection. Morphology is assessed via the PSF–Kron magnitude 
    difference with conservative union logic.
    """
    centering_tolerance   = CATALOG_CONFIGS['PanSTARRS']['centering_tolerance']
    min_separation_arcsec = CATALOG_CONFIGS['PanSTARRS']['min_separation_arcsec']
    min_snr               = CATALOG_CONFIGS['PanSTARRS']['min_snr']
    all_bands             = CATALOG_CONFIGS['PanSTARRS']['all_bands']
    threat_bands          = CATALOG_CONFIGS['PanSTARRS']['mission_threat_bands'][target_mission]

    tap_service         = TAPService("https://tapvizier.u-strasbg.fr/TAPVizieR/tap")
    total_sources       = len(df)
    query_radius_arcsec = search_radius_arcsec + centering_tolerance
    radius_deg          = query_radius_arcsec / 3600.0

    if verbose:
        print(f"Checking {total_sources} candidates against Pan-STARRS DR1 "
              f"(checking {search_radius_arcsec}\" around target)...\n")
        if ignore_psf_contaminants:
            print("-> Ignoring PSF contaminants (Only flagging extended sources/Unknowns).")
        print(f"-> SPHEREx-relevant bands checked for contamination: {', '.join(threat_bands)}")
        print(f"-> Flagging neighbours detected in ANY band above SNR = {min_snr}")
        print(f"-> Ignoring ultra-close de-blending artefacts at distance < {min_separation_arcsec}\"")

    is_contaminated = []
    notes           = []

    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sid = row['source_id']
        ra  = row['ra']
        dec = row['dec']

        query = f"""
            SELECT objID, RAJ2000 as ra, DEJ2000 as dec,
                   ymag, e_ymag, yKmag, zmag, e_zmag, zKmag,
                   imag, e_imag, iKmag, rmag, e_rmag, rKmag,
                   gmag, e_gmag, gKmag
            FROM "II/349/ps1"
            WHERE 1=CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
        """

        success      = False
        attempts     = 0
        target_found = False

        while not success and attempts < max_retries:
            try:
                result    = tap_service.search(query)
                cone_data = result.to_table().to_pandas()
                n_total   = len(cone_data)

                if n_total > 0:
                    theoretical_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
                    cat_coords        = SkyCoord(ra=cone_data['ra'].values,
                                                 dec=cone_data['dec'].values,
                                                 unit=(u.deg, u.deg), frame='icrs')
                    cone_data['dist_arcsec'] = theoretical_coord.separation(cat_coords).arcsec

                    # ── SNR AND MORPHOLOGY PER BAND ──
                    for b in all_bands:
                        e_mag = pd.to_numeric(cone_data[f'e_{b}mag'], errors='coerce').fillna(0)
                        mag   = pd.to_numeric(cone_data[f'{b}mag'],   errors='coerce').fillna(np.nan)
                        kmag  = pd.to_numeric(cone_data[f'{b}Kmag'],  errors='coerce').fillna(np.nan)
                        cone_data[f'snr_{b}'] = np.where(e_mag > 0, 1.0857 / e_mag, 0)
                        
                        cone_data[f'ext_{b}'] = np.where(
                            np.isfinite(mag) & np.isfinite(kmag),
                            (mag - kmag) > 0.05,
                            np.nan
                        )

                    # best_snr/best_band — TARGET LOGGING ONLY
                    conditions             = [cone_data[f'snr_{b}'] > 0 for b in all_bands]
                    cone_data['best_snr']  = np.select(conditions, [cone_data[f'snr_{b}'] for b in all_bands], default=0)
                    cone_data['best_band'] = np.select(conditions, all_bands, default='none')

                    # ── MORPHOLOGY: CONSERVATIVE UNION ACROSS SPHEREx BANDS ──
                    ext_df           = cone_data[[f'ext_{b}' for b in threat_bands]]
                    has_any_extended = (ext_df == True).any(axis=1)
                    has_any_stellar  = (ext_df == False).any(axis=1)
                    cone_data['type_clean'] = np.where(has_any_extended, 'Galaxy',
                                             np.where(has_any_stellar,   'Star', 'Unknown'))

                    # ── NEAREST NEIGHBOUR LOGIC ──
                    min_idx      = cone_data['dist_arcsec'].idxmin()
                    min_dist     = cone_data.loc[min_idx, 'dist_arcsec']
                    target_found = min_dist <= centering_tolerance

                    if target_found:
                        target_coord = SkyCoord(ra=cone_data.loc[min_idx, 'ra'],
                                                dec=cone_data.loc[min_idx, 'dec'],
                                                unit=(u.deg, u.deg), frame='icrs')
                        cone_data['sep_from_target'] = target_coord.separation(cat_coords).arcsec
                        contaminants = cone_data[
                            (cone_data['objID'] != cone_data.loc[min_idx, 'objID']) &
                            (cone_data['sep_from_target'] <= search_radius_arcsec)
                        ]
                        dist_col                = 'sep_from_target'
                        target_snr, target_band = (cone_data.loc[min_idx, 'best_snr'],
                                                   cone_data.loc[min_idx, 'best_band'])
                    else:
                        contaminants = cone_data[cone_data['dist_arcsec'] <= search_radius_arcsec]
                        dist_col     = 'dist_arcsec'

                    # ── THE FILTERS ──
                    if ignore_psf_contaminants:
                        contaminants = contaminants[contaminants['type_clean'] != 'Star']

                    # ── OMNI-BAND SPHEREx CONTAMINATION DETECTION ──
                    contaminants   = contaminants[
                        contaminants[[f'snr_{b}' for b in threat_bands]].ge(min_snr).any(axis=1)
                    ]
                    contaminants   = contaminants[contaminants[dist_col] >= min_separation_arcsec]
                    n_contaminants = len(contaminants)

                else:
                    n_contaminants = 0
                    contaminants   = pd.DataFrame()

                # ── SAFE DECISION BLOCK ──
                if n_contaminants == 0:
                    temp_is_contam = False
                    if n_total == 0:
                        msg = f"⚠️ Source {sid}: No data found (likely outside footprint). Kept as clean."
                    elif not target_found:
                        msg = f"⚠️ Source {sid}: Target not found in field. No other sources found nearby. Kept as clean."
                    else:
                        msg = f"🟢 Source {sid}: Clean. (Target SNR_{target_band}: {target_snr:.1f})"
                else:
                    if not target_found:
                        if n_contaminants == 1:
                            temp_is_contam = False
                            msg = f"⚠️ Source {sid}: Target missing AND {n_contaminants} PanSTARRS neighbour found. Kept as clean."
                        else:
                            temp_is_contam = True
                            t_list = _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr)
                            msg = f"🔴 Flagging Source {sid}: Target missing AND {n_contaminants} PanSTARRS neighbour(s) found {t_list}."
                    else:
                        temp_is_contam = True
                        t_list = _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr)
                        msg = f"🔴 Flagging Source {sid}: {n_contaminants} PanSTARRS neighbour(s) found {t_list}."
                        
                if verbose: print(f"[{i}/{total_sources}] {msg}")
                
                is_contaminated.append(temp_is_contam)
                notes.append(msg)
                success = True

            except Exception as e:
                attempts += 1
                if attempts < max_retries:
                    time.sleep(2)
                else:
                    msg = f"❌ Gave up on Source {sid} after {max_retries} attempts. Kept as clean. Error: {e}"
                    print(msg)
                    is_contaminated.append(False)
                    notes.append(msg)

        time.sleep(0.5)

    df = df.copy()
    df['is_contaminated_panstarrs'] = is_contaminated
    df['note_panstarrs']            = notes

    clean_count   = (~df['is_contaminated_panstarrs']).sum()
    flagged_count =   df['is_contaminated_panstarrs'].sum()

    if verbose:
        print(f"\n✅ Flagging complete!")
        print(f"Started with: {len(df)} sources")
        print(f"Clean:        {clean_count} isolated sources")
        print(f"Flagged:      {flagged_count} blended/error sources")

    return df


# ── SDSS DR16 ──────────────────────────────────────────────────────────────────

def remove_sdss_blends(df, target_mission='SPHEREx', search_radius_arcsec=9.3, max_retries=3,
                       ignore_psf_contaminants=False, verbose=False):
    """
    Flags candidates with neighbouring sources in SDSS DR16.
    Omni-Band Threat Detection. Unclassified objects (NaN class) are 
    retained as 'Unknown'.
    """
    centering_tolerance   = CATALOG_CONFIGS['SDSS']['centering_tolerance']
    min_separation_arcsec = CATALOG_CONFIGS['SDSS']['min_separation_arcsec']
    min_snr               = CATALOG_CONFIGS['SDSS']['min_snr']
    all_bands             = CATALOG_CONFIGS['SDSS']['all_bands']
    threat_bands          = CATALOG_CONFIGS['SDSS']['mission_threat_bands'][target_mission]

    tap_service         = TAPService("https://tapvizier.u-strasbg.fr/TAPVizieR/tap")
    total_sources       = len(df)
    query_radius_arcsec = search_radius_arcsec + centering_tolerance
    radius_deg          = query_radius_arcsec / 3600.0

    if verbose:
        print(f"Checking {total_sources} candidates against SDSS DR16 "
              f"(checking {search_radius_arcsec}\" around target)...\n")
        if ignore_psf_contaminants:
            print("-> Ignoring PSF contaminants (Only flagging class=3 Galaxies/Unknowns).")
        print(f"-> SPHEREx-relevant bands checked for contamination: {', '.join(threat_bands)}")
        print(f"-> Flagging neighbours detected in ANY band above SNR = {min_snr}")
        print(f"-> Ignoring ultra-close de-blending artefacts at distance < {min_separation_arcsec}\"")

    is_contaminated = []
    notes           = []

    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sid = row['source_id']
        ra  = row['ra']
        dec = row['dec']

        query = f"""
            SELECT objID, RA_ICRS as ra, DE_ICRS as dec, class,
                   zmag, e_zmag, imag, e_imag, rmag, e_rmag,
                   gmag, e_gmag, umag, e_umag
            FROM "V/154/sdss16"
            WHERE 1=CONTAINS(POINT('ICRS', RA_ICRS, DE_ICRS), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
        """

        success      = False
        attempts     = 0
        target_found = False

        while not success and attempts < max_retries:
            try:
                result    = tap_service.search(query)
                cone_data = result.to_table().to_pandas()
                n_total   = len(cone_data)

                if n_total > 0:
                    theoretical_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
                    cat_coords        = SkyCoord(ra=cone_data['ra'].values,
                                                 dec=cone_data['dec'].values,
                                                 unit=(u.deg, u.deg), frame='icrs')
                    cone_data['dist_arcsec'] = theoretical_coord.separation(cat_coords).arcsec
                    
                    cone_data['type_clean']  = cone_data['class'].map({3: 'Galaxy', 6: 'Star'}).fillna('Unknown')

                    # ── SNR PER BAND ──
                    for b in all_bands:
                        e_mag = pd.to_numeric(cone_data[f'e_{b}mag'], errors='coerce').fillna(0)
                        cone_data[f'snr_{b}'] = np.where(e_mag > 0, 1.0857 / e_mag, 0)

                    # best_snr/best_band — TARGET LOGGING ONLY
                    conditions             = [cone_data[f'snr_{b}'] > 0 for b in all_bands]
                    cone_data['best_snr']  = np.select(conditions, [cone_data[f'snr_{b}'] for b in all_bands], default=0)
                    cone_data['best_band'] = np.select(conditions, all_bands, default='none')

                    # ── NEAREST NEIGHBOUR LOGIC ──
                    min_idx      = cone_data['dist_arcsec'].idxmin()
                    min_dist     = cone_data.loc[min_idx, 'dist_arcsec']
                    target_found = min_dist <= centering_tolerance

                    if target_found:
                        target_coord = SkyCoord(ra=cone_data.loc[min_idx, 'ra'],
                                                dec=cone_data.loc[min_idx, 'dec'],
                                                unit=(u.deg, u.deg), frame='icrs')
                        cone_data['sep_from_target'] = target_coord.separation(cat_coords).arcsec
                        contaminants = cone_data[
                            (cone_data['objID'] != cone_data.loc[min_idx, 'objID']) &
                            (cone_data['sep_from_target'] <= search_radius_arcsec)
                        ]
                        dist_col                = 'sep_from_target'
                        target_snr, target_band = (cone_data.loc[min_idx, 'best_snr'],
                                                   cone_data.loc[min_idx, 'best_band'])
                    else:
                        contaminants = cone_data[cone_data['dist_arcsec'] <= search_radius_arcsec]
                        dist_col     = 'dist_arcsec'

                    # ── THE FILTERS ──
                    if ignore_psf_contaminants:
                        contaminants = contaminants[contaminants['type_clean'] != 'Star']

                    # ── OMNI-BAND SPHEREx CONTAMINATION DETECTION ──
                    contaminants   = contaminants[
                        contaminants[[f'snr_{b}' for b in threat_bands]].ge(min_snr).any(axis=1)
                    ]
                    contaminants   = contaminants[contaminants[dist_col] >= min_separation_arcsec]
                    n_contaminants = len(contaminants)

                else:
                    n_contaminants = 0
                    contaminants   = pd.DataFrame()

                # ── SAFE DECISION BLOCK ──
                if n_contaminants == 0:
                    temp_is_contam = False
                    if n_total == 0:
                        msg = f"⚠️ Source {sid}: No data found (likely outside footprint). Kept as clean."
                    elif not target_found:
                        msg = f"⚠️ Source {sid}: Target not found in field. No other sources found nearby. Kept as clean."
                    else:
                        msg = f"🟢 Source {sid}: Clean. (Target SNR_{target_band}: {target_snr:.1f})"
                else:
                    if not target_found:
                        if n_contaminants == 1:
                            temp_is_contam = False
                            msg = f"⚠️ Source {sid}: Target missing AND {n_contaminants} SDSS neighbour found. Kept as clean."
                        else:
                            temp_is_contam = True
                            t_list = _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr)
                            msg = f"🔴 Flagging Source {sid}: Target missing AND {n_contaminants} SDSS neighbour(s) found {t_list}."
                    else:
                        temp_is_contam = True
                        t_list = _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr)
                        msg = f"🔴 Flagging Source {sid}: {n_contaminants} SDSS neighbour(s) found {t_list}."
                        
                if verbose: print(f"[{i}/{total_sources}] {msg}")
                
                is_contaminated.append(temp_is_contam)
                notes.append(msg)
                success = True

            except Exception as e:
                attempts += 1
                if attempts < max_retries:
                    time.sleep(2)
                else:
                    msg = f"❌ Gave up on Source {sid} after {max_retries} attempts. Kept as clean. Error: {e}"
                    print(msg)
                    is_contaminated.append(False)
                    notes.append(msg)

        time.sleep(0.5)

    df = df.copy()
    df['is_contaminated_sdss'] = is_contaminated
    df['note_sdss']            = notes

    clean_count   = (~df['is_contaminated_sdss']).sum()
    flagged_count =   df['is_contaminated_sdss'].sum()

    if verbose:
        print(f"\n✅ Flagging complete!")
        print(f"Started with: {len(df)} sources")
        print(f"Clean:        {clean_count} isolated sources")
        print(f"Flagged:      {flagged_count} blended/error sources")

    return df


# ── unWISE ─────────────────────────────────────────────────────────────────────

def remove_unwise_blends(df, target_mission='SPHEREx', search_radius_arcsec=9.3, max_retries=3,
                         ignore_psf_contaminants=False, verbose=False):
    """
    Flags candidates with neighbouring sources in the unWISE Catalog.
    Omni-Band Threat Detection across W2 and W1. 
    """
    centering_tolerance   = CATALOG_CONFIGS['unWISE']['centering_tolerance']
    min_separation_arcsec = CATALOG_CONFIGS['unWISE']['min_separation_arcsec']
    min_snr               = CATALOG_CONFIGS['unWISE']['min_snr']
    all_bands             = CATALOG_CONFIGS['unWISE']['all_bands']
    threat_bands          = CATALOG_CONFIGS['unWISE']['mission_threat_bands'][target_mission]

    if ignore_psf_contaminants:
        warnings.warn(
            "unWISE does not provide morphological classification. "
            "'ignore_psf_contaminants' has no effect.",
            UserWarning, stacklevel=2
        )

    tap_service         = TAPService("https://datalab.noirlab.edu/tap")
    total_sources       = len(df)
    query_radius_arcsec = search_radius_arcsec + centering_tolerance
    radius_deg          = query_radius_arcsec / 3600.0

    if verbose:
        print(f"Checking {total_sources} candidates against unWISE "
              f"(checking {search_radius_arcsec}\" around target)...\n")
        print(f"-> SPHEREx-relevant bands checked for contamination: {', '.join(threat_bands)}")
        print(f"-> Flagging neighbours detected in ANY band above SNR = {min_snr}")
        print(f"-> Ignoring ultra-close de-blending artefacts at distance < {min_separation_arcsec}\"")

    is_contaminated = []
    notes           = []

    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sid = row['source_id']
        ra  = row['ra']
        dec = row['dec']

        query = f"""
            SELECT unwise_objid, ra, dec,
                   flux_w1, dflux_w1, flux_w2, dflux_w2,
                   Q3C_DIST(ra, dec, {ra}, {dec}) * 3600.0 AS dist_arcsec
            FROM   unwise_dr1.object
            WHERE  't' = Q3C_RADIAL_QUERY(ra, dec, {ra}, {dec}, {radius_deg})
        """

        success      = False
        attempts     = 0
        target_found = False

        while not success and attempts < max_retries:
            try:
                result    = tap_service.search(query)
                cone_data = result.to_table().to_pandas()
                n_total   = len(cone_data)

                if n_total > 0:
                    cone_data['dist_arcsec'] = pd.to_numeric(cone_data['dist_arcsec'], errors='coerce')
                    cone_data['type_clean']  = 'IR_Source'

                    # ── SNR PER BAND ──
                    for b in all_bands:
                        f   = pd.to_numeric(cone_data[f'flux_{b}'],  errors='coerce').fillna(0)
                        err = pd.to_numeric(cone_data[f'dflux_{b}'], errors='coerce').fillna(0)
                        cone_data[f'snr_{b}'] = np.where(err > 0, f / err, 0)

                    # best_snr/best_band — TARGET LOGGING ONLY
                    conditions             = [cone_data[f'snr_{b}'] > 0 for b in all_bands]
                    cone_data['best_snr']  = np.select(conditions, [cone_data[f'snr_{b}'] for b in all_bands], default=0)
                    cone_data['best_band'] = np.select(conditions, all_bands, default='none')

                    # ── NEAREST NEIGHBOUR LOGIC ──
                    min_idx      = cone_data['dist_arcsec'].idxmin()
                    min_dist     = cone_data.loc[min_idx, 'dist_arcsec']
                    target_found = min_dist <= centering_tolerance

                    if target_found:
                        target_coord = SkyCoord(ra=cone_data.loc[min_idx, 'ra'],
                                                dec=cone_data.loc[min_idx, 'dec'],
                                                unit=(u.deg, u.deg), frame='icrs')
                        cat_coords   = SkyCoord(ra=cone_data['ra'].values,
                                                dec=cone_data['dec'].values,
                                                unit=(u.deg, u.deg), frame='icrs')
                        cone_data['sep_from_target'] = target_coord.separation(cat_coords).arcsec
                        contaminants = cone_data[
                            (cone_data['unwise_objid'] != cone_data.loc[min_idx, 'unwise_objid']) &
                            (cone_data['sep_from_target'] <= search_radius_arcsec)
                        ]
                        dist_col                = 'sep_from_target'
                        target_snr, target_band = (cone_data.loc[min_idx, 'best_snr'],
                                                   cone_data.loc[min_idx, 'best_band'])
                    else:
                        contaminants = cone_data[cone_data['dist_arcsec'] <= search_radius_arcsec]
                        dist_col     = 'dist_arcsec'

                    # ── OMNI-BAND SPHEREx CONTAMINATION DETECTION ──
                    contaminants   = contaminants[
                        contaminants[[f'snr_{b}' for b in threat_bands]].ge(min_snr).any(axis=1)
                    ]
                    contaminants   = contaminants[contaminants[dist_col] >= min_separation_arcsec]
                    n_contaminants = len(contaminants)

                else:
                    n_contaminants = 0
                    contaminants   = pd.DataFrame()

                # ── SAFE DECISION BLOCK ──
                if n_contaminants == 0:
                    temp_is_contam = False
                    if n_total == 0:
                        msg = f"⚠️ Source {sid}: No data found (likely outside footprint). Kept as clean."
                    elif not target_found:
                        msg = f"⚠️ Source {sid}: Target not found in field. No other sources found nearby. Kept as clean."
                    else:
                        msg = f"🟢 Source {sid}: Clean. (Target SNR_{target_band}: {target_snr:.1f})"
                else:
                    if not target_found:
                        if n_contaminants == 1:
                            temp_is_contam = False
                            msg = f"⚠️ Source {sid}: Target missing AND {n_contaminants} unWISE neighbour found. Kept as clean."
                        else:
                            temp_is_contam = True
                            t_list = _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr)
                            msg = f"🔴 Flagging Source {sid}: Target missing AND {n_contaminants} unWISE neighbour(s) found {t_list}."
                    else:
                        temp_is_contam = True
                        t_list = _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr)
                        msg = f"🔴 Flagging Source {sid}: {n_contaminants} unWISE neighbour(s) found {t_list}."
                        
                if verbose: print(f"[{i}/{total_sources}] {msg}")
                
                is_contaminated.append(temp_is_contam)
                notes.append(msg)
                success = True

            except Exception as e:
                attempts += 1
                if attempts < max_retries:
                    time.sleep(2)
                else:
                    msg = f"❌ Gave up on Source {sid} after {max_retries} attempts. Kept as clean. Error: {e}"
                    print(msg)
                    is_contaminated.append(False)
                    notes.append(msg)

        time.sleep(0.5)

    df = df.copy()
    df['is_contaminated_unwise'] = is_contaminated
    df['note_unwise']            = notes

    clean_count   = (~df['is_contaminated_unwise']).sum()
    flagged_count =   df['is_contaminated_unwise'].sum()

    if verbose:
        print(f"\n✅ Flagging complete!")
        print(f"Started with: {len(df)} sources")
        print(f"Clean:        {clean_count} isolated sources")
        print(f"Flagged:      {flagged_count} blended/error sources")

    return df


# ── VHS DR5 ────────────────────────────────────────────────────────────────────

def remove_vhs_blends(df, target_mission='SPHEREx', search_radius_arcsec=9.3, max_retries=3,
                      ignore_psf_contaminants=False, verbose=False):
    """
    Flags candidates with neighbouring sources in the VISTA VHS DR5.
    Omni-Band Threat Detection. 
    (Fixed Schema to safely request Jap3, Hap3, Ksap3 default aperture photometry).
    """
    centering_tolerance   = CATALOG_CONFIGS['VHS']['centering_tolerance']
    min_separation_arcsec = CATALOG_CONFIGS['VHS']['min_separation_arcsec']
    min_snr               = CATALOG_CONFIGS['VHS']['min_snr']
    all_bands             = CATALOG_CONFIGS['VHS']['all_bands']
    threat_bands          = CATALOG_CONFIGS['VHS']['mission_threat_bands'][target_mission]

    tap_service         = TAPService("https://tapvizier.u-strasbg.fr/TAPVizieR/tap")
    total_sources       = len(df)
    query_radius_arcsec = search_radius_arcsec + centering_tolerance
    radius_deg          = query_radius_arcsec / 3600.0

    if verbose:
        print(f"Checking {total_sources} candidates against VHS DR5 "
              f"(checking {search_radius_arcsec}\" around target)...\n")
        if ignore_psf_contaminants:
            print("-> Ignoring PSF contaminants (Only flagging extended sources/Unknowns).")
        print(f"-> SPHEREx-relevant bands checked for contamination: {', '.join(threat_bands)}")
        print(f"-> Flagging neighbours detected in ANY band above SNR = {min_snr}")
        print(f"-> Ignoring ultra-close de-blending artefacts at distance < {min_separation_arcsec}\"")

    is_contaminated = []
    notes           = []

    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sid = row['source_id']
        ra  = row['ra']
        dec = row['dec']

        # [FIXED] Safely restored VizieR schema aliases to prevent ADQL crash
        query = f"""
            SELECT SrcID as objid, RAJ2000 as ra, DEJ2000 as dec,
                   Jap3 as Jmag, e_Jap3 as e_Jmag, Hap3 as Hmag, e_Hap3 as e_Hmag, Ksap3 as Ksmag, e_Ksap3 as e_Ksmag,
                   pStar
            FROM "II/367/vhs_dr5"
            WHERE 1=CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
        """

        success      = False
        attempts     = 0
        target_found = False

        while not success and attempts < max_retries:
            try:
                result    = tap_service.search(query)
                cone_data = result.to_table().to_pandas()
                n_total   = len(cone_data)

                if n_total > 0:
                    theoretical_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
                    cat_coords        = SkyCoord(ra=cone_data['ra'].values,
                                                 dec=cone_data['dec'].values,
                                                 unit=(u.deg, u.deg), frame='icrs')
                    cone_data['dist_arcsec'] = theoretical_coord.separation(cat_coords).arcsec

                    # pStar NaN → Unknown (not Galaxy) to avoid masking unclassified sources
                    cone_data['type_clean'] = np.where(cone_data['pStar'].isna(),    'Unknown',
                                             np.where(cone_data['pStar'] > 0.9, 'Star', 'Galaxy'))

                    # ── SNR PER BAND ──
                    for b in all_bands:
                        e_mag = pd.to_numeric(cone_data[f'e_{b}mag'], errors='coerce').fillna(0)
                        cone_data[f'snr_{b}'] = np.where(e_mag > 0, 1.0857 / e_mag, 0)

                    # best_snr/best_band — TARGET LOGGING ONLY
                    conditions             = [cone_data[f'snr_{b}'] > 0 for b in all_bands]
                    cone_data['best_snr']  = np.select(conditions, [cone_data[f'snr_{b}'] for b in all_bands], default=0)
                    cone_data['best_band'] = np.select(conditions, all_bands, default='none')

                    # ── NEAREST NEIGHBOUR LOGIC ──
                    min_idx      = cone_data['dist_arcsec'].idxmin()
                    min_dist     = cone_data.loc[min_idx, 'dist_arcsec']
                    target_found = min_dist <= centering_tolerance

                    if target_found:
                        target_coord = SkyCoord(ra=cone_data.loc[min_idx, 'ra'],
                                                dec=cone_data.loc[min_idx, 'dec'],
                                                unit=(u.deg, u.deg), frame='icrs')
                        cone_data['sep_from_target'] = target_coord.separation(cat_coords).arcsec
                        contaminants = cone_data[
                            (cone_data['objid'] != cone_data.loc[min_idx, 'objid']) &
                            (cone_data['sep_from_target'] <= search_radius_arcsec)
                        ]
                        dist_col                = 'sep_from_target'
                        target_snr, target_band = (cone_data.loc[min_idx, 'best_snr'],
                                                   cone_data.loc[min_idx, 'best_band'])
                    else:
                        contaminants = cone_data[cone_data['dist_arcsec'] <= search_radius_arcsec]
                        dist_col     = 'dist_arcsec'

                    # ── THE FILTERS ──
                    if ignore_psf_contaminants:
                        contaminants = contaminants[contaminants['type_clean'] != 'Star']

                    # ── OMNI-BAND SPHEREx CONTAMINATION DETECTION ──
                    contaminants   = contaminants[
                        contaminants[[f'snr_{b}' for b in threat_bands]].ge(min_snr).any(axis=1)
                    ]
                    contaminants   = contaminants[contaminants[dist_col] >= min_separation_arcsec]
                    n_contaminants = len(contaminants)

                else:
                    n_contaminants = 0
                    contaminants   = pd.DataFrame()

                # ── SAFE DECISION BLOCK ──
                if n_contaminants == 0:
                    temp_is_contam = False
                    if n_total == 0:
                        msg = f"⚠️ Source {sid}: No data found (likely outside footprint). Kept as clean."
                    elif not target_found:
                        msg = f"⚠️ Source {sid}: Target not found in field. No other sources found nearby. Kept as clean."
                    else:
                        msg = f"🟢 Source {sid}: Clean. (Target SNR_{target_band}: {target_snr:.1f})"
                else:
                    if not target_found:
                        if n_contaminants == 1:
                            temp_is_contam = False
                            msg = f"⚠️ Source {sid}: Target missing AND {n_contaminants} VHS neighbour found. Kept as clean."
                        else:
                            temp_is_contam = True
                            t_list = _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr)
                            msg = f"🔴 Flagging Source {sid}: Target missing AND {n_contaminants} VHS neighbour(s) found {t_list}."
                    else:
                        temp_is_contam = True
                        t_list = _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr)
                        msg = f"🔴 Flagging Source {sid}: {n_contaminants} VHS neighbour(s) found {t_list}."
                        
                if verbose: print(f"[{i}/{total_sources}] {msg}")
                
                is_contaminated.append(temp_is_contam)
                notes.append(msg)
                success = True

            except Exception as e:
                attempts += 1
                if attempts < max_retries:
                    time.sleep(2)
                else:
                    msg = f"❌ Gave up on Source {sid} after {max_retries} attempts. Kept as clean. Error: {e}"
                    print(msg)
                    is_contaminated.append(False)
                    notes.append(msg)

        time.sleep(0.5)

    df = df.copy()
    df['is_contaminated_vhs'] = is_contaminated
    df['note_vhs']            = notes

    clean_count   = (~df['is_contaminated_vhs']).sum()
    flagged_count =   df['is_contaminated_vhs'].sum()

    if verbose:
        print(f"\n✅ Flagging complete!")
        print(f"Started with: {len(df)} sources")
        print(f"Clean:        {clean_count} isolated sources")
        print(f"Flagged:      {flagged_count} blended/error sources")

    return df


# ── DES DR2 ────────────────────────────────────────────────────────────────────

def remove_des_blends(df, target_mission='SPHEREx', search_radius_arcsec=9.3, max_retries=3,
                      ignore_psf_contaminants=False, verbose=False):
    """
    Flags candidates with neighbouring sources in DES DR2.
    Omni-Band Threat Detection across y (~1.0 µm), z (~0.9 µm), i (~0.75 µm).
    """
    centering_tolerance   = CATALOG_CONFIGS['DES']['centering_tolerance']
    min_separation_arcsec = CATALOG_CONFIGS['DES']['min_separation_arcsec']
    min_snr               = CATALOG_CONFIGS['DES']['min_snr']
    all_bands             = CATALOG_CONFIGS['DES']['all_bands']
    threat_bands          = CATALOG_CONFIGS['DES']['mission_threat_bands'][target_mission]

    tap_service         = TAPService("https://datalab.noirlab.edu/tap")
    total_sources       = len(df)
    query_radius_arcsec = search_radius_arcsec + centering_tolerance
    radius_deg          = query_radius_arcsec / 3600.0

    if verbose:
        print(f"Checking {total_sources} candidates against DES DR2 "
              f"(checking {search_radius_arcsec}\" around target)...\n")
        if ignore_psf_contaminants:
            print("-> Ignoring PSF contaminants (Only flagging extended sources/Unknowns).")
        print(f"-> SPHEREx-relevant bands checked for contamination: {', '.join(threat_bands)}")
        print(f"-> Flagging neighbours detected in ANY band above SNR = {min_snr}")
        print(f"-> Ignoring ultra-close de-blending artefacts at distance < {min_separation_arcsec}\"")

    is_contaminated = []
    notes           = []

    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sid = row['source_id']
        ra  = row['ra']
        dec = row['dec']

        query = f"""
            SELECT coadd_object_id as objid, ra, dec,
                   wavg_flux_psf_y, wavg_fluxerr_psf_y,
                   wavg_flux_psf_z, wavg_fluxerr_psf_z,
                   wavg_flux_psf_i, wavg_fluxerr_psf_i,
                   wavg_flux_psf_r, wavg_fluxerr_psf_r,
                   wavg_flux_psf_g, wavg_fluxerr_psf_g,
                   extended_class_coadd, Q3C_DIST(ra, dec, {ra}, {dec}) * 3600.0 AS dist_arcsec
            FROM   des_dr2.main
            WHERE  't' = Q3C_RADIAL_QUERY(ra, dec, {ra}, {dec}, {radius_deg})
        """

        success      = False
        attempts     = 0
        target_found = False

        while not success and attempts < max_retries:
            try:
                result    = tap_service.search(query)
                cone_data = result.to_table().to_pandas()
                n_total   = len(cone_data)

                if n_total > 0:
                    cone_data['dist_arcsec'] = pd.to_numeric(cone_data['dist_arcsec'], errors='coerce')
                    cone_data['type_clean']  = np.where(cone_data['extended_class_coadd'].isna(), 'Unknown',
                                             np.where(cone_data['extended_class_coadd'] <= 1, 'Star', 'Galaxy'))

                    # ── SNR PER BAND ──
                    for b in all_bands:
                        f   = pd.to_numeric(cone_data[f'wavg_flux_psf_{b}'],    errors='coerce').fillna(0)
                        err = pd.to_numeric(cone_data[f'wavg_fluxerr_psf_{b}'], errors='coerce').fillna(0)
                        cone_data[f'snr_{b}'] = np.where(err > 0, f / err, 0)

                    # best_snr/best_band — TARGET LOGGING ONLY
                    conditions             = [cone_data[f'snr_{b}'] > 0 for b in all_bands]
                    cone_data['best_snr']  = np.select(conditions, [cone_data[f'snr_{b}'] for b in all_bands], default=0)
                    cone_data['best_band'] = np.select(conditions, all_bands, default='none')

                    # ── NEAREST NEIGHBOUR LOGIC ──
                    min_idx      = cone_data['dist_arcsec'].idxmin()
                    min_dist     = cone_data.loc[min_idx, 'dist_arcsec']
                    target_found = min_dist <= centering_tolerance

                    if target_found:
                        target_coord = SkyCoord(ra=cone_data.loc[min_idx, 'ra'],
                                                dec=cone_data.loc[min_idx, 'dec'],
                                                unit=(u.deg, u.deg), frame='icrs')
                        cat_coords   = SkyCoord(ra=cone_data['ra'].values,
                                                dec=cone_data['dec'].values,
                                                unit=(u.deg, u.deg), frame='icrs')
                        cone_data['sep_from_target'] = target_coord.separation(cat_coords).arcsec
                        contaminants = cone_data[
                            (cone_data['objid'] != cone_data.loc[min_idx, 'objid']) &
                            (cone_data['sep_from_target'] <= search_radius_arcsec)
                        ]
                        dist_col                = 'sep_from_target'
                        target_snr, target_band = (cone_data.loc[min_idx, 'best_snr'],
                                                   cone_data.loc[min_idx, 'best_band'])
                    else:
                        contaminants = cone_data[cone_data['dist_arcsec'] <= search_radius_arcsec]
                        dist_col     = 'dist_arcsec'

                    # ── THE FILTERS ──
                    if ignore_psf_contaminants:
                        contaminants = contaminants[contaminants['type_clean'] != 'Star']

                    # ── OMNI-BAND SPHEREx CONTAMINATION DETECTION ──
                    contaminants   = contaminants[
                        contaminants[[f'snr_{b}' for b in threat_bands]].ge(min_snr).any(axis=1)
                    ]
                    contaminants   = contaminants[contaminants[dist_col] >= min_separation_arcsec]
                    n_contaminants = len(contaminants)

                else:
                    n_contaminants = 0
                    contaminants   = pd.DataFrame()

                # ── SAFE DECISION BLOCK ──
                if n_contaminants == 0:
                    temp_is_contam = False
                    if n_total == 0:
                        msg = f"⚠️ Source {sid}: No data found (likely outside footprint). Kept as clean."
                    elif not target_found:
                        msg = f"⚠️ Source {sid}: Target not found in field. No other sources found nearby. Kept as clean."
                    else:
                        msg = f"🟢 Source {sid}: Clean. (Target SNR_{target_band}: {target_snr:.1f})"
                else:
                    if not target_found:
                        if n_contaminants == 1:
                            temp_is_contam = False
                            msg = f"⚠️ Source {sid}: Target missing AND {n_contaminants} DES neighbour found. Kept as clean."
                        else:
                            temp_is_contam = True
                            t_list = _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr)
                            msg = f"🔴 Flagging Source {sid}: Target missing AND {n_contaminants} DES neighbour(s) found {t_list}."
                    else:
                        temp_is_contam = True
                        t_list = _format_contaminant_list(contaminants, dist_col, threat_bands, min_snr)
                        msg = f"🔴 Flagging Source {sid}: {n_contaminants} DES neighbour(s) found {t_list}."
                        
                if verbose: print(f"[{i}/{total_sources}] {msg}")
                
                is_contaminated.append(temp_is_contam)
                notes.append(msg)
                success = True

            except Exception as e:
                attempts += 1
                if attempts < max_retries:
                    time.sleep(2)
                else:
                    msg = f"❌ Gave up on Source {sid} after {max_retries} attempts. Kept as clean. Error: {e}"
                    print(msg)
                    is_contaminated.append(False)
                    notes.append(msg)

        time.sleep(0.5)

    df = df.copy()
    df['is_contaminated_des'] = is_contaminated
    df['note_des']            = notes

    clean_count   = (~df['is_contaminated_des']).sum()
    flagged_count =   df['is_contaminated_des'].sum()

    if verbose:
        print(f"\n✅ Flagging complete!")
        print(f"Started with: {len(df)} sources")
        print(f"Clean:        {clean_count} isolated sources")
        print(f"Flagged:      {flagged_count} blended/error sources")

    return df