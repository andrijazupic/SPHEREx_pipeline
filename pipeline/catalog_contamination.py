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

# Silence the annoying "INFO: Query finished" spam
logging.getLogger('astroquery').setLevel(logging.WARNING)


def remove_gaia_blends(df, search_radius_arcsec=9.3, max_retries=3, verbose=False):
    total_sources = len(df)

    if verbose:
        print(f"Checking {total_sources} candidates using Synchronous ADQL...\n")
    
    is_contaminated = []
    notes = []
    
    # Convert arcseconds to degrees for the ADQL query
    radius_deg = search_radius_arcsec / 3600.0 
    
    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sid = row['source_id']
        ra = row['ra']
        dec = row['dec']
        
        # Direct ADQL query: "Find all sources within a circle around this RA/DEC"
        query = f"""
            SELECT source_id 
            FROM gaiadr3.gaia_source 
            WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
        """
        
        success = False
        attempts = 0
        
        while not success and attempts < max_retries:
            try:
                # launch_job is SYNCHRONOUS. It waits for the data directly.
                job = Gaia.launch_job(query)
                cone_data = job.get_results()
                
                total_objects = len(cone_data)
                
                if total_objects == 1:
                    is_contaminated.append(False)
                    if verbose:
                        print(f"[{i}/{total_sources}] 🟢 Source {sid}: Clean isolated target")
                    notes.append(f"Found {total_objects - 1} extra neighbor(s).")
                else:
                    if verbose:
                        print(f"[{i}/{total_sources}] 🔴 Flagging Source {sid}: Found {total_objects - 1} extra neighbor(s).")
                    is_contaminated.append(True)
                    notes.append(f"Found {total_objects - 1} extra neighbor(s).")
                
                success = True # It worked, break out of retry loop
                
            except Exception as e:
                attempts += 1
                if attempts < max_retries:
                    time.sleep(2) # Wait and retry on genuine server timeouts
                else:
                    print(f"❌ Gave up on Source {sid} after {max_retries} attempts. Kept as clean. Error: {e}")
                    is_contaminated.append(False) # Safe default if it fails
                    notes.append(f"Error: {e}")
        
        # Polite delay to prevent IP rate-limiting
        time.sleep(0.5) 

    # Append columns to the original dataframe
    df = df.copy()
    df['is_contaminated_gaia'] = is_contaminated
    df['note_gaia'] = notes
    
    clean_count = (~df['is_contaminated_gaia']).sum()
    flagged_count = df['is_contaminated_gaia'].sum()
    
    if verbose:
        print(f"\n✅ Flagging complete!")
        print(f"Started with: {len(df)} sources")
        print(f"Clean:        {clean_count} isolated sources")
        print(f"Flagged:      {flagged_count} blended/error sources")
    
    return df


def remove_desi_blends(df, search_radius_arcsec=9.3, centering_tolerance=2.0, min_separation_arcsec=1.0, 
                       max_retries=3, ignore_psf_contaminants=False, min_snr=11.0, verbose=False):
    """
    Flags candidates with neighboring sources in the DESI DR10 Tractor catalog.
    Finds the central Target WD first, then calculates distances to other sources relative to the WD.
    Filters out noise (SNR), ignores PSFs, and ignores de-blending artifacts too close to the core.
    """
    tap_service = TAPService("https://datalab.noirlab.edu/tap")

    total_sources = len(df)
    
    query_radius_arcsec = search_radius_arcsec + centering_tolerance
    radius_deg = query_radius_arcsec / 3600.0

    if verbose:
        print(f"Checking {total_sources} candidates against DESI DR10 Tractor (checking {search_radius_arcsec}\" around target)...\n")
        if ignore_psf_contaminants:
            print("-> Ignoring PSF contaminants (Only flagging extended galaxies: REX, EXP, DEV, SER).")
        print(f"-> Filtering out noise artifacts with Signal-to-Noise Ratio (SNR) < {min_snr}")
        print(f"-> Ignoring ultra-close de-blending artifacts at distance < {min_separation_arcsec}\"")

    is_contaminated = []
    notes = []

    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sid  = row['source_id']
        ra   = row['ra']
        dec  = row['dec']

        query = f"""
            SELECT objid, ra, dec, type, flux_r, flux_ivar_r, Q3C_DIST(ra, dec, {ra}, {dec}) * 3600.0 AS dist_arcsec
            FROM   ls_dr10.tractor
            WHERE  't' = Q3C_RADIAL_QUERY(ra, dec, {ra}, {dec}, {radius_deg})
        """

        success  = False
        attempts = 0
        target_found = False 

        while not success and attempts < max_retries:
            try:
                result     = tap_service.search(query)
                cone_data  = result.to_table().to_pandas()
                
                n_total = len(cone_data)
                
                if n_total > 0:
                    cone_data['dist_arcsec'] = pd.to_numeric(cone_data['dist_arcsec'], errors='coerce')
                    
                    cone_data['flux_r'] = pd.to_numeric(cone_data['flux_r'], errors='coerce').fillna(0)
                    cone_data['flux_ivar_r'] = pd.to_numeric(cone_data['flux_ivar_r'], errors='coerce').fillna(0)
                    
                    cone_data['snr_r'] = cone_data['flux_r'] * np.sqrt(np.clip(cone_data['flux_ivar_r'], 0, None))
                    
                    cone_data['type_clean'] = cone_data['type'].apply(
                        lambda t: t.decode('utf-8').strip() if isinstance(t, bytes) else str(t).strip()
                    )
                    
                    # --- NEAREST NEIGHBOR LOGIC ---
                    min_idx = cone_data['dist_arcsec'].idxmin()
                    min_dist = cone_data.loc[min_idx, 'dist_arcsec']
                    
                    target_found = min_dist <= centering_tolerance
                    
                    if target_found:
                        target_ra_desi = cone_data.loc[min_idx, 'ra']
                        target_dec_desi = cone_data.loc[min_idx, 'dec']
                        target_objid = cone_data.loc[min_idx, 'objid']
                        
                        target_coord = SkyCoord(ra=target_ra_desi, dec=target_dec_desi, unit=(u.deg, u.deg), frame='icrs')
                        cat_coords = SkyCoord(ra=cone_data['ra'].values, dec=cone_data['dec'].values, unit=(u.deg, u.deg), frame='icrs')
                        
                        cone_data['sep_from_target'] = target_coord.separation(cat_coords).arcsec
                        
                        contaminants = cone_data[
                            (cone_data['objid'] != target_objid) & 
                            (cone_data['sep_from_target'] <= search_radius_arcsec)
                        ]
                        dist_col = 'sep_from_target' # Set dynamic distance column
                        
                    else:
                        contaminants = cone_data.iloc[0:0] 
                        dist_col = 'dist_arcsec' 
                    
                    # --- THE FILTERS ---
                    if ignore_psf_contaminants:
                        contaminants = contaminants[contaminants['type_clean'] != 'PSF']
                        
                    contaminants = contaminants[contaminants['snr_r'] >= min_snr]
                    contaminants = contaminants[contaminants[dist_col] >= min_separation_arcsec]
                        
                    n_contaminants = len(contaminants)
                        
                else:
                    n_contaminants = 0
                    contaminants = pd.DataFrame()

                # --- DECISION ---
                if n_contaminants == 0:
                    is_contaminated.append(False)
                    if n_total == 0:
                        msg = f"⚠️ Source {sid}: No data found (likely outside footprint). Kept as clean."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)
                    elif not target_found:
                        msg = f"⚠️ Source {sid}: Target WD not found. Skipping contaminant check and kept as clean."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)
                    else:
                        msg = f"🟢 Source {sid}: Clean isolated target."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)
                else:
                    is_contaminated.append(True)
                    types_with_dists = []
                    for t, d, s in zip(contaminants['type_clean'], contaminants[dist_col], contaminants['snr_r']):
                        types_with_dists.append(f"{t} ({d:.2f}\", SNR:{s:.1f})")
                    
                    if not target_found:
                        msg = f"🔴 Flagging Source {sid}: Target WD missing AND {n_contaminants} real extended neighbour(s) found {types_with_dists}."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)
                    else:
                        msg = f"🔴 Flagging Source {sid}: {n_contaminants} real Tractor neighbour(s) found {types_with_dists}."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)

                success = True

            except Exception as e:
                attempts += 1
                if attempts < max_retries:
                    time.sleep(2)
                else:
                    msg = f"❌ Gave up on Source {sid} after {max_retries} attempts. Kept as clean. Error: {e}"
                    print(msg)
                    is_contaminated.append(False) # Safe default to flag on failure
                    notes.append(msg)

        time.sleep(0.5)   # polite rate-limit delay

    # Append columns to the original dataframe
    df = df.copy()
    df['is_contaminated_desi'] = is_contaminated
    df['note_desi'] = notes

    clean_count = (~df['is_contaminated_desi']).sum()
    flagged_count = df['is_contaminated_desi'].sum()

    if verbose:
        print(f"\n✅ Flagging complete!")
        print(f"Started with: {len(df)} sources")
        print(f"Clean:        {clean_count} isolated sources")
        print(f"Flagged:      {flagged_count} blended/error sources")

    return df


def remove_panstarrs_blends(df, search_radius_arcsec=9.3, centering_tolerance=2.0, min_separation_arcsec=1.0, 
                            max_retries=3, ignore_psf_contaminants=False, min_snr=9, verbose=False):
    """
    Flags candidates with neighboring sources in the Pan-STARRS DR1 catalog via VizieR.
    Uses identical spatial/anchoring logic to the DESI filter.
    Separates stars from galaxies using the mathematical difference between PSF and Kron magnitudes.
    """
    tap_service = TAPService("https://tapvizier.u-strasbg.fr/TAPVizieR/tap")

    total_sources = len(df)
    
    query_radius_arcsec = search_radius_arcsec + centering_tolerance
    radius_deg = query_radius_arcsec / 3600.0

    if verbose:
        print(f"Checking {total_sources} candidates against Pan-STARRS DR1 (checking {search_radius_arcsec}\" around target)...\n")
        if ignore_psf_contaminants:
            print("-> Ignoring PSF contaminants (Only flagging extended sources: r_PSF - r_Kron > 0.05).")
        print(f"-> Filtering out noise artifacts with Signal-to-Noise Ratio (SNR) < {min_snr}")
        print(f"-> Ignoring ultra-close de-blending artifacts at distance < {min_separation_arcsec}\"")

    is_contaminated = []
    notes = []

    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sid  = row['source_id']
        ra   = row['ra']
        dec  = row['dec']

        # Pan-STARRS VizieR Table: "II/349/ps1"
        # rmag = PSF magnitude | rKmag = Kron (extended) magnitude
        query = f"""
            SELECT objID, RAJ2000 as ra, DEJ2000 as dec, rmag, e_rmag, rKmag
            FROM "II/349/ps1"
            WHERE 1=CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
        """

        success  = False
        attempts = 0
        target_found = False 

        while not success and attempts < max_retries:
            try:
                result     = tap_service.search(query)
                cone_data  = result.to_table().to_pandas()
                
                n_total = len(cone_data)
                
                if n_total > 0:
                    theoretical_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
                    cat_coords = SkyCoord(ra=cone_data['ra'].values, dec=cone_data['dec'].values, unit=(u.deg, u.deg), frame='icrs')
                    cone_data['dist_arcsec'] = theoretical_coord.separation(cat_coords).arcsec
                    
                    # Safely calculate Signal-to-Noise Ratio from the magnitude error
                    cone_data['e_rmag'] = pd.to_numeric(cone_data['e_rmag'], errors='coerce')
                    cone_data['snr_r'] = np.where(cone_data['e_rmag'] > 0, 1.0857 / cone_data['e_rmag'], 0)
                    
                    # Morphology Logic: r_PSF - r_Kron
                    cone_data['rmag'] = pd.to_numeric(cone_data['rmag'], errors='coerce')
                    cone_data['rKmag'] = pd.to_numeric(cone_data['rKmag'], errors='coerce')
                    
                    # If PSF is >0.05 magnitudes fainter than Kron, it's fuzzy!
                    is_extended = (cone_data['rmag'] - cone_data['rKmag']) > 0.05
                    cone_data['type_clean'] = np.where(is_extended, 'Galaxy', 'Star')
                    
                    # --- NEAREST NEIGHBOR LOGIC ---
                    min_idx = cone_data['dist_arcsec'].idxmin()
                    min_dist = cone_data.loc[min_idx, 'dist_arcsec']
                    
                    target_found = min_dist <= centering_tolerance
                    
                    if target_found:
                        target_ra_ps1 = cone_data.loc[min_idx, 'ra']
                        target_dec_ps1 = cone_data.loc[min_idx, 'dec']
                        target_objid = cone_data.loc[min_idx, 'objID']
                        
                        target_snr = cone_data.loc[min_idx, 'snr_r']
                        
                        # ANCHOR TO TARGET
                        target_coord = SkyCoord(ra=target_ra_ps1, dec=target_dec_ps1, unit=(u.deg, u.deg), frame='icrs')
                        cone_data['sep_from_target'] = target_coord.separation(cat_coords).arcsec
                        
                        contaminants = cone_data[
                            (cone_data['objID'] != target_objid) & 
                            (cone_data['sep_from_target'] <= search_radius_arcsec)
                        ]
                        dist_col = 'sep_from_target'
                        
                    else:
                        # Target missing. Ignore rest of field.
                        contaminants = cone_data.iloc[0:0] 
                        dist_col = 'dist_arcsec'
                    
                    # --- THE FILTERS ---
                    if ignore_psf_contaminants:
                        contaminants = contaminants[contaminants['type_clean'] == 'Galaxy']
                        
                    contaminants = contaminants[contaminants['snr_r'] >= min_snr]
                    contaminants = contaminants[contaminants[dist_col] >= min_separation_arcsec]
                        
                    n_contaminants = len(contaminants)
                        
                else:
                    n_contaminants = 0
                    contaminants = pd.DataFrame()

                # --- DECISION ---
                if n_contaminants == 0:
                    is_contaminated.append(False)
                    if n_total == 0:
                        msg = f"⚠️ Source {sid}: No data found (likely outside footprint). Kept as clean."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)
                    elif not target_found:
                        msg = f"⚠️ Source {sid}: Target WD not found. Skipping contaminant check and kept as clean."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)
                    else:
                        msg = f"🟢 Source {sid}: Clean. Target WD found (SNR: {target_snr:.1f})."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)
                else:
                    is_contaminated.append(True)
                    types_with_dists = []
                    for t, d, s in zip(contaminants['type_clean'], contaminants[dist_col], contaminants['snr_r']):
                        types_with_dists.append(f"{t} ({d:.2f}\", SNR:{s:.1f})")
                    
                    if not target_found:
                        msg = f"🔴 Flagging Source {sid}: Target WD missing AND {n_contaminants} real extended neighbour(s) found {types_with_dists}."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)
                    else:
                        msg = f"🔴 Flagging Source {sid}: Target WD found (SNR: {target_snr:.1f}). {n_contaminants} real Pan-STARRS neighbour(s) found {types_with_dists}."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
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

    # Append columns to the original dataframe
    df = df.copy()
    df['is_contaminated_panstarrs'] = is_contaminated
    df['note_panstarrs'] = notes

    clean_count = (~df['is_contaminated_panstarrs']).sum()
    flagged_count = df['is_contaminated_panstarrs'].sum()

    if verbose:
        print(f"\n✅ Flagging complete!")
        print(f"Started with: {len(df)} sources")
        print(f"Clean:        {clean_count} isolated sources")
        print(f"Flagged:      {flagged_count} blended/error sources")

    return df


def remove_sdss_blends(df, search_radius_arcsec=9.3, centering_tolerance=2.5, min_separation_arcsec=1.5, 
                       max_retries=3, ignore_psf_contaminants=False, min_snr=9, verbose=False):
    """
    Flags candidates with neighboring sources in the SDSS DR16 catalog via VizieR.
    Uses exact identical spatial logic to the DESI filter, but adapted for SDSS schema.
    """
    # VizieR TAP service handles SDSS gracefully
    tap_service = TAPService("https://tapvizier.u-strasbg.fr/TAPVizieR/tap")

    total_sources = len(df)
    
    query_radius_arcsec = search_radius_arcsec + centering_tolerance
    radius_deg = query_radius_arcsec / 3600.0

    if verbose:
        print(f"Checking {total_sources} candidates against SDSS DR16 (checking {search_radius_arcsec}\" around target)...\n")
        if ignore_psf_contaminants:
            print("-> Ignoring PSF contaminants (Only flagging class=3 Galaxies).")
        print(f"-> Filtering out noise artifacts with Signal-to-Noise Ratio (SNR) < {min_snr}")
        print(f"-> Ignoring ultra-close de-blending artifacts at distance < {min_separation_arcsec}\"")

    is_contaminated = []
    notes = []

    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sid  = row['source_id']
        ra   = row['ra']
        dec  = row['dec']

        # SDSS VizieR Table: "V/154/sdss16"
        # class: 3 = Galaxy (Extended), 6 = Star (PSF)
        query = f"""
            SELECT objID, RA_ICRS as ra, DE_ICRS as dec, class, rmag, e_rmag
            FROM "V/154/sdss16"
            WHERE 1=CONTAINS(POINT('ICRS', RA_ICRS, DE_ICRS), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
        """

        success  = False
        attempts = 0
        target_found = False 

        while not success and attempts < max_retries:
            try:
                result     = tap_service.search(query)
                cone_data  = result.to_table().to_pandas()
                
                n_total = len(cone_data)
                
                if n_total > 0:
                    # Calculate distances to theoretical center
                    theoretical_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
                    cat_coords = SkyCoord(ra=cone_data['ra'].values, dec=cone_data['dec'].values, unit=(u.deg, u.deg), frame='icrs')
                    cone_data['dist_arcsec'] = theoretical_coord.separation(cat_coords).arcsec
                    
                    # Convert magnitude errors to SNR (handling NaNs/Zeros safely)
                    cone_data['e_rmag'] = pd.to_numeric(cone_data['e_rmag'], errors='coerce')
                    cone_data['snr_r'] = np.where(cone_data['e_rmag'] > 0, 1.0857 / cone_data['e_rmag'], 0)
                    
                    # Clean up SDSS classes for printing
                    cone_data['type_clean'] = cone_data['class'].map({3: 'Galaxy', 6: 'Star'}).fillna('Unknown')
                    
                    # --- NEAREST NEIGHBOR LOGIC ---
                    min_idx = cone_data['dist_arcsec'].idxmin()
                    min_dist = cone_data.loc[min_idx, 'dist_arcsec']
                    
                    target_found = min_dist <= centering_tolerance
                    
                    if target_found:
                        target_ra_sdss = cone_data.loc[min_idx, 'ra']
                        target_dec_sdss = cone_data.loc[min_idx, 'dec']
                        target_objid = cone_data.loc[min_idx, 'objID']
                        
                        target_snr = cone_data.loc[min_idx, 'snr_r']
                        
                        target_coord = SkyCoord(ra=target_ra_sdss, dec=target_dec_sdss, unit=(u.deg, u.deg), frame='icrs')
                        cone_data['sep_from_target'] = target_coord.separation(cat_coords).arcsec
                        
                        contaminants = cone_data[
                            (cone_data['objID'] != target_objid) & 
                            (cone_data['sep_from_target'] <= search_radius_arcsec)
                        ]
                        dist_col = 'sep_from_target'
                        
                    else:
                        contaminants = cone_data.iloc[0:0] 
                        dist_col = 'dist_arcsec'
                    
                    # --- THE FILTERS ---
                    if ignore_psf_contaminants:
                        # Keep only SDSS class 3 (Galaxies)
                        contaminants = contaminants[contaminants['class'] == 3]
                        
                    contaminants = contaminants[contaminants['snr_r'] >= min_snr]
                    contaminants = contaminants[contaminants[dist_col] >= min_separation_arcsec]
                        
                    n_contaminants = len(contaminants)
                        
                else:
                    n_contaminants = 0
                    contaminants = pd.DataFrame()

                # --- DECISION ---
                if n_contaminants == 0:
                    is_contaminated.append(False)
                    if n_total == 0:
                        msg = f"⚠️ Source {sid}: No data found (likely outside footprint). Kept as clean."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)
                    elif not target_found:
                        msg = f"⚠️ Source {sid}: Target WD not found. Skipping contaminant check and kept as clean."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)
                    else:
                        msg = f"🟢 Source {sid}: Clean. Target WD found (SNR: {target_snr:.1f})."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)
                else:
                    is_contaminated.append(True)
                    types_with_dists = []
                    for t, d, s in zip(contaminants['type_clean'], contaminants[dist_col], contaminants['snr_r']):
                        types_with_dists.append(f"{t} ({d:.2f}\", SNR:{s:.1f})")
                    
                    if not target_found:
                        msg = f"🔴 Flagging Source {sid}: Target WD missing AND {n_contaminants} real extended neighbour(s) found {types_with_dists}."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
                        notes.append(msg)
                    else:
                        msg = f"🔴 Flagging Source {sid}: Target WD found (SNR: {target_snr:.1f}). {n_contaminants} real SDSS neighbour(s) found {types_with_dists}."
                        if verbose:
                            print(f"[{i}/{total_sources}] {msg}")
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

    # Append columns to the original dataframe
    df = df.copy()
    df['is_contaminated_sdss'] = is_contaminated
    df['note_sdss'] = notes

    clean_count = (~df['is_contaminated_sdss']).sum()
    flagged_count = df['is_contaminated_sdss'].sum()

    if verbose:
        print(f"\n✅ Flagging complete!")
        print(f"Started with: {len(df)} sources")
        print(f"Clean:        {clean_count} isolated sources")
        print(f"Flagged:      {flagged_count} blended/error sources")

    return df