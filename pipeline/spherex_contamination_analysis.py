from .catalog_contamination import *
from .image_contamination import *
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
import numpy as np
import pandas as pd

def spherex_contamination_analysis(df, search_radius_arcsec=9.3, remove_contaminated=True, verbose=False):
    """
    Executes a multi-survey contamination pipeline for SPHEREx targets.
    
    This function sequentially passes the input catalog through 
    a series of optical and near-infrared catalog filters (Gaia, DESI, 
    Pan-STARRS, and SDSS), followed by a final image-level WCS 
    PSF fitting filter. It is designed to identify and/or remove unresolved 
    background contaminants that could blend into the large SPHEREx pixels.

    Args:
        df (pandas.DataFrame): The input catalog of targets. Must contain at least 
            'source_id', 'ra', and 'dec' columns. It is highly recommended to also 
            include a 'phot_g_mean_mag' column (Gaia G-band magnitude), which is 
            required for the image-level processing logic. If this column is missing, 
            the function will automatically query the Gaia database to fetch the magnitude 
            for each source.
        search_radius_arcsec (float, optional): The radial distance in arcseconds 
            to check for neighboring sources. Defaults to 9.3.
        remove_contaminated (bool, optional): If True, drops rows 
            from the DataFrame at each step if they are flagged as contaminated. 
            If False, retains all rows and only appends the boolean flag columns 
            (e.g., 'is_contaminated_gaia', 'note_gaia', etc.). Defaults to True.
        verbose (bool, optional): If True, prints progress and logging 
            messages to the console. Defaults to False.

    Returns:
        pandas.DataFrame: The processed DataFrame. If `remove_contaminated` is 
            True, this returns only the isolated targets. If False, 
            returns the original targets appended with contamination flags, 
            investigation notes from each survey, and a final master 
            'is_contaminated' boolean column.
    """
    
    # 1. Gaia DR3 Synchronous ADQL Check
    if verbose:
        print("------------------------------ Gaia DR3 Synchronous ADQL Check ------------------------------")
    clean_gaia_df = remove_gaia_blends(df, search_radius_arcsec=search_radius_arcsec, verbose=verbose)
    if remove_contaminated:
        clean_gaia_df = clean_gaia_df[clean_gaia_df["is_contaminated_gaia"] == False]

    # 2. DESI Legacy DR10 Tractor Catalog Check
    if verbose:
        print("\n\n-------------------------- DESI Legacy DR10 Tractor Catalog Check --------------------------")
    clean_desi_df = remove_desi_blends(clean_gaia_df, search_radius_arcsec=search_radius_arcsec, verbose=verbose)
    if remove_contaminated:
        clean_desi_df = clean_desi_df[clean_desi_df["is_contaminated_desi"] == False]

    # 3. Pan-STARRS DR1 VizieR Check
    if verbose:
        print("\n\n-------------------------------- Pan-STARRS DR1 VizieR Check --------------------------------")
    clean_panstarrs_df = remove_panstarrs_blends(clean_desi_df, search_radius_arcsec=search_radius_arcsec, verbose=verbose)
    if remove_contaminated:
        clean_panstarrs_df = clean_panstarrs_df[clean_panstarrs_df["is_contaminated_panstarrs"] == False]

    # 4. SDSS DR16 VizieR Check
    if verbose:
        print("\n\n----------------------------------- SDSS DR16 VizieR Check -----------------------------------")
    clean_sdss_df = remove_sdss_blends(clean_panstarrs_df, search_radius_arcsec=search_radius_arcsec, verbose=verbose)
    if remove_contaminated:
        clean_sdss_df = clean_sdss_df[clean_sdss_df["is_contaminated_sdss"] == False]

    # 5. Direct Optical Image Astrometric/WCS Check
    if verbose:
        print("\n\n----------------------------- Optical Image Astrometric/WCS Check -----------------------------")
        
    if not clean_sdss_df.empty:
        # Helper function to fetch G mag by coordinate
        def get_g_mag_by_coord(row):
            import time # Make sure this is imported!
            
            # If the dataframe already has it, don't fetch it again!
            if 'phot_g_mean_mag' in row and not pd.isna(row['phot_g_mean_mag']):
                return row['phot_g_mean_mag']
            
            # Try up to 3 times to handle random server hiccups
            for attempt in range(3):
                try:
                    time.sleep(0.3) # Give the Gaia server a 0.3-second breather
                    
                    coord = SkyCoord(ra=row['ra'], dec=row['dec'], unit=(u.degree, u.degree), frame='icrs')
                    result = Gaia.cone_search_async(coord, radius=u.Quantity(2.0, u.arcsec))
                    res_df = result.get_results().to_pandas()
                    
                    if not res_df.empty:
                        res_df['dist'] = np.hypot(res_df['ra'] - row['ra'], res_df['dec'] - row['dec'])
                        best_match = res_df.sort_values('dist').iloc[0]
                        return best_match['phot_g_mean_mag']
                    
                    # If it successfully queried but found nothing, break the retry loop
                    break 
                    
                except Exception:
                    if attempt == 2:
                        print(f"⚠️ Gaia API timeout for RA={row['ra']:.4f} after 3 attempts.")
            return np.nan

        if verbose:
            print("Ensuring Gaia G magnitudes are present for saturation cutoff...")
            
        # Create a copy to prevent SettingWithCopy warnings
        working_df = clean_sdss_df.copy()
        working_df['phot_g_mean_mag'] = working_df.apply(get_g_mag_by_coord, axis=1)
        
        # Split into safe (G >= 13 or NaN fallback) and bright (G < 13)
        bright_df = working_df[working_df['phot_g_mean_mag'] < 13].copy()
        to_process_df = working_df[(working_df['phot_g_mean_mag'] >= 13) | (working_df['phot_g_mean_mag'].isna())].copy()

        if verbose:
            print(f"Skipping image check for {len(bright_df)} bright sources (G < 13).")
            print(f"Running image check for {len(to_process_df)} sources.")

        # Process the faint sources normally
        if not to_process_df.empty:
            processed_df = flag_image_contamination(to_process_df, search_radius_arcsec=search_radius_arcsec, verbose=verbose)
        else:
            processed_df = to_process_df
            processed_df['is_contaminated_image'] = False # Ensure flag exists

        # Bypass the bright sources (they automatically pass)
        if not bright_df.empty:
            bright_df['is_contaminated_image'] = False

        # Recombine the dataframe
        clean_final_df = pd.concat([processed_df, bright_df], ignore_index=True)

        if remove_contaminated:
            clean_final_df = clean_final_df[clean_final_df["is_contaminated_image"] == False]
    else:
        clean_final_df = clean_sdss_df.copy()

    # --- FINAL MASTER CONTAMINATION FLAG ---
    # Dynamically find all the specific flag columns that were successfully appended
    flag_columns = [col for col in clean_final_df.columns if col.startswith('is_contaminated_')]
    
    if flag_columns:
        # If any of the specific survey flags are True, the master flag is True.
        # It evaluates to False ONLY if all individual flags are False.
        clean_final_df['is_contaminated'] = clean_final_df[flag_columns].any(axis=1)
    else:
        # Safe fallback in case the pipeline logic skips or fails
        clean_final_df['is_contaminated'] = False

    return clean_final_df