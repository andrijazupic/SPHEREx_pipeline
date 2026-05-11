from catalog_contamination import *
from image_contamination import *

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
            'source_id', 'ra', and 'dec' columns.
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
    clean_final_df = flag_image_contamination(clean_sdss_df, search_radius_arcsec=search_radius_arcsec, verbose=verbose)
    if remove_contaminated:
        clean_final_df = clean_final_df[clean_final_df["is_contaminated_image"] == False]

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