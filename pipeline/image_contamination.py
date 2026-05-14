import numpy as np
import pandas as pd
import requests
import warnings
import time
from io import BytesIO
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.visualization import ZScaleInterval
from astroquery.gaia import Gaia
import scipy.ndimage as ndimage


# Silence warnings from photutils (empty images) and WCS (non-standard headers)
warnings.filterwarnings('ignore', module='photutils')
warnings.filterwarnings('ignore', module='astropy.wcs')

# ── Survey Registry ────────────────────────────────────────────────────────────
SURVEY_CONFIGS = {
    'Legacy': {
        'bands':               ['z', 'r', 'g'],
        'hips_base':           None,
        'pixscale':            0.262,
        'fwhm_arcsec':         2.0,
        'detection_sigma':     6.1,
        'centering_tolerance': 2.0,
        'label':               'DESI Legacy DR10',
    },
    'PanSTARRS': {
        'bands':               ['y', 'z', 'i', 'r', 'g'],
        'hips_base':           'CDS/P/PanSTARRS/DR1/',
        'pixscale':            0.25,
        'fwhm_arcsec':         2.0,
        'detection_sigma':     6.1,
        'centering_tolerance': 2.0,
        'label':               'PanSTARRS DR1',
    },
    'SDSS': {
        'bands':               ['z', 'i', 'r', 'g', 'u'],
        'hips_base':           'CDS/P/SDSS9/',
        'pixscale':            0.396,
        'fwhm_arcsec':         2.5,
        'detection_sigma':     5.0,
        'centering_tolerance': 2.5,
        'label':               'SDSS DR9',
    },
    'DSS2': {
        'bands':               ['NIR', 'red', 'blue'],
        'hips_base':           'CDS/P/DSS2/',
        'pixscale':            1.0,
        'fwhm_arcsec':         3.0,
        'detection_sigma':     4.0,
        'centering_tolerance': 5.0,
        'label':               'DSS2',
    },
}

FALLBACK_CHAIN = ['Legacy', 'PanSTARRS', 'SDSS', 'DSS2']

# ── 0. The Profile Checker ─────────────────────────────────────────────────────

def is_real_source(data, target_x, target_y, contam_x, contam_y, min_prominence=0.10):
    """
    Extracts a 1D pixel profile between the target star and a potential contaminant.
    Returns True if the contaminant has a distinct 'valley' separating it from the target.
    """
    num_points = 100
    x_line = np.linspace(target_x, contam_x, num_points)
    y_line = np.linspace(target_y, contam_y, num_points)
    
    profile = ndimage.map_coordinates(data, np.vstack((y_line, x_line)))
    contam_peak = profile[-1]
    
    if contam_peak <= 0:
        return False
        
    valley_val = np.min(profile[10:-10])
    bump_height = contam_peak - valley_val
    
    if bump_height <= 0:
        return False
        
    prominence_ratio = bump_height / contam_peak
    if prominence_ratio < min_prominence:
        return False
        
    return True


# ── 1. The Headless Fitter ─────────────────────────────────────────────────────

def count_sources_in_image(data, header, theoretical_ra, theoretical_dec, survey_name, search_radius_arcsec, wing_multiplier, g_mag=np.nan):
    """
    Finds the true observed center of the target first, then anchors all
    distance calculations relative to that specific object.
    """
    if not pd.isna(g_mag) and g_mag < 13.0:
        return 1, True

    if not np.any(np.isfinite(data)):
        return 0, False

    data_clean = np.nan_to_num(data, nan=np.nanmedian(data))

    mean, median, std = sigma_clipped_stats(data_clean, sigma=3.0)
    std = max(std, 0.001) 
    
    wcs = WCS(header).celestial
    true_pixscale_deg = proj_plane_pixel_scales(wcs)[0]
    true_pixscale_arcsec = true_pixscale_deg * 3600.0

    cfg = SURVEY_CONFIGS[survey_name]
    fwhm_pixels = cfg['fwhm_arcsec'] / true_pixscale_arcsec
    threshold_val = cfg['detection_sigma'] * std
    centering_tolerance = cfg['centering_tolerance']

    # ── DYNAMIC HALO & SENSITIVITY THRESHOLDS ──
    if not pd.isna(g_mag) and g_mag < 15.0:
        # ZONE 2: Bright Stars.
        prominence_thresh = 0.15
        halo_check_radius = 7.0
        do_heal_debounce = False
    else:
        # ZONE 3: Faint Stars.
        prominence_thresh = 0.1
        halo_check_radius = 4.0
        do_heal_debounce = False

    # ── IMAGE HEALING ──
    if do_heal_debounce:
        det_data = ndimage.grey_closing(data_clean, size=(9, 9))
        det_data = ndimage.gaussian_filter(det_data, sigma=1.0)
    else:
        det_data = data_clean

    # ── SHAPE FILTERED DETECTION ──
    daofind = DAOStarFinder(
        fwhm=fwhm_pixels, 
        threshold=threshold_val,
        sharplo=0.4, sharphi=2.0,   
        roundlo=-1.0, roundhi=1.0   
    )
    sources = daofind(det_data - median)
    
    if sources is None or len(sources) == 0:
        return 0, False 

    # ── SPATIAL DEBOUNCING (SHRAPNEL FILTER) ──
    if do_heal_debounce:
        sources.sort('peak', reverse=True)
        valid_indices = []
        debounce_radius = fwhm_pixels * 1.5
        for i in range(len(sources)):
            keep = True
            for j in valid_indices:
                dist = np.hypot(sources['xcentroid'][i] - sources['xcentroid'][j],
                                sources['ycentroid'][i] - sources['ycentroid'][j])
                if dist < debounce_radius:
                    keep = False
                    break
            if keep:
                valid_indices.append(i)
        sources = sources[valid_indices]
    
    sigma_pixels = fwhm_pixels / 2.35482 
    theoretical_coord = SkyCoord(ra=theoretical_ra, dec=theoretical_dec, unit=(u.deg, u.deg), frame='icrs')
    
    x_positions = sources['xcentroid']
    y_positions = sources['ycentroid']
    peak_values = sources['peak'] 
    
    # 1. Calculate distance from the theoretical map center to find the target
    detected_coords = wcs.pixel_to_world(x_positions, y_positions)
    separations_from_theoretical = theoretical_coord.separation(detected_coords).arcsec
    
    valid_sources_in_zone = 0
    found_central_source = False
    
    # --- FIND THE OBSERVED TARGET ---
    min_sep_index = np.argmin(separations_from_theoretical)
    min_sep_val = separations_from_theoretical[min_sep_index]
    target_index = -1
    
    if min_sep_val <= centering_tolerance:
        found_central_source = True
        target_index = min_sep_index
        valid_sources_in_zone += 1  # Count the target itself
        
        # ANCHOR TO THE TARGET: Recompute distances relative to the actual star
        anchor_coord = detected_coords[target_index]
        separations_to_use = anchor_coord.separation(detected_coords).arcsec
    else:
        # If no target found, just measure from the center of the image
        separations_to_use = separations_from_theoretical
    
    # --- ITERATE SOURCES USING THE ANCHORED DISTANCES ---
    for i, (sep, peak, cx, cy) in enumerate(zip(separations_to_use, peak_values, x_positions, y_positions)):
        if i == target_index:
            continue 

        # --- THE PROFILE CHECK ---
        # Only check sources inside the dynamic halo radius if we actually found a central star
        if found_central_source and sep < halo_check_radius:
            actual_target_x = x_positions[target_index]
            actual_target_y = y_positions[target_index]
            # Pass original data (minus median) for the raw profile check
            if not is_real_source(data_clean - median, actual_target_x, actual_target_y, cx, cy, min_prominence=prominence_thresh):
                continue # Skip this source, it's just halo shine!
        # -------------------------

        if peak > threshold_val:
            wing_radius_pixels = sigma_pixels * np.sqrt(2 * np.log(peak / threshold_val))
        else:
            wing_radius_pixels = 0
            
        wing_radius_arcsec = (wing_radius_pixels * true_pixscale_arcsec) * wing_multiplier
        
        if (sep - wing_radius_arcsec) <= search_radius_arcsec:
            valid_sources_in_zone += 1

    return valid_sources_in_zone, found_central_source


# ── 2. The Fetcher ─────────────────────────────────────────────────────────────

def get_optical_fits(ra, dec, fov_arcsec, search_radius_arcsec, wing_multiplier, g_mag=np.nan):
    """
    Downloads FITS and validates the central source using the Fallback Chain.
    Implements Dynamic Red-Fallback for missing survey coverage.
    """
    for survey_name in FALLBACK_CHAIN:
        cfg = SURVEY_CONFIGS[survey_name]
        pixscale = cfg['pixscale']
        size = max(15, int(np.ceil(fov_arcsec / pixscale)))
        
        for band in cfg['bands']:
            try:
                if survey_name == 'Legacy':
                    url = (
                        f"https://www.legacysurvey.org/viewer/fits-cutout?"
                        f"ra={ra}&dec={dec}&size={size}&layer=ls-dr10"
                        f"&pixscale={pixscale}&bands={band}"
                    )
                else:
                    hips = requests.utils.quote(f"{cfg['hips_base']}{band}", safe='/')
                    fov_deg = fov_arcsec / 3600.0
                    url = (
                        f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits?"
                        f"hips={hips}&width={size}&height={size}"
                        f"&fov={fov_deg}&projection=TAN&coordsys=icrs"
                        f"&ra={ra}&dec={dec}&format=fits"
                    )

                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    hdul = fits.open(BytesIO(resp.content))
                    data = np.squeeze(hdul[0].data)
                    header = hdul[0].header
                    
                    # --- STRICT PRE-FLIGHT VALIDATION ---
                    # Reject if data is missing, completely flat, or mostly NaNs/Zeros
                    if data is None or data.size == 0:
                        continue
                    
                    # Count valid (non-NaN, non-zero) pixels
                    valid_pixels = np.isfinite(data) & (data != 0.0)
                    if np.sum(valid_pixels) < (0.05 * data.size) or np.ptp(data[valid_pixels]) == 0:
                        continue # Reject empty footprints masquerading as valid FITS
                    # ------------------------------------

                    header['BANDUSED'] = band 
                    
                    num_opt, found = count_sources_in_image(
                        data, header, ra, dec, survey_name, search_radius_arcsec, wing_multiplier, g_mag
                    )
                    
                    if found or survey_name == FALLBACK_CHAIN[-1]:
                         return data, header, survey_name, num_opt, found
                         
            except Exception:
                pass # If it fails, silently move to the next bluest band

    return None, None, 'Failed', 0, False


# ── 3. The Main Dataframe Iterator ─────────────────────────────────────────────

def flag_image_contamination(df, fov_arcsec=30, search_radius_arcsec=9.3, wing_multiplier=1.0, verbose=False):
    total_sources = len(df)
    if verbose:
        print(f"Running Optical Astrometric (WCS) PSF filtering on {total_sources} sources...")
        print(f"Exact Search Radius: {search_radius_arcsec} arcsec.")
        print(f"Hierarchy: {' -> '.join(FALLBACK_CHAIN)}\n")
    
    is_contaminated, notes = [], []
    
    for i, (index, row) in enumerate(df.iterrows(), start=1):
        sid = row['source_id']
        ra = row['ra']
        dec = row['dec']
        
        # Smartly grab the G magnitude if it exists, otherwise fetch it.
        g_mag = np.nan
        if 'phot_g_mean_mag' in row and not pd.isna(row['phot_g_mean_mag']):
            g_mag = row['phot_g_mean_mag']
        else:
            try:
                coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
                res = Gaia.cone_search_async(coord, radius=u.Quantity(2.0, u.arcsec)).get_results().to_pandas()
                if not res.empty:
                    res['dist'] = np.hypot(res['ra'] - ra, res['dec'] - dec)
                    g_mag = res.sort_values('dist').iloc[0]['phot_g_mean_mag']
            except Exception:
                pass
        
        # Pass the g_mag to the FITS fetcher
        data_opt, head_opt, survey_opt, num_opt, found_center = get_optical_fits(ra, dec, fov_arcsec, search_radius_arcsec, wing_multiplier, g_mag)
        
        if data_opt is not None:
            # Extract the band from the header for nice logging
            band = head_opt.get('BANDUSED', '')
            band_str = f" ({band}-band)" if band else ""
            survey_label = f"{survey_opt}{band_str}"

            if not found_center:
                if num_opt <= 1:
                    is_contaminated.append(False)
                    msg = f"⚠️ Source {sid}: Target missing from center in {survey_label}, {num_opt} other source(s) found. Kept as clean."
                    if verbose: print(f"[{i}/{total_sources}] {msg}")
                    notes.append(msg)
                else:
                    is_contaminated.append(True)
                    msg = f"🔴 Source {sid}: Target missing from center in {survey_label}, {num_opt} other source(s) found."
                    if verbose: print(f"[{i}/{total_sources}] {msg}")
                    notes.append(msg)
            else:
                if num_opt==1:
                    is_contaminated.append(False)
                    msg = f"🟢 Source {sid}: Clean isolated target in {survey_label}"
                    if verbose: print(f"[{i}/{total_sources}] {msg}")
                    notes.append(msg)
                else:
                    is_contaminated.append(True)
                    msg = f"🔴 Source {sid}: {num_opt-1} contaminants found in {survey_label}"
                    if verbose: print(f"[{i}/{total_sources}] {msg}")
                    notes.append(msg)
        else:
            is_contaminated.append(False)
            msg = f"❌ Source {sid}: All optical downloads failed or returned empty data. Kept as clean."
            print(msg)
            notes.append(msg)

        time.sleep(0.3) 

    df = df.copy()
    
    df['is_contaminated_image'] = is_contaminated
    df['note_image'] = notes
    
    if verbose:
        print("\n✅ Flagging complete!")
    
    return df


FALLBACK_CHAIN_SLOT1 = ['Legacy',  'PanSTARRS', 'SDSS', 'DSS2']
FALLBACK_CHAIN_SLOT2 = ['PanSTARRS', 'SDSS', 'DSS2']

# ── Fetchers ───────────────────────────────────────────────────────────────────

def fetch_survey_fits(ra, dec, fov_arcsec, survey_name):
    """Downloads FITS cutout using Dynamic Red-Fallback and NaN-safe validation."""
    cfg = SURVEY_CONFIGS.get(survey_name)
    if cfg is None:
        return None, None, None

    pixscale = cfg['pixscale']
    size     = max(20, int(np.ceil(fov_arcsec / pixscale)))

    for band in cfg['bands']:
        try:
            if survey_name == 'Legacy':
                url = (
                    f"https://www.legacysurvey.org/viewer/fits-cutout?"
                    f"ra={ra}&dec={dec}&size={size}&layer=ls-dr10"
                    f"&pixscale={pixscale}&bands={band}"
                )
            else:
                hips    = requests.utils.quote(f"{cfg['hips_base']}{band}", safe='/')
                fov_deg = fov_arcsec / 3600.0
                url = (
                    f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits?"
                    f"hips={hips}&width={size}&height={size}"
                    f"&fov={fov_deg}&projection=TAN&coordsys=icrs"
                    f"&ra={ra}&dec={dec}&format=fits"
                )

            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                hdul = fits.open(BytesIO(resp.content))
                data = np.squeeze(hdul[0].data)
                header = hdul[0].header

                # --- NaN-SAFE VALIDATION ---
                if data is None or data.size == 0:
                    continue
                
                valid_pixels = np.isfinite(data) & (data != 0.0)
                if np.sum(valid_pixels) < (0.05 * data.size) or np.ptp(data[valid_pixels]) == 0:
                    continue # Reject mostly empty or perfectly flat images
                # ---------------------------

                header['BANDUSED'] = band # Attach the band to the header for the plotter
                return data, header, survey_name

        except Exception:
            pass

    return None, None, None


def fetch_with_fallback(ra, dec, fov_arcsec, chain, exclude_list=None):
    """Walks the chain, skipping exclusions, returning the first valid image."""
    if exclude_list is None:
        exclude_list = []
        
    for survey_name in chain:
        if survey_name in exclude_list:
            print(f"    Skipping {survey_name} (Already showing in other slot)")
            continue
            
        print(f"    Trying {survey_name} …", end=" ")
        data, header, name = fetch_survey_fits(ra, dec, fov_arcsec, survey_name)
        if data is not None:
            band = header.get('BANDUSED', '')
            print(f"✓ ({band}-band)")
            return data, header, name
        print("✗ (no data / no coverage)")
    return None, None, None


# ── Main plotting function ─────────────────────────────────────────────────────

def plot_survey_comparison(source_id, ra, dec, g_mag=None, fov_arcsec=30, search_radius_arcsec=9.3, wing_multiplier=1.0):
    """
    Fetches and plots a side-by-side visual comparison of the target across multiple optical surveys.
    
    This function utilizes a fallback hierarchy to fetch the two highest-quality 
    available images for a given coordinate (e.g., trying DESI Legacy first, then 
    PanSTARRS, SDSS, and DSS2). It applies World Coordinate System (WCS) astrometry 
    to pinpoint the theoretical target location, detects observed sources using 
    DAOStarFinder, and visualizes potential contamination by drawing dynamic 
    Point Spread Function (PSF) wings based on each survey's optical properties.

    To ensure high accuracy, the function implements a magnitude-dependent artifact 
    filtering strategy. It automatically skips detection for heavily saturated bright 
    stars (G < 13) to avoid "shredding" artifacts, and applies dynamic saddle-point 
    deblending (profile checking) to distinguish real faint companions from artificial 
    halo shine on fainter targets.

    Args:
        source_id (int or str): The unique identifier for the target (used for the plot title).
        ra (float): Right Ascension of the target in degrees (ICRS).
        dec (float): Declination of the target in degrees (ICRS).
        g_mag (float, optional): The Gaia G-band mean magnitude. Dictates artifact 
            filter sensitivity and whether image fitting is bypassed. If None, it 
            will be automatically queried from the Gaia database prior to plotting.
        fov_arcsec (float, optional): The field of view for the downloaded FITS 
            cutouts in arcseconds. Defaults to 30.
        search_radius_arcsec (float, optional): The radial distance in arcseconds to 
            check for neighboring sources. Defaults to 9.3.
        wing_multiplier (float, optional): A tuning parameter that scales the 
            calculated physical extent of the PSF wings. A value > 1.0 makes the 
            contamination check more conservative/aggressive. Defaults to 1.0.

    Returns:
        None: This function does not return any variables. It directly displays 
        a matplotlib figure showing the two image panels with overlaid astrometric 
        and contamination diagnostics.
    """
    print(f"\nFetching images for Source {source_id} (ra={ra:.5f}, dec={dec:.5f})")
    
    # ── 1. Fetch Gaia Magnitude by RA/DEC ──
    if g_mag is None or pd.isna(g_mag):
        g_mag = np.nan
        try:
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
            res = Gaia.cone_search_async(coord, radius=u.Quantity(2.0, u.arcsec)).get_results().to_pandas()
            if not res.empty:
                res['dist'] = np.hypot(res['ra'] - ra, res['dec'] - dec)
                g_mag = res.sort_values('dist').iloc[0]['phot_g_mean_mag']
        except Exception:
            pass

    is_bright = False
    if not np.isnan(g_mag) and g_mag < 13:
        is_bright = True
        print(f"  ⚠️ Target is bright (G={g_mag:.2f}). Skipping DAOStarFinder fitting to avoid shredding.")
    
    print("  Slot 1:")
    data1, head1, name1 = fetch_with_fallback(ra, dec, fov_arcsec, FALLBACK_CHAIN_SLOT1)
    
    print("  Slot 2:")
    exclude = [name1] if name1 else []
    data2, head2, name2 = fetch_with_fallback(ra, dec, fov_arcsec, FALLBACK_CHAIN_SLOT2, exclude_list=exclude)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    def plot_axis(ax, data, header, survey_name):
        if survey_name is None or data is None:
            ax.text(0.5, 0.5, "All surveys exhausted\n(no data / no sky coverage)",
                    ha='center', va='center', transform=ax.transAxes, color='red', fontsize=12)
            ax.set_title("N/A", fontweight='bold')
            ax.axis('off')
            return

        cfg = SURVEY_CONFIGS[survey_name]
        
        # Extract the dynamically selected band from the FITS header
        band = header.get('BANDUSED', '')
        band_str = f" ({band}-band)" if band else ""
        
        # ── Astrometry ──
        wcs = WCS(header).celestial
        true_pixscale_arcsec = proj_plane_pixel_scales(wcs)[0] * 3600.0

        vmin, vmax = ZScaleInterval().get_limits(data)
        ax.imshow(data, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)

        target_coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
        target_x, target_y = wcs.world_to_pixel(target_coord)

        # ALWAYS plot the central marker and bounds, even if skipping detection
        ax.add_patch(patches.Circle((target_x, target_y), search_radius_arcsec / true_pixscale_arcsec,
                                    linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'))
        ax.plot(target_x, target_y, 'x', color='black', markersize=10, markeredgewidth=4)
        ax.plot(target_x, target_y, 'x', color='blue', markersize=6, markeredgewidth=2)

        # ── Bypass Detection for Bright Sources ──
        if is_bright:
            info = f"Fitting Skipped\nTarget too bright (G={g_mag:.2f})"
            ax.text(0.95, 0.05, info, ha='right', va='bottom', transform=ax.transAxes, 
                    color='blue', fontweight='bold', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))
        else:
            # ── Standard Detection ──
            data_clean = np.nan_to_num(data, nan=np.nanmedian(data))
            
            fwhm_pix = cfg['fwhm_arcsec'] / true_pixscale_arcsec
            _, median, std = sigma_clipped_stats(data_clean, sigma=3.0)
            std = max(std, 0.001) 
            
            threshold = cfg['detection_sigma'] * std
            daofind = DAOStarFinder(
                fwhm=fwhm_pix, 
                threshold=threshold,
                sharplo=0.4, sharphi=2.0,   
                roundlo=-1.0, roundhi=1.0   
            )
            sources = daofind(data_clean - median)

            found_tgt, n_contam, total_sources = False, 0, 0
            
            if sources is not None and len(sources) > 0:
                total_sources = len(sources)
                x_col, y_col, peak_col = sources['xcentroid'], sources['ycentroid'], sources['peak']
                sigma_pix = fwhm_pix / 2.35482

                # Find distance to theoretical center to identify target
                seps_theoretical = [np.hypot(x_col[i] - target_x, y_col[i] - target_y) * true_pixscale_arcsec for i in range(total_sources)]
                min_idx = int(np.argmin(seps_theoretical))
                
                # ANCHOR LOGIC: Re-measure distances relative to the observed star
                if seps_theoretical[min_idx] <= cfg['centering_tolerance']:
                    found_tgt = True
                    tgt_idx = min_idx
                    actual_target_x = x_col[tgt_idx]
                    actual_target_y = y_col[tgt_idx]
                    seps = [np.hypot(x_col[i] - actual_target_x, y_col[i] - actual_target_y) * true_pixscale_arcsec for i in range(total_sources)]
                else:
                    tgt_idx = -1
                    seps = seps_theoretical

                # Determine how aggressive the artifact filter should be based on target brightness
                if not np.isnan(g_mag) and g_mag < 15.0:
                    # ZONE 2: Bright Stars. Large noisy halo, aggressive filtering.
                    prominence_thresh = 0.15
                    halo_check_radius = 7.0
                else:
                    # ZONE 3: Faint Stars. Small halo, high sensitivity.
                    prominence_thresh = 0.1
                    halo_check_radius = 4.0

                for i in range(total_sources):
                    cx, cy, cp, sep = x_col[i], y_col[i], peak_col[i], seps[i]

                    if i == tgt_idx:
                        ax.plot(cx, cy, 'o', color='black', markersize=8, markeredgewidth=0)
                        ax.plot(cx, cy, 'o', color='blue', mfc='none', markersize=8, markeredgewidth=2)
                        continue

                    # --- THE PROFILE CHECK ---
                    # Only check sources inside the dynamic halo radius
                    if found_tgt and sep < halo_check_radius:
                        # Pass the dynamic prominence threshold to the function
                        if not is_real_source(data_clean - median, actual_target_x, actual_target_y, cx, cy, min_prominence=prominence_thresh):
                            continue # Skip this source, it's just halo shine!
                    # -------------------------

                    wing_pix = (sigma_pix * np.sqrt(2 * np.log(cp / threshold)) if cp > threshold else 0)
                    wing_arcsec = wing_pix * true_pixscale_arcsec * wing_multiplier

                    is_contam = (sep - wing_arcsec) <= search_radius_arcsec
                    color = 'red' if is_contam else 'green'
                    if is_contam: n_contam += 1

                    ax.plot(cx, cy, 'x', color='black', markersize=8, markeredgewidth=3)
                    ax.plot(cx, cy, 'x', color=color, markersize=6, markeredgewidth=2)
                    ax.add_patch(patches.Circle((cx, cy), wing_pix * wing_multiplier,
                                                linewidth=2, edgecolor=color, facecolor='none', linestyle=':'))

                # --- MATCH HEADLESS DECISION LOGIC ---
                if found_tgt:
                    if n_contam == 0:
                        status_text = "Clean"
                        txt_color = 'green'
                    else:
                        status_text = "Contaminated"
                        txt_color = 'red'
                else:
                    if n_contam <= 1:
                        status_text = "Missing (Kept Clean)"
                        txt_color = 'orange'
                    else:
                        status_text = "Missing & Contaminated"
                        txt_color = 'red'

                info = f"Sources found: {total_sources}\nTarget found: {found_tgt}\nContaminants: {n_contam}\nStatus: {status_text}"
                
                ax.text(0.95, 0.05, info, ha='right', va='bottom', transform=ax.transAxes, 
                        color=txt_color, fontweight='bold', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor=txt_color))
            else:
                ax.text(0.95, 0.05, "No Sources Found", ha='right', va='bottom', transform=ax.transAxes,
                        color='red', fontweight='bold', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

        ax.set_title(f"{cfg['label']}{band_str} ({cfg['detection_sigma']}-sigma)\n({true_pixscale_arcsec:.3f}\"/px) | Bounds: {search_radius_arcsec}\" | Wing: {wing_multiplier}", 
                     fontsize=11, fontweight='bold')
        ax.axis('off')

    plot_axis(axes[0], data1, head1, name1)
    plot_axis(axes[1], data2, head2, name2)

    fig.suptitle(f"Target: {source_id} | Multi-Survey Astrometric Check", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()