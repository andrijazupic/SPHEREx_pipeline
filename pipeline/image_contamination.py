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


# Silence warnings from photutils (empty images) and WCS (non-standard headers)
warnings.filterwarnings('ignore', module='photutils')
warnings.filterwarnings('ignore', module='astropy.wcs')

# ── Survey Registry ────────────────────────────────────────────────────────────
SURVEY_CONFIGS = {
    'Legacy': {
        'hips':                None,
        'pixscale':            0.262,
        'fwhm_arcsec':         2.0,
        'detection_sigma':     6.1,
        'centering_tolerance': 2.0,
        'label':               'DESI Legacy DR10',
    },
    'PanSTARRS': {
        'hips':                'CDS/P/PanSTARRS/DR1/r',
        'pixscale':            0.25,
        'fwhm_arcsec':         2.0,
        'detection_sigma':     6.1,
        'centering_tolerance': 2.0,
        'label':               'PanSTARRS DR1',
    },
    'SDSS': {
        'hips':                'CDS/P/SDSS9/r',
        'pixscale':            0.396,
        'fwhm_arcsec':         2.5,#?
        'detection_sigma':     5.0,#?
        'centering_tolerance': 2.5,#?
        'label':               'SDSS DR9',
    },
    'DSS2': {
        'hips':                'CDS/P/DSS2/red',
        'pixscale':            1.0,
        'fwhm_arcsec':         3.0,
        'detection_sigma':     4.0,
        'centering_tolerance': 5.0,
        'label':               'DSS2',
    },
}

FALLBACK_CHAIN = ['Legacy', 'PanSTARRS', 'SDSS', 'DSS2']

# ── 1. The Headless Fitter ─────────────────────────────────────────────────────

def count_sources_in_image(data, header, theoretical_ra, theoretical_dec, survey_name, search_radius_arcsec, wing_multiplier):
    """
    PUBLICATION-GRADE WCS PSF FITTER.
    Finds the true observed center of the target first, then anchors all
    distance calculations relative to that specific object.
    """
    if not np.any(np.isfinite(data)):
        return 0, False

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    std = max(std, 0.001) 
    
    wcs = WCS(header).celestial
    true_pixscale_deg = proj_plane_pixel_scales(wcs)[0]
    true_pixscale_arcsec = true_pixscale_deg * 3600.0

    cfg = SURVEY_CONFIGS[survey_name]
    fwhm_pixels = cfg['fwhm_arcsec'] / true_pixscale_arcsec
    threshold_val = cfg['detection_sigma'] * std
    centering_tolerance = cfg['centering_tolerance']
    
    daofind = DAOStarFinder(fwhm=fwhm_pixels, threshold=threshold_val)
    sources = daofind(data - median)
    
    if sources is None or len(sources) == 0:
        return 0, False 
    
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
    for i, (sep, peak) in enumerate(zip(separations_to_use, peak_values)):
        if i == target_index:
            continue 

        if peak > threshold_val:
            wing_radius_pixels = sigma_pixels * np.sqrt(2 * np.log(peak / threshold_val))
        else:
            wing_radius_pixels = 0
            
        wing_radius_arcsec = (wing_radius_pixels * true_pixscale_arcsec) * wing_multiplier
        
        if (sep - wing_radius_arcsec) <= search_radius_arcsec:
            valid_sources_in_zone += 1

    return valid_sources_in_zone, found_central_source


# ── 2. The Fetcher ─────────────────────────────────────────────────────────────

def get_optical_fits(ra, dec, fov_arcsec, search_radius_arcsec, wing_multiplier):
    """
    Downloads FITS and validates the central source using the Fallback Chain.
    """
    for survey_name in FALLBACK_CHAIN:
        cfg = SURVEY_CONFIGS[survey_name]
        pixscale = cfg['pixscale']
        size = max(15, int(np.ceil(fov_arcsec / pixscale)))
        
        try:
            if survey_name == 'Legacy':
                url = (
                    f"https://www.legacysurvey.org/viewer/fits-cutout?"
                    f"ra={ra}&dec={dec}&size={size}&layer=ls-dr10"
                    f"&pixscale={pixscale}&bands=r"
                )
            else:
                hips = requests.utils.quote(cfg['hips'], safe='/')
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
                
                if data is not None and data.size > 0 and np.any(np.isfinite(data)):
                    num_opt, found = count_sources_in_image(
                        data, header, ra, dec, survey_name, search_radius_arcsec, wing_multiplier
                    )
                    
                    if found or survey_name == FALLBACK_CHAIN[-1]:
                         return data, header, survey_name
        except Exception:
            pass

    return None, None, 'Failed'


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
        
        data_opt, head_opt, survey_opt = get_optical_fits(ra, dec, fov_arcsec, search_radius_arcsec, wing_multiplier)
        
        if data_opt is not None:
            num_opt, found_center = count_sources_in_image(
                data_opt, head_opt, ra, dec, survey_opt, search_radius_arcsec, wing_multiplier
            )

            if not found_center:
                if num_opt <= 1:
                    is_contaminated.append(False)
                    msg = f"⚠️ Source {sid}: Target missing from center in {survey_opt}, {num_opt} other source(s) found. Kept as clean."
                    if verbose:
                        print(f"[{i}/{total_sources}] {msg}")
                    notes.append(msg)
                else:
                    is_contaminated.append(True)
                    msg = f"🔴 Source {sid}: Target missing from center in {survey_opt}, {num_opt} other source(s) found."
                    if verbose:
                        print(f"[{i}/{total_sources}] {msg}")
                    notes.append(msg)
            else:
                if num_opt==1:
                    is_contaminated.append(False)
                    msg = f"🟢 Source {sid}: Clean isolated target in {survey_opt}"
                    if verbose:
                        print(f"[{i}/{total_sources}] {msg}")
                    notes.append(msg)
                else:
                    is_contaminated.append(True)
                    msg = f"🔴 Source {sid}: {num_opt-1} contaminants found in {survey_opt}"
                    if verbose:
                        print(f"[{i}/{total_sources}] {msg}")
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
    """Downloads FITS cutout. Legacy uses bespoke API, others use HiPS2FITS."""
    cfg = SURVEY_CONFIGS.get(survey_name)
    if cfg is None:
        return None, None, None

    pixscale = cfg['pixscale']
    size     = max(20, int(np.ceil(fov_arcsec / pixscale)))

    try:
        if survey_name == 'Legacy':
            url = (
                f"https://www.legacysurvey.org/viewer/fits-cutout?"
                f"ra={ra}&dec={dec}&size={size}&layer=ls-dr10"
                f"&pixscale={pixscale}&bands=r"
            )
        else:
            hips    = requests.utils.quote(cfg['hips'], safe='/')
            fov_deg = fov_arcsec / 3600.0
            url = (
                f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits?"
                f"hips={hips}&width={size}&height={size}"
                f"&fov={fov_deg}&projection=TAN&coordsys=icrs"
                f"&ra={ra}&dec={dec}&format=fits"
            )

        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None, None, None

        hdul = fits.open(BytesIO(resp.content))
        data = np.squeeze(hdul[0].data)
        header = hdul[0].header

        if data is None or data.size == 0 or not np.any(np.isfinite(data)):
            return None, None, None

        return data, header, survey_name

    except Exception:
        return None, None, None


def fetch_with_fallback(ra, dec, fov_arcsec, chain, exclude_list=[]):
    """Walks the chain, skipping exclusions, returning the first valid image."""
    for survey_name in chain:
        if survey_name in exclude_list:
            print(f"    Skipping {survey_name} (Already showing in other slot)")
            continue
            
        print(f"    Trying {survey_name} …", end=" ")
        data, header, name = fetch_survey_fits(ra, dec, fov_arcsec, survey_name)
        if data is not None:
            print("✓")
            return data, header, name
        print("✗ (no data / no coverage)")
    return None, None, None


# ── Main plotting function ─────────────────────────────────────────────────────

def plot_survey_comparison(source_id, ra, dec, fov_arcsec=30, search_radius_arcsec=9.3, wing_multiplier=1.0):
    """
    Fetches and plots a side-by-side visual comparison of the target across multiple optical surveys.
    
    This function utilizes a fallback hierarchy to fetch the two highest-quality 
    available images for a given coordinate (e.g., trying DESI Legacy first, then 
    PanSTARRS, SDSS, and DSS2). It applies World Coordinate System (WCS) astrometry 
    to pinpoint the theoretical target location, detects observed sources using 
    DAOStarFinder, and visualizes potential contamination by drawing dynamic 
    Point Spread Function (PSF) wings based on each survey's optical properties.

    Args:
        source_id (int or str): The unique identifier for the target (used for the plot title).
        ra (float): Right Ascension of the target in degrees (ICRS).
        dec (float): Declination of the target in degrees (ICRS).
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
        
        # ── Astrometry ──
        wcs = WCS(header).celestial
        true_pixscale_arcsec = proj_plane_pixel_scales(wcs)[0] * 3600.0

        vmin, vmax = ZScaleInterval().get_limits(data)
        ax.imshow(data, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)

        target_coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
        target_x, target_y = wcs.world_to_pixel(target_coord)

        # ── Detection ──
        fwhm_pix = cfg['fwhm_arcsec'] / true_pixscale_arcsec
        _, median, std = sigma_clipped_stats(data, sigma=3.0)
        std = max(std, 0.001) 
        
        # FIXED: Now pulling threshold from the config file!
        threshold = cfg['detection_sigma'] * std

        sources = DAOStarFinder(fwhm=fwhm_pix, threshold=threshold)(data - median)

        ax.add_patch(patches.Circle((target_x, target_y), search_radius_arcsec / true_pixscale_arcsec,
                                    linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'))
        
        ax.plot(target_x, target_y, 'x', color='black', markersize=10, markeredgewidth=4)
        ax.plot(target_x, target_y, 'x', color='blue', markersize=6, markeredgewidth=2)

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

            for i in range(total_sources):
                cx, cy, cp, sep = x_col[i], y_col[i], peak_col[i], seps[i]

                if i == tgt_idx:
                    ax.plot(cx, cy, 'o', color='black', markersize=8, markeredgewidth=0)
                    ax.plot(cx, cy, 'o', color='blue', mfc='none', markersize=8, markeredgewidth=2)
                    continue

                wing_pix = (sigma_pix * np.sqrt(2 * np.log(cp / threshold)) if cp > threshold else 0)
                wing_arcsec = wing_pix * true_pixscale_arcsec * wing_multiplier

                is_contam = (sep - wing_arcsec) <= search_radius_arcsec
                color = 'red' if is_contam else 'green'
                if is_contam: n_contam += 1

                ax.plot(cx, cy, 'x', color='black', markersize=8, markeredgewidth=3)
                ax.plot(cx, cy, 'x', color=color, markersize=6, markeredgewidth=2)
                ax.add_patch(patches.Circle((cx, cy), wing_pix * wing_multiplier,
                                            linewidth=2, edgecolor=color, facecolor='none', linestyle=':'))

            info = f"Sources found: {total_sources}\nTarget found: {found_tgt}\nContaminated: {n_contam}"
            txt_color = 'green' if found_tgt and n_contam == 0 else 'orange'
            ax.text(0.95, 0.05, info, ha='right', va='bottom', transform=ax.transAxes, 
                    color=txt_color, fontweight='bold', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=txt_color))
        else:
            ax.text(0.95, 0.05, "No Sources Found", ha='right', va='bottom', transform=ax.transAxes,
                    color='red', fontweight='bold', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

        ax.set_title(f"{cfg['label']} ({cfg['detection_sigma']}-sigma)\n({true_pixscale_arcsec:.3f}\"/px) | Bounds: {search_radius_arcsec}\" | Wing: {wing_multiplier}", 
                     fontsize=11, fontweight='bold')
        ax.axis('off')

    plot_axis(axes[0], data1, head1, name1)
    plot_axis(axes[1], data2, head2, name2)

    fig.suptitle(f"Target: {source_id} | Multi-Survey Astrometric Check", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()