import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import shutil
from pathlib import Path
from spxquery.core.pipeline import run_pipeline

# ==========================================
# 1. Your Saved Calibration Data
# ==========================================
SAVED_CALIBRATION = {
    1: {'slope': -0.030978, 'intercept': 1.443219},
    2: {'slope': -0.032349, 'intercept': 1.457172},
    3: {'slope': -0.042253, 'intercept': 1.496206},
    4: {'slope': -0.024838, 'intercept': 1.424380},
    5: {'slope': -0.047032, 'intercept': 1.523252},
    6: {'slope': -0.023907, 'intercept': 1.429536},
    7: {'slope': -0.063772, 'intercept': 1.585484},
    8: {'slope': -0.020332, 'intercept': 1.419779},
    9: {'slope': -0.053909, 'intercept': 1.536003},
    10: {'slope': 0.052778, 'intercept': 1.060467},
}

STANDARD_BIN_EDGES = [0.74, 0.95, 1.15, 1.40, 1.65, 2.05, 2.45, 2.90, 3.50, 4.15, 5.05]

# Calculate the exact center of each bin for interpolation
BIN_CENTERS = [(STANDARD_BIN_EDGES[i] + STANDARD_BIN_EDGES[i+1]) / 2 for i in range(10)]

# ==========================================
# 2. The Smooth Interpolating Engine
# ==========================================
def apply_calibration_to_dataframe(df, cal_funcs=SAVED_CALIBRATION, bin_centers=BIN_CENTERS):
    """
    Takes a raw SPXQuery DataFrame and applies a smoothly interpolated
    correction factor to avoid discontinuities at bin edges.
    """
    df = df.copy()
    
    # Pre-extract the slopes and intercepts into fast numpy arrays
    slopes = np.array([cal_funcs[i]['slope'] for i in range(1, 11)])
    intercepts = np.array([cal_funcs[i]['intercept'] for i in range(1, 11)])
    
    def get_smooth_correction(row):
        w = row['wavelength']
        flux = row['flux']
        
        # Safety check for negative/zero flux
        if flux <= 0:
            return 1.0
            
        # 1. Calculate the ideal factor for ALL 10 bins simultaneously based on this specific flux
        log_flux = np.log10(flux)
        all_bin_factors = slopes * log_flux + intercepts
        
        # 2. Interpolate the exact factor based on where 'w' falls between the bin centers
        # np.interp automatically handles wavelengths that fall outside the very first or last center 
        # by clamping them to the nearest edge (extrapolation).
        interpolated_factor = np.interp(w, bin_centers, all_bin_factors)
        
        # 3. Physics Clamp
        return max(1.0, interpolated_factor)

    # Apply the function to every row
    df['correction_factor'] = df.apply(get_smooth_correction, axis=1)
    
    # Calculate final corrected fluxes and errors
    df['corrected_flux'] = df['flux'] * df['correction_factor']
    df['corrected_flux_error'] = df['flux_error'] * df['correction_factor']
    
    return df


# ==========================================
# 3. Your Updated Extraction Function (FLATTENED OUTPUT)
# ==========================================
import pandas as pd
import shutil
from pathlib import Path

def spxquery_get_spectrum_calibrated(source_id, ra, dec, calibrate=True, outdir="spxquery_results", verbose=False):
    """
    Queries SPHEREx data. 
    If calibrate=True, applies empirical aperture calibration and adds 3 new columns.
    If calibrate=False, moves the raw file untouched.
    Preserves all original columns and the '#' metadata header.
    Deletes ALL temporary pipeline folders.
    """
    source_id_str = str(source_id)
    base_dir = Path("./" + outdir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # The final target file: spxquery_results/318381458887086976.csv
    final_csv_path = base_dir / f"{source_id_str}.csv"
    
    # The temporary directories used by the pipeline
    temp_output_dir = base_dir / source_id_str
    temp_csv_path = temp_output_dir / "results" / "lightcurve.csv"

    # 1. Check if we already have the finalized file
    if final_csv_path.exists():
        if verbose:
            print(f"Skipping pipeline: {final_csv_path.name} already exists.")
        return pd.read_csv(final_csv_path, comment="#")

    # 2. Run Pipeline (writes to temporary nested folder)
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Assuming 'run_pipeline' is defined elsewhere in your code
        run_pipeline(
            ra=ra, dec=dec, source_name=source_id_str,
            output_dir=temp_output_dir, max_processing_workers=10, cutout_size="20px"
        )
    except Exception as e:
        print(f"Error running pipeline for {source_id_str}: {e}")
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)
        return pd.DataFrame()

    # 3. Extract, Calibrate (if requested), Save to Root, and Erase Temp Folders
    if temp_csv_path.exists():
        
        if not calibrate:
            # ---> RAW MODE: Just move the file untouched <---
            shutil.move(str(temp_csv_path), str(final_csv_path))
            if verbose:
                print(f"Skipping Aperture Correction. Saved raw data to {final_csv_path.name}")
            final_df = pd.read_csv(final_csv_path, comment="#")
            
        else:
            # ---> CALIBRATION MODE: Preserve header, append new columns <---
            
            # Step A: Extract the original '#' header block
            header_lines = []
            with open(temp_csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#'):
                        header_lines.append(line)
                    else:
                        break # Stop reading once we hit the data
            
            # Step B: Load the full dataframe (all columns preserved)
            df = pd.read_csv(temp_csv_path, comment="#")
            if verbose:
                print(f"Applying Aperture Correction to {len(df)} observations...")
                
            # Apply your smooth calibration function
            final_df = apply_calibration_to_dataframe(df)
            
            # Rename the columns to exactly match your requested names
            if 'corrected_flux' in final_df.columns:
                final_df.rename(columns={
                    'corrected_flux': 'flux_calib', 
                    'corrected_flux_error': 'flux_error_calib'
                }, inplace=True)
            
            # Step C: Write the file back with the original header
            with open(final_csv_path, 'w', encoding='utf-8') as f:
                f.writelines(header_lines)
                final_df.to_csv(f, index=False)
                
            if verbose:
                print(f"Saved calibrated data to {final_csv_path.name}")

        # ---> FULL CLEANUP: Delete the entire source folder tree <---
        shutil.rmtree(temp_output_dir)
        if verbose:
            print(f"Cleanup: Deleted all temporary pipeline folders for {source_id_str}.")
        
        return final_df
    
    # Fallback cleanup if the pipeline ran but failed to create lightcurve.csv
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)
        print(f"Pipeline failed to produce data for {source_id_str}. Deleted temp folders.")

    return pd.DataFrame()