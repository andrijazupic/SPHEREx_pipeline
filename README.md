# SPHEREx Pipeline

A data pipeline designed to flag blended SPHEREx sources using high-resolution optical catalogs, and retrieve fast, bulk spectra via SPXQuery with empirical calibrations to match SPIFF PSF-photometry.

## Installation

Clone the repository to your local machine and install the required dependencies:

```bash
git clone [https://github.com/andrijazupic/SPHEREx_pipeline.git](https://github.com/andrijazupic/SPHEREx_pipeline.git)
cd SPHEREx_pipeline
# Recommended: Create and activate a virtual environment (venv/conda) here
pip install -r requirements.txt

```

## Source Decontamination

Due to the large pixel scale of SPHEREx, isolated targets can easily be contaminated by unresolved background sources. The contamination pipeline cross-references targets against high-resolution optical catalogs and direct FITS images to flag or remove blended sources. A tutorial is available in `tutorial_contamination.ipynb`.

### 1. Catalog-Level Filtering

The pipeline queries four successive survey catalogs to identify neighboring sources within a defined search radius:

* **Gaia DR3:** Executes synchronous ADQL queries to identify basic positional blends within the search radius.
* **DESI Legacy DR10 (Tractor):** Anchors to the central target to evaluate nearest neighbors. Filters noise using Signal-to-Noise Ratio (SNR) limits and identifies extended background sources using Tractor morphological classifications and r-band flux.
* **Pan-STARRS DR1:** Anchors to the central target to evaluate nearest neighbors. Filters noise using Signal-to-Noise Ratio (SNR) limits and identifies extended background sources using the difference between PSF and Kron magnitudes in the r-band.
* **SDSS DR16:** Anchors to the central target to evaluate nearest neighbors. Filters noise using Signal-to-Noise Ratio (SNR) limits and identifies extended background sources using SDSS morphological classifications in the r-band.

### 2. Image-Level Filtering

For sources that pass catalog checks, the pipeline performs direct image analysis using a World Coordinate System (WCS) PSF fitter:

* **Fallback Hierarchy:** Sequentially attempts to download optical FITS cutouts from the highest available resolution survey: DESI Legacy $\rightarrow$ PanSTARRS $\rightarrow$ SDSS $\rightarrow$ DSS2.
* **Magnitude-Dependent Strategy:** Utilizes Gaia G-band magnitudes to dynamically adjust the detection algorithms and prevent false positives from bright star artifacts:
* **Bright (G < 13):** Bypasses image-level fitting entirely to prevent diffraction spikes and saturation bleeding from being misidentified as contaminants.
* **Moderate (13 $\le$ G < 15):** Applies morphological closing and spatial debouncing before source detection to heal saturated cores and suppress halo noise.
* **Faint (G $\ge$ 15):** Operates at maximum sensitivity without debouncing. Applies shape filtering and saddle-point deblending (profile checking) to distinguish genuine faint background companions from the gentle gradient of a stellar halo.


* **Dynamic Flagging:** Identifies the true observed center of the target, computes dynamic Point Spread Function (PSF) wings for all surrounding detections based on the survey's pixel scale and FWHM, and flags sources whose wings intersect the target's exclusion radius.

### Usage

The pipeline is executed via the `spherex_contamination_analysis` wrapper function, which runs the input dataframe through all catalog and image filters sequentially.

**Key Parameters:**

* `df`: The input pandas DataFrame. Must contain `source_id`, `ra`, and `dec` columns. It is highly recommended to also include a `phot_g_mean_mag` (Gaia G-band) column to drive the image-level filtering logic. If missing, the pipeline will automatically query the Gaia database to fetch the magnitude.
* `search_radius_arcsec`: The radial distance to check for blending (default: 9.3").
* `remove_contaminated`:
* If `True`: Drops contaminated rows from the dataframe at each step, returning only purely isolated sources.
* If `False`: Retains all rows. Appends boolean flag columns (e.g., `is_contaminated_gaia`) and diagnostic notes for each survey, culminating in a master `is_contaminated` flag.


* `verbose`: Enables step-by-step console logging.

**Example Implementation:**

```python
import pandas as pd
from spherex_contamination_analysis import spherex_contamination_analysis

# Load your target catalog
df = pd.read_csv("my_targets.csv")

# Run the contamination pipeline
result_df = spherex_contamination_analysis(
    df, 
    search_radius_arcsec=9.3, 
    remove_contaminated=False, 
    verbose=True
)

# Inspect results
print(result_df[['source_id', 'is_contaminated', 'note_image']].head())

```

### Visual Diagnostics

The pipeline includes a `plot_survey_comparison` function to visually verify the automated flagging. It fetches the two highest-resolution available optical cutouts, applies the same magnitude-dependent artifact filtering used in the main pipeline, calculates dynamic PSF wings, and overlays the 9.3" SPHEREx exclusion zone.

**Example Implementation:**

```python
from image_contamination import plot_survey_comparison

# Visualize the two highest-resolution available optical cutouts for a single source.
# Note: If g_mag is not explicitly provided, the function will auto-fetch it from Gaia.
plot_survey_comparison(
    source_id=3108936854882971904, 
    ra=105.29821, 
    dec=-2.431236, 
    g_mag=16.6781,
    search_radius_arcsec=9.3
)

```

**Example: Contaminated Source**
*(DSS2 and Pan-STARRS cutouts showing background sources within the exclusion zone)*
![Contaminated Source Example](images/contaminated.png)

**Example: Clean Source**
*(DESI Legacy and Pan-STARRS cutouts confirming an isolated target)*
![Uncontaminated Source Example](images/uncontaminated.png)


## Spectrum Querying and Extraction

A tutorial is available in `tutorial_SPXQuery.ipynb`. This pipeline utilizes a custom wrapper around the [spxquery](https://github.com/WenkeRen/spxquery) package for fast bulk data retrieval.

### The Aperture vs. PSF Discrepancy

While `spxquery` is highly efficient for bulk queries, it relies on fixed 3-pixel (9.3") aperture photometry. This misses faint light in the source wings, resulting in lower total flux compared to [SPIFF](https://github.com/jgagneastro/SPIFF), an extraction tool that uses highly accurate PSF fitting and proper motion tracking. However, SPIFF is computationally expensive, often requiring over two hours to extract a single source.

**The Discrepancy:** Raw `spxquery` aperture extraction (blue) systemically underestimates flux compared to SPIFF's PSF fitting (red) due to lost light in the source wings.
![Raw SPIFF vs SPXQuery](images/SPIFF_vs_SPXQuery.png)


**The Correction:** Applying the empirical calibration wrapper scales the `spxquery` output to match SPIFF photometry, bridging the gap between processing speed and extraction accuracy.
![Calibrated SPIFF vs SPXQuery](images/SPIFF_vs_SPXQuery_calib.png)

### The Calibrated Wrapper

To combine the speed of `spxquery` with the accuracy of SPIFF, the wrapper applies an empirical calibration. It calculates a magnitude-dependent correction factor across 10 wavelength bins, utilizing smooth interpolation to prevent discontinuities at bin edges. This effectively scales the fast aperture extraction to match SPIFF's PSF photometry.

The wrapper manages all data handling locally: it downloads the necessary 20x20 pixel cutouts, performs the spectrophotometry, and then automatically deletes all temporary directories to preserve disk space, leaving only the final parsed `.csv` file.

### Usage

The extraction is handled by `spxquery_get_spectrum_calibrated`.

**Key Parameters:**

* `calibrate`:
* If `True`: Applies the PSF correction. The output file retains the original `flux` and `flux_error` columns, and appends `flux_calib`, `flux_error_calib`, and the specific `correction_factor` applied to each row.
* If `False`: Bypasses the calibration and simply moves the raw `spxquery` output file untouched.


* `outdir`: The destination directory for the final `[source_id].csv` files.
* Original headers (denoted by `#`) from the `spxquery` output are preserved in both modes.

**Example Implementation:**

```python
from spxquery_wrapper import spxquery_get_spectrum_calibrated

# Extract and calibrate a single source from a dataframe row
spectrum_df = spxquery_get_spectrum_calibrated(
    source_id=3108936854882971904,
    ra=105.29821,
    dec=-2.431236,
    calibrate=True, 
    outdir="spxquery_results_calib",
    verbose=True
)

# Inspect the calibrated columns
print(spectrum_df[['wavelength', 'flux', 'flux_calib', 'correction_factor']].head())

```