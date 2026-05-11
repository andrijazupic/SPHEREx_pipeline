# SPHEREx Pipeline

Data pipeline for decontaminating SPHEREx sources and retrieving spectra using the SPXQuery and SPIFF packages.

## Installation

Clone the repository to your local machine and install the required dependencies:

```bash
git clone https://github.com/andrijazupic/SPHEREx_pipeline.git
cd SPHEREx_pipeline
pip install -r requirements.txt

```

## Source Decontamination

Due to the large pixel scale of SPHEREx, isolated targets can easily be contaminated by unresolved background sources. The contamination pipeline cross-references targets against high-resolution optical catalogs and direct FITS images to flag or remove blended sources. A full walkthrough is available in `tutorial_contamination.ipynb`.

### 1. Catalog-Level Filtering

The pipeline queries four successive survey catalogs to identify neighboring sources within a defined search radius:

* **Gaia DR3:** Synchronous ADQL spatial queries to identify basic positional blends.
* **DESI Legacy DR10 (Tractor):** Anchors the search to the central target and evaluates nearest neighbors. Filters noise artifacts using Signal-to-Noise Ratio (SNR) limits and identifies extended background sources.
* **Pan-STARRS DR1:** Separates stars from extended galaxies using the mathematical difference between PSF and Kron magnitudes in the r-band.
* **SDSS DR16:** Applies exact spatial logic and SNR limits using SDSS morphological classifications.

### 2. Image-Level Filtering

For sources that pass catalog checks, the pipeline performs direct image analysis using a headless World Coordinate System (WCS) PSF fitter:

* **Fallback Hierarchy:** Sequentially attempts to download optical FITS cutouts from the highest available resolution survey: DESI Legacy $\rightarrow$ PanSTARRS $\rightarrow$ SDSS $\rightarrow$ DSS2.
* **Source Fitting:** Uses `photutils.DAOStarFinder` to detect observed sources in the cutout.
* **Dynamic Flagging:** Identifies the true observed center of the target, then computes dynamic Point Spread Function (PSF) wings for all surrounding detections based on the specific survey's pixel scale and FWHM. Sources whose calculated wings intersect the target's exclusion radius are flagged.

### Usage

The pipeline is executed via the `spherex_contamination_analysis` wrapper function, which runs the input dataframe through all catalog and image filters sequentially.

**Key Parameters:**

* `df`: The input pandas DataFrame. Must contain `source_id`, `ra`, and `dec` columns.
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