import os
import shutil
import subprocess
import pandas as pd
from pathlib import Path

def spiff_get_spectrum(source_id, ra, dec, pmra, pmdec, epoch=2016.0, outdir="./spiff_results", verbose=False):
    """
    Runs SPIFF for a single source, flattens the output results.csv to {source_id}.csv, 
    and deletes the temporary nested folders.
    """
    source_id_str = str(source_id)
    base_dir = Path(outdir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # The final target file: spiff_results/319875081009093632.csv
    final_csv_path = base_dir / f"{source_id_str}.csv"
    
    # 1. Short-circuit if we already processed this source
    if final_csv_path.exists():
        print(f"Skipping SPIFF: {final_csv_path.name} already exists.")
        return pd.read_csv(final_csv_path)
        
    target_name = f"target_{source_id_str}"
    
    command = [
        "spiff-lv2",
        "--ra", str(ra),
        "--dec", str(dec),
        "--reference-crd-epoch-yr", str(epoch),
        "--reference-pmra-masyr", str(pmra),
        "--reference-pmdec-masyr", str(pmdec),
        "--target-name", target_name,
        "--outdir", str(base_dir),
        "--scipy-only" 
    ]
    
    # 2. Run the pipeline
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"⚠️ Extraction failed for {source_id_str}. Error: {result.stderr.strip()}")
        return None

    # 3. Extract, Rename, and Cleanup
    try:
        # Find the auto-generated folder (e.g., target_319875081009093632_RA19.82..._DEC...)
        matching_folders = list(base_dir.glob(f"{target_name}*"))
        if not matching_folders:
            return None
            
        temp_dir = matching_folders[0]
        
        # Look strictly for results.csv
        temp_csv = temp_dir / "results.csv"
        
        if temp_csv.exists():
            # Load the data so we can return it to the script
            df = pd.read_csv(temp_csv)
            
            # Move and rename the file to the root outdir
            shutil.move(str(temp_csv), str(final_csv_path))
            
            # Nuke the entire temporary directory
            shutil.rmtree(temp_dir)
            if verbose:
                print(f"✅ Saved flattened SPIFF data to {final_csv_path.name} and cleaned up.")
            
            return df
        else:
            # Cleanup the folder even if it failed to produce a valid CSV
            shutil.rmtree(temp_dir)
            print(f"⚠️ No results.csv found in output. Deleted temporary folder for {source_id_str}.")
            return None
            
    except Exception as e:
        print(f"❌ Error processing results for {source_id_str}: {e}")
        return None