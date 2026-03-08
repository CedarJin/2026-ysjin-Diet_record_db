#!/usr/bin/env python3
"""
Read and display a SAS dataset file (.sas7bdat)

This script reads SAS dataset files and displays their contents.
It requires either pandas+pyreadstat or sas7bdat library.

To install dependencies:
  pip install pandas pyreadstat
  OR
  pip install sas7bdat
"""

import sys
from pathlib import Path

# Try to import required libraries
USE_PYREADSTAT = False
USE_SAS7BDAT = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyreadstat
    USE_PYREADSTAT = True
except ImportError:
    try:
        import sas7bdat
        USE_SAS7BDAT = True
    except ImportError:
        pass

if not (USE_PYREADSTAT or USE_SAS7BDAT):
    print("=" * 80)
    print("ERROR: Required libraries not found!")
    print("=" * 80)
    print("\nPlease install one of the following:")
    print("  1. pip install pandas pyreadstat")
    print("  2. pip install sas7bdat")
    print("\nIf you're on macOS and get 'externally-managed-environment' error:")
    print("  python3 -m venv venv")
    print("  source venv/bin/activate")
    print("  pip install pandas pyreadstat")
    print("=" * 80)
    sys.exit(1)


def read_sas_file(file_path: str, verbose: bool = True, save_csv: bool = True):
    """Read a SAS dataset file and optionally display its contents."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return None
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Reading SAS file: {file_path.name}")
        print('='*80)
    
    try:
        if USE_PYREADSTAT:
            df, meta = pyreadstat.read_sas7bdat(str(file_path))
            if verbose:
                print(f"Using pyreadstat library...")
                print(f"\nDataset: {meta.table_name if hasattr(meta, 'table_name') and meta.table_name else 'N/A'}")
        elif USE_SAS7BDAT:
            if verbose:
                print("Using sas7bdat library...")
            with sas7bdat.SAS7BDAT(str(file_path)) as reader:
                df = reader.to_data_frame()
        elif HAS_PANDAS:
            if verbose:
                print("Using pandas read_sas...")
            df = pd.read_sas(file_path, encoding='latin-1')
        
        if verbose:
            print(f"Number of rows: {len(df):,}")
            print(f"Number of columns: {len(df.columns)}")
            print(f"\nColumn names and types:")
            for col in df.columns:
                print(f"  - {col}: {df[col].dtype}")
            
            print(f"\n\nFirst 5 rows:")
            print("-" * 80)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 50)
            print(df.head(5).to_string())
        
        # Save to CSV
        if save_csv:
            output_csv = file_path.with_suffix('.csv')
            if verbose:
                print(f"\nSaving to CSV: {output_csv}")
            df.to_csv(output_csv, index=False)
            if verbose:
                print(f"✓ Saved {len(df):,} rows to {output_csv}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading {file_path.name}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def process_directory(directory: str, verbose: bool = True):
    """Process all SAS7BDAT files in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory not found: {dir_path}")
        return
    
    sas_files = sorted(dir_path.glob("*.sas7bdat"))
    if not sas_files:
        print(f"No SAS7BDAT files found in {dir_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(sas_files)} SAS file(s) to process")
    print('='*80)
    
    results = []
    for sas_file in sas_files:
        # Skip if CSV already exists (unless user wants to overwrite)
        csv_file = sas_file.with_suffix('.csv')
        if csv_file.exists():
            print(f"\n⏭️  Skipping {sas_file.name} (CSV already exists)")
            continue
        
        df = read_sas_file(sas_file, verbose=verbose, save_csv=True)
        if df is not None:
            results.append({
                'file': sas_file.name,
                'rows': len(df),
                'columns': len(df.columns),
                'csv': csv_file.name
            })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    if results:
        print(f"\nSuccessfully processed {len(results)} file(s):\n")
        for r in results:
            print(f"  ✓ {r['file']:30s} → {r['rows']:>8,} rows, {r['columns']:>3} cols → {r['csv']}")
    else:
        print("\nNo new files processed (all CSVs already exist)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if Path(arg).is_dir():
            # Process directory
            process_directory(arg)
        elif Path(arg).is_file():
            # Process single file
            read_sas_file(arg)
        else:
            print(f"Error: {arg} is not a valid file or directory")
    else:
        # Default: process all SAS files in the FNDDS directory
        default_dir = "db/fndds/FNDDS_2021-2023_SAS"
        process_directory(default_dir)
