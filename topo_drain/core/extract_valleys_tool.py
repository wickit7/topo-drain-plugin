# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: extract_valleys_tool.py
#
# Purpose: Python tool to create valley lines (river network) based on WhiteboxTools
#
# -----------------------------------------------------------------------------

import os
import rasterio
#from topo_drain.core.topo_drain_core import extract_valleys  # <-- updated import
from topo_drain_core import extract_valleys  # <-- updated import
import geopandas as gpd

def extract_valleys_tool(
    dtm: str,
    valley_output_path: str,
    filled_output_path: str = None,
    fdir_output_path: str = None,
    facc_output_path: str = None,
    facc_log_output_path: str = None,
    accumulation_threshold: int = 1000,
    dist_facc: float = 0
) -> str:
    """
    Wrapper to extract valleys for use in a QGIS toolbox script or standalone call.

    Args:
        dtm (str):
            Path to input DTM (GeoTIFF).
        valley_output_path (str):
            Output path for valley lines (Shapefile or GeoPackage, “.shp” or “.gpkg”).
        filled_output_path (str, optional):
            Path to save the depression-filled DTM (GeoTIFF, “.tif”).
        fdir_output_path (str, optional):
            Path to save the flow-direction raster (GeoTIFF, “.tif”).
        facc_output_path (str, optional):
            Path to save the flow-accumulation raster (GeoTIFF, “.tif”).
        facc_log_output_path (str, optional):
            Path to save the log-scaled accumulation raster (GeoTIFF, “.tif”).
        accumulation_threshold (int):
            Threshold for valley extraction (flow accumulation units).
        dist_facc (float):
            Maximum search distance for breach paths in cells.

    Returns:
        str: Path to the output valley lines file (same as `valley_output_path`).
    """
    # Read CRS from the DTM
    with rasterio.open(dtm) as src:
        dtm_crs = src.crs

    # validate output directories
    for p in (filled_output_path, fdir_output_path, facc_output_path, facc_log_output_path):
        if p and not os.path.isdir(os.path.dirname(p)):
            raise FileNotFoundError(f"[Config Error] directory not found: {os.path.dirname(p)}")
    if not os.path.isdir(os.path.dirname(valley_output_path)):
        raise FileNotFoundError(f"[Config Error] directory not found: {os.path.dirname(valley_output_path)}")

    # run the core extraction
    gdf_valleys = extract_valleys(
        dtm_path=dtm,
        filled_output_path=filled_output_path,
        fdir_output_path=fdir_output_path,
        facc_output_path=facc_output_path,
        facc_log_output_path=facc_log_output_path,
        accumulation_threshold=accumulation_threshold,
        dist_facc=dist_facc
    )

    # assign the CRS from the input DTM
    gdf_valleys = gdf_valleys.set_crs(dtm_crs, allow_override=True)

    # save result
    try:
        gdf_valleys.to_file(valley_output_path)
        print(f"[ExtractValleysTool] valley lines saved to: {valley_output_path}")
    except Exception as e:
        raise RuntimeError(f"[ExtractValleysTool] failed to save valley output: {e}")

    return valley_output_path

def main():
    """
    Run this script as standalone for testing outside of QGIS.
    """
    # Input: Digital terrain model
    dtm_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Originaldaten/swissAlti3d/swissalti3d_perimter_2.tif"

    # Required output path
    valley_output_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/valley_lines.shp"

    # Optional overrides
    filled_output = None
    fdir_output = None
    facc_output = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/facc_valleys.tif"
    facc_log_output = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/facc_valleys_log.tif"

    # Optional parameters
    accumulation_threshold_valleys = 2000
    dist_facc_valleys = 0

    # Run the tool
    result_path = extract_valleys_tool(
        dtm=dtm_path,
        valley_output_path=valley_output_path,
        filled_output_path=filled_output,
        fdir_output_path=fdir_output,
        facc_output_path=facc_output,
        facc_log_output_path=facc_log_output,
        accumulation_threshold=accumulation_threshold_valleys,
        dist_facc=dist_facc_valleys
    )

    print(f"Valley lines written to: {result_path}")


if __name__ == "__main__":
    main()