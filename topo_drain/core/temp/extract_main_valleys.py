# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: extract_main_valleys.py
#
# Purpose: Python tool to create and extract main valley lines based on WhiteboxTools
#
# -----------------------------------------------------------------------------

import os
import geopandas as gpd
from topo_drain import extract_main_valleys

def extract_main_valleys_tool(
    valley_lines_path: str,
    facc_path: str,
    perimeter: str,
    nr_main: int = 2,
    output_path: str = None,
    clip_to_perimeter: bool = True
) -> str:
    """
    Wrapper to extract main valleys for use in a QGIS toolbox script or standalone call.

    Args:
        valley_lines_path (str): Path to the input valley line feature class with fields 'FID', 'TRIB_ID', 'DS_LINK_ID' - output of [ExtractValleys] (SHP or GeoPackage).
        facc_path (str): Path to the flow accumulation raster - output of [ExtractValleys](GeoTIFF).
        perimeter (str): Path to the input polygon feature class of interest (SHP or GeoPackage).
        nr_main (int): Number of main valleys to extract.
        output_path (str): Output path for main valley lines (SHP or GeoPackage).
        clip_to_perimeter (bool): If True, clips output to perimeter polygon.

    Returns:
        str: Path to output main valley shapefile (output_path).
    """

    # --- Validate file and directory paths ---
    if not os.path.isfile(valley_lines_path):
        raise FileNotFoundError(f"[Config Error] valley line file not found: {valley_lines_path}")
    if not os.path.isfile(facc_path):
        raise FileNotFoundError(f"[Config Error] flow accumulation file not found: {facc_path}")     
    if not os.path.isdir(os.path.dirname(output_path)):
        raise FileNotFoundError(f"[Config Error] main valley output directory not found: {os.path.dirname(output_path)}")

    try:
        gdf_valley_lines = gpd.read_file(valley_lines_path)
        print(f"[ExtractMainValleysTool] valley loaded to geodataframe")
    except Exception as e:
        raise RuntimeError(f"[ExtractMainValleysTool] failed to load valley lines as geodataframe: {e}")
    try:
        gdf_perimeter = gpd.read_file(perimeter)
        print(f"[ExtractMainValleysTool] perimeter loaded to geodataframe")
    except Exception as e:
        raise RuntimeError(f"[ExtractMainValleysTool] Failed to load perimter as geodataframe: {e}")


    # Run valley extraction
    gdf_main_valleys = extract_main_valleys(
        valley_lines=gdf_valley_lines,
        facc_path=facc_path,
        perimeter=gdf_perimeter,
        nr_main=nr_main,
        clip_to_perimeter=clip_to_perimeter
    )

    try:
        gdf_main_valleys.to_file(output_path)
        print(f"[ExtractMainValleysTool] Valley lines saved to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"[ExtractMainValleysTool] Failed to save valley output: {e}")

    return output_path


def main():
    """
    Run this script as standalone for testing outside of QGIS.
    """
    # Valley lines input feature class
    valley_lines_input_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/valley_lines.shp"
    # Flow accumulation input raster file
    facc_input_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/facc_valleys.tif"
    # Perimeter input feature class
    perimeter_input_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Daten/perimeter_2.shp"


    # Output paths (user-defined)
    main_valleys_output_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/main_valley_lines.shp"

    # Optional parameters
    nr_main = 3
    clip_to_perimeter = True

    # Run the tool
    result_path = extract_main_valleys_tool(
        valley_lines_path=valley_lines_input_path,
        facc_path=facc_input_path,
        perimeter=perimeter_input_path,
        nr_main=nr_main,
        output_path=main_valleys_output_path,
        clip_to_perimeter=clip_to_perimeter
    )

    print(f"Main valley lines written to: {result_path}")

if __name__ == "__main__":
    main()