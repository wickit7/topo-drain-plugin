# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: extract_main_ridges.py
#
# Purpose: Python tool to create and extract main valley lines based on WhiteboxTools
#
# -----------------------------------------------------------------------------
import os
from extract_main_valleys import extract_main_valleys_tool

def extract_main_ridges_tool(
    ridge_lines_path: str,
    inv_facc_path: str,
    perimeter: str,
    nr_main: int = 2,
    output_path: str = None,
    clip_to_perimeter: bool = True
) -> str:
    """
    Wrapper to extract main ridges for use in a QGIS toolbox script or standalone call.

    Args:
        ridge_lines_path (str): Path to the input ridge line feature class with fields 'FID', 'TRIB_ID', 'DS_LINK_ID' - output of [ExtractRidges] (SHP or GeoPackage).
        inv_facc_path (str): Path to the inverted flow accumulation raster (respectively ridge flow accumullation) - output of [ExtractRidges] (GeoTIFF).
        perimeter (str): Path to the input polygon feature class of interest (SHP or GeoPackage).
        nr_main (int): Number of main valleys to extract.
        output_path (str): Output path for main ridge lines (SHP or GeoPackage).
        clip_to_perimeter (bool): If True, clips output to perimeter polygon.

    Returns:
        str: Path to output main valley shapefile (output_path).
    """
    print(f"[ExtractMainRidgesTool] Start ExtractMainValleyTool with ExtractMainRidgeTool input paramters...")

    output_path = extract_main_valleys_tool(
        valley_lines_path=ridge_lines_path,
        facc_path=inv_facc_path,
        perimeter=perimeter,
        nr_main=nr_main,
        output_path=output_path,
        clip_to_perimeter=clip_to_perimeter
    )

    return output_path



def main():
    """
    Run this script as standalone for testing outside of QGIS.
    """
    # Ridge lines input feature class
    ridge_lines_input_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/ridge_lines.shp"
    # Flow accumulation input raster file
    facc_ridge_input_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/facc_ridges.tif"
    # Perimeter input feature class
    perimeter_input_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Daten/perimeter_2.shp"

    # Output paths (user-defined)
    main_ridges_output_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/main_ridge_lines.shp"

    # Optional parameters
    nr_main = 2
    clip_to_perimeter = True

    # Run the tool
    result_path = extract_main_ridges_tool(
        ridge_lines_path=ridge_lines_input_path,
        inv_facc_path=facc_ridge_input_path,
        perimeter=perimeter_input_path,
        nr_main=nr_main,
        output_path=main_ridges_output_path,
        clip_to_perimeter=clip_to_perimeter
    )

    print(f"Main ridge lines written to: {result_path}")

if __name__ == "__main__":
    main()