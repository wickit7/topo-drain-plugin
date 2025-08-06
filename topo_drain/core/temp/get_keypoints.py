# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: get_keypoints.py
#
# Purpose: Python tool to extract keypoints (local convexity) from valley lines using DTM
#
# -----------------------------------------------------------------------------

import os
import geopandas as gpd
from topo_drain import get_keypoints

def get_keypoints_tool(
    main_valley_lines_path: str,
    dtm_path: str,
    output_path: str,
    sampling_distance: float = 2.0,
    smoothing_window: int = 9,
    polyorder: int = 2,
    top_n: int = 100,
    min_distance: float = 10.0,
    find_window_distance: float = 10.0,
    plot_debug: bool = False
) -> str:
    """
    Wrapper to extract keypoints from valley lines for QGIS or standalone use.

    Args:
        main_valley_lines_path (str): Path to the main valley line shapefile (with FID).
        dtm_path (str): Path to the input DTM (GeoTIFF).
        output_path (str): Path to save the keypoints as shapefile/GeoPackage.
        sampling_distance (float): Distance between elevation samples (m).
        smoothing_window (int): Window size for Savitzky-Golay filter (odd int).
        polyorder (int): Polynomial order for smoothing.
        top_n (int): Number of keypoints per valley line. Use large values to get points about every min_distance meters.
        min_distance (float): Minimum distance between keypoints (m).
        find_window_distance (float): Window size for second derivative search (m).
        plot_debug (bool): If True, show plots for debugging.

    Returns:
        str: Path to saved keypoints file.
    """
    # --- Validate inputs ---
    if not os.path.isfile(main_valley_lines_path):
        raise FileNotFoundError(f"[Config Error] Valley lines file not found: {main_valley_lines_path}")
    if not os.path.isfile(dtm_path):
        raise FileNotFoundError(f"[Config Error] DTM raster file not found: {dtm_path}")
    if not os.path.isdir(os.path.dirname(output_path)):
        raise FileNotFoundError(f"[Config Error] Output directory not found: {os.path.dirname(output_path)}")

    print("[GetKeypointsTool] Loading valley lines...")
    try:
        gdf_valleys = gpd.read_file(main_valley_lines_path)
    except Exception as e:
        raise RuntimeError(f"[GetKeypointsTool] Failed to load valley lines: {e}")

    print("[GetKeypointsTool] Detecting keypoints...")
    gdf_keypoints = get_keypoints(
        valley_lines=gdf_valleys,
        dtm_path=dtm_path,
        sampling_distance=sampling_distance,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
        top_n=top_n,
        min_distance=min_distance,
        find_window_distance=find_window_distance,
        plot_debug=plot_debug
    )

    print(f"[GetKeypointsTool] {len(gdf_keypoints)} keypoints detected. Saving to file...")
    try:
        gdf_keypoints.to_file(output_path)
    except Exception as e:
        raise RuntimeError(f"[GetKeypointsTool] Failed to save output: {e}")

    print(f"[GetKeypointsTool] Keypoints saved to: {output_path}")
    return output_path

def main():
    """
    Standalone test run for keypoint detection tool.
    """
    main_valley_lines_input = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/main_valley_lines.shp"
    dtm_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Originaldaten/swissAlti3d/swissalti3d_perimter_2.tif"
    keypoints_output = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/keypoints.shp"

    # Optional params
    sampling_distance = 1.0
    smoothing_window = 40
    polyorder = 2
    top_n = 1
    min_distance = 20.0
    find_window_distance = 20.0
    plot_debug = True

    result = get_keypoints_tool(
        main_valley_lines_path=main_valley_lines_input,
        dtm_path=dtm_path,
        output_path=keypoints_output,
        sampling_distance=sampling_distance,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
        top_n=top_n,
        min_distance=min_distance,
        find_window_distance=find_window_distance,
        plot_debug=plot_debug
    )

    print(f"[Main] Keypoints written to: {result}")


if __name__ == "__main__":
    main()