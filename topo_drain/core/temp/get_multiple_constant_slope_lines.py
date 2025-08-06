# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: get_multiple_constant_slope_lines.py
#
# Purpose: Python tool to create constant slope lines from points
# -----------------------------------------------------------------------------

import os
import geopandas as gpd
from topo_drain import get_constant_slope_lines, add_m_to_gdf

def get_multiple_constant_slope_lines_tool(
    dtm_path: str,
    start_points_path: str,
    destination_features_paths: list,
    output_path: str,
    slope: float = 0.01,
    barrier_features_paths: list = None
) -> str:
    """
    Wrapper to create constant slope lines for use in QGIS toolbox or standalone.

    Args:
        dtm_path (str): Path to the digital terrain model (GeoTIFF).
        start_points_path (str): Path to point feature class (SHP or GeoPackage).
        destination_features_paths (list): List of vector paths (SHP/GeoPackage) for destination features.
        output_path (str): Output path for slope lines (SHP or GeoPackage).
        slope (float): Desired slope (e.g., 0.01 for 1%).
        barrier_features_paths (list, optional): List of vector paths that define barriers.

    Returns:
        str: Path to output slope line shapefile.
    """

    ###### Additional Input: Dictionary: FROM (relative Distance of line i.e. 0 means from start of line), TO (relative Distance of line i.e. 0.5 means to 50% of Lenght of line, SLOPE
    # Validate that intervall between 0 and 1 is coverd from all FROM and TO values
     
    ###### for first slope (FROM: 0) use get_constant_slope_lines_tool


    # --- Validate input ---
    if not os.path.isfile(dtm_path):
        raise FileNotFoundError(f"[Config Error] DTM file not found: {dtm_path}")
    if not os.path.isfile(start_points_path):
        raise FileNotFoundError(f"[Config Error] Start points file not found: {start_points_path}")
    if not os.path.isdir(os.path.dirname(output_path)):
        raise FileNotFoundError(f"[Config Error] Output directory not found: {os.path.dirname(output_path)}")

    # --- Load input data ---
    try:
        gdf_start_points = gpd.read_file(start_points_path)
        print("[SlopeLinesTool] Start points loaded.")
    except Exception as e:
        raise RuntimeError(f"[SlopeLinesTool] Failed to load start points: {e}")

    try:
        destination_features = [gpd.read_file(path) for path in destination_features_paths]
        print(f"[SlopeLinesTool] Loaded {len(destination_features)} destination feature sets.")
    except Exception as e:
        raise RuntimeError(f"[SlopeLinesTool] Failed to load destination features: {e}")

    barrier_features = None
    if barrier_features_paths:
        try:
            barrier_features = [gpd.read_file(path) for path in barrier_features_paths]
            print(f"[SlopeLinesTool] Loaded {len(barrier_features)} barrier feature sets.")
        except Exception as e:
            raise RuntimeError(f"[SlopeLinesTool] Failed to load barrier features: {e}")

    # # --- Run slope line creation ---
    # gdf_slope_lines = get_constant_slope_lines(
    #     dtm_path=dtm_path,
    #     start_points=gdf_start_points,
    #     destination_features=destination_features,
    #     slope=slope,
    #     barrier_features=barrier_features
    # )

    # try:
    #     gdf_slope_lines.to_file(output_path)
    #     print(f"[SlopeLinesTool] Slope lines saved to: {output_path}")
    # except Exception as e:
    #     raise RuntimeError(f"[SlopeLinesTool] Failed to save slope lines: {e}")

    # --- Run the core slope‐line algorithm ---
    gdf_lines = get_constant_slope_lines(
        dtm_path=dtm_path,
        start_points=gdf_start_points,
        destination_features=destination_features,
        slope=slope,
        barrier_features=barrier_features
    )
    print(f"[SlopeLinesTool] Generated {len(gdf_lines)} slope lines (no M).")

    # --- Add cumulative-distance 'm' into the Z dimension ---
    try:
        gdf_lines = add_m_to_gdf(
            gdf_lines,
            geom_col="geometry",
            out_col="geometry_m"
        )
        # Make the new 3D column the active geometry
        gdf_lines = gdf_lines.set_geometry("geometry_m")
        # **Drop the old 2D geometry column** so to_file() only sees one geometry
        if "geometry" in gdf_lines.columns:
            gdf_lines = gdf_lines.drop(columns=["geometry"])
        print("[SlopeLinesTool] Added M values to geometries.")
    except Exception as e:
        raise RuntimeError(f"[SlopeLinesTool] Failed to add M values: {e}")

    # --- Save final output with M-aware geometries ---
    try:
        gdf_lines.to_file(output_path)
        print(f"[SlopeLinesTool] Final slope lines with M saved to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"[SlopeLinesTool] Failed to save final slope lines: {e}")
    
    # --- Save final output with M-aware geometries ---


    return output_path

    ###### Adjust gdf_lines with FORM Distant to TO-Distance (Line only to this distance)

    ###### for next slopes in dictioray (sorted FROM values) use get_constant_slope_lines_tool with new starting starting point at FROM m-value of line output of preivous run ans so on.





def main():
    """
    Run as standalone script for testing outside QGIS.
    """
    dtm_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Originaldaten/swissAlti3d/swissalti3d_perimter_2.tif"
    start_points_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/keypoints.shp"
    destination_features_paths = [
        r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/main_ridge_lines.shp",
        r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Daten/perimeter_2.shp"
    ]
    barrier_features_paths = [
        r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/main_valley_lines.shp"
    ]
    output_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/constant_slope_lines.shp"
    slope = 0.01

    result_path = get_constant_slope_lines_tool(
        dtm_path=dtm_path,
        start_points_path=start_points_path,
        destination_features_paths=destination_features_paths,
        output_path=output_path,
        slope=slope,
        barrier_features_paths=barrier_features_paths
    )

    print(f"Constant slope lines written to: {result_path}")

if __name__ == "__main__":
    main()


