import os
import rasterio
import geopandas as gpd
from topo_drain import extract_ridges

def extract_ridges_tool(
    dtm: str,
    ridge_output_path: str,
    inverted_filled_output_path: str = None,
    inverted_fdir_output_path: str = None,
    inverted_facc_output_path: str = None,
    inverted_facc_log_output_path: str = None,
    accumulation_threshold: int = 1000,
    dist_facc: float = 0
) -> str:
    """
    Wrapper to extract ridge lines for use in a QGIS toolbox script or standalone call.

    Args:
        dtm (str):
            Path to input DTM (GeoTIFF).
        ridge_output_path (str):
            Output path for ridge lines (Shapefile or GeoPackage, “.shp” or “.gpkg”).
        inverted_filled_output_path (str, optional):
            Path to save the inverted‐DTM’s filled DEM (GeoTIFF, “.tif”).
        inverted_fdir_output_path (str, optional):
            Path to save the inverted‐DTM’s flow‐direction raster (GeoTIFF, “.tif”).
        inverted_facc_output_path (str, optional):
            Path to save the inverted‐DTM’s flow‐accumulation raster (GeoTIFF, “.tif”).
        inverted_facc_log_output_path (str, optional):
            Path to save the inverted‐DTM’s log‐scaled accumulation raster (GeoTIFF, “.tif”).
        accumulation_threshold (int):
            Threshold for ridge extraction (flow accumulation units).
        dist_facc (float):
            Maximum search distance for breach paths in cells.

    Returns:
        str:
            Path to the output ridge lines file (same as `ridge_output_path`).
    """
    # 1) Read CRS from input DTM
    with rasterio.open(dtm) as src:
        dtm_crs = src.crs

    # 2) Validate output directories
    if not os.path.isdir(os.path.dirname(ridge_output_path)):
        raise FileNotFoundError(
            f"[Config Error] ridge output directory not found: {os.path.dirname(ridge_output_path)}"
        )
    for p in (
        inverted_filled_output_path,
        inverted_fdir_output_path,
        inverted_facc_output_path,
        inverted_facc_log_output_path
    ):
        if p and not os.path.isdir(os.path.dirname(p)):
            raise FileNotFoundError(
                f"[Config Error] directory not found: {os.path.dirname(p)}"
            )

    # 3) Run ridge extraction
    gdf_ridges = extract_ridges(
        dtm_path=dtm,
        inverted_filled_output_path=inverted_filled_output_path,
        inverted_fdir_output_path=inverted_fdir_output_path,
        inverted_facc_output_path=inverted_facc_output_path,
        inverted_facc_log_output_path=inverted_facc_log_output_path,
        accumulation_threshold=accumulation_threshold,
        dist_facc=dist_facc
        )

    # 4) Assign the DTM CRS to the GeoDataFrame
    gdf_ridges = gdf_ridges.set_crs(dtm_crs, allow_override=True)

    # 5) Save result
    try:
        gdf_ridges.to_file(ridge_output_path)
        print(f"[ExtractRidgesTool] ridge lines saved to: {ridge_output_path}")
    except Exception as e:
        raise RuntimeError(f"[ExtractRidgesTool] failed to save ridge output: {e}")

    return ridge_output_path


def main():
    """
    Run this script as standalone for testing outside of QGIS.
    """
    # Input: Digital terrain model
    dtm_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Originaldaten/swissAlti3d/swissalti3d_perimter_2.tif"

    # Required output path
    ridge_output_path = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/ridge_lines.shp"

    # Optional overrides
    filled_out = None
    fdir_out   = None
    facc_out   = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/facc_ridges.tif"
    facc_log_out = r"/Users/aquaplus_tiw/Documents/Dokumente_TIW/Projekte_laufend/Land-schafft Wasser/Wettbewerb/Resultate/perimter 2/facc_ridges_log.tif"

    # Optional parameters
    accumulation_threshold_ridges = 2000
    dist_facc_ridges = 0

    result_path = extract_ridges_tool(
        dtm=dtm_path,
        ridge_output_path=ridge_output_path,
        inverted_filled_output_path=filled_out,
        inverted_fdir_output_path=fdir_out,
        inverted_facc_output_path=facc_out,
        inverted_facc_log_output_path=facc_log_out,
        accumulation_threshold=accumulation_threshold_ridges,
        dist_facc=dist_facc_ridges
    )

    print(f"Ridge lines written to: {result_path}")


if __name__ == "__main__":
    main()