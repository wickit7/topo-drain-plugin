# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: topo_drain_core.py
#
# Purpose: Script with python functions of topo drain qgis plugin
#
# -----------------------------------------------------------------------------
import os
import sys
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.sample import sample_gen
from rasterio.features import rasterize
from rasterio import Affine
from rasterio.enums import Resampling
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, nearest_points, unary_union
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# ---  Configuration ---
_thisdir = os.path.dirname(__file__)

# WhiteBoxTools directory
whitebox_dir = os.path.join(_thisdir, "WBT")
print(f"[TOPO DRAIN CORE] Using WhiteboxTools directory: {whitebox_dir}")

# Temporary and working directories
TEMP_DIRECTORY = None
WORKING_DIRECTORY = None
NODATA = -32768  # Default NoData value for raster operations

def set_temp_and_working_dir(temp_dir, working_dir):
    global TEMP_DIRECTORY, WORKING_DIRECTORY
    print(f"[TopoDrainCore] Setting TEMP_DIRECTORY: {temp_dir}")
    TEMP_DIRECTORY = temp_dir
    print(f"[TopoDrainCore] Setting WhiteboxTools WORKING_DIRECTORY: {working_dir}")
    WORKING_DIRECTORY = working_dir
    if wbt is not None:
        if WORKING_DIRECTORY:
            wbt.set_working_dir(WORKING_DIRECTORY)

def set_nodata_value(no_data_val):
    global NODATA
    NODATA = no_data_val
    # if wbt is not None:
    #     wbt.set_nodata_value(NODATA)
        
# --- Ensure we can import the bundled or configured WhiteboxTools ---
if whitebox_dir not in sys.path:
    sys.path.insert(0, whitebox_dir)
from topo_drain.core.WBT.whitebox_tools import WhiteboxTools

# --- Instantiate and configure WBT ---
wbt = WhiteboxTools()
wbt.set_whitebox_dir(whitebox_dir)

def add_m_to_gdf(
    gdf: gpd.GeoDataFrame,
    geom_col: str = "geometry",
    out_col: str = "geometry_m"
) -> gpd.GeoDataFrame:
    """
    For each LineString in gdf[geom_col], compute a cumulative-distance 'm'
    at each vertex and return a new 3D LineString in gdf[out_col].
    """

    def _line_with_m(ls: LineString) -> LineString:
        # 1) extract 2D coords
        coords = list(ls.coords)
        # 2) compute segment lengths
        seg_lens = np.hypot(
            np.diff([c[0] for c in coords]),
            np.diff([c[1] for c in coords])
        )
        # 3) cumulative distance (m), starting at 0
        m_vals = np.insert(np.cumsum(seg_lens), 0, 0.0)
        # 4) build 3D coords
        coords3 = [(x, y, m) for (x, y), m in zip(coords, m_vals)]
        return LineString(coords3)

    # apply to each geometry
    gdf[out_col] = gdf[geom_col].apply(_line_with_m)
    return gdf


def stitch_multilinestring(geom, preserve_original=False):
    """
    Turn a MultiLineString (or a multipart LineString) into a single LineString
    by concatenating segments end-to-end at their nearest endpoints.

    Args:
        geom (LineString or MultiLineString):
            If a simple LineString, it’s returned unchanged.
            If a MultiLineString, its parts are stitched together.
        preserve_original (bool):
            If True, returns the original geom when it's already a LineString;
            if False, always returns a new LineString.

    Returns:
        LineString:
            One continuous LineString whose coordinate sequence visits every part
            of the original geometry.  Segments are appended in order of minimal
            jump distance between ends.  Gaps are not interpolated.

    Raises:
        TypeError: if `geom` is not LineString or MultiLineString.
    """
    # nothing to do for a single LineString (unless user explicitly wants a copy)
    if isinstance(geom, LineString) and preserve_original:
        return geom

    # flatten a MultiLineString into a list of LineStrings
    if isinstance(geom, MultiLineString):
        parts = list(geom.geoms)
    elif isinstance(geom, LineString):
        return LineString(geom.coords)  # make a fresh copy
    else:
        raise TypeError("geom must be LineString or MultiLineString")

    if not parts:
        return None

    # start with the longest segment
    parts.sort(key=lambda seg: seg.length, reverse=True)
    base_coords = list(parts.pop(0).coords)

    # greedily attach the next segment whose one endpoint is closest to our current end
    while parts:
        tail = Point(base_coords[-1])
        best_idx, best_flip, best_dist = None, False, float("inf")

        for i, seg in enumerate(parts):
            s, e = Point(seg.coords[0]), Point(seg.coords[-1])
            d_s, d_e = tail.distance(s), tail.distance(e)
            if d_s < best_dist:
                best_dist, best_idx, best_flip = d_s, i, False
            if d_e < best_dist:
                best_dist, best_idx, best_flip = d_e, i, True

        seg = parts.pop(best_idx)
        seq = list(seg.coords)
        if best_flip:
            seq.reverse()

        # append, skipping the duplicate connecting vertex
        base_coords.extend(seq[1:])

    return LineString(base_coords)

def get_abs_path(relative_path):
    """
    Convert a relative file path to an absolute path, relative to the script location.

    Args:
        relative_path (str): Relative path to the file or directory.

    Returns:
        str: Absolute path based on the script's directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))


def smooth_linestring(geom, sigma: float = 1.0):
    """
    Smooth a LineString or MultiLineString geometry using a Gaussian filter.

    Args:
        geom (LineString|MultiLineString): Input geometry to smooth.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        LineString or MultiLineString: Smoothed geometry.
    """
    # Handle MultiLineString by smoothing each part
    if isinstance(geom, MultiLineString):
        smoothed_parts = [smooth_linestring(part, sigma) for part in geom.geoms]
        return MultiLineString(smoothed_parts)

    # Must be a LineString from here on
    if not isinstance(geom, LineString):
        raise TypeError(f"Unsupported geometry type: {type(geom)}")

    coords = np.array(geom.coords)
    if len(coords) < 3 or sigma <= 0:
        # nothing to smooth
        return geom

    # apply gaussian filter separately to x and y
    x_smooth = gaussian_filter1d(coords[:, 0], sigma=sigma)
    y_smooth = gaussian_filter1d(coords[:, 1], sigma=sigma)

    # rebuild as a LineString
    smoothed = LineString(np.column_stack([x_smooth, y_smooth]))
    return smoothed

def mask_raster(raster_path: str, mask: gpd.GeoDataFrame, out_path: str) -> str:
    """
    Mask (clip) a raster file using a polygon mask.

    Args:
        raster_path (str): Path to the input raster (e.g., DTM GeoTIFF).
        mask (GeoDataFrame): Polygon(s) to use as mask.
        out_path (str): Desired path for output masked raster.

    Returns:
        str: Path to the masked raster.
    """
    try:
        if mask.empty:
            raise ValueError("The provided GeoDataFrame is empty. Cannot mask raster.")

        with rasterio.open(raster_path) as src:
            out_image, out_transform = rio_mask(src, mask.geometry, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)

        return out_path

    except ValueError as ve:
        raise RuntimeError(f"Masking error: {ve}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during raster masking: {e}")


def vector_to_mask(
    features: list[gpd.GeoDataFrame],
    reference_raster_path: str
) -> np.ndarray:
    """
    Convert one or more GeoDataFrames to a binary raster mask (1 = feature, 0 = background).

    Args:
        features (list[GeoDataFrame]): List of GeoDataFrames (polygon or line geometries).
        reference_raster_path (str): Path to a reference raster for shape and transform.

    Returns:
        np.ndarray: Binary mask with the same shape as the reference raster.
    """
    with rasterio.open(reference_raster_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform

    all_shapes = []
    for gdf in features:
        if gdf.empty:
            continue
        for geom in gdf.geometry:
            all_shapes.append((geom, 1))

    mask = rasterize(
        shapes=all_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )

    return mask

def invert_dtm(dtm_path: str, output_path: str) -> str:
    """
    Create an inverted DTM (multiply by -1) to extract ridges.

    Args:
        dtm_path (str): Path to original DTM.

    Returns:
        str: Path to inverted DTM.
    """
    if wbt is None:
        raise RuntimeError("WhiteboxTools not initialized.")

    wbt.multiply(input1=dtm_path, input2=-1.0, output=output_path)

    return output_path

def line_to_raster(
    gdf: gpd.GeoDataFrame,
    reference_raster: str,
    value_field: str = None,
    output_path: str = None
) -> np.ndarray:
    """
    Rasterize geometries from a GeoDataFrame to match the resolution and extent of a reference raster.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing lines, polygons, or points.
        reference_raster (str): Path to a GeoTIFF file used as spatial reference (resolution, shape, transform).
        value_field (str, optional): Column name to use for raster values. If None or invalid, all values are set to 1.
        output_path (str, optional): If provided, saves the raster to this path as a GeoTIFF.

    Returns:
        np.ndarray: Rasterized array (int32).
    """
    with rasterio.open(reference_raster) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

    if value_field and value_field in gdf.columns:
        shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf[value_field])]
    else:
        shapes = [(geom, 1) for geom in gdf.geometry]

    raster = rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='int32',
        all_touched=True
    )

    if output_path:
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=out_shape[0],
            width=out_shape[1],
            count=1,
            dtype='int32',
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(raster, 1)

    return raster

def boundary_to_raster(
    gdf: gpd.GeoDataFrame,
    reference_raster: str,
    output_path: str = None
) -> np.ndarray:
    """
    Rasterize the outer boundary (line) of a polygon GeoDataFrame to match a reference raster.

    Args:
        gdf (GeoDataFrame): GeoDataFrame with polygon geometries.
        reference_raster (str): Path to a GeoTIFF file used as reference (resolution, extent, transform).
        output_path (str, optional): If provided, the resulting mask will be saved as GeoTIFF to this path.

    Returns:
        np.ndarray: Binary mask with 1 for boundary pixels, 0 elsewhere.
    """
    with rasterio.open(reference_raster) as src:
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        out_shape = (src.height, src.width)

    # Extract the outer boundary as a single LineString or MultiLineString
    perimeter_line = gdf.geometry.boundary.unary_union

    # Rasterize the boundary line
    mask = rasterize(
        [(perimeter_line, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='uint8',
        all_touched=True
    )

    # Optional: write to GeoTIFF
    if output_path:
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=out_shape[0],
            width=out_shape[1],
            count=1,
            dtype='uint8',
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(mask, 1)

    return mask

def log_raster(
    input_raster: str,
    output_path: str,
    overwrite: bool = True,
    val_band: int = 1,
    nodata: float = -32768
) -> str:
    """
    Computes the natural logarithm of a specified band in a raster,
    and either overwrites it or appends the result as a new band.

    Args:
        input_raster (str): Path to the input raster.
        output_path (str): Path to the output raster.
        overwrite (bool): If True, replaces the selected band with log values.
                          If False, appends the log values as a new band.
        val_band (int): 1-based index of the band to compute the logarithm from.
        nodata (float): Nodata value to use in the log band.

    Returns:
        str: Path to the output raster.
    """
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        dtype = 'float32'
        band_count = src.count

        if not (1 <= val_band <= band_count):
            raise ValueError(f"val_band={val_band} is out of range. Input raster has {band_count} band(s).")

        # Read the band to compute log from
        data = src.read(val_band).astype(dtype)

        # Compute log(x) only for valid values > 0
        log_data = np.where(data > 0, np.log(data), nodata)

        if overwrite:
            # Overwrite selected band with log values, drop other bands
            profile.update(count=1, dtype=dtype, nodata=nodata)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(log_data, 1)
        else:
            # Append log band to original bands
            existing_data = src.read().astype(dtype)  # shape: (bands, rows, cols)
            new_band_count = band_count + 1
            profile.update(count=new_band_count, dtype=dtype, nodata=nodata)

            with rasterio.open(output_path, 'w', **profile) as dst:
                for i in range(band_count):
                    dst.write(existing_data[i], i + 1)
                dst.write(log_data, new_band_count)

    return output_path


def modify_dtm_with_mask(
    dtm_path: str,
    mask: np.ndarray,
    elevation_add: float,
    output_path: str
) -> str:
    """
    Modify DTM by adding elevation to masked cells.

    Args:
        dtm_path (str): Path to original DTM.
        mask (np.ndarray): Binary mask where elevation should be modified.
        elevation_add (float): Value to add to masked cells.
        output_path (str): Path to save modified DTM.

    Returns:
        str: Path to the modified DTM.
    """
    with rasterio.open(dtm_path) as src:
        data = src.read(1)
        meta = src.meta.copy()

    modified = data.copy()
    modified[mask == 1] += elevation_add

    meta.update(dtype="float32")

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(modified.astype("float32"), 1)

    return output_path

def create_contour_lines(
    dtm_path: str,
    spacing: float = 1.0,
    base: float = 0.0,
    smooth: int = 9,
    tolerance: float = 10.0,
    output_path: str = None
) -> gpd.GeoDataFrame:
    """
    Generate smooth contour lines from a DTM using WhiteboxTools.

    Args:
        dtm_path (str): Path to the raster file.
        spacing (float): Contour interval.
        base (float): Base elevation (usually 0).
        smooth (int): Smoothing factor (higher = smoother).
        tolerance (float): Simplification tolerance.
        output_path (str, optional): If provided, use this path for output shapefile.
                                     Otherwise, a temporary path will be generated.

    Returns:
        GeoDataFrame: Smoothed contour lines.

    Raises:
        RuntimeError: If contour generation fails.
    """
    try:
        if wbt is None:
            raise RuntimeError("WhiteboxTools not initialized.")

        if output_path is None:
            output_path = os.path.join(TEMP_DIRECTORY, f"contours.shp")

        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        wbt.contours_from_raster(
            i=dtm_path,
            output=output_path,
            interval=spacing,
            base=base,
            smooth=smooth,
            tolerance=tolerance
        )

        return gpd.read_file(output_path)

    except Exception as e:
        raise RuntimeError(f"Failed to create contour lines: {e}")


def raster_to_linestring_wbt(raster_path: str) -> LineString:
    """
    Uses WhiteboxTools to vectorize a raster and return a merged LineString or MultiLineString.

    Args:
        raster_path (str): Path to a raster where 1-valued pixels form your keyline.

    Returns:
        LineString or MultiLineString, or None if empty.
    """
    if wbt is None:
        raise RuntimeError("WhiteboxTools not initialized.")

    vector_path = raster_path.replace(".tif", ".shp")
    wbt.raster_to_vector_lines(i=raster_path, output=vector_path)

    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        warnings.warn(f"No vector features found in {vector_path}.")
        return None

    # 1) union all pieces
    all_union = unary_union(list(gdf.geometry))

    # 2) if that's already a single LineString, return it
    if isinstance(all_union, LineString):
        return all_union

    # 3) if it's a MultiLineString (or GeometryCollection), merge touching parts
    try:
        merged = linemerge(all_union)
    except ValueError:
        # fallback: try merging the raw list of geometries
        merged = linemerge(list(gdf.geometry))

    # 4) warn if still multipart
    if isinstance(merged, MultiLineString):
        warnings.warn(
            f"Raster vectorized to {len(merged.geoms)} disjoint parts; returning a MultiLineString."
        )
    return merged

def extract_valleys(
    dtm_path: str,
    filled_output_path: str = None,
    fdir_output_path: str = None,
    facc_output_path: str = None,
    facc_log_output_path: str = None,
    accumulation_threshold: int = 1000,
    dist_facc: float = 50,
    postfix: str = None,
    feedback=None
) -> gpd.GeoDataFrame:
    """
    Extract valley lines using WhiteboxTools. You can override the filled DEM,
    flow-direction, and flow-accumulation outputs; all other intermediate files
    (streams rasters and vectors, network) use defaults in TEMP_DIRECTORY.

    Args:
        dtm_path (str):
            Path to input DTM GeoTIFF (e.g. "/path/to/dtm.tif").
        filled_output_path (str, optional):
            Path to save the depression-filled DTM GeoTIFF (".tif").
        fdir_output_path (str, optional):
            Path to save the flow-direction raster GeoTIFF (".tif").
        facc_output_path (str, optional):
            Path to save the flow-accumulation raster GeoTIFF (".tif").
        facc_log_output_path (str, optional):
            Path to save the log-scaled accumulation raster GeoTIFF (".tif").
        accumulation_threshold (int):
            Threshold for stream extraction (flow accumulation units).
        dist_facc (float):
            Maximum breach distance (in raster units) for depression filling.
        postfix (str, optional):
            Optional string to include in default output filenames.
        feedback (QgsProcessingFeedback, optional):
            Optional feedback object for progress reporting/logging (for QGIS Plugin).

    Returns:
        GeoDataFrame:
            Extracted stream (valley) network with attributes.
    """
    if wbt is None:
        raise RuntimeError("WhiteboxTools not initialized.")

    if feedback:
        feedback.pushInfo("[ExtractValleys] Starting valley extraction process...")
    else:
        print("[ExtractValleys] Starting valley extraction process...")

    # Build defaults for everything
    if not postfix:
        d = lambda name: os.path.join(TEMP_DIRECTORY, name)
        defaults = {
            "filled":         d("filled.tif"),
            "fdir":           d("fdir.tif"),
            "streams":        d("streams.tif"),
            "streams_vec":    d("streams.shp"),
            "streams_linked": d("streams_linked.shp"),
            "facc":           d("facc.shp"),
            "facc_log":       d("facc_log.shp"),
            "network":        d("stream_network.shp"),
        }
    else:
        d = lambda base: os.path.join(TEMP_DIRECTORY, f"{base}_{postfix}")
        defaults = {
            "filled":         d("filled") + ".tif",
            "fdir":           d("fdir") + ".tif",
            "streams":        d("streams") + ".tif",
            "streams_vec":    d("streams") + ".shp",
            "streams_linked": d("streams_linked") + ".shp",
            "facc":           d("facc") + ".shp",
            "facc_log":       d("facc_log") + ".shp",
            "network":        d("stream_network") + ".shp",
        }

    print(f"[ExtractValleys] Define paths for outputs")
    # Only these four can be overridden
    filled_output_path   = filled_output_path   or defaults["filled"]
    fdir_output_path     = fdir_output_path     or defaults["fdir"]
    facc_output_path     = facc_output_path     or defaults["facc"]
    facc_log_output_path = facc_log_output_path or defaults["facc_log"]

    # intermediate paths always use defaults:
    streams_output_path        = defaults["streams"]
    streams_vec_output_path    = defaults["streams_vec"]
    streams_linked_output_path = defaults["streams_linked"]
    stream_network_output_path = defaults["network"]

    try:
        if feedback:
            feedback.pushInfo(f"[ExtractValleys] Filling depressions → {filled_output_path}")
        else:
            print(f"[ExtractValleys] Filling depressions → {filled_output_path}")
        try:
            ret = wbt.breach_depressions_least_cost(
                dem=dtm_path,
                output=filled_output_path,
                dist=int(dist_facc)
            )
            if ret != 0 or not os.path.exists(filled_output_path):
                raise RuntimeError(f"[ExtractValleys] Depression filling failed: WhiteboxTools returned {ret}, output not found at {filled_output_path}")
        except Exception as e:
            raise RuntimeError(f"[ExtractValleys] Depression filling failed: {e}")

        if feedback:
            feedback.pushInfo(f"[ExtractValleys] Flow direction → {fdir_output_path}")
        else:
            print(f"[ExtractValleys] Flow direction → {fdir_output_path}")
        try:
            ret = wbt.d8_pointer(dem=filled_output_path, output=fdir_output_path)
            if ret != 0 or not os.path.exists(fdir_output_path):
                raise RuntimeError(f"[ExtractValleys] Flow direction failed: WhiteboxTools returned {ret}, output not found at {fdir_output_path}")
        except Exception as e:
            raise RuntimeError(f"[ExtractValleys] Flow direction failed: {e}")

        if feedback:
            feedback.pushInfo(f"[ExtractValleys] Flow accumulation → {facc_output_path}")
        else:
            print(f"[ExtractValleys] Flow accumulation → {facc_output_path}")
        try:
            ret = wbt.d8_flow_accumulation(
                i=filled_output_path,
                output=facc_output_path,
                out_type="specific contributing area"
            )
            if ret != 0 or not os.path.exists(facc_output_path):
                raise RuntimeError(f"[ExtractValleys] Flow accumulation failed: WhiteboxTools returned {ret}, output not found at {facc_output_path}")
        except Exception as e:
            raise RuntimeError(f"[ExtractValleys] Flow accumulation failed: {e}")

        if feedback:
            feedback.pushInfo(f"[ExtractValleys] Log-scaled accumulation → {facc_log_output_path}")
        else:
            print(f"[ExtractValleys] Log-scaled accumulation → {facc_log_output_path}")
        try:
            log_raster(input_raster=facc_output_path, output_path=facc_log_output_path, nodata=float(NODATA))
            if not os.path.exists(facc_log_output_path):
                raise RuntimeError(f"[ExtractValleys] Log-scaled accumulation output not found at {facc_log_output_path}")
        except Exception as e:
            raise RuntimeError(f"[ExtractValleys] Log-scaled accumulation failed: {e}")

        if feedback:
            feedback.pushInfo(f"[ExtractValleys] Extracting streams (threshold={accumulation_threshold})")
        else:
            print(f"[ExtractValleys] Extracting streams (threshold={accumulation_threshold})")
        try:
            ret = wbt.extract_streams(
                flow_accum=facc_output_path,
                output=streams_output_path,
                threshold=accumulation_threshold
            )
            if ret != 0 or not os.path.exists(streams_output_path):
                raise RuntimeError(f"[ExtractValleys] Stream extraction failed: WhiteboxTools returned {ret}, output not found at {streams_output_path}")
        except Exception as e:
            raise RuntimeError(f"[ExtractValleys] Stream extraction failed: {e}")

        if feedback:
            feedback.pushInfo("[ExtractValleys] Vectorizing streams")
        else:
            print("[ExtractValleys] Vectorizing streams")
        try:
            ret = wbt.raster_streams_to_vector(
                streams=streams_output_path,
                d8_pntr=fdir_output_path,
                output=streams_vec_output_path
            )
            if ret != 0 or not os.path.exists(streams_vec_output_path):
                raise RuntimeError(f"[ExtractValleys] Vectorizing streams failed: WhiteboxTools returned {ret}, output not found at {streams_vec_output_path}")
        except Exception as e:
            raise RuntimeError(f"[ExtractValleys] Vectorizing streams failed: {e}")

        streams_vec_id = streams_linked_output_path.replace(".shp", "_id.tif")
        try:
            if feedback:
                feedback.pushInfo("[ExtractValleys] Identifying stream links")
            else:
                print("[ExtractValleys] Identifying stream links")
            ret = wbt.stream_link_identifier(
                d8_pntr=fdir_output_path,
                streams=streams_output_path,
                output=streams_vec_id
            )
            if ret != 0 or not os.path.exists(streams_vec_id):
                raise RuntimeError(f"[ExtractValleys] Stream link identifier failed: WhiteboxTools returned {ret}, output not found at {streams_vec_id}")
        except Exception as e:
            raise RuntimeError(f"[ExtractValleys] Stream link identifier failed: {e}")

        try:
            if feedback:
                feedback.pushInfo("[ExtractValleys] Converting linked streams")
            else:
                print("[ExtractValleys] Converting linked streams")
            wbt.raster_streams_to_vector(
                streams=streams_vec_id,
                d8_pntr=fdir_output_path,
                output=streams_linked_output_path
            )
        except Exception as e:
            raise RuntimeError(f"[ExtractValleys] Converting linked streams failed: {e}")

        try:
            if feedback:
                feedback.pushInfo("[ExtractValleys] Network analysis")
            else:
                print("[ExtractValleys] Network analysis")
            wbt.vector_stream_network_analysis(
                streams=streams_linked_output_path, 
                dem=fdir_output_path,
                output=stream_network_output_path
            )
        except Exception as e:
            raise RuntimeError(f"[ExtractValleys] Network analysis failed: {e}")

        if feedback:
            feedback.pushInfo(f"[ExtractValleys] Loading network from {stream_network_output_path}")
        else:
            print(f"[ExtractValleys] Loading network from {stream_network_output_path}")

        # Check if the file exists and is non-empty before reading
        if not os.path.exists(stream_network_output_path):
            raise RuntimeError(f"[ExtractValleys] Network output file not found: {stream_network_output_path}")
        gdf = gpd.read_file(stream_network_output_path)
        if gdf.empty:
            raise RuntimeError(f"[ExtractValleys] Network output file is empty: {stream_network_output_path}")

        if feedback:
            feedback.pushInfo(f"[ExtractValleys] Completed: {len(gdf)} features extracted.")
        else:
            print(f"[ExtractValleys] Completed: {len(gdf)} features extracted.")

        return gdf

    except Exception as e:
        raise RuntimeError(f"[ExtractValleys] Failed to extract valleys: {e}")
    

def extract_ridges(
    dtm_path: str,
    inverted_filled_output_path: str = None,
    inverted_fdir_output_path: str = None,
    inverted_facc_output_path: str = None,
    inverted_facc_log_output_path: str = None,
    accumulation_threshold: int = 1000,
    dist_facc: float = 50,
    postfix: str = "inverted"
) -> gpd.GeoDataFrame:
    """
    Extract ridge lines (watershed divides) from a DTM by inverting the terrain
    and running the valley‐extraction workflow.

    Args:
        dtm_path (str):
            Path to the input DTM GeoTIFF.
        inverted_filled_output_path (str, optional):
            Where to save the inverted‐DTM’s filled DEM (GeoTIFF, “.tif”).
        inverted_fdir_output_path (str, optional):
            Where to save the inverted‐DTM’s flow‐direction raster (GeoTIFF).
        inverted_facc_output_path (str, optional):
            Where to save the inverted‐DTM’s flow‐accumulation raster (GeoTIFF).
        inverted_facc_log_output_path (str, optional):
            Where to save the inverted‐DTM’s log‐scaled accumulation raster (GeoTIFF).
        accumulation_threshold (int):
            Threshold for ridge extraction (analogous to stream threshold).
        dist_facc (float):
            Maximum breach distance (in raster units) for depression filling.
        postfix (str):
            Postfix for naming intermediate files (default “inverted”).

    Returns:
        GeoDataFrame:
            Extracted ridge (divide) network as vector geometries.
    """
    if wbt is None:
        raise RuntimeError("WhiteboxTools not initialized.")

    # 1) Invert the DTM
    print("[ExtractRidges] Inverting DTM…")
    inverted_dtm = os.path.join(TEMP_DIRECTORY, f"inverted_dtm_{postfix}.tif")
    inverted_dtm = invert_dtm(dtm_path, inverted_dtm)
    print(f"[ExtractRidges] Inversion complete: {inverted_dtm}")

    # 2) Compute defaults for the four inverted outputs
    #    We leverage extract_valleys’ own default logic by passing these params through.
    print("[ExtractRidges] Preparing inverted-output paths…")
    # If the user did not supply, leave as None—extract_valleys will pick its defaults (which include postfix).
    inv_filled = inverted_filled_output_path
    inv_fdir   = inverted_fdir_output_path
    inv_facc   = inverted_facc_output_path
    inv_facc_log = inverted_facc_log_output_path

    # 3) Call extract_valleys on the inverted DTM
    print("[ExtractRidges] Running valley-extraction on inverted DTM…")
    ridges = extract_valleys(
        dtm_path=inverted_dtm,
        filled_output_path=inv_filled,
        fdir_output_path=inv_fdir,
        facc_output_path=inv_facc,
        facc_log_output_path=inv_facc_log,
        accumulation_threshold=accumulation_threshold,
        dist_facc=dist_facc,
        postfix=postfix
    )

    print(f"[ExtractRidges] Ridge extraction complete: {len(ridges)} features")
    return ridges


def extract_main_valleys(
    valley_lines: gpd.GeoDataFrame,
    facc_path: str,
    perimeter: gpd.GeoDataFrame,
    nr_main: int = 2,
    clip_to_perimeter: bool = True
) -> gpd.GeoDataFrame:
    """
    Identify and merge main valley lines based on the highest flow accumulation,
    using only points uniquely associated with one TRIB_ID (to avoid confluent points).
    """
    print("[ExtractMainValleys] Starting main valley extraction...")

    print("[ExtractMainValleys] Reading flow accumulation raster...")
    with rasterio.open(facc_path) as src:
        facc = src.read(1)
        transform = src.transform
        facc_crs = src.crs

    print("[ExtractMainValleys] Clipping valley lines to perimeter...")
    valley_clipped = gpd.overlay(valley_lines, perimeter, how="intersection")

    print("[ExtractMainValleys] Rasterizing valley lines...")
    valley_raster_path = os.path.join(TEMP_DIRECTORY, "valley_mask.tif")
    valley_mask = line_to_raster(
        gdf=valley_clipped.geometry,
        reference_raster=facc_path,
        output_path=valley_raster_path
    )

    print("[ExtractMainValleys] Extracting facc > 0 points on valley lines...")
    mask = (valley_mask == 1) & (facc > 0)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        raise RuntimeError("[ExtractMainValleys] No valley cells with flow accumulation > 0 found inside perimeter.")

    coords = [rasterio.transform.xy(transform, row, col) for row, col in zip(rows, cols)]
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*zip(*coords)), crs=facc_crs)
    points["facc"] = facc[rows, cols]

    print("[ExtractMainValleys] Performing spatial join with valley lines...")
    points_joined = gpd.sjoin(
        points,
        valley_clipped[["geometry", "FID", "TRIB_ID", "DS_LINK_ID"]],
        how="inner",
        predicate="intersects"
    ).drop(columns="index_right")

    print("[ExtractMainValleys] Filtering ambiguous facc points...")
    points_joined["geom_wkt"] = points_joined.geometry.to_wkt()
    geom_counts = points_joined.groupby("geom_wkt")["TRIB_ID"].nunique()
    valid_geoms = geom_counts[geom_counts == 1].index
    points_unique = points_joined[points_joined["geom_wkt"].isin(valid_geoms)].copy()

    if points_unique.empty:
        raise RuntimeError("[ExtractMainValleys] No unique valley points (with single TRIB_ID) found.")

    print("[ExtractMainValleys] Selecting top TRIB_IDs by max flow accumulation...")
    points_sorted = points_unique.sort_values("facc", ascending=False)
    points_top = points_sorted.drop_duplicates(subset="TRIB_ID").head(nr_main)

    if points_top.empty:
        raise RuntimeError("[ExtractMainValleys] No main valley lines could be selected.")

    selected_trib_ids = points_top["TRIB_ID"].unique()
    print(f"[ExtractMainValleys] Selected TRIB_IDs: {list(selected_trib_ids)}")

    print("[ExtractMainValleys] Merging valley line segments per TRIB_ID...")
    merged_records = []
    for i, trib_id in enumerate(selected_trib_ids):
        lines = valley_lines[valley_lines["TRIB_ID"] == trib_id]

        cleaned = []
        for geom in lines.geometry:
            if geom.is_empty:
                continue
            if isinstance(geom, LineString):
                cleaned.append(geom)
            elif isinstance(geom, MultiLineString):
                cleaned.extend([g for g in geom.geoms if isinstance(g, LineString)])

        if cleaned:
            try:
                merged_line = linemerge(cleaned)
                merged_records.append({
                    "geometry": merged_line,
                    "TRIB_ID": trib_id,
                    "FID": i + 1
                })
                print(f"[ExtractMainValleys] Merged TRIB_ID={trib_id}, segments={len(cleaned)}")
            except Exception as e:
                raise RuntimeError(f"[ExtractMainValleys] Failed to merge lines for TRIB_ID={trib_id}: {e}")

    if valley_lines.crs:
        gdf = gpd.GeoDataFrame(merged_records, crs=valley_lines.crs)
    else:
        gdf = gpd.GeoDataFrame(merged_records, crs=facc_crs)

    if clip_to_perimeter:
        print("[ExtractMainValleys] Clipping final valley lines to perimeter...")
        gdf = gpd.overlay(gdf, perimeter, how="intersection")

    print(f"[ExtractMainValleys] Main valley extraction complete. {len(gdf)} valleys extracted.")
    return gdf


def extract_main_ridges(
    ridge_lines: gpd.GeoDataFrame,
    facc_path: str,
    perimeter: gpd.GeoDataFrame,
    nr_main: int = 2,
    clip_to_perimeter: bool = True
) -> gpd.GeoDataFrame:
    """
    Identify and trace the main ridge lines (watershed divides) using the same logic as main valley detection.

    Args:
        ridge_lines (GeoDataFrame): Ridge line network with 'FID', 'TRIB_ID', and 'DS_LINK_ID' attributes.
        facc_path (str): Path to the flow accumulation raster (based on inverted DTM).
        perimeter (GeoDataFrame): Polygon defining the area boundary.
        nr_main (int): Number of main ridges to select.
        clip_to_perimeter (bool): If True, clips output to boundary polygon of perimeter.

    Returns:
        GeoDataFrame: Traced main ridge lines.
    """
    print("[ExtractMainRidges] Starting main ExtractMainValleys with extract main ridges input...")

    gdf = extract_main_valleys(
        valley_lines=ridge_lines,
        facc_path=facc_path,
        perimeter=perimeter,
        nr_main=nr_main,
        clip_to_perimeter=clip_to_perimeter
    )

    return gdf


def find_inflection_candidates(curvature: np.ndarray, window: int) -> list:
    """
    Detect inflection points where the curvature changes from concave to convex,
    using a moving average window. If none found, return point of strongest convexity.

    Args:
        curvature (np.ndarray): Smoothed 2nd derivative of elevation profile.
        window (int): Number of points before/after to average.

    Returns:
        list of tuples: Sorted (index, strength) candidates (at least one guaranteed).
    """
    candidates = []

    for i in range(window, len(curvature) - window):
        before_avg = np.mean(curvature[i - window:i])
        after_avg = np.mean(curvature[i + 1:i + 1 + window])
        if before_avg < 0 and after_avg > 0:
            strength = abs(before_avg) + abs(after_avg)
            candidates.append((i, strength))

    # Fallback: wenn keine echten Übergänge gefunden wurden
    if not candidates:
        warnings.warn("No clear concave → convex inflection points found. Using strongest average sign change as fallback.")
        best_strength = -np.inf
        best_index = None
        for i in range(window, len(curvature) - window):
            before_avg = np.mean(curvature[i - window:i])
            after_avg = np.mean(curvature[i + 1:i + 1 + window])
            strength = after_avg - before_avg
            if strength > best_strength:
                best_strength = strength
                best_index = i
        candidates = [(best_index, best_strength)]

    # Sortieren nach Stärke
    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return sorted_candidates


def get_keypoints(
    valley_lines: gpd.GeoDataFrame,
    dtm_path: str,
    sampling_distance: float = 2.0,
    smoothing_window: int = 9,
    polyorder: int = 2,
    top_n: int = 100,
    min_distance: float = 10.0,
    find_window_distance: float = 10.0,
    plot_debug: bool = False
    ) -> gpd.GeoDataFrame:
    """
    Detect keypoints along valley lines based on curvature of elevation profiles
    (second derivative). Keypoints are locations with high convexity, typically
    indicating a local change from concave to convex profile shape.

    The elevation profile is extracted along each valley line using the DTM and
    smoothed using a Savitzky-Golay filter. The second derivative is then computed,
    and the top N points with the strongest convex curvature (i.e., highest values
    of the second derivative) are selected as keypoints.

    Threfore, the function does not require an actual sign change in curvature. It returns
    the top N most convex locations regardless of whether an inflection point
    (from concave to convex) is present.

    Args:
        valley_lines (GeoDataFrame): Valley centerlines with geometries and unique FID.
        dtm_path (str): Path to the input DTM raster.
        sampling_distance (float): Distance between elevation samples along each line (in meters).
        smoothing_window (int): Window size for Savitzky-Golay filter (must be odd).
        polyorder (int): Polynomial order for Savitzky-Golay smoothing.
        top_n (int): Maximum number of keypoints to retain per valley line. Use large values to get points about every min_distance meters.
        min_distance (float): Minimum distance between selected keypoints (in meters).
        find_window_distance (float): Distance used for the prominence window to find concave and convex transitions or almost transitions (curvature) ordered according to the second derivative (in meters).
        plot_debug (bool): If True, plot elevation profiles and keypoints for visual inspection.

    Returns:
        GeoDataFrame: Detected keypoints as point geometries with metadata.
    """
    results = []

    with rasterio.open(dtm_path) as src:
        pixel_size = src.res[0]
        dtm_crs = src.crs
        for idx, row in valley_lines.iterrows():
            line = row.geometry
            line_id = row.FID
            length = line.length
            num_samples = int(length // sampling_distance)

            if num_samples < smoothing_window or num_samples < 10:
                continue

            distances = np.linspace(0, length, num=num_samples)
            sample_points = [line.interpolate(d) for d in distances]
            coords = [(pt.x, pt.y) for pt in sample_points]
            elevations = [val[0] for val in sample_gen(src, coords)]

            # Smooth elevation profile
            elev_smooth = savgol_filter(elevations, smoothing_window, polyorder)

            # Second derivative = curvature
            curvature = savgol_filter(elevations, smoothing_window, polyorder, deriv=2)

            # Find concave→convex transitions
            find_window = max(3, int(round(find_window_distance / pixel_size)))  # mindestens 3 Zellen
            candidates = find_inflection_candidates(curvature, window=find_window)

            # Sort and select strongest candidates
            sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

            # Check for minimum distance between keypoints
            accepted = []
            for i, strength in sorted_candidates:
                pt = sample_points[i]
                if all(pt.distance(p[0]) >= min_distance for p in accepted):
                    accepted.append((pt, strength, i))
                if len(accepted) >= top_n:
                    break

            for rank, (pt, _, idx_pt) in enumerate(accepted, start=1):
                results.append({
                    "geometry": Point(pt),
                    "valley_id": row["FID"],
                    "elev_index": idx_pt,
                    "rank": rank,
                    "curvature": curvature[idx_pt]
                })

            # Optional Plot
            if plot_debug:
                plt.figure(figsize=(10, 4))
                plt.plot(distances, elevations, label="Raw Elevation", linewidth=2)
                plt.plot(distances, elev_smooth, label="Smoothed Elevation", linewidth=2)
                for pt, _, i in accepted:
                    plt.axvline(x=distances[i], color='red', linestyle='--', alpha=0.7)

                for rank, (pt, _, i) in enumerate(accepted, start=1):
                    x = distances[i]
                    if rank == 1:
                        plt.axvline(x=x, color='red', linestyle='--', alpha=0.7, label="Keypoints (Nr=rank)")
                    else:
                        plt.axvline(x=x, color='red', linestyle='--', alpha=0.7)
                    plt.axvline(x=x, color='red', linestyle='--', alpha=0.7)
                    plt.text(x, elev_smooth[i], str(rank), color='red', fontsize=10, ha='left', va='bottom')

                plt.title(f"Valley Line '{line_id}': Inflection Points (Concave → Convex)")
                plt.xlabel("Distance along line (m)")
                plt.ylabel("Elevation")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    if dtm_crs:
        gdf = gpd.GeoDataFrame(results, geometry="geometry", crs=dtm_crs)
    else:
        gdf = gpd.GeoDataFrame(results, geometry="geometry", crs=valley_lines.crs)

    return gdf

def get_orthogonal_directions_start_points(
    barrier_raster_path: str,
    point: Point,
    line_geom: LineString,
    max_offset: int = 5
) -> tuple[Point, Point]:
    """
    Determine two start points to the left and right of a given point, orthogonal to an input line.

    The function searches along the orthogonal direction from a given point until it finds
    a non-barrier cell in the provided raster.

    Args:
        barrier_raster_path (str): Path to binary raster (GeoTIFF) with 1 = barrier, 0 = free.
        point (Point): The reference point (typically a keypoint on a valley line).
        line_geom (LineString): Reference line geometry used to determine orientation.
        max_offset (int): Maximum number of cells to move outward when searching.

    Returns:
        tuple: (left_point, right_point), or (None, None) if no valid points found.
    """
    with rasterio.open(barrier_raster_path) as src:
        barrier_mask = src.read(1)
        rows, cols = barrier_mask.shape
        res = src.res[0]  # assumes square pixels

        # Find nearest segment and compute tangent vector
        nearest_pt = nearest_points(point, line_geom)[1]
        coords = list(line_geom.coords)

        min_dist = float("inf")
        tangent = None
        for i in range(1, len(coords)):
            seg = LineString([coords[i - 1], coords[i]])
            dist = seg.distance(nearest_pt)
            if dist < min_dist:
                min_dist = dist
                dx = coords[i][0] - coords[i - 1][0]
                dy = coords[i][1] - coords[i - 1][1]
                norm = np.linalg.norm([dx, dy])
                if norm > 0:
                    tangent = np.array([dx, dy]) / norm

        if tangent is None:
            return None, None

        # Compute orthogonal direction vectors
        ortho_left = np.array([-tangent[1], tangent[0]])
        ortho_right = np.array([tangent[1], -tangent[0]])

        def find_valid_point(direction_vec):
            for i in range(1, max_offset + 1):
                offset = res * i
                test_x = point.x + direction_vec[0] * offset
                test_y = point.y + direction_vec[1] * offset

                # use br.index to get the correct row/col
                row_idx, col_idx = src.index(test_x, test_y)

                # now bounds-check
                if not (0 <= row_idx < rows and 0 <= col_idx < cols):
                    continue

                # barrier == 1 means forbidden
                if barrier_mask[row_idx, col_idx] != 1:
                    return Point(test_x, test_y)

            return None

        left_pt = find_valid_point(ortho_left)
        right_pt = find_valid_point(ortho_right)

        return left_pt, right_pt
    
    
def create_slope_cost_raster(
    dtm_path: str,
    start_point: Point,
    output_cost_raster_path: str,
    slope: float = 0.01,
    barrier_mask: np.ndarray = None,
    penalty_exp: float = 4.0
) -> str:
    """
    Create a raster with cost values based on deviation from desired slope.
    You can now set penalty_exp>1 to punish larger deviations more heavily.

    Args:
        dtm_path (str): Path to the digital terrain model (GeoTIFF).
        start_point (Point): Starting point of the constant slope line.
        output_cost_raster_path (str): Path to output cost raster.
        slope (float): Desired slope (1% downhill = 0.01).
        barrier_mask (np.ndarray): Binary mask of barriers (1=barrier).
        penalty_exp (float): Exponent on the absolute deviation (>=1). 2.0 => quadratic penalty.

    Returns:
        str: Path to the written cost raster.
    """
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1).astype(np.float32)
        nodata = src.nodata
        dtm[dtm == nodata] = np.nan

        rows, cols = dtm.shape
        rr, cc = np.indices((rows, cols))
        key_row, key_col = src.index(start_point.x, start_point.y)

        # elevation difference and horizontal distance
        dz = dtm - dtm[key_row, key_col]
        dist = np.hypot(rr - key_row, cc - key_col) * src.res[0]
        expected_dz = -dist * slope

        # linear deviation
        deviation = np.abs(dz - expected_dz)

        # apply exponentiation for stronger penalty
        cost = deviation ** penalty_exp

        # enforce NoData
        cost[np.isnan(dtm)] = 1e6

        # enforce barrier
        if barrier_mask is not None:
            cost[barrier_mask.astype(bool)] = 1e6

        # zero‐cost at the true start
        cost[key_row, key_col] = 0

        profile = src.profile
        profile.update(dtype=rasterio.float32, nodata=1e6)

    # write out
    with rasterio.open(output_cost_raster_path, "w", **profile) as dst:
        dst.write(cost, 1)

    return output_cost_raster_path


def create_source_raster(
    reference_raster_path: str,
    source_point: Point,
    output_source_raster_path: str,
    ) -> str:
    """
    Create a binary raster marking the source cell (value = 1) based on a given Point.
    All other cells are set to 0.

    Args:
        reference_raster_path (str): Path to the reference raster (e.g., DTM, GeoTIFF).
        source_point (Point): Shapely Point marking the source location.
        output_source_raster_path (str): Path to output binary raster (GeoTIFF).

    Returns:
        str: Path to the saved binary raster file.
    """
    with rasterio.open(reference_raster_path) as src:
        data = np.zeros(src.shape, dtype=np.uint8)

        # Convert Point to raster indices
        row, col = src.index(source_point.x, source_point.y)

        # Set source cell to 1
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            data[row, col] = 1
        else:
            raise ValueError("Source point is outside the bounds of the reference raster.")

        # Update profile
        profile = src.profile
        profile.update(dtype=rasterio.uint8, nodata=0)

        with rasterio.open(output_source_raster_path, "w", **profile) as dst:
            dst.write(data, 1)

    return output_source_raster_path

def select_best_destination_cell(
    accum_raster_path: str,
    destination_raster_path: str,
    output_best_destination_raster_path: str
) -> str:
    """
    Select the best destination cell from a binary destination raster based on
    minimum accumulated cost, and write it as a single-cell binary raster.

    Args:
        accum_raster_path (str): Path to the cost accumulation raster (GeoTIFF).
        destination_raster_path (str): Path to the binary destination raster (1 = destination, 0 = background).
        output_best_destination_raster_path (str): Path to output raster with only the best cell marked (GeoTIFF).

    Returns:
        str: Path to the output best destination raster (GeoTIFF).
    """
    with rasterio.open(accum_raster_path) as acc_src, rasterio.open(destination_raster_path) as dest_src:
        acc_data = acc_src.read(1)
        dest_data = dest_src.read(1)

        # Consider only destination cells (where value == 1)
        mask = (dest_data == 1)
        acc_masked = np.where(mask, acc_data, np.nan)

        # Identify cell with minimum accumulated cost
        if np.all(np.isnan(acc_masked)):
            raise RuntimeError("No valid destination cell found.")

        min_idx = np.unravel_index(np.nanargmin(acc_masked), acc_masked.shape)

        # Create output raster marking only the best cell
        best_dest = np.zeros_like(dest_data, dtype=np.uint8)
        best_dest[min_idx] = 1

        profile = dest_src.profile
        profile.update(dtype=rasterio.uint8, nodata=0)

        with rasterio.open(output_best_destination_raster_path, "w", **profile) as dst:
            dst.write(best_dest, 1)

        return output_best_destination_raster_path
    
def get_constant_slope_line(
    dtm_path: str,
    start_point: Point,
    destination_mask: np.ndarray,
    slope: float = 0.01,
    barrier_mask: np.ndarray = None
) -> LineString:
    """
    Trace lines with constant slope starting from a given point using a cost-distance approach based on slope deviation.

    This function creates a cost raster that penalizes deviation from the desired slope,
    runs a least-cost-path analysis using WhiteboxTools, and returns the resulting line.

    Args:
        dtm_path (str): Path to the digital terrain model (GeoTIFF).
        start_point (Point): Starting point of the constant slope line (usually located on a valley line).
        destination_mask (np.ndarray): Binary mask indicating destination cells (1 = destination).
        slope (float): Desired slope for the line (e.g., 0.01 for 1% downhill or -0.01 for uphill).
        barrier_mask (np.ndarray): Optional binary mask of cells that should not be crossed (1 = barrier).

    Returns:
        LineString: Least-cost slope path as a Shapely LineString, or None if no path found.
    """
    if wbt is None:
        raise RuntimeError("WhiteboxTools not initialized.")

    # --- File paths ---
    cost_raster_path = os.path.join(TEMP_DIRECTORY, "cost.tif")
    source_raster_path = os.path.join(TEMP_DIRECTORY, "source.tif")
    destination_raster_path = os.path.join(TEMP_DIRECTORY, "destination.tif")
    accum_raster_path = os.path.join(TEMP_DIRECTORY, "accum.tif")
    backlink_raster_path = os.path.join(TEMP_DIRECTORY, "backlink.tif")
    best_destination_path = os.path.join(TEMP_DIRECTORY, "destination_best.tif")
    pathway_raster_path = os.path.join(TEMP_DIRECTORY, "pathway.tif")

    # --- Create cost raster ---
    cost_raster_path = create_slope_cost_raster(
        dtm_path=dtm_path,
        start_point=start_point,
        output_cost_raster_path=cost_raster_path,
        slope=slope,
        barrier_mask=barrier_mask
    )

    # --- Create source raster ---
    source_raster_path = create_source_raster(
        reference_raster_path=dtm_path,
        source_point=start_point,
        output_source_raster_path=source_raster_path
    )

    # --- Create binary destination raster ---
    with rasterio.open(dtm_path) as src:
        data = np.zeros(src.shape, dtype=np.uint8)
        data[destination_mask > 0] = 1  # Apply binary mask
        profile = src.profile
        profile.update(dtype=rasterio.uint8, nodata=0)
        with rasterio.open(destination_raster_path, "w", **profile) as dst:
            dst.write(data, 1)

    # --- Run cost-distance analysis ---
    wbt.cost_distance(
        source=source_raster_path,
        cost=cost_raster_path,
        out_accum=accum_raster_path,
        out_backlink=backlink_raster_path
    )

    # --- Select best destination cell ---
    best_destination_path = select_best_destination_cell(
        accum_raster_path=accum_raster_path,
        destination_raster_path=destination_raster_path,
        output_best_destination_raster_path=best_destination_path
    )

    # --- Trace least-cost pathway ---
    wbt.cost_pathway(
        destination=best_destination_path,
        backlink=backlink_raster_path,
        output=pathway_raster_path
    )

    # --- Set correct NoData value for pathway raster ---
    with rasterio.open(backlink_raster_path) as src:
        nodata_value = src.nodata
    with rasterio.open(pathway_raster_path, 'r+') as dst:
        dst.nodata = nodata_value

    # --- Convert to LineString ---
    line = raster_to_linestring_wbt(pathway_raster_path)
    if line is None:
        print("[SlopeLine] No valid line could be extracted from pathway raster.")
        return None

    # --- Optional smoothing ---
    line = smooth_linestring(line, sigma=1.0)

    return line

def get_constant_slope_lines(
    dtm_path: str,
    start_points: gpd.GeoDataFrame,
    destination_features: list[gpd.GeoDataFrame],
    slope: float = 0.01,
    barrier_features: list[gpd.GeoDataFrame] = None
) -> gpd.GeoDataFrame:
    """
    Trace lines with constant slope starting from given points using a cost-distance approach
    based on slope deviation, snapping true original start-points only when they overlapped barrier lines.
    All barrier_features (lines, polygons, points) are rasterized into barrier_mask,
    but only the line geometries are used for splitting and offsetting start points.
    """
    print("[ConstantSlopeLines] Starting tracing")
    original_pts = start_points.copy()

    # --- Destination mask ---
    print("[ConstantSlopeLines] Building destination mask…")
    destination_processed = []
    for gdf in destination_features:
        if gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            g = gdf.copy()
            g["geometry"] = g.boundary
            destination_processed.append(g)
        else:
            destination_processed.append(gdf)
    destination_mask = vector_to_mask(destination_processed, dtm_path)
    print("[ConstantSlopeLines] Destination mask ready")

    # Raster metadata
    with rasterio.open(dtm_path) as src:
        profile = src.profile.copy()
        res = src.res[0]
        dtm_crs = src.crs

    # --- Barrier handling ---
    if barrier_features:
        # 1) split into line vs non-line
        print("[ConstantSlopeLines] Separating line vs non-line barriers")
        line_geoms = []
        non_line_gdfs = []
        for gdf in barrier_features:
            # collect only the geometries themselves
            lines = [geom for geom in gdf.geometry if geom.geom_type in ("LineString", "MultiLineString")]
            if lines:
                line_geoms.extend(lines)
            # everything goes into non-line list, too:
            non_line = gdf[~gdf.geom_type.isin(["LineString","MultiLineString"])]
            if not non_line.empty:
                non_line_gdfs.append(non_line)

        print(f"[ConstantSlopeLines]  → {len(line_geoms)} total barrier line geometries, "
              f"{len(non_line_gdfs)} non-line GeoDataFrames")

        # 2) buffer *all* barriers for mask: buffer lines slightly, leave others as-is
        print("[ConstantSlopeLines] Buffering lines and preparing mask layers")
        buf_dist = res + 0.01 ################ Später vielleicht ohne Buffer und nur mit all_touched in Rasterio Rasterize in vector_to_mask (funktioniert nicht wie erwartet. wbt Versuche noch VectorPolygonsToRaster)
        buffered_line_gdf = gpd.GeoDataFrame(geometry=[ln.buffer(buf_dist) for ln in line_geoms],
                                             crs=start_points.crs)
        mask_layers = [buffered_line_gdf] + non_line_gdfs

        print("[ConstantSlopeLines] Rasterizing all barrier features into mask")
        barrier_mask = vector_to_mask(mask_layers, dtm_path)
        barrier_raster = os.path.join(TEMP_DIRECTORY, "barrier_mask.tif")
        with rasterio.open(barrier_raster, "w", **profile) as dst:
            dst.write(barrier_mask.astype("uint8"), 1)
        print(f"[ConstantSlopeLines] Barrier raster saved to {barrier_raster}")

        # 3) fast-overlap on *unbuffered* lines only
        if line_geoms:
            print("[ConstantSlopeLines] Fast-check: which start points intersect barrier lines?")
            union_lines = unary_union(line_geoms)
            # buffer outward by a tiny amount
            tol = res * 0.1
            buffered_lines = union_lines.buffer(tol)
            # now intersects is more robust
            overlaps = original_pts.geometry.intersects(buffered_lines)
            overlapping = original_pts[overlaps]
            non_overlapping  = original_pts[~overlaps]
            print(f"[ConstantSlopeLines]  → {len(overlapping)} overlapping, {len(non_overlapping)} non-overlapping")
        else:
            print("[ConstantSlopeLines] No line barriers → all points non-overlapping")
            overlapping = original_pts.iloc[0:0]
            non_overlapping = original_pts.copy()

        # 4) build adjusted start points only for the true overlaps
        if not overlapping.empty:
            print("[ConstantSlopeLines] Generating adjusted start points for overlaps…")
            adjusted_records = []
            tol = 1e-6
            for orig_idx, row in overlapping.iterrows():
                pt = row.geometry
                for line in line_geoms:
                    if pt.distance(line) < tol:
                        print(f"[ConstantSlopeLines]  Point {orig_idx} on a barrier line → offsetting")
                        left_pt, right_pt = get_orthogonal_directions_start_points(
                            barrier_raster_path=barrier_raster,
                            point=pt,
                            line_geom=line
                        )
                        if left_pt:
                            adjusted_records.append({"geometry": left_pt, "orig_index": orig_idx})
                            print(f"[ConstantSlopeLines]   → Left offset for {orig_idx}")
                        if right_pt:
                            adjusted_records.append({"geometry": right_pt, "orig_index": orig_idx})
                            print(f"[ConstantSlopeLines]   → Right offset for {orig_idx}")
                        break
                else:
                    print(f"[ConstantSlopeLines]   No precise line match for {orig_idx} → treated as non-overlap")
                    non_overlapping = non_overlapping.append(row)

            print(f"[ConstantSlopeLines] Created {len(adjusted_records)} adjusted start points")
            adj_gdf = gpd.GeoDataFrame(adjusted_records, crs=start_points.crs).reset_index(drop=True)
            # build mapping if needed
            orig_to_adjusted = defaultdict(list)
            for adj_idx, rec in adj_gdf.iterrows():
                orig_to_adjusted[rec.orig_index].append(adj_idx)
        else:
            print("[ConstantSlopeLines] No overlapping start points → skipping adjustment step")
            adj_gdf = gpd.GeoDataFrame(columns=["geometry","orig_index"], crs=start_points.crs)
            orig_to_adjusted = {}

    else:
        print("[ConstantSlopeLines] No barrier features provided")
        barrier_mask = None
        adj_gdf = gpd.GeoDataFrame(columns=["geometry","orig_index"], crs=start_points.crs)
        non_overlapping = original_pts.copy()
        orig_to_adjusted = {}

    # --- Trace slope lines ---
    print("[ConstantSlopeLines] Tracing from adjusted points…")
    results = []
    with rasterio.open(dtm_path) as src:
        # adjusted
        for adj_idx, row in adj_gdf.iterrows():
            pt, orig_idx = row.geometry, row.orig_index
            orig_attrs = original_pts.loc[orig_idx].drop(labels="geometry").to_dict()

            print(f"[ConstantSlopeLines]  Adjusted pt {adj_idx} (orig {orig_idx})…")
            r, c = src.index(pt.x, pt.y)
            if barrier_mask is not None and 0 <= r < barrier_mask.shape[0] and 0 <= c < barrier_mask.shape[1]:
                if barrier_mask[r, c] == 1:
                    warnings.warn(f"[ConstantSlopeLines] Adjusted pt {adj_idx} on barrier cell")

            raw_line = get_constant_slope_line(
                dtm_path=dtm_path,
                start_point=pt,
                destination_mask=destination_mask,
                slope=slope,
                barrier_mask=barrier_mask
            )
            if not raw_line:
                print(f"[ConstantSlopeLines]   → No line for adjusted {adj_idx}")
                continue

            # if we got a MultiLineString, stitch it into one LineString
            if isinstance(raw_line, MultiLineString):
                raw_line = stitch_multilinestring(raw_line)
            # snap original at start
            coords = list(raw_line.coords)
            s_pt, e_pt = Point(coords[0]), Point(coords[-1])
            if original_pts.loc[orig_idx].geometry.distance(s_pt) <= original_pts.loc[orig_idx].geometry.distance(e_pt):
                if original_pts.loc[orig_idx].geometry != s_pt:
                    new_coords = [original_pts.loc[orig_idx].geometry.coords[0]] + coords
                else:
                    new_coords = coords
            else:
                if original_pts.loc[orig_idx].geometry != e_pt:
                    new_coords = coords + [original_pts.loc[orig_idx].geometry.coords[0]]
                else:
                    new_coords = coords
                new_coords.reverse()

            results.append({
                "geometry": LineString(new_coords),
                "orig_index": orig_idx,
                "adj_index": adj_idx,
                **orig_attrs
            })
            print(f"[ConstantSlopeLines]   → Line created for adjusted {adj_idx}")

        # non-overlapping
        print("[ConstantSlopeLines] Tracing from non-overlapping points…")
        for orig_idx, row in non_overlapping.iterrows():
            orig_pt = row.geometry
            orig_attrs = original_pts.loc[orig_idx].drop(labels="geometry").to_dict()
            print(f"[ConstantSlopeLines]  Orig pt {orig_idx} (no barrier)…")

            raw_line = get_constant_slope_line(
                dtm_path=dtm_path,
                start_point=orig_pt,
                destination_mask=destination_mask,
                slope=slope,
                barrier_mask=barrier_mask
            )

            if raw_line:
                # if we got a MultiLineString, stitch it into one LineString
                if isinstance(raw_line, MultiLineString):
                    raw_line = stitch_multilinestring(raw_line)

                coords = list(raw_line.coords)
                s_pt, e_pt = Point(coords[0]), Point(coords[-1])
                if orig_pt.distance(s_pt) <= orig_pt.distance(e_pt):
                    if orig_pt != s_pt:
                        new_coords = [orig_pt.coords[0]] + coords
                    else:
                        new_coords = coords
                else:
                    if orig_pt != e_pt:
                        new_coords = coords + [orig_pt.coords[0]]
                    else:
                        new_coords = coords
                    new_coords.reverse()

                results.append({
                    "geometry": LineString(new_coords),
                    "orig_index": orig_idx,
                    **orig_attrs
                })
                print(f"[ConstantSlopeLines]   → Line created for orig {orig_idx} (snapped)")
            else:
                print(f"[ConstantSlopeLines]   → No line for orig {orig_idx}")

    if not results:
        raise RuntimeError("No slope lines could be created.")
    print(f"[ConstantSlopeLines] Done: generated {len(results)} lines")

    # build GeoDataFrame including all original attributes
    out_gdf = gpd.GeoDataFrame(results, crs=dtm_crs)
    if barrier_features and line_geoms:
        out_gdf.orig_to_adjusted = dict(orig_to_adjusted)
    return out_gdf



if __name__ == "__main__":
    print("No main part")
