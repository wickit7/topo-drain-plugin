# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: topo_drain_core.py
#
# Purpose: Script with python functions of topo drain qgis plugin
#
# -----------------------------------------------------------------------------
import os
import sys
import importlib.util
from typing import Union
import warnings
import uuid
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box
from shapely.ops import linemerge, nearest_points, substring
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import re
from osgeo import gdal, ogr

# ---  Class TopoDrainCore ---
class TopoDrainCore:
    def __init__(self, whitebox_directory=None, nodata=None, crs=None, temp_directory=None, working_directory=None):
        print("[TopoDrainCore] Initializing TopoDrainCore...")
        self._thisdir = os.path.dirname(__file__)
        print(f"[TopoDrainCore] Module directory: {self._thisdir}")
        
        # Handle None whitebox_directory gracefully
        if whitebox_directory is None:
            print("[TopoDrainCore] No WhiteboxTools directory provided")
            self.whitebox_directory = None  # Keep as None to trigger lazy loading
        else:
            self.whitebox_directory = whitebox_directory
        print(f"[TopoDrainCore] WhiteboxTools directory: {self.whitebox_directory if self.whitebox_directory else 'Not set'}")
        
        self.nodata = nodata if nodata is not None else -32768
        print(f"[TopoDrainCore] NoData value set to: {self.nodata}")
        self.crs = crs if crs is not None else "EPSG:4326"
        print(f"[TopoDrainCore] crs value set to: {self.crs}")
        self.temp_directory = temp_directory if temp_directory is not None else None
        print(f"[TopoDrainCore] Temp directory set to: {self.temp_directory if self.temp_directory else 'Not set'}")
        self.working_directory = working_directory if working_directory is not None else None
        print(f"[TopoDrainCore] Working directory set to: {self.working_directory if self.working_directory else 'Not set'}")
        
        # Define supported GDAL driver mappings for raster formats (has to be compatible with available raster formats for WhiteboxTools)
        self.gdal_driver_mapping = {
            '.tif': 'GTiff',
            '.tiff': 'GTiff',
            '.hdr': 'EHdr',
            '.asc': 'AAIGrid',
            '.bil': 'EHdr',
            '.gpkg': 'GPKG',
            '.sdat': 'SAGA',
            '.sgrd': 'SAGA',
            '.rdc': 'RDxC',
            '.rst': 'RST'

        }
        print(f"[TopoDrainCore] Configured GDAL driver support for {len(self.gdal_driver_mapping)} raster formats")
        
        # Define supported OGR driver mappings for vector formats
        self.ogr_driver_mapping = {
            '.shp': 'ESRI Shapefile',
            '.gpkg': 'GPKG',
            '.geojson': 'GeoJSON',
            '.json': 'GeoJSON',
            '.gml': 'GML'
        }
        print(f"[TopoDrainCore] Configured OGR driver support for {len(self.ogr_driver_mapping)} vector formats")
        
        # Configure GDAL settings for the entire class
        self._configure_gdal()
        
        # Try to initialize WhiteboxTools, but don't fail if it's not available
        self.wbt = self._init_whitebox_tools(self.whitebox_directory)
        print(f"[TopoDrainCore] WhiteboxTools initialized: {self.wbt is not None}")
        print("[TopoDrainCore] Initialization complete.")

    def _init_whitebox_tools(self, whitebox_directory):
        """
        Initialize WhiteboxTools with graceful fallback for plugin loading order issues.
        Returns None if WhiteboxTools cannot be initialized (will be configured later).
        """
        if whitebox_directory is None:
            print("[TopoDrainCore] WhiteboxTools directory not provided - will be configured when available")
            return None
            
        try:
            # Add WhiteboxTools directory to sys path
            if whitebox_directory not in sys.path:
                sys.path.insert(0, whitebox_directory)
                
            # Try to import from the provided location
            wbt_path = os.path.join(whitebox_directory, "whitebox_tools.py")
            if not os.path.exists(wbt_path):
                print(f"[TopoDrainCore] WhiteboxTools not found at {wbt_path}")
                return None
            
            # Step 1: Create the specification for WhiteboxTools module
            spec = importlib.util.spec_from_file_location("whitebox_tools", wbt_path)
            if spec is None or spec.loader is None:
                print(f"[TopoDrainCore] Could not create spec for WhiteboxTools from {wbt_path}")
                return None
            
            # Step 2: Create empty module from spec
            whitebox_tools_mod = importlib.util.module_from_spec(spec)
            # Step 3: Register module in sys.modules
            sys.modules["whitebox_tools"] = whitebox_tools_mod
            # Step 4: Actually execute the module's code
            spec.loader.exec_module(whitebox_tools_mod)
            # Step 5: Access classes/functions from the loaded module
            WhiteboxTools = whitebox_tools_mod.WhiteboxTools
            # Step 6: Reference WhiteboxTools as wbt
            wbt = WhiteboxTools()
            if self.working_directory:
                wbt.set_working_dir(self.working_directory)
            print(f"[TopoDrainCore] Using WhiteboxTools from directory: {whitebox_directory}")
            return wbt
            
        except Exception as e:
            print(f"[TopoDrainCore] WhiteboxTools initialization failed: {e}")
            print("[TopoDrainCore] WhiteboxTools will be configured when available")
            return None

    ## GDAL/OGR setting functions
    def _configure_gdal(self):
        """
        Configure GDAL/OGR settings for the entire class instance.
        Sets up exception handling and error management for all GDAL/OGR operations.
        """
        try:
            # Enable GDAL/OGR exceptions for better error handling
            # Note: gdal.UseExceptions() enables exceptions for both GDAL and OGR
            gdal.UseExceptions()
            
            # Set quiet error handler to suppress console messages
            # We'll capture errors using GetLastErrorMsg() instead
            gdal.PushErrorHandler('CPLQuietErrorHandler')
            
            print("[TopoDrainCore] GDAL/OGR configured: exceptions enabled, quiet error handler set")
            
        except Exception as e:
            print(f"[TopoDrainCore] Warning: Failed to configure GDAL/OGR settings: {e}")
            print("[TopoDrainCore] GDAL/OGR operations may have less detailed error reporting")

    def __del__(self):
        """
        Cleanup method to restore GDAL error handler when object is destroyed.
        """
        try:
            gdal.PopErrorHandler()  # Restore default error handler
        except:
            pass  # Ignore errors during cleanup

    @staticmethod
    def _get_gdal_error_message():
        """
        Helper method to get the last GDAL/OGR error message with proper formatting.
        Note: GDAL and OGR share the same error system (CPL), so this works for both.
        
        Returns:
            str: Formatted GDAL/OGR error message or empty string if no error
        """
        try:
            error_msg = gdal.GetLastErrorMsg()
            return f" GDAL/OGR Error: {error_msg}" if error_msg else ""
        except:
            return ""

    @staticmethod
    def _check_ogr_error(operation_name: str, error_code):
        """
        Helper method to check OGR operation return codes and raise appropriate errors.
        
        Args:
            operation_name (str): Name of the operation for error messages
            error_code: The return code from OGR operation
            
        Raises:
            RuntimeError: If the operation failed
        """
        if error_code != ogr.OGRERR_NONE:
            error_msg = TopoDrainCore._get_gdal_error_message()
            raise RuntimeError(f"{operation_name} failed.{error_msg} (Error code: {error_code})")

    def _get_gdal_driver_from_path(self, file_path: str) -> str:
        """
        Get appropriate GDAL driver name based on file extension.
        
        Args:
            file_path (str): Path to the output file
            
        Returns:
            str: GDAL driver name corresponding to the file extension
        """
        ext = os.path.splitext(file_path)[1].lower()
        return self.gdal_driver_mapping.get(ext, 'GTiff')  # Default to GTiff if unknown extension

    def _get_ogr_driver_from_path(self, file_path: str) -> str:
        """
        Get appropriate OGR driver name based on file extension.
        
        Args:
            file_path (str): Path to the output file
            
        Returns:
            str: OGR driver name corresponding to the file extension
        """
        ext = os.path.splitext(file_path)[1].lower()
        return self.ogr_driver_mapping.get(ext, 'ESRI Shapefile')  # Default to Shapefile if unknown extension


    ## Setters for class configuration
    def set_temp_directory(self, temp_dir):
        self.temp_directory = temp_dir

    def set_working_directory(self, working_dir):
        self.working_directory = working_dir
        if self.wbt is not None and self.working_directory:
            self.wbt.set_working_dir(self.working_directory)

    def set_nodata_value(self, nodata):
        self.nodata = nodata

    def set_crs(self, crs):
        self.crs = crs

    ## WhiteboxTools helper functions
    def _execute_wbt(self, tool_name, feedback=None, report_progress=True, **kwargs):
        """
        Execute a WhiteboxTools command using the Python API.
        
        Args:
            tool_name (str): Name of the WhiteboxTools command
            feedback (QgsProcessingFeedback, optional): Feedback object for progress reporting
            report_progress (bool): Whether to report progress via setProgress calls. Default True.
            **kwargs: Tool parameters as keyword arguments
            
        Returns:
            int: Return code (0 = success)
            
        Raises:
            RuntimeError: If WhiteboxTools is not properly configured
        """
        # Check if WhiteboxTools is available
        if self.wbt is None:
            error_msg = "WhiteboxTools is not properly configured. Please ensure WhiteboxTools plugin is loaded and configured."
            if feedback:
                feedback.reportError(error_msg)
            raise RuntimeError(error_msg)
        
        # Helper function to determine if a message should be displayed
        def should_show_message(msg):
            if '%' not in msg:
                return True  # Show all non-percentage messages
            
            try:
                parts = msg.split('%')
                if len(parts) > 1:
                    progress_part = parts[0].strip().split()[-1]
                    progress = int(progress_part)
                    
                    # Only show Initializing/Progress messages at 25% intervals
                    if ("Initializing:" in msg or "Progress:" in msg):
                        return progress % 25 == 0 or progress == 100
                    else:
                        return True  # Show other percentage messages
                else:
                    return True  # Show messages that contain % but can't be parsed
            except (ValueError, IndexError):
                return True  # Show messages that contain % but can't be parsed as progress

        # Create callback function for progress reporting and logging
        def callback_func(message):
            # Check for cancellation first - this allows immediate cancellation during WhiteboxTools execution
            if feedback and feedback.isCanceled():
                raise RuntimeError("[WBT] Process cancelled by user during WhiteboxTools execution.")
            
            if feedback:
                # Parse progress percentage if available and report_progress is True
                if '%' in message and report_progress:
                    try:
                        # Extract progress from messages like "Progress: 45%" or "Initializing: 45%"
                        parts = message.split('%')
                        if len(parts) > 1:
                            progress_part = parts[0].strip().split()[-1]
                            progress = int(progress_part)
                            
                            # Only report progress at 25% intervals (0, 25, 50, 75, 100) to reduce noise
                            if progress % 25 == 0 or progress == 100:
                                feedback.setProgress(progress)
                    except (ValueError, IndexError):
                        pass  # If parsing fails, just ignore progress
                
                # Filter and send console output
                if message.strip() and should_show_message(message):
                    feedback.pushConsoleInfo(message.strip())
            else:
                # Print to console when no feedback available - also apply filtering
                if message.strip() and should_show_message(message):
                    print(f"[WBT] {message.strip()}")
        
        # Get the tool method from WhiteboxTools object
        tool_method = getattr(self.wbt, tool_name, None)
        if tool_method is not None:
            # Use the convenience method if available (cleaner and more reliable)
            try:
                return tool_method(callback=callback_func, **kwargs)
            except Exception as e:
                if feedback:
                    feedback.reportError(f"WhiteboxTools error: {e}")
                raise RuntimeError(f"WhiteboxTools error: {e}")
        else:
            # Fallback to run_tool method for tools without convenience methods
            # Build arguments list for the tool
            args = []
            for param, value in kwargs.items():
                if value is not None:
                    args.append(f"--{param}='{value}'")
            
            try:
                return self.wbt.run_tool(tool_name, args, callback=callback_func)
            except Exception as e:
                if feedback:
                    feedback.reportError(f"WhiteboxTools error: {e}")
                raise RuntimeError(f"WhiteboxTools error: {e}")

    ## Vector geometry functions
    @staticmethod
    def _merge_lines_by_distance(line_geometries):
        """
        Merge multiple LineString geometries into a single LineString by
        connecting them based on closest endpoint distances.
        
        Can handle:
        - List of LineString/MultiLineString objects
        - Single LineString 
        - Single MultiLineString

        Args:
            line_geometries (list|LineString|MultiLineString): 
                Input geometries to merge.

        Returns:
            LineString: Single merged LineString, or None if merging fails.
        """
        print(f"[_merge_lines_by_distance] Start merge lines...")
        # Handle different input types
        if isinstance(line_geometries, LineString):
            return line_geometries
        elif isinstance(line_geometries, MultiLineString):
            line_list = list(line_geometries.geoms)
        elif isinstance(line_geometries, list):
            # Flatten any MultiLineString objects in the list
            line_list = []
            for geom in line_geometries:
                if isinstance(geom, LineString):
                    line_list.append(geom)
                elif isinstance(geom, MultiLineString):
                    line_list.extend(list(geom.geoms))
                else:
                    warnings.warn(f"[_merge_lines_by_distance] Warning: Skipping unsupported geometry type: {type(geom)}")
        else:
            warnings.warn(f"[_merge_lines_by_distance] Error: Unsupported input type: {type(line_geometries)}")
            return None

        
        if not line_list:
            return None
        
        if len(line_list) == 1:
            return line_list[0]
        
        # Convert to list to avoid modifying original
        remaining_lines = line_list.copy()

        remaining_lines.sort(key=lambda line: line.length, reverse=True)
        longest_line = remaining_lines.pop(0)
        merged_coords = list(longest_line.coords)
        iteration = 0
        while remaining_lines:
            iteration += 1
            
            # Get endpoints of current merged line
            start_point = Point(merged_coords[0])
            end_point = Point(merged_coords[-1])
            
            best_line_idx = None
            best_connection = None
            best_distance = float('inf')
            
            # Find the closest connection among all remaining lines
            for idx, line in enumerate(remaining_lines):
                line_start = Point(line.coords[0])
                line_end = Point(line.coords[-1])
                
                # Check all possible connections
                connections = [
                    ('start_to_start', start_point.distance(line_start), 'prepend_reversed'),
                    ('start_to_end', start_point.distance(line_end), 'prepend_normal'),
                    ('end_to_start', end_point.distance(line_start), 'append_normal'),
                    ('end_to_end', end_point.distance(line_end), 'append_reversed')
                ]
                
                for conn_type, distance, action in connections:
                    if distance < best_distance:
                        best_distance = distance
                        best_line_idx = idx
                        best_connection = action
            
            # Connect the best line
            if best_line_idx is not None:
                best_line = remaining_lines.pop(best_line_idx)
                line_coords = list(best_line.coords)
                
                if best_connection == 'prepend_normal':
                    # Add line to start of merged (line_end connects to merged_start)
                    merged_coords = line_coords[:-1] + merged_coords
                elif best_connection == 'prepend_reversed':
                    # Add reversed line to start of merged (line_start connects to merged_start)
                    line_coords.reverse()
                    merged_coords = line_coords[:-1] + merged_coords
                elif best_connection == 'append_normal':
                    # Add line to end of merged (merged_end connects to line_start)
                    merged_coords = merged_coords[:-1] + line_coords
                elif best_connection == 'append_reversed':
                    # Add reversed line to end of merged (merged_end connects to line_end)
                    line_coords.reverse()
                    merged_coords = merged_coords[:-1] + line_coords
                
            else:
                # No valid connection found, break
                break
        
        result = LineString(merged_coords)        

        return result

    @staticmethod
    def _smooth_linestring(geom, sigma: float = 1.0):
        """
        Smooth a LineString or MultiLineString geometry using a Gaussian filter,
        preserving the first and last point of the original geometry.

        Args:
            geom (LineString|MultiLineString): Input geometry to smooth.
            sigma (float): Standard deviation for Gaussian kernel.

        Returns:
            LineString or MultiLineString: Smoothed geometry.
        """
        # Handle MultiLineString by smoothing each part
        if isinstance(geom, MultiLineString):
            smoothed_parts = [TopoDrainCore._smooth_linestring(part, sigma) for part in geom.geoms]
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

        # Ensure first and last points are exactly as in the original
        x_smooth[0], y_smooth[0] = coords[0, 0], coords[0, 1]
        x_smooth[-1], y_smooth[-1] = coords[-1, 0], coords[-1, 1]

        # rebuild as a LineString
        smoothed = LineString(np.column_stack([x_smooth, y_smooth]))
        return smoothed

    @staticmethod
    def _flatten_to_linestrings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Flatten a GeoDataFrame to individual LineString geometries by:
        1. Converting polygons to boundaries
        2. Extracting individual LineStrings from MultiLineStrings
        3. Ignoring other geometry types
        
        Args:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame to process
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with only individual LineString geometries
        """
        # Convert polygons to boundaries if needed
        if gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            g = gdf.copy()
            g["geometry"] = g.boundary
        else:
            g = gdf.copy()
            
        # Flatten to individual LineStrings
        line_geoms = []
        for geom in g.geometry:
            print(f"[FlattenToLinestrings] Processing geometry: {geom.geom_type}")
            if geom.geom_type == "LineString":
                line_geoms.append(geom)
            elif geom.geom_type == "MultiLineString":
                line_geoms.extend(list(geom.geoms))  # Flatten MultiLineString to individual LineStrings
            # Ignore other geometry types
            
        return gpd.GeoDataFrame(geometry=line_geoms, crs=gdf.crs)


    @staticmethod
    def _features_to_single_linestring(features: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        """
        Convert features to individual LineString geometries by:
        1. Converting polygons to boundaries
        2. Flattening MultiLineStrings to individual LineStrings
        3. Ignoring other geometry types
        
        Args:
            features (list[gpd.GeoDataFrame]): List of GeoDataFrames to process
            
        Returns:
            gpd.GeoDataFrame: Single GeoDataFrame with only LineString geometries from all input features
        """
        all_line_geoms = []
        print(f"[FeaturesToSingleLinestring] Processing {len(features)} GeoDataFrames")
        
        for gdf in features:
            if gdf.empty:
                continue
                
            # Get CRS from first non-empty GeoDataFrame
            crs = gdf.crs
                
            # Use the helper function to flatten to LineStrings
            flattened_gdf = TopoDrainCore._flatten_to_linestrings(gdf)
            
            if not flattened_gdf.empty:
                all_line_geoms.extend(flattened_gdf.geometry.tolist())
                
        if not all_line_geoms:
            # Return empty GeoDataFrame with same structure
            return gpd.GeoDataFrame(geometry=[], crs=crs)
            
        result_gdf = gpd.GeoDataFrame(geometry=all_line_geoms, crs=crs)
        print(f"[FeaturesToSingleLinestring] Processed {len(result_gdf)} LineString geometries")
        return result_gdf

    @staticmethod
    def _snap_line_to_point(line: LineString, snap_point: Point, position: str = None) -> LineString:
        """
        Snap the closest endpoint of a line to a given Point.

        Args:
            line (LineString): Input line geometry.
            snap_point (Point): Point to snap to.
            position (str, optional): "start", "end", or None. 
                - "start" snaps the start of the line to the point, if endpoint is closer reverse line direction first.
                - "end" snaps the end of the line to the point, if startpoint is closer reverse line direction first.
                - None snaps the closer endpoint to the point.

        Returns:
            LineString: Line with endpoint snapped to the given point.
        """
        if not isinstance(line, LineString):
            warnings.warn("Warning: Cannot snap endpoint for MultiLineString geometry")
            return line

        coords = list(line.coords)
        if len(coords) < 2:
            return line

        dist_start = snap_point.distance(Point(coords[0]))
        dist_end = snap_point.distance(Point(coords[-1]))

        # Always snap to the closest endpoint
        if dist_start <= dist_end:
            coords[0] = (snap_point.x, snap_point.y)
            if position == "end":
                # Reverse line direction if snapping to end
                coords.reverse()
        else:
            coords[-1] = (snap_point.x, snap_point.y)
            if position == "start":
                # Reverse line direction if snapping to start
                coords.reverse()

        return LineString(coords)

    def _perimeter_from_features(self, input_features: list[gpd.GeoDataFrame], buffer_distance: float = 0) -> gpd.GeoDataFrame:
        """
        Create a bounding box perimeter polygon from input features with an optional buffer.
        
        Args:
            input_features (list[gpd.GeoDataFrame]): List of GeoDataFrames to create perimeter from
            buffer_distance (float, optional): Buffer distance to add around bounding box. If 0, no buffer is applied. Default 0.
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing a single polygon perimeter with buffer
            
        Raises:
            ValueError: If no valid features are provided to create bounding box
        """
        print("[PerimeterFromFeatures] Creating perimeter from input features...")
        
        # Filter out None or empty features
        valid_features = []
        for feature in input_features:
            if feature is not None and not feature.empty:
                valid_features.append(feature)
        
        if not valid_features:
            print("[PerimeterFromFeatures] Warning: No valid features provided to create bounding box")
            raise ValueError("No valid features provided to create bounding box")
        
        # Combine all features to get total bounds
        all_features = gpd.GeoDataFrame(pd.concat(valid_features, ignore_index=True))
        
        # Get bounding box and calculate buffer distance
        bounds = all_features.total_bounds  # [minx, miny, maxx, maxy]
        
        # Create bounding box polygon with buffer (if buffer_distance is 0, no buffer is applied)
        bbox_geom = box(
            bounds[0] - buffer_distance,  # minx
            bounds[1] - buffer_distance,  # miny  
            bounds[2] + buffer_distance,  # maxx
            bounds[3] + buffer_distance   # maxy
        )
        
        # Create GeoDataFrame for the bounding box perimeter
        perimeter = gpd.GeoDataFrame([{'geometry': bbox_geom}], crs=self.crs)
        
        print(f"[PerimeterFromFeatures] Created bounding box perimeter with {buffer_distance:.2f}m buffer")
        return perimeter
    
    def _vector_to_mask_raster(
        self,
        features: list[gpd.GeoDataFrame],
        reference_raster_path: str,
        output_path: str = None,
        unique_values: bool = False,
        flatten_lines: bool = True,
        buffer_lines: bool = False
        ) -> Union[str, tuple[str, dict]]:
        """
        Convert one or more GeoDataFrames to a binary raster mask (1 = feature, 0 = background)
        or a multi-value mask with unique values for each geometry and save as raster file.

        Args:
            features (list[GeoDataFrame]): List of GeoDataFrames (polygon or line geometries).
            reference_raster_path (str): Path to a reference raster for shape and transform.
            output_path (str, optional): Path to save the mask raster. If None, generates a temporary path.
            unique_values (bool): If True, assigns unique values (1, 2, 3, ...) to cells for each individual geometry.
                                If False, all features get value 1 (default behavior).
            flatten_lines (bool): If True, flattens MultiLineStrings to individual LineStrings before rasterization.
            buffer_lines (bool): If True, buffers lines by a small distance to ensure no diagonal gaps in rasterization.

        Returns:
            str | tuple[str, dict]: 
                - If unique_values=False: Path to the saved mask raster file
                - If unique_values=True: Tuple of (path, geometry_mapping) where geometry_mapping 
                  is a dict {raster_value: geometry} for each unique geometry
        """
        # If any input GeoDataFrame contains a 'LINK_ID' field, sort that GeoDataFrame
        # by LINK_ID to ensure deterministic ordering when assigning unique raster values.
        if unique_values:
            # Sort features by preferred ID column for deterministic unique value assignment
            sort_cols = ['LINK_ID', 'FID', 'fid', 'id']
            sorted_features = []
            for gdf in features:
                if isinstance(gdf, gpd.GeoDataFrame):
                    sort_col = next((col for col in sort_cols if col in gdf.columns), None)
                    if sort_col:
                        try:
                            sorted_gdf = gdf.sort_values(sort_col).reset_index(drop=True)
                        except Exception:
                            sorted_gdf = gdf
                        sorted_features.append(sorted_gdf)
                    else:
                        sorted_features.append(gdf)
                else:
                    sorted_features.append(gdf)
            features = sorted_features

        # Read reference raster information using GDAL
        ref_ds = gdal.Open(reference_raster_path, gdal.GA_ReadOnly)
        if ref_ds is None:
            raise RuntimeError(f"Cannot open reference raster: {reference_raster_path}.{self._get_gdal_error_message()}")
            
        try:
            # Get raster dimensions and geotransform
            width = ref_ds.RasterXSize
            height = ref_ds.RasterYSize
            geotransform = ref_ds.GetGeoTransform()
            projection = ref_ds.GetProjection()
            
            # Get complete spatial reference system
            srs = ref_ds.GetSpatialRef()
            
            res = abs(geotransform[1])  # pixel width
            
            # Validate geotransform
            if geotransform is None or len(geotransform) != 6:
                raise RuntimeError(f"Invalid geotransform in reference raster: {reference_raster_path}.{self._get_gdal_error_message()}")
                
        finally:
            ref_ds = None  # Close dataset

        buffer_distance = res + 0.01  # Small buffer to ensure rasterization of lines has no diagonal gaps ### with gdal version still needed? Debug later...

        # Collect all geometries from the provided GeoDataFrames
        all_geometries = [] 
        for gdf in features:
            if gdf.empty:
                continue
            else:
                # Use the helper function to flatten to LineStrings
                if flatten_lines and gdf.geometry.geom_type.isin(["MultiLineString", "LineString"]).any():
                    flattened_gdf = TopoDrainCore._flatten_to_linestrings(gdf) # flatten all line geometries to individual LineStrings
                    if not flattened_gdf.empty:
                        all_geometries.extend(flattened_gdf.geometry)
                else:
                    all_geometries.extend(gdf.geometry)

        # Generate output path if not provided
        if output_path is None:
            output_path = os.path.join(self.temp_directory, f"mask_{uuid.uuid4().hex[:8]}.tif")

        # Determine GDAL driver based on output file extension
        driver_name = self._get_gdal_driver_from_path(output_path)
        
        # Create empty output raster using GDAL with best practices
        driver = gdal.GetDriverByName(driver_name)
        if driver is None:
            raise RuntimeError(f"{driver_name} driver not available.{self._get_gdal_error_message()}")
            
        creation_options = [
            'COMPRESS=LZW',
            'TILED=YES',
            'BIGTIFF=IF_SAFER'
        ]
        
        out_ds = driver.Create(output_path, width, height, 1, gdal.GDT_UInt32, 
                              options=creation_options)
        if out_ds is None:
            raise RuntimeError(f"Failed to create output raster: {output_path}.{self._get_gdal_error_message()}")
            
        # Initialize geometry mapping for return value
        geometry_mapping = {}
            
        try:
            out_ds.SetGeoTransform(geotransform)
            
            # Set complete spatial reference system
            if srs is not None:
                out_ds.SetSpatialRef(srs)
            elif projection:
                # Fallback to projection string if SRS not available
                out_ds.SetProjection(projection)
            
            # Initialize raster with zeros
            out_band = out_ds.GetRasterBand(1)
            out_band.SetNoDataValue(0) # maybe use self.nodata? or fine for mask raster?
            out_band.Fill(0)

            # Create memory vector layer for rasterization
            mem_driver = ogr.GetDriverByName('Memory')
            if mem_driver is None:
                raise RuntimeError(f"Memory driver not available.{self._get_gdal_error_message()}")
                
            mem_ds = mem_driver.CreateDataSource('')
            if mem_ds is None:
                raise RuntimeError(f"Failed to create memory datasource.{self._get_gdal_error_message()}")
            
            try:
                mem_layer = mem_ds.CreateLayer('temp', None, ogr.wkbUnknown)
                if mem_layer is None:
                    raise RuntimeError(f"Failed to create memory layer.{self._get_gdal_error_message()}")
                
                # Add attribute field for raster values
                field_defn = ogr.FieldDefn('burn_value', ogr.OFTInteger)
                self._check_ogr_error("Create burn_value field", mem_layer.CreateField(field_defn))
                
                # Add geometries to layer
                for i, geom in enumerate(all_geometries):
                    if unique_values:
                        mask_value = i + 1  # Assign unique values to each individual geometry. Start from 1
                        geometry_mapping[mask_value] = geom  # Store the mapping
                    else:
                        mask_value = 1  # Default value for all geometries (binary mask)
                    
                    # Apply buffering to lines if requested
                    if geom.geom_type in ("LineString", "MultiLineString") and buffer_lines:
                        geom = geom.buffer(buffer_distance)
                    
                    # Create OGR feature
                    feature = ogr.Feature(mem_layer.GetLayerDefn())
                    feature.SetField('burn_value', int(mask_value))
                    
                    # Convert Shapely geometry to OGR geometry
                    ogr_geom = ogr.CreateGeometryFromWkt(geom.wkt)
                    if ogr_geom is None:
                        warnings.warn(f"Warning: Failed to convert geometry {i} to OGR format, skipping.{self._get_gdal_error_message()}")
                        continue
                    
                    # Validate geometry before setting
                    if not ogr_geom.IsValid():
                        warnings.warn(f"Warning: Invalid geometry {i}, attempting to fix or skipping.{self._get_gdal_error_message()}")
                        # Try to make it valid
                        try:
                            ogr_geom = ogr_geom.Buffer(0)  # Common trick to fix invalid geometries
                            if ogr_geom is None or not ogr_geom.IsValid():
                                ogr_geom = None
                                continue
                        except:
                            ogr_geom = None
                            continue
                        
                    feature.SetGeometry(ogr_geom)
                    
                    # Use the error checking helper for CreateFeature
                    result = mem_layer.CreateFeature(feature)
                    if result != ogr.OGRERR_NONE:
                        warnings.warn(f"Warning: Failed to create feature {i}, skipping.{self._get_gdal_error_message()} (Error code: {result})")
                    
                    feature = None  # Clean up feature
                    ogr_geom = None  # Clean up geometry
                
                # Rasterize the vector layer
                result = gdal.RasterizeLayer(out_ds, [1], mem_layer, 
                                           options=['ATTRIBUTE=burn_value', 'ALL_TOUCHED=YES'])
                if result != gdal.CE_None:
                    raise RuntimeError(f"Rasterization failed.{self._get_gdal_error_message()}")
                    
            finally:
                mem_layer = None
                mem_ds = None
                
        finally:
            out_ds = None  # Clean up output dataset

        # Return path only for binary mask, or tuple with geometry mapping for unique values
        if unique_values:
            return output_path, geometry_mapping
        else:
            return output_path

    ## Raster functions
    @staticmethod
    def _pixel_indices_to_coords(rows, cols, geotransform):
        """
        Convert pixel row,col indices to world coordinates using GDAL geotransform.
        Returns coordinates at the center of each pixel.
        
        GDAL Geotransform parameters:
        GT(0) x-coordinate of the upper-left corner of the upper-left pixel
        GT(1) w-e pixel resolution / pixel width
        GT(2) row rotation (typically zero)
        GT(3) y-coordinate of the upper-left corner of the upper-left pixel
        GT(4) column rotation (typically zero)
        GT(5) n-s pixel resolution / pixel height (negative value for a north-up image)
        
        Args:
            rows (array-like): Row indices of pixels
            cols (array-like): Column indices of pixels
            geotransform (tuple): GDAL geotransform parameters (6 values)
            
        Returns:
            list: List of (x, y) coordinate tuples in world coordinates
        """
        coords = []
        for row, col in zip(rows, cols):
            # Convert pixel indices to world coordinates (center of pixel)
            x = geotransform[0] + (col + 0.5) * geotransform[1] + (row + 0.5) * geotransform[2]
            y = geotransform[3] + (col + 0.5) * geotransform[4] + (row + 0.5) * geotransform[5]
            coords.append((x, y))
        return coords

    @staticmethod
    def _coords_to_pixel_indices(coords, geotransform):
        """
        Convert world coordinates to pixel indices using GDAL geotransform.
        
        GDAL Geotransform parameters:
        GT(0) x-coordinate of the upper-left corner of the upper-left pixel
        GT(1) w-e pixel resolution / pixel width
        GT(2) row rotation (typically zero)
        GT(3) y-coordinate of the upper-left corner of the upper-left pixel
        GT(4) column rotation (typically zero)
        GT(5) n-s pixel resolution / pixel height (negative value for a north-up image)
        
        Args:
            coords (list): List of (x, y) coordinate tuples in world coordinates
            geotransform (tuple): GDAL geotransform parameters (6 values)
            
        Returns:
            list: List of (px, py) pixel index tuples
        """
        pixel_indices = []
        for x, y in coords:
            px = int((x - geotransform[0]) / geotransform[1])
            py = int((y - geotransform[3]) / geotransform[5])
            pixel_indices.append((px, py))
        return pixel_indices

    # _clip_raster not used yet and therefore not debugged
    def _clip_raster(self, raster_path: str, mask: gpd.GeoDataFrame, out_path: str) -> str:
        """
        Clip a raster file using a polygon mask using GDAL.

        Args:
            raster_path (str): Path to input raster (supports all GDAL formats in gdal_driver_mapping).
            mask (GeoDataFrame): Polygon(s) to use as mask.
            out_path (str): Desired path for output masked raster.

        Returns:
            str: Path to the masked raster.
        """
        if mask.empty:
            raise ValueError("The provided GeoDataFrame is empty. Cannot mask raster.")

        # Create memory vector layer from mask GeoDataFrame
        mem_driver = ogr.GetDriverByName('Memory')
        if mem_driver is None:
            raise RuntimeError(f"Memory driver not available.{self._get_gdal_error_message()}")
            
        mem_ds = mem_driver.CreateDataSource('')
        if mem_ds is None:
            raise RuntimeError(f"Failed to create memory datasource.{self._get_gdal_error_message()}")
        
        try:
            mem_layer = mem_ds.CreateLayer('mask', None, ogr.wkbPolygon)
            if mem_layer is None:
                raise RuntimeError(f"Failed to create memory layer.{self._get_gdal_error_message()}")
            
            # Add mask geometries to layer
            for _, row in mask.iterrows():
                geom = row.geometry
                
                # Create OGR feature
                feature = ogr.Feature(mem_layer.GetLayerDefn())
                
                # Convert Shapely geometry to OGR geometry
                ogr_geom = ogr.CreateGeometryFromWkt(geom.wkt)
                if ogr_geom is None:
                    continue
                
                feature.SetGeometry(ogr_geom)
                
                # Add feature to layer
                result = mem_layer.CreateFeature(feature)
                if result != ogr.OGRERR_NONE:
                    continue
                
                feature = None  # Clean up feature
                ogr_geom = None  # Clean up geometry

            # Use gdal.Warp to crop and mask the raster
            # Ensure output directory exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # Determine GDAL driver based on output file extension
            driver_name = self._get_gdal_driver_from_path(out_path)
            
            # Configure warp options
            warp_options = gdal.WarpOptions(
                format=driver_name,
                cutlineDSName=mem_ds,  # Use memory dataset as cutline
                cutlineLayer='mask',
                cropToCutline=True,
                creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
            )
            
            # Perform the warp operation
            result_ds = gdal.Warp(out_path, raster_path, options=warp_options)
            if result_ds is None:
                raise RuntimeError(f"Failed to mask raster: {raster_path}.{self._get_gdal_error_message()}")
            
            result_ds = None  # Close result dataset
            
        finally:
            mem_layer = None
            mem_ds = None

        return out_path

    def _invert_dtm(self, dtm_path: str, output_path: str, feedback=None) -> str:
        """
        Create an inverted DTM (multiply by -1) to extract ridges.

        Args:
            dtm_path (str): Path to input DTM raster (supports all GDAL formats in gdal_driver_mapping).
            output_path (str): Path to output inverted DTM raster.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting.

        Returns:
            str: Path to inverted DTM raster.
        """
        if self.wbt is None:
            raise RuntimeError("WhiteboxTools not initialized. Check WhiteboxTools configuration: QGIS settings -> Options -> Processing -> Provider -> WhiteboxTools -> WhiteboxTools executable.")

        try:
            ret = self._execute_wbt(
                'multiply',
                feedback=feedback,
                report_progress=False,  # Don't override main progress bar
                input1=dtm_path,
                input2=-1.0,
                output=output_path
            )
            
            if ret != 0 or not os.path.exists(output_path):
                raise RuntimeError(f"DTM inversion failed: WhiteboxTools returned {ret}, output not found at {output_path}")
        except Exception as e:
            # Check if cancellation was the cause
            if feedback and feedback.isCanceled():
                feedback.reportError("Process cancelled by user during DTM inversion.")
                raise RuntimeError('Process cancelled by user.')
            raise RuntimeError(f"DTM inversion failed: {e}")

        return output_path

    def _log_raster(
        self,
        input_raster: str,
        output_path: str,
        overwrite: bool = True,
        val_band: int = 1
            ) -> str:
        """
        Computes the natural logarithm of a specified band in a raster,
        and either overwrites it or appends the result as a new band.

        Args:
            input_raster (str): Path to input raster (supports all GDAL formats in gdal_driver_mapping).
            output_path (str): Path to output raster (format determined by file extension).
            overwrite (bool): If True, replaces the selected band with log values.
                    If False, appends the log values as a new band.
            val_band (int): 1-based index of the band to compute the logarithm from.

        Returns:
            str: Path to the output raster.
        """
        # Read input raster using GDAL
        input_ds = gdal.Open(input_raster, gdal.GA_ReadOnly)
        if input_ds is None:
            raise RuntimeError(f"Cannot open input raster: {input_raster}.{self._get_gdal_error_message()}")
            
        try:
            # Get raster information
            width = input_ds.RasterXSize
            height = input_ds.RasterYSize
            band_count = input_ds.RasterCount
            geotransform = input_ds.GetGeoTransform()
            projection = input_ds.GetProjection()
            srs = input_ds.GetSpatialRef()
            
            # Validate band index
            if not (1 <= val_band <= band_count):
                raise ValueError(f"val_band={val_band} is out of range. Input raster has {band_count} band(s).")
            
            # Get the source band and its properties
            src_band = input_ds.GetRasterBand(val_band)
            nodata = src_band.GetNoDataValue()
            if nodata is None:
                nodata = self.nodata
            
            # Read the band data
            data = src_band.ReadAsArray().astype(np.float32)
            
            # Compute log(x) only for valid values > 0
            log_data = np.where(data > 0, np.log(data), nodata)
            
        finally:
            input_ds = None  # Close input dataset

        # Determine GDAL driver based on output file extension
        driver_name = self._get_gdal_driver_from_path(output_path)
        
        # Create output raster using GDAL with best practices
        driver = gdal.GetDriverByName(driver_name)
        if driver is None:
            raise RuntimeError(f"{driver_name} driver not available.{self._get_gdal_error_message()}")
            
        creation_options = [
            'COMPRESS=LZW',
            'TILED=YES',
            'BIGTIFF=IF_SAFER'
        ]
        
        if overwrite:
            # Create single-band output raster
            out_ds = driver.Create(output_path, width, height, 1, gdal.GDT_Float32, 
                                  options=creation_options)
        else:
            # Create multi-band output raster (original bands + log band)
            new_band_count = band_count + 1
            out_ds = driver.Create(output_path, width, height, new_band_count, gdal.GDT_Float32, 
                                  options=creation_options)
        
        if out_ds is None:
            raise RuntimeError(f"Failed to create output raster: {output_path}.{self._get_gdal_error_message()}")
            
        try:
            out_ds.SetGeoTransform(geotransform)
            
            # Set complete spatial reference system
            if srs is not None:
                out_ds.SetSpatialRef(srs)
            elif projection:
                # Fallback to projection string if SRS not available
                out_ds.SetProjection(projection)
            
            if overwrite:
                # Write only the log-transformed band
                out_band = out_ds.GetRasterBand(1)
                out_band.SetNoDataValue(nodata)
                out_band.WriteArray(log_data)
            else:
                # Re-open input raster to copy all original bands
                input_ds = gdal.Open(input_raster, gdal.GA_ReadOnly)
                if input_ds is None:
                    raise RuntimeError(f"Cannot re-open input raster: {input_raster}.{self._get_gdal_error_message()}")
                    
                try:
                    # Copy all original bands
                    for i in range(band_count):
                        src_band = input_ds.GetRasterBand(i + 1)
                        out_band = out_ds.GetRasterBand(i + 1)
                        
                        # Copy band data and properties
                        band_data = src_band.ReadAsArray().astype(np.float32)
                        out_band.WriteArray(band_data)
                        out_band.SetNoDataValue(src_band.GetNoDataValue())
                    
                    # Write the new log band
                    log_band = out_ds.GetRasterBand(new_band_count)
                    log_band.SetNoDataValue(nodata)
                    log_band.WriteArray(log_data)
                    
                finally:
                    input_ds = None  # Close input dataset again
                    
        finally:
            out_ds = None  # Close output dataset

        return output_path

    # Alternatively define constant value of absolute elevation at masked cells or input raster contains absolute elevation values
    # Not tested yet
    def _modify_dtm_with_mask(
        self,
        dtm_path: str,
        mask: np.ndarray,
        elevation_add: float,
        output_path: str
        ) -> str:
        """
        Modify DTM by adding elevation to masked cells using GDAL.

        Args:
            dtm_path (str): Path to input DTM raster (supports all GDAL formats in gdal_driver_mapping).
            mask (np.ndarray): Binary mask where elevation should be modified.
            elevation_add (float): Value to add to masked cells.
            output_path (str): Path to save modified DTM raster (format determined by file extension).

        Returns:
            str: Path to the modified DTM raster.
        """
        # Read input raster using GDAL
        input_ds = gdal.Open(dtm_path, gdal.GA_ReadOnly)
        if input_ds is None:
            raise RuntimeError(f"Cannot open input DTM raster: {dtm_path}.{self._get_gdal_error_message()}")
            
        try:
            # Get raster information
            width = input_ds.RasterXSize
            height = input_ds.RasterYSize
            geotransform = input_ds.GetGeoTransform()
            projection = input_ds.GetProjection()
            srs = input_ds.GetSpatialRef()
            
            # Read the first band data
            src_band = input_ds.GetRasterBand(1)
            nodata = src_band.GetNoDataValue()
            if nodata is None:
                nodata = self.nodata
                
            data = src_band.ReadAsArray().astype(np.float32)
            
        finally:
            input_ds = None  # Close input dataset

        # Modify data using the mask
        modified = data.copy()
        modified[mask == 1] += elevation_add

        # Determine GDAL driver based on output file extension
        driver_name = self._get_gdal_driver_from_path(output_path)
        
        # Create output raster using GDAL with best practices
        driver = gdal.GetDriverByName(driver_name)
        if driver is None:
            raise RuntimeError(f"{driver_name} driver not available.{self._get_gdal_error_message()}")
            
        creation_options = [
            'COMPRESS=LZW',
            'TILED=YES',
            'BIGTIFF=IF_SAFER'
        ]
        
        out_ds = driver.Create(output_path, width, height, 1, gdal.GDT_Float32, 
                              options=creation_options)
        if out_ds is None:
            raise RuntimeError(f"Failed to create output raster: {output_path}.{self._get_gdal_error_message()}")
            
        try:
            out_ds.SetGeoTransform(geotransform)
            
            # Set complete spatial reference system
            if srs is not None:
                out_ds.SetSpatialRef(srs)
            elif projection:
                # Fallback to projection string if SRS not available
                out_ds.SetProjection(projection)
            
            # Write the modified data
            out_band = out_ds.GetRasterBand(1)
            out_band.SetNoDataValue(nodata)
            out_band.WriteArray(modified)
            
        finally:
            out_ds = None  # Close output dataset

        return output_path

    def _raster_to_linestring_wbt(self, raster_path: str, snap_to_start_point: Point = None, snap_to_endpoint: Point = None, output_vector_path: str = None, feedback=None) -> LineString:
        """
        Uses WhiteboxTools to vectorize a raster and return a merged LineString or MultiLineString.
        Optionally snaps the endpoint to the center of a destination cell.

        Args:
            raster_path (str): Path to input raster where 1-valued pixels form your keyline (supports all GDAL formats in gdal_driver_mapping).
            snap_to_start_point (Point, optional): Point to snap the start of the line to.
            snap_to_endpoint (Point, optional): Point to snap the endpoint of the line to.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting.

        Returns:
            LineString or MultiLineString, or None if empty.
        """
        if self.wbt is None:
            raise RuntimeError("WhiteboxTools not initialized. Check WhiteboxTools configuration: QGIS settings -> Options -> Processing -> Provider -> WhiteboxTools -> WhiteboxTools executable.")

        if not output_vector_path:
            base, _ = os.path.splitext(raster_path)
            output_vector_path = base + ".shp"
            
        try:
            ret = self._execute_wbt(
                'raster_to_vector_lines',
                feedback=feedback,
                report_progress=False,  # Don't override main progress bar
                i=raster_path,
                output=output_vector_path
            )
            
            if ret != 0 or not os.path.exists(output_vector_path):
                raise RuntimeError(f"Raster to vector lines failed: WhiteboxTools returned {ret}, output not found at {output_vector_path}")
        except Exception as e:
            # Check if cancellation was the cause
            if feedback and feedback.isCanceled():
                feedback.reportError("Process cancelled by user during raster to vector conversion.")
                raise RuntimeError('Process cancelled by user.')
            raise RuntimeError(f"Raster to vector lines failed: {e}")

        gdf = gpd.read_file(output_vector_path)
        
        if gdf.empty:
            warnings.warn(f"Warning: No vector features found in {output_vector_path}.")
            return None

        all_geometries = list(gdf.geometry) 
        # First linemerge on all geometries
        merged_geom = linemerge(all_geometries)

        # Extract line geometries and filter valid ones
        line_geometries = []
        if isinstance(merged_geom, LineString):
            line_geometries.append(merged_geom)
        elif isinstance(merged_geom, MultiLineString):
            line_geometries.extend(list(merged_geom.geoms))
        else:
            # Fallback: process individual geometries if linemerge didn't work
            for geom in gdf.geometry:
                if isinstance(geom, LineString):
                    line_geometries.append(geom)
                elif isinstance(geom, MultiLineString):
                    # Add individual parts of MultiLineString
                    line_geometries.extend(list(geom.geoms))
        
        if not line_geometries:
            warnings.warn("Warning: No valid LineString geometries found after vectorization.")
            return None
        
        # Merge to one single LineString
        if len(line_geometries) == 1:
            # If only one line, no need to merge
            single_part_line = line_geometries[0]
        else:
            # Merge lines using distance-based approach
            single_part_line = TopoDrainCore._merge_lines_by_distance(line_geometries)
            
        # If single_part_line is empty, return None
        if not single_part_line:
            warnings.warn("Warning: No valid line segments found after vectorization.")
            return None
        
        # 5) Snap start to destination cell center if requested, and ensure correct line direction
        if snap_to_start_point:
            single_part_line = TopoDrainCore._snap_line_to_point(single_part_line, snap_to_start_point, "start")

        # 6) Snap endpoint to destination cell center if requested
        if snap_to_endpoint:
            single_part_line = TopoDrainCore._snap_line_to_point(single_part_line, snap_to_endpoint, "end")
        
        return single_part_line

    ## TopoDrainCore functions
    @staticmethod
    def _find_inflection_candidates(curvature: np.ndarray, window: int) -> list:
        """
        Detect inflection points where the curvature changes from convex to concave,
        using a moving average window. If none found, return point of strongest transition.

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
            # Look for convex → concave transitions (positive to negative curvature)
            if before_avg > 0 and after_avg < 0:
                strength = abs(before_avg) + abs(after_avg)
                candidates.append((i, strength))

        # Fallback: if no clear convex → concave transitions found
        if not candidates:
            warnings.warn("Warning: No clear convex → concave inflection points found. Using strongest transition as fallback.")
            best_strength = -np.inf
            best_index = None
            for i in range(window, len(curvature) - window):
                before_avg = np.mean(curvature[i - window:i])
                after_avg = np.mean(curvature[i + 1:i + 1 + window])
                # Look for any significant transition (prioritize convex → concave)
                strength = before_avg - after_avg  # Higher when going from positive to negative
                if strength > best_strength:
                    best_strength = strength
                    best_index = i
            candidates = [(best_index, best_strength)]

        # Sort by strength (strongest transitions first)
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        return sorted_candidates

    ## Core functions
    def extract_valleys(
        self,
        dtm_path: str,
        filled_output_path: str = None,
        fdir_output_path: str = None,
        facc_output_path: str = None,
        facc_log_output_path: str = None,
        streams_output_path: str  = None,
        accumulation_threshold: int = 1000,
        dist_facc: float = 50,
        postfix: str = None,
        feedback=None
    ) -> gpd.GeoDataFrame:
        """
        Extract valley lines using WhiteboxTools. 

        Args:
            dtm_path (str):
                Path to input DTM raster (supports all GDAL formats in gdal_driver_mapping).
            filled_output_path (str, optional):
                Path to save the depression-filled DTM raster (format determined by file extension).
            fdir_output_path (str, optional):
                Path to save the flow-direction raster (format determined by file extension).
            facc_output_path (str, optional):
                Path to save the flow-accumulation raster (format determined by file extension).
            facc_log_output_path (str, optional):
                Path to save the log-scaled accumulation raster (format determined by file extension).
            streams_output_path (str, optional):
                Path to save the extracted stream raster (format determined by file extension).
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
        if self.wbt is None:
            raise RuntimeError("WhiteboxTools not initialized. Check WhiteboxTools configuration: QGIS settings -> Options -> Processing -> Provider -> WhiteboxTools -> WhiteboxTools executable.")

        if feedback:
            feedback.pushInfo("[ExtractValleys] Starting valley extraction process...")
            feedback.pushInfo("[ExtractValleys] *Detailed WhiteboxTools output can be viewed in the Python Console")
            feedback.setProgress(0)
        else:
            print("[ExtractValleys] Starting valley extraction process...")

        # Build defaults for everything
        if not postfix:
            d = lambda name: os.path.join(self.temp_directory, name)
            defaults = {
                "filled":         d("filled.tif"),
                "fdir":           d("fdir.tif"),
                "streams":        d("streams.tif"),
                "streams_vec":    d("streams.shp"),
                "streams_linked": d("streams_linked.shp"),
                "facc":           d("facc.tif"),
                "facc_log":       d("facc_log.tif"),
                "network":        d("stream_network.shp"),
            }
        else:
            d = lambda base: os.path.join(self.temp_directory, f"{base}_{postfix}")
            defaults = {
                "filled":         d("filled") + ".tif",
                "fdir":           d("fdir") + ".tif",
                "streams":        d("streams") + ".tif",
                "streams_vec":    d("streams") + ".shp",
                "streams_linked": d("streams_linked") + ".shp",
                "facc":           d("facc") + ".tif",
                "facc_log":       d("facc_log") + ".tif",
                "network":        d("stream_network") + ".shp",
            }

        print(f"[ExtractValleys] Define paths for outputs")
        # Only these four can be overridden
        filled_output_path   = filled_output_path   or defaults["filled"]
        fdir_output_path     = fdir_output_path     or defaults["fdir"]
        facc_output_path     = facc_output_path     or defaults["facc"]
        facc_log_output_path = facc_log_output_path or defaults["facc_log"]
        streams_output_path  = streams_output_path  or defaults["streams"]

        # intermediate paths always use defaults:
        streams_vec_output_path    = defaults["streams_vec"]
        streams_linked_output_path = defaults["streams_linked"]
        stream_network_output_path = defaults["network"]

        try:
            if feedback:
                feedback.pushInfo(f"[ExtractValleys] Step 1/7: Filling depressions → {filled_output_path}")
                feedback.setProgress(10)
                # Check if user has canceled process
                if feedback.isCanceled():
                    feedback.reportError("Process cancelled by user after step 1 initialization.")
                    raise RuntimeError('Process cancelled by user.')
            else:
                print(f"[ExtractValleys] Step 1/7: Filling depressions → {filled_output_path}")
            try:
                ret = self._execute_wbt(
                    'breach_depressions_least_cost',
                    feedback=feedback,  # Pass feedback to enable cancellation during execution
                    report_progress=False,  # Don't override main progress bar
                    dem=dtm_path,
                    output=filled_output_path,
                    dist=int(dist_facc),
                    fill=True,
                    min_dist=True
                )
                if ret != 0 or not os.path.exists(filled_output_path):
                    raise RuntimeError(f"[ExtractValleys] Depression filling failed: WhiteboxTools returned {ret}, output not found at {filled_output_path}")
            except Exception as e:
                # Check if cancellation was the cause
                if feedback and feedback.isCanceled():
                    feedback.reportError("Process cancelled by user during depression filling.")
                    raise RuntimeError('Process cancelled by user.')
                raise RuntimeError(f"[ExtractValleys] Depression filling failed: {e}")

            if feedback:
                feedback.pushInfo(f"[ExtractValleys] Step 2/7: Computing flow direction → {fdir_output_path}")
                feedback.setProgress(25)
            else:
                print(f"[ExtractValleys] Step 2/7: Computing flow direction → {fdir_output_path}")
            try:
                ret = self._execute_wbt(
                    'd8_pointer',
                    feedback=feedback,  # Pass feedback to enable cancellation during execution
                    report_progress=False,  # Don't override main progress bar
                    dem=filled_output_path,
                    output=fdir_output_path
                )
                if ret != 0 or not os.path.exists(fdir_output_path):
                    raise RuntimeError(f"[ExtractValleys] Flow direction failed: WhiteboxTools returned {ret}, output not found at {fdir_output_path}")
            except Exception as e:
                # Check if cancellation was the cause
                if feedback and feedback.isCanceled():
                    feedback.reportError("Process cancelled by user during flow direction computation.")
                    raise RuntimeError('Process cancelled by user.')
                raise RuntimeError(f"[ExtractValleys] Flow direction failed: {e}")

            if feedback:
                feedback.pushInfo(f"[ExtractValleys] Step 3/7: Computing flow accumulation → {facc_output_path}")
                feedback.setProgress(40)
            else:
                print(f"[ExtractValleys] Step 3/7: Computing flow accumulation → {facc_output_path}")
            try:
                ret = self._execute_wbt(
                    'd8_flow_accumulation',
                    feedback=feedback,  # Pass feedback to enable cancellation during execution
                    report_progress=False,  # Don't override main progress bar
                    i=filled_output_path,
                    output=facc_output_path,
                    out_type="specific contributing area"
                )
                if ret != 0 or not os.path.exists(facc_output_path):
                    raise RuntimeError(f"[ExtractValleys] Flow accumulation failed: WhiteboxTools returned {ret}, output not found at {facc_output_path}")
            except Exception as e:
                # Check if cancellation was the cause
                if feedback and feedback.isCanceled():
                    feedback.reportError("Process cancelled by user during flow accumulation computation.")
                    raise RuntimeError('Process cancelled by user.')
                raise RuntimeError(f"[ExtractValleys] Flow accumulation failed: {e}")

            if feedback:
                feedback.pushInfo(f"[ExtractValleys] Step 4/7: Creating log-scaled accumulation → {facc_log_output_path}")
                feedback.setProgress(55)
            else:
                print(f"[ExtractValleys] Step 4/7: Creating log-scaled accumulation → {facc_log_output_path}")
            try:
                self._log_raster(input_raster=facc_output_path, output_path=facc_log_output_path)
                if not os.path.exists(facc_log_output_path):
                    raise RuntimeError(f"[ExtractValleys] Log-scaled accumulation output not found at {facc_log_output_path}")
            except Exception as e:
                msg = f"[ExtractValleys] Warning: Log-scaled accumulation failed: {e}"
                if feedback:
                    feedback.pushWarning(msg)
                else:
                    warnings.warn(msg)

            if feedback:
                feedback.pushInfo(f"[ExtractValleys] Step 5/7: Extracting streams (threshold={accumulation_threshold})")
                feedback.setProgress(70)
            else:
                print(f"[ExtractValleys] Step 5/7: Extracting streams (threshold={accumulation_threshold})")
            try:
                ret = self._execute_wbt(
                    'extract_streams',
                    feedback=feedback,  # Pass feedback to enable cancellation during execution
                    report_progress=False,  # Don't override main progress bar
                    flow_accum=facc_output_path,
                    output=streams_output_path,
                    threshold=accumulation_threshold
                )
                if ret != 0 or not os.path.exists(streams_output_path):
                    raise RuntimeError(f"[ExtractValleys] Stream extraction failed: WhiteboxTools returned {ret}, output not found at {streams_output_path}")
            except Exception as e:
                # Check if cancellation was the cause
                if feedback and feedback.isCanceled():
                    feedback.reportError("Process cancelled by user during stream extraction.")
                    raise RuntimeError('Process cancelled by user.')
                raise RuntimeError(f"[ExtractValleys] Stream extraction failed: {e}")

            if feedback:
                feedback.pushInfo("[ExtractValleys] Step 6/7: Vectorizing streams")
                feedback.setProgress(80)
            else:
                print("[ExtractValleys] Step 6/7: Vectorizing streams")
            try:
                ret = self._execute_wbt(
                    'raster_streams_to_vector',
                    feedback=feedback,  # Pass feedback to enable cancellation during execution
                    report_progress=False,  # Don't override main progress bar
                    streams=streams_output_path,
                    d8_pntr=fdir_output_path,
                    output=streams_vec_output_path
                )
                if ret != 0 or not os.path.exists(streams_vec_output_path):
                    raise RuntimeError(f"[ExtractValleys] Vectorizing streams failed: WhiteboxTools returned {ret}, output not found at {streams_vec_output_path}")
            except Exception as e:
                # Check if cancellation was the cause
                if feedback and feedback.isCanceled():
                    feedback.reportError("Process cancelled by user during stream vectorization.")
                    raise RuntimeError('Process cancelled by user.')
                raise RuntimeError(f"[ExtractValleys] Vectorizing streams failed: {e}")

            streams_vec_id = streams_linked_output_path.replace(".shp", "_id.tif")
            try:
                if feedback:
                    feedback.pushInfo("[ExtractValleys] Step 7/7: Processing network topology - Identifying stream links")
                    feedback.setProgress(85)
                else:
                    print("[ExtractValleys] Step 7/7: Processing network topology - Identifying stream links")
                ret = self._execute_wbt(
                    'stream_link_identifier',
                    feedback=feedback,  # Pass feedback to enable cancellation during execution
                    report_progress=False,  # Don't override main progress bar
                    d8_pntr=fdir_output_path,
                    streams=streams_output_path,
                    output=streams_vec_id
                )
                if ret != 0 or not os.path.exists(streams_vec_id):
                    raise RuntimeError(f"[ExtractValleys] Stream link identifier failed: WhiteboxTools returned {ret}, output not found at {streams_vec_id}")
            except Exception as e:
                # Check if cancellation was the cause
                if feedback and feedback.isCanceled():
                    feedback.reportError("Process cancelled by user during stream link identification.")
                    raise RuntimeError('Process cancelled by user.')
                raise RuntimeError(f"[ExtractValleys] Stream link identifier failed: {e}")

            try:
                if feedback:
                    feedback.pushInfo("[ExtractValleys] Converting linked streams to vectors")
                    feedback.setProgress(90)
                else:
                    print("[ExtractValleys] Converting linked streams to vectors")
                ret = self._execute_wbt(
                    'raster_streams_to_vector',
                    feedback=feedback,  # Pass feedback to enable cancellation during execution
                    report_progress=False,  # Don't override main progress bar
                    streams=streams_vec_id,
                    d8_pntr=fdir_output_path,
                    output=streams_linked_output_path
                )
                if ret != 0 or not os.path.exists(streams_linked_output_path):
                    raise RuntimeError(f"[ExtractValleys] Converting linked streams failed: WhiteboxTools returned {ret}, output not found at {streams_linked_output_path}")
            except Exception as e:
                # Check if cancellation was the cause
                if feedback and feedback.isCanceled():
                    feedback.reportError("Process cancelled by user during linked stream conversion.")
                    raise RuntimeError('Process cancelled by user.')
                raise RuntimeError(f"[ExtractValleys] Converting linked streams failed: {e}")

            try:
                if feedback:
                    feedback.pushInfo("[ExtractValleys] Performing final network analysis")
                    feedback.setProgress(95)
                else:
                    print("[ExtractValleys] Performing final network analysis")
                ret = self._execute_wbt(
                    'VectorStreamNetworkAnalysis',
                    feedback=feedback,  # Pass feedback to enable cancellation during execution
                    report_progress=False,  # Don't override main progress bar
                    streams=streams_linked_output_path,
                    dem=filled_output_path,
                    output=stream_network_output_path
                    )
                if ret != 0 or not os.path.exists(stream_network_output_path):
                    raise RuntimeError(f"[ExtractValleys] Network analysis failed: WhiteboxTools returned {ret}, output not found at {stream_network_output_path}")
            except Exception as e:
                # Check if cancellation was the cause
                if feedback and feedback.isCanceled():
                    feedback.reportError("Process cancelled by user during network analysis.")
                    raise RuntimeError('Process cancelled by user.')
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

            # Always create a reliable LINK_ID field as primary identifier
            # This addresses Windows case-sensitivity and temporary layer issues
            if 'FID' in gdf.columns or 'fid' in gdf.columns:
                fid_col = 'FID' if 'FID' in gdf.columns else 'fid'
                gdf['LINK_ID'] = gdf[fid_col]
            else:
                gdf['LINK_ID'] = range(1, len(gdf) + 1)
            
            if feedback:
                feedback.pushInfo("[ExtractValleys] Created reliable 'LINK_ID' field for cross-platform compatibility")
                feedback.pushInfo(f"[ExtractValleys] Completed: {len(gdf)} valley features extracted successfully!")
                feedback.setProgress(100)
            else:
                print("[ExtractValleys] Created reliable 'LINK_ID' field for cross-platform compatibility")
                print(f"[ExtractValleys] Completed: {len(gdf)} valley features extracted successfully!")
            
            return gdf

        except Exception as e:
            raise RuntimeError(f"[ExtractValleys] Failed to extract valleys: {e}")
        

    def extract_ridges(
        self,
        dtm_path: str,
        inverted_filled_output_path: str = None,
        inverted_fdir_output_path: str = None,
        inverted_facc_output_path: str = None,
        inverted_facc_log_output_path: str = None,
        inverted_streams_output_path: str = None,
        accumulation_threshold: int = 1000,
        dist_facc: float = 50,
        postfix: str = "inverted",
        feedback=None
    ) -> gpd.GeoDataFrame:
        """
        Extract ridge lines (watershed divides) from a DTM by inverting the terrain
        and running the valley‐extraction workflow.

        Args:
            dtm_path (str):
                Path to input DTM raster (supports all GDAL formats in gdal_driver_mapping).
            inverted_filled_output_path (str, optional):
                Where to save the inverted‐DTM’s filled DEM (GeoTIFF, “.tif”).
            inverted_fdir_output_path (str, optional):
                Where to save the inverted‐DTM’s flow‐direction raster (GeoTIFF).
            inverted_facc_output_path (str, optional):
                Where to save the inverted‐DTM’s flow‐accumulation raster (GeoTIFF).
            inverted_facc_log_output_path (str, optional):
                Where to save the inverted‐DTM’s log‐scaled accumulation raster (GeoTIFF).
            inverted_streams_output_path (str, optional):
                Where to save the inverted‐DTM’s extracted streams (GeoTIFF).   
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
        if self.wbt is None:
            raise RuntimeError("WhiteboxTools not initialized. Check WhiteboxTools configuration: QGIS settings -> Options -> Processing -> Provider -> WhiteboxTools -> WhiteboxTools executable.")

        if feedback:
            feedback.pushInfo("[ExtractRidges] Starting ridge extraction process...")
            feedback.setProgress(0)
        else:
            print("[ExtractRidges] Starting ridge extraction process...")

        # 1) Invert the DTM
        if feedback:
            feedback.pushInfo("[ExtractRidges] Inverting DTM for ridge extraction...")
        else:
            print("[ExtractRidges] Inverting DTM for ridge extraction...")
        
        inverted_dtm = os.path.join(self.temp_directory, f"inverted_dtm_{postfix}.tif")
        inverted_dtm = self._invert_dtm(dtm_path, inverted_dtm, feedback=feedback)  # Remove feedback to prevent multiple progress bars
        
        if feedback:
            feedback.pushInfo(f"[ExtractRidges] DTM inversion complete: {inverted_dtm}")
            feedback.setProgress(5)
        else:
            print(f"[ExtractRidges] DTM inversion complete: {inverted_dtm}")

        # 2) Compute defaults for the four inverted outputs
        #    We leverage extract_valleys’ own default logic by passing these params through.
        if feedback:
            feedback.pushInfo("[ExtractRidges] Extracting ridges from inverted DTM (using extract_valleys function)...")
        else:
            print("[ExtractRidges] Extracting ridges from inverted DTM (using extract_valleys function)...")
        # If the user did not supply, leave as None—extract_valleys will pick its defaults (which include postfix).
        inv_filled = inverted_filled_output_path
        inv_fdir   = inverted_fdir_output_path
        inv_facc   = inverted_facc_output_path
        inv_facc_log = inverted_facc_log_output_path
        inv_streams = inverted_streams_output_path

        # 3) Call extract_valleys on the inverted DTM (this will handle its own progress reporting)
        ridges_gdf = self.extract_valleys(
            dtm_path=inverted_dtm,
            filled_output_path=inv_filled,
            fdir_output_path=inv_fdir,
            facc_output_path=inv_facc,
            facc_log_output_path=inv_facc_log,
            streams_output_path=inv_streams,
            accumulation_threshold=accumulation_threshold,
            dist_facc=dist_facc,
            postfix=postfix,
            feedback=feedback 
        )

        if feedback:
            feedback.pushInfo(f"[ExtractRidges] Ridge extraction completed successfully: {len(ridges_gdf)} ridge features extracted!")
        else:
            print(f"[ExtractRidges] Ridge extraction completed successfully: {len(ridges_gdf)} ridge features extracted!")

        return ridges_gdf


    def extract_main_valleys(
        self,
        valley_lines: gpd.GeoDataFrame,
        facc_path: str,
        perimeter: gpd.GeoDataFrame = None,
        nr_main: int = 2,
        clip_to_perimeter: bool = True,
        feedback=None
    ) -> gpd.GeoDataFrame:
        """
        Identify and merge main valley lines based on the highest flow accumulation,
        using only points uniquely associated with one TRIB_ID (to avoid confluent points).

        Args:
            valley_lines (GeoDataFrame): Valley line network with 'LINK_ID', 'TRIB_ID', and 'DS_LINK_ID' attributes.
            facc_path (str): Path to the flow accumulation raster.
            perimeter (GeoDataFrame, optional): Polygon defining the area boundary. If None, uses valley_lines extent.
            nr_main (int): Number of main valleys to select.
            clip_to_perimeter (bool): If True, clips output to boundary polygon of perimeter.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting/logging.

        Returns:
            GeoDataFrame: Main valley lines with TRIB_ID, LINK_ID, RANK, and POLYGON_ID attributes.
        """
        if feedback:
            feedback.pushInfo("[ExtractMainValleys] Starting main valley extraction...")
            feedback.setProgress(0)
            if feedback.isCanceled():
                feedback.reportError('Process cancelled by user at initialization.')
                raise RuntimeError('Process cancelled by user.')
        else:
            print("[ExtractMainValleys] Starting main valley extraction...")

        if valley_lines.empty:
            raise RuntimeError("[ExtractMainValleys] Input valley_lines is empty")

        # Ensure LINK_ID field is present (required)
        if 'LINK_ID' not in valley_lines.columns:
            raise RuntimeError("[ExtractMainValleys] Input valley_lines must have a 'LINK_ID' attribute. Please use valley lines generated by the Create Valleys algorithm.")
        
        # Ensure TRIB_ID field is present in valley_lines
        if 'TRIB_ID' not in valley_lines.columns:
            # Create TRIB_ID field using LINK_ID values as fallback
            valley_lines = valley_lines.copy()  # Avoid modifying original
            valley_lines['TRIB_ID'] = valley_lines['LINK_ID']
            if feedback:
                feedback.pushWarning("[ExtractMainValleys] Warning: 'TRIB_ID' column not found in valley_lines, using 'LINK_ID' values as fallback")
            else:
                warnings.warn("[ExtractMainValleys] Warning: 'TRIB_ID' column not found in valley_lines, using 'LINK_ID' values as fallback")

        # Ensure DS_LINK_ID field is present in valley_lines (optional field, can be null)
        if 'DS_LINK_ID' not in valley_lines.columns:
            # Create DS_LINK_ID field with null values (not critical for main valley extraction)
            valley_lines = valley_lines.copy()  # Avoid modifying original
            valley_lines['DS_LINK_ID'] = None
            if feedback:
                feedback.pushWarning("[ExtractMainValleys] Warning: 'DS_LINK_ID' column not found in valley_lines, created with null values as fallback")
            else:
                warnings.warn("[ExtractMainValleys] Warning: 'DS_LINK_ID' column not found in valley_lines, created with null values as fallback")

        # Create perimeter from valley_lines extent if not provided
        if perimeter is None:
            if feedback:
                feedback.pushInfo("[ExtractMainValleys] No perimeter provided, using valley lines extent...")
            else:
                print("[ExtractMainValleys] No perimeter provided, using valley lines extent...")
            
            # Get the bounding box of valley_lines and create a polygon
            bounds = valley_lines.total_bounds  # [minx, miny, maxx, maxy]
            bbox_polygon = Polygon([
                (bounds[0], bounds[1]),  # bottom-left
                (bounds[2], bounds[1]),  # bottom-right
                (bounds[2], bounds[3]),  # top-right
                (bounds[0], bounds[3]),  # top-left
                (bounds[0], bounds[1])   # close polygon
            ])
            perimeter = gpd.GeoDataFrame([{'geometry': bbox_polygon}], crs=valley_lines.crs)

        if feedback:
            feedback.pushInfo("[ExtractMainValleys] Reading flow accumulation raster...")
            feedback.setProgress(10)
            if feedback.isCanceled():
                feedback.reportError('Process cancelled by user after reading flow accumulation raster.')
                raise RuntimeError('Process cancelled by user.')
        else:
            print("[ExtractMainValleys] Reading flow accumulation raster...")
        
        # Read flow accumulation raster using GDAL
        facc_ds = gdal.Open(facc_path, gdal.GA_ReadOnly)
        if facc_ds is None:
            raise RuntimeError(f"Cannot open flow accumulation raster: {facc_path}.{self._get_gdal_error_message()}")
            
        try:
            facc_band = facc_ds.GetRasterBand(1)
            facc = facc_band.ReadAsArray()
            if facc is None:
                raise RuntimeError(f"Failed to read flow accumulation data from: {facc_path}.{self._get_gdal_error_message()}")
            
            # Get geotransform for coordinate conversion
            geotransform = facc_ds.GetGeoTransform()
            if geotransform is None:
                raise RuntimeError(f"Failed to get geotransform from: {facc_path}.{self._get_gdal_error_message()}")
            
            res = abs(geotransform[1])  # pixel width
            
        finally:
            facc_ds = None  # Close dataset

        # Process each polygon in the perimeter separately
        all_merged_records = []
        global_fid_counter = 1
        
        for poly_idx, poly_row in perimeter.iterrows():
            single_polygon = gpd.GeoDataFrame([poly_row], crs=perimeter.crs)
            
            # Calculate progress based on polygon processing (20-80% range)
            polygon_progress = 20 + int((poly_idx / len(perimeter)) * 60)
            
            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Processing polygon {poly_idx + 1}/{len(perimeter)}...")
                feedback.setProgress(polygon_progress)
                if feedback.isCanceled():
                    feedback.reportError(f'Process cancelled by user while processing polygon {poly_idx + 1}.')
                    raise RuntimeError('Process cancelled by user.')
            else:
                print(f"[ExtractMainValleys] Processing polygon {poly_idx + 1}/{len(perimeter)}...")

            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Clipping valley lines to polygon {poly_idx + 1}...")
            else:
                print(f"[ExtractMainValleys] Clipping valley lines to polygon {poly_idx + 1}...")
            valley_clipped = gpd.overlay(valley_lines, single_polygon, how="intersection")
            
            if valley_clipped.empty:
                if feedback:
                    feedback.pushInfo(f"[ExtractMainValleys] No valley lines found in polygon {poly_idx + 1}, skipping...")
                else:
                    print(f"[ExtractMainValleys] No valley lines found in polygon {poly_idx + 1}, skipping...")
                continue

            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Rasterizing valley lines for polygon {poly_idx + 1}...")
            else:
                print(f"[ExtractMainValleys] Rasterizing valley lines for polygon {poly_idx + 1}...")
            valley_raster_path = os.path.join(self.temp_directory, f"valley_mask_poly_{poly_idx}.tif")
           
           # All valley lines are rasterized together into a single binary mask (1 = valley cell, 0 = background)
            valley_mask_path = self._vector_to_mask_raster(
                features=[valley_clipped],
                reference_raster_path=facc_path,
                output_path=valley_raster_path,
                unique_values=False,
                flatten_lines=False,
                buffer_lines=False
            )
            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Valley mask created at {valley_mask_path}")
            else:
                print(f"[ExtractMainValleys] Valley mask created at {valley_mask_path}")

            # Read the valley mask data from the saved raster file using GDAL
            valley_lines_ds = gdal.Open(valley_mask_path, gdal.GA_ReadOnly)
            if valley_lines_ds is None:
                raise RuntimeError(f"Cannot open valley mask raster: {valley_mask_path}.{self._get_gdal_error_message()}")
                
            try:
                valley_lines_band = valley_lines_ds.GetRasterBand(1)
                valley_mask = valley_lines_band.ReadAsArray()
                if valley_mask is None:
                    raise RuntimeError(f"Failed to read valley mask data from: {valley_mask_path}.{self._get_gdal_error_message()}")
                    
            finally:
                valley_lines_ds = None  # Close dataset

            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Extracting facc > 0 points for polygon {poly_idx + 1}...")
            else:
                print(f"[ExtractMainValleys] Extracting facc > 0 points for polygon {poly_idx + 1}...")
            mask = (valley_mask == 1) & (facc > 0)
            rows, cols = np.where(mask)
            if len(rows) == 0:
                if feedback:
                    feedback.pushInfo(f"[ExtractMainValleys] No valley cells with flow accumulation > 0 found in polygon {poly_idx + 1}, skipping...")
                else:
                    print(f"[ExtractMainValleys] No valley cells with flow accumulation > 0 found in polygon {poly_idx + 1}, skipping...")
                continue

            # Points are created at the center coordinates of the raster cells containing valley lines with facc > 0
            # Convert row,col indices to world coordinates using GDAL geotransform
            coords = self._pixel_indices_to_coords(rows, cols, geotransform)
            points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*zip(*coords)), crs=self.crs)
            points["facc"] = facc[rows, cols]

            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Performing spatial join for polygon {poly_idx + 1}...")
            else:
                print(f"[ExtractMainValleys] Performing spatial join for polygon {poly_idx + 1}...")
            
            # Ensure the required columns exist in valley_clipped (they should after validation above)
            join_columns = ["geometry"]
            for col in ["LINK_ID", "TRIB_ID", "DS_LINK_ID"]:
                if col in valley_clipped.columns:
                    join_columns.append(col)
            
            # Spatial Join with Original Vector Lines using buffered points to ensure all valley lines within raster cells are captured
            # Buffer points by half the cell resolution to catch all lines passing through the raster cell
            buffer_distance = res / 2.0  # Half cell size ensures we capture lines at cell edges
            
            points_buffered = points.copy()
            points_buffered.geometry = points.geometry.buffer(buffer_distance)
            
            points_joined = gpd.sjoin(
                points_buffered,
                valley_clipped[join_columns],
                how="inner"
            ).drop(columns="index_right")
            
            # Restore original point geometries for further processing
            points_joined.geometry = points.geometry[points_joined.index]

            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Filtering ambiguous facc points for polygon {poly_idx + 1}...")
            else:
                print(f"[ExtractMainValleys] Filtering ambiguous facc points for polygon {poly_idx + 1}...")
            
            # Removes any point that belongs to multiple TRIB_IDs. Prevents "Flow Accumulation Theft" at confluences.
            points_joined["geom_wkt"] = points_joined.geometry.apply(lambda geom: geom.wkt)
            geom_counts = points_joined.groupby("geom_wkt")["TRIB_ID"].nunique()
            valid_geoms = geom_counts[geom_counts == 1].index
            points_unique = points_joined[points_joined["geom_wkt"].isin(valid_geoms)].copy()

            if points_unique.empty:
                if feedback:
                    feedback.pushWarning(f"[ExtractMainValleys] Warning: No unique valley points found in polygon {poly_idx + 1}, skipping...")
                else:
                    warnings.warn(f"[ExtractMainValleys] Warning: No unique valley points found in polygon {poly_idx + 1}, skipping...")
                continue

            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Selecting top {nr_main} TRIB_IDs for polygon {poly_idx + 1}...")
            else:
                print(f"[ExtractMainValleys] Selecting top {nr_main} TRIB_IDs for polygon {poly_idx + 1}...")
            points_sorted = points_unique.sort_values("facc", ascending=False)
            points_top = points_sorted.drop_duplicates(subset="TRIB_ID").head(nr_main)

            if points_top.empty:
                if feedback:
                    feedback.pushWarning(f"[ExtractMainValleys] Warning: No main valley lines could be selected for polygon {poly_idx + 1}, skipping...")
                else:
                    warnings.warn(f"[ExtractMainValleys] Warning: No main valley lines could be selected for polygon {poly_idx + 1}, skipping...")
                continue

            selected_trib_ids = points_top["TRIB_ID"].unique()
            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Selected TRIB_IDs for polygon {poly_idx + 1}: {list(selected_trib_ids)}")
            else:
                print(f"[ExtractMainValleys] Selected TRIB_IDs for polygon {poly_idx + 1}: {list(selected_trib_ids)}")

            # Create ranking based on flow accumulation values (highest facc gets rank 1)
            # points_top is already sorted by facc descending since it comes from points_sorted
            trib_id_ranking = {}
            for rank, (_, row) in enumerate(points_top.iterrows(), 1):
                trib_id_ranking[row["TRIB_ID"]] = rank
            
            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] TRIB_ID rankings for polygon {poly_idx + 1}: {trib_id_ranking}")
            else:
                print(f"[ExtractMainValleys] TRIB_ID rankings for polygon {poly_idx + 1}: {trib_id_ranking}")

            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Merging valley line segments for polygon {poly_idx + 1}...")
            else:
                print(f"[ExtractMainValleys] Merging valley line segments for polygon {poly_idx + 1}...") # because maybe split by perimeter
            for trib_id in selected_trib_ids:
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
                        # Get the first matching line to copy attributes from
                        first_line = lines.iloc[0]
                        # Get the rank for this TRIB_ID
                        rank = trib_id_ranking.get(trib_id, 999)  # Default to 999 if not found
                        all_merged_records.append({
                            "geometry": merged_line,
                            "TRIB_ID": trib_id,
                            "LINK_ID": global_fid_counter,
                            "RANK": rank,  # Add ranking based on flow accumulation
                            "POLYGON_ID": poly_idx + 1,
                            # Copy other attributes if they exist
                            "DS_LINK_ID": first_line.get("DS_LINK_ID", None) if hasattr(first_line, 'get') else None
                        })
                        global_fid_counter += 1
                        if feedback:
                            feedback.pushInfo(f"[ExtractMainValleys] Merged TRIB_ID={trib_id} (RANK={rank}) for polygon {poly_idx + 1}, segments={len(cleaned)}")
                        else:
                            print(f"[ExtractMainValleys] Merged TRIB_ID={trib_id} (RANK={rank}) for polygon {poly_idx + 1}, segments={len(cleaned)}")
                    except Exception as e:
                        raise RuntimeError(f"[ExtractMainValleys] Failed to merge lines for TRIB_ID={trib_id} in polygon {poly_idx + 1}: {e}")

        if not all_merged_records:
            raise RuntimeError("[ExtractMainValleys] No main valley lines could be extracted from any polygon.")

        gdf = gpd.GeoDataFrame(all_merged_records, crs=self.crs)

        if clip_to_perimeter:
            if feedback:
                feedback.pushInfo("[ExtractMainValleys] Clipping final valley lines to perimeter...")
                feedback.setProgress(90)
                if feedback.isCanceled():
                    feedback.reportError('Process cancelled by user during final clipping.')
                    raise RuntimeError('Process cancelled by user.')
            else:
                print("[ExtractMainValleys] Clipping final valley lines to perimeter...")
            gdf = gpd.overlay(gdf, perimeter, how="intersection")

        if feedback:
            feedback.pushInfo(f"[ExtractMainValleys] Main valley extraction complete. {len(gdf)} valleys extracted from {len(perimeter)} polygons.")
            feedback.setProgress(100)
            if feedback.isCanceled():
                feedback.reportError('Process cancelled by user at completion.')
                raise RuntimeError('Process cancelled by user.')
        else:
            print(f"[ExtractMainValleys] Main valley extraction complete. {len(gdf)} valleys extracted from {len(perimeter)} polygons.")
        return gdf


    def extract_main_ridges(
        self,
        ridge_lines: gpd.GeoDataFrame,
        facc_path: str,
        perimeter: gpd.GeoDataFrame = None,
        nr_main: int = 2,
        clip_to_perimeter: bool = True,
        feedback=None
    ) -> gpd.GeoDataFrame:
        """
        Identify and trace the main ridge lines (watershed divides) using the same logic as main valley detection.
        Merging based on the highest flow accumulation,
        using only points uniquely associated with one TRIB_ID (to avoid confluent points).

        Args:
            ridge_lines (GeoDataFrame): Ridge line network with 'LINK_ID', 'TRIB_ID', and 'DS_LINK_ID' attributes.
            facc_path (str): Path to the flow accumulation raster (based on inverted DTM).
            perimeter (GeoDataFrame, optional): Polygon defining the area boundary. If None, uses ridge_lines extent.
            nr_main (int): Number of main ridges to select.
            clip_to_perimeter (bool): If True, clips output to boundary polygon of perimeter.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting/logging.

        Returns:
            GeoDataFrame: Traced main ridge lines.
        """
        if feedback:
            feedback.pushInfo("[ExtractMainRidges] Starting main ridge extraction using main valleys logic (extract_main_valleys)...")
        else:
            print("[ExtractMainRidges] Starting main ridge extraction using main valleys logic (extract_main_valleys)...")

        gdf = self.extract_main_valleys(
            valley_lines=ridge_lines,
            facc_path=facc_path,
            perimeter=perimeter,
            nr_main=nr_main,
            clip_to_perimeter=clip_to_perimeter,
            feedback=feedback
        )

        return gdf
    
    def get_keypoints(
        self,
        valley_lines: gpd.GeoDataFrame,
        dtm_path: str,
        smoothing_window: int = 9,
        polyorder: int = 2,
        min_distance: float = 10.0,
        max_keypoints: int = 5,
        find_window_cells: int = 10,
        feedback=None
        ) -> gpd.GeoDataFrame:
        """
        Detect keypoints along valley lines based on curvature of elevation profiles
        (second derivative). Keypoints are locations where the profile changes from 
        convex to concave curvature, indicating morphological transitions like 
        channel heads or slope breaks.

        The elevation profile is extracted along each valley line using the DTM at
        pixel resolution (all values along the line) and smoothed using a Savitzky-Golay 
        filter. The second derivative is then computed, and points with the strongest 
        convex → concave transitions are selected as keypoints.

        Args:
            valley_lines (GeoDataFrame): Valley centerlines with geometries and unique LINK_ID.
            dtm_path (str): Path to the input DTM raster.
            smoothing_window (int): Window size for Savitzky-Golay filter (must be odd).
            polyorder (int): Polynomial order for Savitzky-Golay smoothing.
            min_distance (float): Minimum distance between selected keypoints (in meters).
            max_keypoints (int): Maximum number of keypoints to retain per valley line.
            find_window_cells (int): Number of cells to consider for curvature detection.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting/logging.

        Returns:
            GeoDataFrame: Detected keypoints as point geometries with metadata.
        """
        results = []

        # Validate smoothing window (must be odd)
        if smoothing_window % 2 == 0:
            smoothing_window += 1
            if feedback:
                feedback.pushInfo(f"[GetKeypoints] Smoothing window adjusted to {smoothing_window} (must be odd)")
            else:
                print(f"[GetKeypoints] Smoothing window adjusted to {smoothing_window} (must be odd)")
        if feedback:
            feedback.pushInfo(f"[GetKeypoints] Starting keypoint detection on {len(valley_lines)} valley lines...")
            feedback.setProgress(0)
            if feedback.isCanceled():
                feedback.reportError('Process cancelled by user at initialization.')
                raise RuntimeError('Process cancelled by user.')
        else:
            print(f"[GetKeypoints] Starting keypoint detection on {len(valley_lines)} valley lines...")

        # Read DTM raster using GDAL
        dtm_ds = gdal.Open(dtm_path, gdal.GA_ReadOnly)
        if dtm_ds is None:
            raise RuntimeError(f"Cannot open DTM raster: {dtm_path}.{self._get_gdal_error_message()}")
            
        try:
            # Get raster information
            geotransform = dtm_ds.GetGeoTransform()
            if geotransform is None:
                raise RuntimeError(f"Cannot get geotransform from DTM raster: {dtm_path}.{self._get_gdal_error_message()}")
            
            res = abs(geotransform[1])  # pixel width
            dtm_band = dtm_ds.GetRasterBand(1)
            
            # Auto-calculate find_window_distance based on pixel size
            processed_lines = 0
            skipped_lines = 0
            total_lines = len(valley_lines)
            total_keypoints = 0
            
            for idx, row in valley_lines.iterrows():
                line = row.geometry
                line_id = row.LINK_ID
                length = line.length
                # Sample at pixel resolution - use pixel size as sampling distance
                sampling_distance = res
                num_samples = max(int(length / sampling_distance), 2)  # At least 2 samples

                # Progress reporting for every line (or every 5 lines for large datasets)
                current_line = idx + 1
                progress_pct = int((current_line / total_lines) * 100)
                
                if feedback:
                    # Update progress bar for every line
                    feedback.setProgress(progress_pct)
                    if feedback.isCanceled():
                        feedback.reportError(f'Process cancelled by user while processing line {current_line}/{total_lines}.')
                        raise RuntimeError('Process cancelled by user.')
                    # Detailed info for smaller datasets or periodic updates for large datasets
                    if total_lines <= 20 or current_line % 5 == 0 or current_line == total_lines:
                        feedback.pushInfo(f"[GetKeypoints] Processing line {current_line}/{total_lines} ({progress_pct}%) - Line ID: {line_id}, Length: {length:.1f}m, Samples: {num_samples}")
                else:
                    if total_lines <= 20 or current_line % 5 == 0 or current_line == total_lines:
                        print(f"[GetKeypoints] Processing line {current_line}/{total_lines} ({progress_pct}%) - Line ID: {line_id}, Length: {length:.1f}m, Samples: {num_samples}")
                processed_lines += 1

                distances = np.linspace(0, length, num=num_samples)
                sample_points = [line.interpolate(d) for d in distances]
                coords = [(pt.x, pt.y) for pt in sample_points]
                
                # Convert world coordinates to pixel coordinates using utility function
                pixel_indices = TopoDrainCore._coords_to_pixel_indices(coords, geotransform)
                
                # Sample elevations using GDAL
                elevations = []
                for px, py in pixel_indices:
                    # Read elevation value at pixel location
                    try:
                        elevation_array = dtm_band.ReadAsArray(px, py, 1, 1)
                        if elevation_array is not None and elevation_array.size > 0:
                            elevations.append(float(elevation_array[0, 0]))
                        else:
                            # Use a default value if pixel is outside raster bounds
                            elevations.append(0.0)
                    except:
                        # Handle any GDAL errors
                        elevations.append(0.0)

                # Smooth elevation profile first
                elev_smooth = savgol_filter(elevations, smoothing_window, polyorder)
                
                # Calculate curvature (second derivative) directly from smoothed data using numpy gradient
                # This avoids double-smoothing while still working with clean data
                curvature = np.gradient(np.gradient(elev_smooth))
                
                # Alternative approaches:
                # Option 1 - Direct savgol on raw data (single smoothing): 
                # curvature = savgol_filter(elevations, smoothing_window, polyorder, deriv=2)
                # Option 2 - Double smoothing (may over-smooth):
                # curvature = savgol_filter(elev_smooth, smoothing_window, polyorder, deriv=2)

                # Find convex→concave transitions (keypoints)
                find_window = max(3, find_window_cells)  # At least 3 pixels, maybe later as input parameter? e.g. Nr. of cells?
                candidates = TopoDrainCore._find_inflection_candidates(curvature, window=find_window)

                # Sort and select strongest candidates
                sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

                # Check for minimum distance between keypoints
                accepted = []
                for i, strength in sorted_candidates:
                    pt = sample_points[i]
                    if all(pt.distance(p[0]) >= min_distance for p in accepted):
                        accepted.append((pt, strength, i))
                    if len(accepted) >= max_keypoints:
                        break

                for rank, (pt, _, idx_pt) in enumerate(accepted, start=1):
                    results.append({
                        "geometry": Point(pt),
                        "VALLEY_ID": row["LINK_ID"],
                        "ELEV_INDEX": idx_pt,
                        "RANK": rank,
                        "CURVATURE": curvature[idx_pt]
                    })

                # Update keypoint count and provide feedback
                line_keypoints = len(accepted)
                total_keypoints += line_keypoints
                
                if feedback:
                    if line_keypoints > 0:
                        feedback.pushInfo(f"[GetKeypoints] Line {line_id}: found {line_keypoints} keypoints (total: {total_keypoints})")
                    else:
                        feedback.pushInfo(f"[GetKeypoints] Line {line_id}: no keypoints found")
                else:
                    if line_keypoints > 0:
                        print(f"[GetKeypoints] Line {line_id}: found {line_keypoints} keypoints (total: {total_keypoints})")
                    else:
                        print(f"[GetKeypoints] Line {line_id}: no keypoints found")
                        
        finally:
            dtm_ds = None  # Close GDAL dataset

        gdf = gpd.GeoDataFrame(results, geometry="geometry", crs=self.crs)

        if feedback:
            feedback.pushInfo(f"[GetKeypoints] Keypoint detection complete:")
            feedback.pushInfo(f"[GetKeypoints] - Total valley lines: {total_lines}")
            feedback.pushInfo(f"[GetKeypoints] - Processed lines: {processed_lines}")
            feedback.pushInfo(f"[GetKeypoints] - Skipped lines: {skipped_lines}")
            feedback.pushInfo(f"[GetKeypoints] - Total keypoints found: {len(gdf)}")
            feedback.setProgress(100)
            if feedback.isCanceled():
                feedback.reportError('Process cancelled by user at completion.')
                raise RuntimeError('Process cancelled by user.')
        else:
            print(f"[GetKeypoints] Keypoint detection complete: {len(gdf)} keypoints found from {processed_lines}/{total_lines} valley lines (skipped: {skipped_lines})")

        return gdf

    def get_points_along_lines(
        self,
        input_lines: gpd.GeoDataFrame,
        reference_points: gpd.GeoDataFrame = None,
        distance_between_points: float = 10.0,
        feedback=None
        ) -> gpd.GeoDataFrame:
        """
        Distribute points along input lines at specified intervals. If reference points are provided, 
        the points will be placed in an optimal way: So that the distance is at least the specified 
        distance from the nearest reference point and as much points as possible can be created.

        Args:
            input_lines (GeoDataFrame): Input lines (e.g. valley centerlines).
            reference_points (GeoDataFrame, optional): Input reference points (e.g. keypoints). 
                If provided, points will be placed optimally to maintain minimum distance from reference points.
            distance_between_points (float): Distance between points to be created along lines (in meters). Default 10.0.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting/logging.

        Returns:
            GeoDataFrame: Points along lines with preserved line attributes plus additional point metadata.
        """
        if feedback:
            feedback.pushInfo(f"[GetPointsAlongLines] Starting point distribution along {len(input_lines)} lines...")
            feedback.setProgress(0)
        else:
            print(f"[GetPointsAlongLines] Starting point distribution along {len(input_lines)} lines...")

        if input_lines.empty:
            if feedback:
                feedback.pushWarning("[GetPointsAlongLines] No input lines provided")
            else:
                warnings.warn("[GetPointsAlongLines] Warning: No input lines provided")
            return gpd.GeoDataFrame(crs=self.crs)

        # Validate distance parameter
        if distance_between_points <= 0:
            raise ValueError("distance_between_points must be positive")

        all_points = []
        total_lines = len(input_lines)

        for line_idx, line_row in input_lines.iterrows():
            line_geom = line_row.geometry
            
            # Progress reporting
            if feedback:
                progress = int((line_idx / total_lines) * 90)  # Reserve 10% for final processing
                feedback.setProgress(progress)
                if feedback.isCanceled():
                    feedback.reportError("[GetPointsAlongLines] Process cancelled by user")
                    raise RuntimeError("Process cancelled by user")
                feedback.pushInfo(f"[GetPointsAlongLines] Processing line {line_idx + 1}/{total_lines}")
            else:
                print(f"[GetPointsAlongLines] Processing line {line_idx + 1}/{total_lines}")

            # Handle different geometry types
            if isinstance(line_geom, LineString):
                line_points = self._distribute_points_on_line(
                    line_geom, reference_points, distance_between_points, line_row, line_idx
                )
                all_points.extend(line_points)
            elif isinstance(line_geom, MultiLineString):
                for sub_line_idx, sub_line in enumerate(line_geom.geoms):
                    line_points = self._distribute_points_on_line(
                        sub_line, reference_points, distance_between_points, line_row, 
                        f"{line_idx}_{sub_line_idx}"
                    )
                    all_points.extend(line_points)
            else:
                if feedback:
                    feedback.pushWarning(f"[GetPointsAlongLines] Skipping unsupported geometry type: {type(line_geom)}")
                else:
                    warnings.warn(f"[GetPointsAlongLines] Warning: Skipping unsupported geometry type: {type(line_geom)}")

        # Create result GeoDataFrame
        if all_points:
            result_gdf = gpd.GeoDataFrame(all_points, crs=self.crs)
        else:
            result_gdf = gpd.GeoDataFrame(crs=self.crs)

        if feedback:
            feedback.setProgress(100)
            feedback.pushInfo(f"[GetPointsAlongLines] Point distribution complete: {len(result_gdf)} points created from {total_lines} lines")
        else:
            print(f"[GetPointsAlongLines] Point distribution complete: {len(result_gdf)} points created from {total_lines} lines")

        return result_gdf

    def _distribute_points_on_line(
        self, 
        line_geom: LineString, 
        reference_points: gpd.GeoDataFrame, 
        distance_between_points: float, 
        line_row, 
        line_id
        ) -> list:
        """
        Distribute points along a single LineString. If reference points are provided,
        optimizes placement starting from reference points and expanding outward.
        
        Args:
            line_geom (LineString): The line geometry to distribute points along
            reference_points (GeoDataFrame): Reference points to maintain distance from
            distance_between_points (float): Target distance between points
            line_row: The original line row data to preserve attributes
            line_id: Identifier for the line
            
        Returns:
            list: List of point dictionaries with geometry and attributes
        """
        line_length = line_geom.length
        if line_length < distance_between_points:
            # Line too short for any points with the specified distance
            return []

        # Find reference points that are ON this line (within very small tolerance for floating point precision)
        ref_distances_on_line = []
        if reference_points is not None and not reference_points.empty:
            ref_point_tolerance = 0.1  # Very small tolerance for floating point precision (10 cm)
            
            for _, ref_row in reference_points.iterrows():
                # Check if reference point is actually on this line (not just nearby)
                ref_distance_to_line = line_geom.distance(ref_row.geometry)
                if ref_distance_to_line <= ref_point_tolerance:
                    # Project reference point onto the line to get its distance along the line
                    distance_along = line_geom.project(ref_row.geometry)
                    ref_distances_on_line.append(distance_along)

        # Simple case: no reference points or no reference points found ON this line
        if not ref_distances_on_line:
            num_points = int(line_length // distance_between_points)
            if num_points == 0:
                return []
                
            points = []
            for i in range(num_points):
                distance_along = (i + 1) * distance_between_points
                if distance_along <= line_length:
                    point_geom = line_geom.interpolate(distance_along)
                    
                    # Create point with line attributes plus additional metadata
                    point_data = line_row.drop('geometry').to_dict()  # Preserve original line attributes
                    point_data.pop('RANK', None)  # Remove RANK attribute if present
                    point_data.pop('DS_LINK_ID', None)  # Remove DS_LINK_ID attribute if present
                    point_data['geometry'] = point_geom
                    point_data['line_id'] = line_id
                    point_data['distance_along_line'] = distance_along
                    point_data['point_index'] = i
                    point_data['is_reference_point'] = False
                    points.append(point_data)
            return points
        
        # Complex case: optimize placement considering reference points
        
        # Remove duplicates and sort reference distances along the line
        ref_distances_on_line = sorted(set(ref_distances_on_line))
        
        # Generate all candidate distances starting from each reference point
        all_candidate_distances = set()
        
        for ref_distance in ref_distances_on_line:
            # Always include the reference point location itself
            all_candidate_distances.add(ref_distance)
            
            # Expand forward (toward end of line) from reference point
            current_dist = ref_distance + distance_between_points
            while current_dist <= line_length:
                all_candidate_distances.add(current_dist)
                current_dist += distance_between_points
            
            # Expand backward (toward start of line) from reference point
            current_dist = ref_distance - distance_between_points
            while current_dist >= 0:
                all_candidate_distances.add(current_dist)
                current_dist -= distance_between_points
        
        # Convert to sorted list for processing
        candidate_distances = sorted([dist for dist in all_candidate_distances if 0 <= dist <= line_length])
        
        # Select final points ensuring minimum distance between them
        # Priority: reference points first, then other candidates
        selected_distances = []
        
        # First pass: Add all reference points (they have highest priority)
        for ref_distance in ref_distances_on_line:
            selected_distances.append(ref_distance)
        
        # Second pass: Add other candidates if they maintain minimum distance
        for candidate_dist in candidate_distances:
            # Skip if this is already a reference point
            if candidate_dist in ref_distances_on_line:
                continue
                
            # Check if this candidate maintains minimum distance from all selected points
            valid = True
            for selected_dist in selected_distances:
                if abs(candidate_dist - selected_dist) < distance_between_points:
                    valid = False
                    break
            
            if valid:
                selected_distances.append(candidate_dist)
        
        # Sort final distances for consistent output
        selected_distances.sort()
        
        # Create point objects from selected distances
        points = []
        for i, dist in enumerate(selected_distances):
            point_geom = line_geom.interpolate(dist)
            
            # Check if this point is at a reference point location
            is_reference_point = False
            for ref_distance in ref_distances_on_line:
                if abs(dist - ref_distance) < 0.1:  # Small tolerance for floating point comparison
                    is_reference_point = True
                    break
            
            # Create point with line attributes plus additional metadata
            point_data = line_row.drop('geometry').to_dict()  # Preserve original line attributes
            point_data.pop('RANK', None)  # Remove RANK attribute if present
            point_data.pop('DS_LINK_ID', None)  # Remove DS_LINK_ID attribute if present
            point_data['geometry'] = point_geom
            point_data['line_id'] = line_id
            point_data['distance_along_line'] = dist
            point_data['point_index'] = i
            point_data['is_reference_point'] = is_reference_point  # Flag indicating if this was a reference point
            points.append(point_data)
        
        return points

    @staticmethod
    def _get_orthogonal_directions_start_points(
        barrier_raster_path: str,
        point: Point,
        line_geom: LineString,
        max_offset: int = 10
    ) -> tuple[Point, Point]:
        """
        Determine two start points to the left and right of a given point, orthogonal to an input line using GDAL.

        The function searches along the orthogonal direction from a given point until it finds
        a non-barrier cell in the provided raster.

        Args:
            barrier_raster_path (str): Path to binary raster with 1 = barrier, 0 = free (supports all GDAL formats in gdal_driver_mapping).
            point (Point): The reference point (typically a keypoint on a valley line).
            line_geom (LineString): Reference line geometry used to determine orientation, e.g. valley line.
            max_offset (int): Maximum number of cells to move outward when searching.

        Returns:
            tuple: (left_point, right_point), or (None, None) if no valid points found.
        """
        print(f"[GetOrthogonalDirectionsStartPoints] Checking point {point}, max_offset={max_offset}")
        
        # Read barrier raster using GDAL
        barrier_ds = gdal.Open(barrier_raster_path, gdal.GA_ReadOnly)
        if barrier_ds is None:
            print(f"[GetOrthogonalDirectionsStartPoints] Cannot open barrier raster: {barrier_raster_path}")
            return None, None
            
        try:
            # Get raster information
            rows = barrier_ds.RasterYSize
            cols = barrier_ds.RasterXSize
            geotransform = barrier_ds.GetGeoTransform()
            if geotransform is None:
                print(f"[GetOrthogonalDirectionsStartPoints] Cannot get geotransform from: {barrier_raster_path}")
                return None, None

            res = abs(geotransform[1])  # pixel width (assumes square pixels)

            # Read barrier mask data
            barrier_band = barrier_ds.GetRasterBand(1)
            barrier_mask = barrier_band.ReadAsArray()
            if barrier_mask is None:
                print(f"[GetOrthogonalDirectionsStartPoints] Cannot read barrier data from: {barrier_raster_path}")
                return None, None
                
        finally:
            barrier_ds = None  # Close dataset

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

                # Convert world coordinates to pixel indices using GDAL geotransform
                px = int((test_x - geotransform[0]) / geotransform[1])
                py = int((test_y - geotransform[3]) / geotransform[5])

                # Bounds check
                if not (0 <= py < rows and 0 <= px < cols):
                    continue

                # barrier >= 1 means forbidden
                if barrier_mask[py, px] >= 1:
                    print(f"[GetOrthogonalDirectionsStartPoints] Still on barrier at offset {i}.")
                else:
                    new_point = Point(test_x, test_y)
                    print(f"[GetOrthogonalDirectionsStartPoints] Found valid point at offset {i}: {new_point.wkt}")
                    return new_point
                
            print("[GetOrthogonalDirectionsStartPoints] No valid point found within max_offset.")
            return None

        left_pt = find_valid_point(ortho_left)
        right_pt = find_valid_point(ortho_right)

        return left_pt, right_pt
        
    @staticmethod
    def _get_linedirection_start_point(
        barrier_raster_path: str,
        line_geom: LineString,
        max_offset: int = 10,
        reverse: bool = False
    ) -> Point:
        """
        Determine a start point in the proceeding direction of a given input line.

        The function searches from the endpoint of the line along its direction until it finds
        a non-barrier cell. If reverse=True, follows the line geometry backwards from the endpoint.
        For forward mode, prioritizes perpendicular direction to the local barrier orientation.

        Args:
            barrier_raster_path (str): Path to binary raster (GeoTIFF) with 1 = barrier, 0 = free.
            line_geom (LineString): Reference line geometry used to determine orientation, e.g. keyline.
            max_offset (int): Maximum number of cells to move outward when searching.
            reverse (bool): If True, follow line geometry backwards from endpoint, if False search forward.

        Returns:
            Point or None: The new start point beyond the barrier, or None if not found.
        """
        coords = list(line_geom.coords)
        if len(coords) >= 2:
            end_point = coords[-1]    # Always start from the endpoint
            ref_point = coords[-2]    # Reference point for direction
        else:
            raise ValueError("LineString must have at least two coordinates to determine direction.")
        
        print(f"[GetLinedirectionStartPoint]: Checking endpoint {end_point}, reverse={reverse}")
        print(f"[GetLinedirectionStartPoint] barrier_raster_path: {barrier_raster_path}")

        # Read barrier raster using GDAL
        barrier_ds = gdal.Open(barrier_raster_path, gdal.GA_ReadOnly)
        if barrier_ds is None:
            print(f"[GetLinedirectionStartPoint] Cannot open barrier raster: {barrier_raster_path}")
            return None
            
        try:
            # Get raster information
            rows = barrier_ds.RasterYSize
            cols = barrier_ds.RasterXSize
            geotransform = barrier_ds.GetGeoTransform()
            if geotransform is None:
                print(f"[GetLinedirectionStartPoint] Cannot get geotransform from: {barrier_raster_path}")
                return None

            res = abs(geotransform[1])  # pixel width (assumes square pixels)

            # Read barrier mask data
            barrier_band = barrier_ds.GetRasterBand(1)
            barrier_mask = barrier_band.ReadAsArray()
            if barrier_mask is None:
                print(f"[GetLinedirectionStartPoint] Cannot read barrier data from: {barrier_raster_path}")
                return None
                
        finally:
            barrier_ds = None  # Close dataset

        # Convert world coordinates to pixel indices using utility function
        pixel_coords = TopoDrainCore._coords_to_pixel_indices([end_point], geotransform)
        col_ep, row_ep = pixel_coords[0]
        print(f"[GetLinedirectionStartPoint] Endpoint raster index: row={row_ep}, col={col_ep}")
        
        # Bounds check for endpoint
        if not (0 <= row_ep < rows and 0 <= col_ep < cols):
            print("[GetLinedirectionStartPoint] Endpoint is outside raster bounds.")
            return None
            
        if not barrier_mask[row_ep, col_ep] >= 1:
            print("[GetLinedirectionStartPoint] No barrier at endpoint, returning None.")
            return None  # no barrier at endpoint, so no need to search

        if reverse:
                # For reverse, follow the line geometry backwards
                print("[GetLinedirectionStartPoint] Reverse mode: following line geometry backwards")
                
                # Get total line length and work backwards from endpoint
                total_length = line_geom.length
                
                def find_valid_point_along_line():
                    for i in range(1, max_offset + 1):
                        # Calculate distance to move back along the line
                        back_distance = res * i
                        
                        # Calculate position along line (from start = 0 to end = total_length)
                        # We want to go backwards from the end, so subtract from total_length
                        target_distance = max(0, total_length - back_distance)
                        
                        # Get point at this distance along the line
                        try:
                            test_point = line_geom.interpolate(target_distance)
                            test_x, test_y = test_point.x, test_point.y
                            
                            # Convert world coordinates to pixel indices using utility function
                            pixel_coords = TopoDrainCore._coords_to_pixel_indices([(test_x, test_y)], geotransform)
                            col_idx, row_idx = pixel_coords[0]
                            print(f"[GetLinedirectionStartPoint] Checking offset {i} along line: ({test_x}, {test_y}) -> row={row_idx}, col={col_idx}")

                            if not (0 <= row_idx < rows and 0 <= col_idx < cols):
                                print(f"[GetLinedirectionStartPoint] Offset {i} out of raster bounds.")
                                continue

                            if barrier_mask[row_idx, col_idx] > 0:
                                print(f"[GetLinedirectionStartPoint] Still on barrier at offset {i} along line.")
                            else:
                                new_point = Point(test_x, test_y)
                                print(f"[GetLinedirectionStartPoint] Found valid point at offset {i} along line: {new_point.wkt}")
                                return new_point

                        except Exception as e:
                            print(f"[GetLinedirectionStartPoint] Error interpolating at distance {target_distance}: {e}")
                            continue
                            
                        # If we've reached the start of the line, stop
                        if target_distance <= 0:
                            print("[GetLinedirectionStartPoint] Reached start of line without finding valid point.")
                            break
                    
                    print("[GetLinedirectionStartPoint] No valid point found following line backwards.")
                    return None
                
                new_pt = find_valid_point_along_line()
                
        else:
                # For forward, use mean direction of the last two line segments
                print("[GetLinedirectionStartPoint] Forward mode: using line direction")
                
                tangent = None
                
                # Use simple mean direction of last two line segments
                if tangent is None:
                    print("[GetLinedirectionStartPoint] Using mean direction of last two line segments")
                    
                    # Calculate tangent vectors for the last two segments (if available)
                    num_segments_to_use = min(2, len(coords) - 1)
                    tangent_vectors = []
                    
                    for i in range(num_segments_to_use):
                        seg_end_idx = len(coords) - 1 - i  # Start from the end
                        seg_start_idx = seg_end_idx - 1
                        
                        if seg_start_idx >= 0:
                            dx = coords[seg_end_idx][0] - coords[seg_start_idx][0]
                            dy = coords[seg_end_idx][1] - coords[seg_start_idx][1]
                            
                            segment_length = np.sqrt(dx*dx + dy*dy)
                            
                            if segment_length > 0:
                                # Normalize to unit vector
                                unit_vector = np.array([dx, dy]) / segment_length
                                tangent_vectors.append(unit_vector)
                    
                    if not tangent_vectors:
                        print("[GetLinedirectionStartPoint] No valid segments found for tangent calculation.")
                        return None
                    
                    # Calculate simple mean of tangent vectors (unweighted average)
                    if len(tangent_vectors) > 0:
                        mean_tangent = np.mean(tangent_vectors, axis=0)
                        norm = np.linalg.norm(mean_tangent)
                        
                        if norm > 0:
                            tangent = mean_tangent / norm
                            print(f"[GetLinedirectionStartPoint] Mean tangent vector from {len(tangent_vectors)} segments: {tangent}")
                        else:
                            print("[GetLinedirectionStartPoint] Zero-length mean tangent vector, cannot proceed.")
                            return None
                    else:
                        print("[GetLinedirectionStartPoint] No tangent vectors calculated.")
                        return None

                def find_valid_point_forward():
                    for i in range(1, max_offset + 1):
                        offset = res * i
                        test_x = end_point[0] + tangent[0] * offset
                        test_y = end_point[1] + tangent[1] * offset

                        # Convert world coordinates to pixel indices using utility function
                        pixel_coords = TopoDrainCore._coords_to_pixel_indices([(test_x, test_y)], geotransform)
                        col_idx, row_idx = pixel_coords[0]
                        print(f"[GetLinedirectionStartPoint] Checking offset {i}: ({test_x}, {test_y}) -> row={row_idx}, col={col_idx}")

                        if not (0 <= row_idx < rows and 0 <= col_idx < cols):
                            print(f"[GetLinedirectionStartPoint] Offset {i} out of raster bounds.")
                            continue

                        # barrier >= 1 means forbidden
                        if barrier_mask[row_idx, col_idx] >= 1:
                            print(f"[GetLinedirectionStartPoint] Still on barrier at offset {i}.")
                        else:
                            new_point = Point(test_x, test_y)
                            print(f"[GetLinedirectionStartPoint] Found valid point at offset {i}: {new_point.wkt}")
                            return new_point

                    print("[GetLinedirectionStartPoint] No valid point found beyond barrier.")
                    return None

                new_pt = find_valid_point_forward()

        if new_pt:
            return new_pt
        else:
            return None

    @staticmethod
    def _reverse_line_direction(input_geometry):
        """
        Reverse the coordinate direction of LineString geometries.
        
        Can handle single LineString or GeoDataFrame with multiple LineString geometries.
        
        Args:
            input_geometry: Either a single LineString or a GeoDataFrame with LineString geometries
            
        Returns:
            Same type as input but with reversed coordinate direction:
            - LineString: Returns reversed LineString
            - GeoDataFrame: Returns GeoDataFrame with all LineString geometries reversed
        """
        if isinstance(input_geometry, LineString):
            # Handle single LineString
            if hasattr(input_geometry, 'coords') and len(input_geometry.coords) >= 2:
                reversed_coords = list(input_geometry.coords)[::-1]
                return LineString(reversed_coords)
            else:
                return input_geometry
                
        elif hasattr(input_geometry, 'geometry') and hasattr(input_geometry, 'iterrows'):
            # Handle GeoDataFrame
            result_gdf = input_geometry.copy()
            reversed_geometries = []
            
            for _, row in input_geometry.iterrows():
                line_geom = row.geometry
                if isinstance(line_geom, LineString) and hasattr(line_geom, 'coords') and len(line_geom.coords) >= 2:
                    reversed_coords = list(line_geom.coords)[::-1]
                    reversed_geometries.append(LineString(reversed_coords))
                else:
                    reversed_geometries.append(line_geom)
                    
            result_gdf.geometry = reversed_geometries
            return result_gdf
            
        else:
            # Return input unchanged if not a recognized type
            return input_geometry

    @staticmethod
    def _create_slope_cost_raster(
        dtm_path: str,
        start_point: Point,
        output_cost_raster_path: str,
        slope: float = 0.01,
        barrier_raster_path: str = None,
        penalty_exp: float = 2.0
    ) -> str:
        """
        Create a raster with cost values based on deviation from desired slope using GDAL.
        You can now set penalty_exp>1 to punish larger deviations more heavily.

        Args:
            dtm_path (str): Path to input DTM raster (supports all GDAL formats in gdal_driver_mapping).
            start_point (Point): Starting point of the constant slope line.
            output_cost_raster_path (str): Path to output cost raster (format determined by file extension).
            slope (float): Desired slope (1% downhill = 0.01).
            barrier_raster_path (str): Path to a binary raster of barriers (1=barrier).
            penalty_exp (float): Exponent on the absolute deviation (>=1) of slope. 2.0 => quadratic penalty --> as higher the exponent as stronger penalty for larger deviations.

        Returns:
            str: Path to the written cost raster.
        """
        # Read DTM raster using GDAL
        dtm_ds = gdal.Open(dtm_path, gdal.GA_ReadOnly)
        if dtm_ds is None:
            raise RuntimeError(f"Cannot open DTM raster: {dtm_path}")
            
        try:
            # Get raster information
            rows = dtm_ds.RasterYSize
            cols = dtm_ds.RasterXSize
            geotransform = dtm_ds.GetGeoTransform()
            if geotransform is None:
                raise RuntimeError(f"Cannot get geotransform from DTM raster: {dtm_path}")
            
            projection = dtm_ds.GetProjection()
            srs = dtm_ds.GetSpatialRef()
            
            # Read DTM data
            dtm_band = dtm_ds.GetRasterBand(1)
            dtm = dtm_band.ReadAsArray().astype(np.float32)
            if dtm is None:
                raise RuntimeError(f"Cannot read DTM data from: {dtm_path}")
            
            # Handle NoData values
            nodata = dtm_band.GetNoDataValue()
            if nodata is not None:
                dtm[dtm == nodata] = np.nan

            # Convert start point coordinates to pixel indices using utility function
            pixel_coords = TopoDrainCore._coords_to_pixel_indices([start_point.coords[0]], geotransform)
            key_col, key_row = pixel_coords[0]
            
            # Bounds check for start point
            if not (0 <= key_row < rows and 0 <= key_col < cols):
                raise ValueError(f"Start point {start_point} is outside raster bounds")

            # Create index arrays for distance calculation
            rr, cc = np.indices((rows, cols))
            
            # Calculate elevation difference and horizontal distance
            dz = dtm - dtm[key_row, key_col]
            res = abs(geotransform[1])  # pixel width
            dist = np.hypot(rr - key_row, cc - key_col) * res
            expected_dz = -dist * slope

            # Calculate linear deviation
            deviation = np.abs(dz - expected_dz)

            # Apply exponentiation for stronger penalty
            cost = deviation ** penalty_exp

            # Enforce NoData areas with high cost
            cost[np.isnan(dtm)] = 1e6

            # Read barrier mask from raster if provided
            if barrier_raster_path is not None:
                barrier_ds = gdal.Open(barrier_raster_path, gdal.GA_ReadOnly)
                if barrier_ds is None:
                    raise RuntimeError(f"Cannot open barrier raster: {barrier_raster_path}")
                    
                try:
                    barrier_band = barrier_ds.GetRasterBand(1)
                    barrier_mask = barrier_band.ReadAsArray()
                    if barrier_mask is None:
                        raise RuntimeError(f"Cannot read barrier data from: {barrier_raster_path}")
                    
                    if barrier_mask.shape != cost.shape:
                        raise ValueError("Barrier raster shape does not match DTM shape.")
                    
                    barrier_mask = barrier_mask.astype(bool)
                    if np.any(barrier_mask):
                        print("[TopoDrainCore] Applying barrier mask to cost raster.")
                        # Set cost to a very high value where barriers are present
                        cost[barrier_mask] = 1e6
                        
                finally:
                    barrier_ds = None  # Close barrier dataset

            # Zero cost at the true start point
            cost[key_row, key_col] = 0
            
        finally:
            dtm_ds = None  # Close DTM dataset

        # Use GTiff driver since we know this is an internal function that creates .tif files
        driver_name = 'GTiff'
        
        # Create output raster using GDAL with best practices
        driver = gdal.GetDriverByName(driver_name)
        if driver is None:
            raise RuntimeError(f"Cannot create GDAL driver for: {driver_name}")
            
        creation_options = [
            'COMPRESS=LZW',
            'TILED=YES',
            'BIGTIFF=IF_SAFER'
        ]
        
        out_ds = driver.Create(output_cost_raster_path, cols, rows, 1, gdal.GDT_Float32, 
                              options=creation_options)
        if out_ds is None:
            raise RuntimeError(f"Cannot create output raster: {output_cost_raster_path}")
            
        try:
            # Set spatial reference and geotransform
            out_ds.SetGeoTransform(geotransform)
            
            # Set complete spatial reference system
            if srs is not None:
                out_ds.SetSpatialRef(srs)
            elif projection:
                # Fallback to projection string if SRS not available
                out_ds.SetProjection(projection)
            
            # Write cost data
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(cost.astype(np.float32))
            out_band.SetNoDataValue(1e6)
            
        finally:
            out_ds = None  # Close output dataset

        return output_cost_raster_path

    @staticmethod
    def _create_source_raster(
        reference_raster_path: str,
        source_point: Point,
        output_source_raster_path: str,
        ) -> str:
        """
        Create a binary raster marking the source cell (value = 1) based on a given Point using GDAL.
        All other cells are set to 0.

        Args:
            reference_raster_path (str): Path to the reference raster (supports all GDAL formats in gdal_driver_mapping).
            source_point (Point): Shapely Point marking the source location.
            output_source_raster_path (str): Path to output binary raster.

        Returns:
            str: Path to the saved binary raster file.
        """
        # Read reference raster using GDAL
        ref_ds = gdal.Open(reference_raster_path, gdal.GA_ReadOnly)
        if ref_ds is None:
            raise RuntimeError(f"Cannot open reference raster: {reference_raster_path}")
            
        try:
            # Get raster information
            rows = ref_ds.RasterYSize
            cols = ref_ds.RasterXSize
            geotransform = ref_ds.GetGeoTransform()
            if geotransform is None:
                raise RuntimeError(f"Cannot get geotransform from reference raster: {reference_raster_path}")
            
            projection = ref_ds.GetProjection()
            srs = ref_ds.GetSpatialRef()
            
            # Convert source point coordinates to pixel indices using utility function
            pixel_coords = TopoDrainCore._coords_to_pixel_indices([source_point.coords[0]], geotransform)
            col, row = pixel_coords[0]
            
            # Bounds check for source point
            if not (0 <= row < rows and 0 <= col < cols):
                raise ValueError(f"Source point {source_point} is outside the bounds of the reference raster.")

            # Create binary data array
            data = np.zeros((rows, cols), dtype=np.uint8)
            data[row, col] = 1
            
        finally:
            ref_ds = None  # Close reference dataset

        # Use GTiff driver since we know this is an internal function that creates .tif files
        driver_name = 'GTiff'
        
        # Create output raster using GDAL with best practices
        driver = gdal.GetDriverByName(driver_name)
        if driver is None:
            raise RuntimeError(f"Cannot create GDAL driver for: {driver_name}")
            
        creation_options = [
            'COMPRESS=LZW',
            'TILED=YES',
            'BIGTIFF=IF_SAFER'
        ]
        
        out_ds = driver.Create(output_source_raster_path, cols, rows, 1, gdal.GDT_Byte, 
                              options=creation_options)
        if out_ds is None:
            raise RuntimeError(f"Cannot create output raster: {output_source_raster_path}")
            
        try:
            # Set spatial reference and geotransform
            out_ds.SetGeoTransform(geotransform)
            
            # Set complete spatial reference system
            if srs is not None:
                out_ds.SetSpatialRef(srs)
            elif projection:
                # Fallback to projection string if SRS not available
                out_ds.SetProjection(projection)
            
            # Write source data
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(data)
            out_band.SetNoDataValue(0)
            
        finally:
            out_ds = None  # Close output dataset

        return output_source_raster_path

    @staticmethod
    def _select_best_destination_cell(
        accum_raster_path: str,
        destination_raster_path: str,
        best_destination_raster_path: str
    ) -> tuple[str, Point]:
        """
        Select the best destination cell from a binary destination raster based on
        minimum accumulated cost, and write it as a single-cell binary raster using GDAL.

        Args:
            accum_raster_path (str): Path to the cost accumulation raster (supports all GDAL formats in gdal_driver_mapping).
            destination_raster_path (str): Path to the binary destination raster (1 = destination, 0 = background).
            best_destination_raster_path (str): Path to output raster with only the best cell marked.

        Returns:
            tuple[str, Point]: Tuple of (path to the output best destination raster, Point with spatial coordinates of the best destination cell).
        """
        # Read accumulation raster using GDAL
        acc_ds = gdal.Open(accum_raster_path, gdal.GA_ReadOnly)
        if acc_ds is None:
            raise RuntimeError(f"Cannot open accumulation raster: {accum_raster_path}")
            
        try:
            # Get accumulation data
            acc_band = acc_ds.GetRasterBand(1)
            acc_data = acc_band.ReadAsArray()
            if acc_data is None:
                raise RuntimeError(f"Cannot read accumulation data from: {accum_raster_path}")
                
            # Get spatial information for coordinate conversion
            geotransform = acc_ds.GetGeoTransform()
            if geotransform is None:
                raise RuntimeError(f"Cannot get geotransform from accumulation raster: {accum_raster_path}")
                
        finally:
            acc_ds = None  # Close accumulation dataset
            
        # Read destination raster using GDAL
        dest_ds = gdal.Open(destination_raster_path, gdal.GA_ReadOnly)
        if dest_ds is None:
            raise RuntimeError(f"Cannot open destination raster: {destination_raster_path}")
            
        try:
            # Get raster information
            rows = dest_ds.RasterYSize
            cols = dest_ds.RasterXSize
            dest_geotransform = dest_ds.GetGeoTransform()
            if dest_geotransform is None:
                raise RuntimeError(f"Cannot get geotransform from destination raster: {destination_raster_path}")
            
            projection = dest_ds.GetProjection()
            srs = dest_ds.GetSpatialRef()
            
            # Get destination data
            dest_band = dest_ds.GetRasterBand(1)
            dest_data = dest_band.ReadAsArray()
            if dest_data is None:
                raise RuntimeError(f"Cannot read destination data from: {destination_raster_path}")
                
        finally:
            dest_ds = None  # Close destination dataset

        # Consider only destination cells (where value == 1)
        mask = (dest_data == 1)
        acc_masked = np.where(mask, acc_data, np.nan)

        # Identify cell with minimum accumulated cost
        if np.all(np.isnan(acc_masked)):
            raise RuntimeError("No valid destination cell found.")

        # Find the minimum value and its index
        min_val = np.nanmin(acc_masked)
        min_indices = np.where(acc_masked == min_val)
        # Take the first occurrence if multiple
        row, col = min_indices[0][0], min_indices[1][0]

        # Calculate spatial coordinates using GDAL geotransform
        coords = TopoDrainCore._pixel_indices_to_coords([row], [col], geotransform)
        x, y = coords[0]
        best_destination_point = Point(x, y)

        # Create output raster marking only the best cell
        best_dest = np.zeros_like(dest_data, dtype=np.uint8)
        best_dest[row, col] = 1

        # Use GTiff driver since we know this is an internal function that creates .tif files
        driver_name = 'GTiff'
        
        # Create output raster using GDAL with best practices
        driver = gdal.GetDriverByName(driver_name)
        if driver is None:
            raise RuntimeError(f"Cannot create GDAL driver for: {driver_name}")
            
        creation_options = [
            'COMPRESS=LZW',
            'TILED=YES',
            'BIGTIFF=IF_SAFER'
        ]
        
        out_ds = driver.Create(best_destination_raster_path, cols, rows, 1, gdal.GDT_Byte, 
                              options=creation_options)
        if out_ds is None:
            raise RuntimeError(f"Cannot create output raster: {best_destination_raster_path}")
            
        try:
            # Set spatial reference and geotransform
            out_ds.SetGeoTransform(dest_geotransform)
            
            # Set complete spatial reference system
            if srs is not None:
                out_ds.SetSpatialRef(srs)
            elif projection:
                # Fallback to projection string if SRS not available
                out_ds.SetProjection(projection)
            
            # Write best destination data
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(best_dest)
            out_band.SetNoDataValue(0)
            
        finally:
            out_ds = None  # Close output dataset

        return best_destination_raster_path, best_destination_point

    def _analyze_slope_deviation_and_cut(
        self,
        line: LineString,
        start_point: Point,
        expected_slope: float,
        slope_deviation_threshold: float
        ) -> Point:
        """
        Analyse the slope deviation of a line compared to the expected slope.
        This method compares the assumed slope (based on euclidean distance) with the 
        real slope (based on actual line distance), without using actual height values.
        
        The least cost algorithm assumes: dz/euclidean_distance = expected_slope
        But the real line gives us: dz/real_distance = actual_slope
        
        Since dz is the same, we can derive: actual_slope = expected_slope * (euclidean_distance / real_distance)
        Cut the line when actual_slope deviates too much from expected_slope.

        Args:
            line (LineString): The line to analyse.
            start_point (Point): Original start point for reference.
            expected_slope (float): Desired slope (e.g., 0.01 for 1% downhill) (assumed euclidean slope).
            slope_deviation_threshold (float): Maximum allowed relative deviation (e.g., 0.1 for 10%).
            
        Returns:
            Point: The point where cutting should occur, or None if no cutting needed.
        """
        # Sample points along the line at regular intervals
        line_length = line.length
        num_samples = max(int(line_length / 5.0), 10)  # Sample every 5 meters or at least 10 points
        
        distances_along_line = np.linspace(0, line_length, num_samples)
        sample_points = [line.interpolate(d) for d in distances_along_line]
        end_point = sample_points[-1]

        print(f"[AnalyzeSlopeDeviation] Analyzing {num_samples} points along {line_length:.1f}m line")
        print(f"[AnalyzeSlopeDeviation] Start point: {start_point}, End point: {end_point}")
        print(f"[AnalyzeSlopeDeviation] Expected slope (euclidean): {expected_slope:.4f}")
        print(f"[AnalyzeSlopeDeviation] Slope deviation threshold: {slope_deviation_threshold:.2f}")

        # Calculate slope deviations based on distance ratios
        for i, (line_distance, point) in enumerate(zip(distances_along_line, sample_points)):
            if line_distance == 0:
                continue  # Skip start point
                
            # Calculate euclidean distance from start point
            euclidean_distance = start_point.distance(point)
            
            if euclidean_distance == 0:
                continue  # Skip if no distance
                
            # Calculate actual slope based on distance ratio
            # actual_slope = expected_slope * (euclidean_distance / real_distance)
            actual_slope = expected_slope * (euclidean_distance / line_distance)
            
            # Calculate relative deviation from expected slope
            if expected_slope != 0:
                slope_deviation_ratio = actual_slope / expected_slope
                relative_deviation = abs(slope_deviation_ratio - 1.0)
            else:
                relative_deviation = 0
            
            print(f"[AnalyzeSlopeDeviation] Point {i}: line_dist={line_distance:.1f}m, "
                  f"euclidean_dist={euclidean_distance:.1f}m, "
                  f"expected_slope={expected_slope:.4f}, actual_slope={actual_slope:.4f}, "
                  f"deviation={relative_deviation:.3f}")

            if relative_deviation > slope_deviation_threshold:
                print(f"[AnalyzeSlopeDeviation] Slope deviation {relative_deviation:.3f} exceeds threshold "
                      f"{slope_deviation_threshold:.3f} at line distance {line_distance:.1f}m")
                print(f"[AnalyzeSlopeDeviation] Expected slope: {expected_slope:.4f}, Actual slope: {actual_slope:.4f}")
                return point

        # No cutting needed - line maintains acceptable slope deviation
        print(f"[AnalyzeSlopeDeviation] Line maintains acceptable slope deviation")
        return None
    

    def _cut_line_at_point(self, line: LineString, cut_point: Point) -> LineString:
        """
        Cut a line at the specified point, returning the portion from start to cut point.
        
        Args:
            line (LineString): The line to cut.
            cut_point (Point): The point where to cut the line.
            
        Returns:
            LineString: The line segment from start to cut point.
        """
        try:
            # Find the distance along the line to the cut point
            cut_distance = line.project(cut_point)
            print(f"[CutLineAtPoint] Cutting line ({line.length:.2f}m) at distance {cut_distance:.2f}m from start to point {cut_point}")

            # Create a new line from start to cut point
            cut_line = substring(line, 0, cut_distance)

            # Ensure the cut line ends exactly at the cut point
            if isinstance(cut_line, LineString) and len(cut_line.coords) >= 2:
                coords = list(cut_line.coords)
                coords[-1] = (cut_point.x, cut_point.y)  # Replace last coordinate with exact cut point
                return LineString(coords)
            else:
                return line  # Fallback to original line if cutting failed
                
        except Exception as e:
            print(f"[CutLineAtPoint] Error cutting line: {e}")
            return line  # Fallback to original line

    def _get_constant_slope_line(
        self,
        dtm_path: str,
        start_point: Point,
        destination_raster_path: str,
        slope: float = 0.01,
        barrier_raster_path: str = None,
        slope_deviation_threshold: float = 0.2,
        max_iterations_slope: int = 30,
        feedback=None
    ) -> LineString:
        """
        Trace lines with constant slope starting from a given point using an iterative approach.
        
        This function traces a line, checks where the line distance deviates too much from 
        the Euclidean distance from start point, cuts the line at that point, and continues 
        from there in subsequent iterations.

        Args:
            dtm_path (str): Path to input DTM raster (supports all GDAL formats in gdal_driver_mapping).
            start_point (Point): Starting point of the constant slope line.
            destination_raster_path (str): Path to the binary raster indicating destination cells (1 = destination).
            slope (float): Desired slope for the line (e.g., 0.01 for 1% downhill or -0.01 for uphill).
            barrier_raster_path (str): Optional path to a binary raster of cells that should not be crossed (1 = barrier).
            slope_deviation_threshold (float): Maximum allowed relative deviation from expected slope (0.0-1.0, e.g., 0.2 for 20% deviation before line cutting). Default 0.2.
            max_iterations_slope (int): Maximum number of iterations for line refinement.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting.

        Returns:
            LineString: Refined constant slope path as a Shapely LineString, or None if no path found.
        """
        if self.wbt is None:
            raise RuntimeError("WhiteboxTools not initialized. Check WhiteboxTools configuration: QGIS settings -> Options -> Processing -> Provider -> WhiteboxTools -> WhiteboxTools executable.")

        print(f"[GetConstantSlopeLine] Starting constant slope line tracing")
        if feedback:
            feedback.pushInfo(f"[GetConstantSlopeLine] Starting constant slope line tracing")
            feedback.pushInfo(f"*for more information see in Python Console")

        print(f"[GetConstantSlopeLine] destination raster path: {destination_raster_path}")
        print(f"[GetConstantSlopeLine] barrier raster path: {barrier_raster_path}")
        print(f"[GetConstantSlopeLine] slope: {slope}, max_iterations_slope: {max_iterations_slope}, slope_deviation_threshold: {slope_deviation_threshold}")

        current_start_point = start_point
        accumulated_line_coords = []
        
        for iteration in range(max_iterations_slope):
            # Check for cancellation at the start of each iteration
            if feedback and feedback.isCanceled():
                feedback.reportError("Operation cancelled by user")
                raise RuntimeError("Operation cancelled by user")
            
            print(f"[GetConstantSlopeLine] *** Slope Iteration {iteration + 1}/max. {max_iterations_slope} ***")
            if feedback:
                feedback.pushInfo(f"[GetConstantSlopeLine] *** Slope Iteration {iteration + 1}/max. {max_iterations_slope} ***")

            # --- Temporary file paths ---
            cost_raster_path = os.path.join(self.temp_directory, f"cost_iter_{iteration}.tif")
            source_raster_path = os.path.join(self.temp_directory, f"source_iter_{iteration}.tif")
            accum_raster_path = os.path.join(self.temp_directory, f"accum_iter_{iteration}.tif")
            backlink_raster_path = os.path.join(self.temp_directory, f"backlink_iter_{iteration}.tif")
            best_destination_raster_path = os.path.join(self.temp_directory, f"destination_best_iter_{iteration}.tif")
            pathway_raster_path = os.path.join(self.temp_directory, f"pathway_iter_{iteration}.tif")
            pathway_vector_path = os.path.join(self.temp_directory, f"pathway_iter_{iteration}.shp")

            print(f"[GetConstantSlopeLine] Create cost slope raster for iteration {iteration + 1}")
            # --- Create cost raster ---
            cost_raster_path = TopoDrainCore._create_slope_cost_raster(
                dtm_path=dtm_path,
                start_point=current_start_point,
                output_cost_raster_path=cost_raster_path,
                slope=slope,
                barrier_raster_path=barrier_raster_path
            )

            print(f"[GetConstantSlopeLine] Cost raster path: {cost_raster_path}")

            print(f"[GetConstantSlopeLine] Create source raster for iteration {iteration + 1}")
            # --- Create source raster ---
            source_raster_path = TopoDrainCore._create_source_raster(
                reference_raster_path=dtm_path,
                source_point=current_start_point,
                output_source_raster_path=source_raster_path
            )

            print(f"[GetConstantSlopeLine] Source raster path: {source_raster_path}")

            print(f"[GetConstantSlopeLine] Starting cost-distance analysis for iteration {iteration + 1}")
            # --- Run cost-distance analysis ---
            try:
                ret = self._execute_wbt(
                    'cost_distance',
                    feedback=feedback,
                    report_progress=False,  # Don't override main progress bar
                    source=source_raster_path,
                    cost=cost_raster_path,
                    out_accum=accum_raster_path,
                    out_backlink=backlink_raster_path
                )
                
                if ret != 0 or not os.path.exists(accum_raster_path) or not os.path.exists(backlink_raster_path):
                    raise RuntimeError(f"Cost distance analysis failed in iteration {iteration + 1}: WhiteboxTools returned {ret}")
            except Exception as e:
                # Check if cancellation was the cause
                if feedback and feedback.isCanceled():
                    feedback.reportError("Process cancelled by user during cost-distance analysis.")
                    raise RuntimeError('Process cancelled by user.')
                raise RuntimeError(f"Cost distance analysis failed in iteration {iteration + 1}: {e}")

            print(f"[GetConstantSlopeLine] Cost accum raster path: {accum_raster_path}")
            print(f"[GetConstantSlopeLine] Cost backlink raster path: {backlink_raster_path}")

            # Check for cancellation after cost-distance analysis
            if feedback and feedback.isCanceled():
                feedback.reportError("Operation cancelled by user")
                raise RuntimeError("Operation cancelled by user")

            print(f"[GetConstantSlopeLine] Selecting best destination cell for iteration {iteration + 1}")
            # --- Select best destination cell ---
            best_destination_raster_path, best_destination_point = TopoDrainCore._select_best_destination_cell(
                accum_raster_path=accum_raster_path,
                destination_raster_path=destination_raster_path,
                best_destination_raster_path=best_destination_raster_path
            )

            print(f"[GetConstantSlopeLine] Best destination raster path: {best_destination_raster_path}")

            print(f"[GetConstantSlopeLine] Tracing least-cost pathway for iteration {iteration + 1}")
            # --- Trace least-cost pathway ---
            try:
                ret = self._execute_wbt(
                    'cost_pathway',
                    feedback=feedback,
                    report_progress=False,  # Don't override main progress bar
                    destination=best_destination_raster_path,
                    backlink=backlink_raster_path,
                    output=pathway_raster_path
                )
                
                if ret != 0 or not os.path.exists(pathway_raster_path):
                    raise RuntimeError(f"Cost pathway analysis failed in iteration {iteration + 1}")
            except Exception as e:
                # Check if cancellation was the cause
                if feedback and feedback.isCanceled():
                    feedback.reportError("Process cancelled by user during pathway tracing.")
                    raise RuntimeError('Process cancelled by user.')
                raise RuntimeError(f"Cost pathway analysis failed in iteration {iteration + 1}: {e}")

            # Check for cancellation after pathway analysis
            if feedback and feedback.isCanceled():
                feedback.reportError("Operation cancelled by user")
                raise RuntimeError("Operation cancelled by user")

            print(f"[GetConstantSlopeLine] pathway raster path: {pathway_raster_path}")

            # --- Set correct NoData value for pathway raster --- ## Noch prüfen, ob das notwendig ist
            # Read NoData value from backlink raster using GDAL
            backlink_ds = gdal.Open(backlink_raster_path, gdal.GA_ReadOnly)
            if backlink_ds is not None:
                try:
                    backlink_band = backlink_ds.GetRasterBand(1)
                    nodata_value = backlink_band.GetNoDataValue()
                finally:
                    backlink_ds = None  # Close dataset
                
                # Set NoData value for pathway raster using GDAL
                pathway_ds = gdal.Open(pathway_raster_path, gdal.GA_Update)
                if pathway_ds is not None:
                    try:
                        pathway_band = pathway_ds.GetRasterBand(1)
                        if nodata_value is not None:
                            pathway_band.SetNoDataValue(nodata_value)
                    finally:
                        pathway_ds = None  # Close dataset

            print(f"[GetConstantSlopeLine] backlink raster path: {backlink_raster_path}")

            print(f"[GetConstantSlopeLine] Converting pathway raster to LineString for iteration {iteration + 1}")
            # --- Convert to LineString ---
            line_segment = self._raster_to_linestring_wbt(
                pathway_raster_path, 
                snap_to_start_point=current_start_point, 
                snap_to_endpoint=best_destination_point, 
                output_vector_path=pathway_vector_path,
                feedback=feedback
            )

            print(f"[GetConstantSlopeLine] pathway vector path: {pathway_vector_path}")

            if line_segment is None:
                print(f"[GetConstantSlopeLine] No valid line found in iteration {iteration + 1}")
                break

            # --- Analyze distance deviation and find cut point ---
            print(f"[GetConstantSlopeLine] Analyzing slope deviation for iteration {iteration + 1}")
            cut_point = self._analyze_slope_deviation_and_cut(
                line=line_segment, 
                start_point=current_start_point, 
                expected_slope=slope,
                slope_deviation_threshold=slope_deviation_threshold
                )

            # Check if the line segment reaches the destination
            if cut_point:
                # Read the destination raster and get the value at the cut point using GDAL
                dest_ds = gdal.Open(destination_raster_path, gdal.GA_ReadOnly)
                if dest_ds is not None:
                    try:
                        # Get raster information
                        rows = dest_ds.RasterYSize
                        cols = dest_ds.RasterXSize
                        geotransform = dest_ds.GetGeoTransform()
                        
                        if geotransform is not None:
                            # Convert cut point coordinates to pixel indices using utility function
                            pixel_coords = TopoDrainCore._coords_to_pixel_indices([cut_point.coords[0]], geotransform)
                            col, row = pixel_coords[0]
                            
                            if 0 <= row < rows and 0 <= col < cols:
                                # Read destination data
                                dest_band = dest_ds.GetRasterBand(1)
                                dest_data = dest_band.ReadAsArray()
                                if dest_data is not None:
                                    dest_value = dest_data[row, col]
                                else:
                                    dest_value = None
                            else:
                                dest_value = None
                        else:
                            dest_value = None
                    finally:
                        dest_ds = None  # Close dataset
                else:
                    dest_value = None
                    
                if dest_value == 1:
                    print(f"[GetConstantSlopeLine] Cut point {cut_point} is a destination cell")
                    reached_destination = True
                else:
                    reached_destination = False

            # Check if last iteration reached
            last_iteration = (iteration == max_iterations_slope - 1)
            
            if last_iteration and cut_point is not None:
                warnings.warn(f"[GetConstantSlopeLine] Warning: Last iteration reached without finding fully valid line segment")
                print(f"[GetConstantSlopeLine] *** End of Slope iteration {iteration+1} ***")
                if feedback:
                    feedback.pushWarning(f"[GetConstantSlopeLine] Warning: Last iteration reached without finding fully valid line segment")
                    feedback.pushInfo(f"[GetConstantSlopeLine] *** End of Slope iteration {iteration+1} ***")

            if last_iteration or cut_point is None or reached_destination:
                # If we are at the last iteration or reached destination, we can add fully line segment instead of doing another iteration
                if accumulated_line_coords:
                    # Skip first coordinate to avoid duplication
                    accumulated_line_coords.extend(line_segment.coords[1:])
                else:
                    accumulated_line_coords.extend(line_segment.coords)
                print(f"[GetConstantSlopeLine] *** End of Slope iteration {iteration+1} ***")
                if feedback:
                    feedback.pushInfo(f"[GetConstantSlopeLine] *** End of Slope iteration {iteration+1} ***")
                break

            else:
                # Cut the line at the identified point
                cut_line = self._cut_line_at_point(line_segment, cut_point)
                if cut_line:
                    if accumulated_line_coords:
                        # Skip first coordinate to avoid duplication
                        accumulated_line_coords.extend(cut_line.coords[1:])
                    else:
                        accumulated_line_coords.extend(cut_line.coords)

                    # Store the cut point for the next iteration
                    current_start_point = cut_point
                    print(f"[GetConstantSlopeLine] *** End of Slope iteration {iteration+1} ***")
                    print(f"[GetConstantSlopeLine] Continuing from cut point: {cut_point}")
                    if feedback:
                        feedback.pushInfo(f"[GetConstantSlopeLine] *** End of Slope iteration {iteration+1} ***")
                    iteration += 1
                else:
                    # If cutting failed, use the whole segment
                    warnings.warn(f"[GetConstantSlopeLine] Warning: Cutting failed, using whole line segment")
                    print(f"[GetConstantSlopeLine] *** End of Slope iteration {iteration+1} ***")
                    if feedback:
                        feedback.pushWarning(f"[GetConstantSlopeLine] Warning: Cutting failed, using whole line segment")
                        feedback.pushInfo(f"[GetConstantSlopeLine] *** End of Slope iteration {iteration+1} ***")
                    if accumulated_line_coords:
                        # Skip first coordinate to avoid duplication
                        accumulated_line_coords.extend(line_segment.coords[1:])
                    else:
                        accumulated_line_coords.extend(line_segment.coords)
                    break
 
        # --- Combine all segments into final line ---
        if not accumulated_line_coords or len(accumulated_line_coords) < 2:
            warnings.warn("[GetConstantSlopeLine] Warning: No valid line segments could be extracted.")
            if feedback:
                feedback.pushWarning("[GetConstantSlopeLine] Warning: No valid line segments could be extracted.")                
            return None
        else:
            print(f"[GetConstantSlopeLine] Completed after {iteration + 1} iterations")
            if feedback:
                feedback.pushInfo(f"[GetConstantSlopeLine] Completed after {iteration + 1} iterations") 

        final_line = LineString(accumulated_line_coords)

        print("[GetConstantSlopeLine] Smoothing the resulting line")
        # --- Optional smoothing ---
        final_line = TopoDrainCore._smooth_linestring(final_line, sigma=1.0)

        print(f"[GetConstantSlopeLine] Finished processing") 
        if feedback:
            feedback.pushInfo(f"[GetConstantSlopeLine] Finished processing")

        return final_line

    def _get_iterative_constant_slope_line(
        self,
        dtm_path: str,
        start_point: Point,
        destination_raster_path: str,
        slope: float,
        barrier_raster_path: str,
        initial_barrier_value: int = None,
        max_iterations_barrier: int = 30,
        slope_deviation_threshold: float = 0.2,
        max_iterations_slope: int = 30,
        feedback=None
    ) -> LineString:
        """
        Trace lines with constant slope starting from a given point using a cost-distance approach based on slope deviation.
        Unlike _get_constant_slope_line, this function allows barriers to act as temporary destinations.
        The barrier raster needs in this case different values (1, 2, ...) for different barrier features.

        This function creates a cost raster that penalizes deviation from the desired slope,
        runs a least-cost-path analysis using WhiteboxTools, and returns the resulting line.

        Args:
            dtm_path (str): Path to the digital terrain model (GeoTIFF).
            start_point (Point): Starting point of the constant slope line (not on a barrier, this is handled in get_constant_slope_lines).
            destination_raster_path (str): Path to the binary raster indicating destination cells (1 = destination).
            slope (float): Desired slope for the line (e.g., 0.01 for 1% downhill or -0.01 for uphill).
            barrier_raster_path (str): Path to a raster of cells that should not be crossed (different barriers have unique values 1, 2, ...).
            initial_barrier_value (int, optional): Initial barrier value to start from. Default None.
            max_iterations_barrier (int): Maximum number of iterations for iterative tracing (nr. of times barriers used as temporary destinations). Default 10.
            slope_deviation_threshold (float): Maximum allowed relative deviation from expected slope (0.0-1.0, e.g., 0.2 for 20% deviation before line cutting).
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting.

        Returns:
            LineString: Least-cost slope path as a Shapely LineString, or None if no path found.
        """
        if self.wbt is None:
            raise RuntimeError("WhiteboxTools not initialized. Check WhiteboxTools configuration: QGIS settings -> Options -> Processing -> Provider -> WhiteboxTools -> WhiteboxTools executable.")
        
        current_iteration = 0
        current_start_point = start_point
        current_barrier_value = initial_barrier_value
        accumulated_line_coords = []
    
        # Read destination raster data using GDAL
        dest_ds = gdal.Open(destination_raster_path, gdal.GA_ReadOnly)
        if dest_ds is None:
            raise RuntimeError(f"Cannot open destination raster: {destination_raster_path}.{self._get_gdal_error_message()}")
            
        try:
            # Get destination raster information
            dest_rows = dest_ds.RasterYSize
            dest_cols = dest_ds.RasterXSize
            dest_geotransform = dest_ds.GetGeoTransform()
            dest_projection = dest_ds.GetProjection()
            dest_srs = dest_ds.GetSpatialRef()
            
            # Read destination data
            dest_band = dest_ds.GetRasterBand(1)
            orig_dest_data = dest_band.ReadAsArray().copy()  # create a copy to avoid modifying the original raster
            if orig_dest_data is None:
                raise RuntimeError(f"Cannot read destination raster data: {destination_raster_path}.{self._get_gdal_error_message()}")
                
        finally:
            dest_ds = None  # Close destination dataset
        
        # Read barrier raster data using GDAL
        if barrier_raster_path is None:
            raise RuntimeError("Barrier raster path is None. _get_iterative_constant_slope_line requires a valid barrier raster path.")
            
        barrier_ds = gdal.Open(barrier_raster_path, gdal.GA_ReadOnly)
        if barrier_ds is None:
            raise RuntimeError(f"Cannot open barrier raster: {barrier_raster_path}.{self._get_gdal_error_message()}")
            
        try:
            # Get barrier raster information
            barrier_rows = barrier_ds.RasterYSize
            barrier_cols = barrier_ds.RasterXSize
            barrier_geotransform = barrier_ds.GetGeoTransform()
            barrier_projection = barrier_ds.GetProjection()
            barrier_srs = barrier_ds.GetSpatialRef()
            
            # Read barrier data
            barrier_band = barrier_ds.GetRasterBand(1)
            orig_barrier_data = barrier_band.ReadAsArray().copy()  # create a copy to avoid modifying the original raster
            if orig_barrier_data is None:
                raise RuntimeError(f"Cannot read barrier raster data: {barrier_raster_path}.{self._get_gdal_error_message()}")
            
        finally:
            barrier_ds = None  # Close barrier dataset
        
        while current_iteration < max_iterations_barrier:
            print(f"[IterativeConstantSlopeLine] *** Barrier Iteration {current_iteration + 1}/max. {max_iterations_barrier} ***")
            if feedback:
                feedback.pushInfo(f"[IterativeConstantSlopeLine] *** Barrier Iteration {current_iteration + 1}/max. {max_iterations_barrier} ***")
                feedback.pushInfo("*for more information see in Python Console")
                
            # Debug: Print start point and extracted value
            print(f"[IterativeConstantSlopeLine] Start point: {current_start_point.wkt}")
            print(f"[IterativeConstantSlopeLine] Start barrier value: {current_barrier_value}")

            print(f"[IterativeConstantSlopeLine] Creating working rasters for _get_constant_slope_line...")
            # --- Create barrier raster for _get_constant_slope_line ---
            # Use dtype information from GDAL data
            
            if current_barrier_value is not None:
                working_barrier_data = np.where(orig_barrier_data == current_barrier_value, 1, 0).astype(orig_barrier_data.dtype) # current barrier act as barrier and not as destination
                working_destination_data = np.where((orig_dest_data == 1) | ((orig_barrier_data >= 1) & (orig_barrier_data != current_barrier_value)), 1, 0).astype(orig_dest_data.dtype)  # all other barriers act as temporary destinations except the current one
            else:
                working_barrier_data = None # all barriers acting as temporary destinations and not as barriers
                working_destination_data = np.where((orig_dest_data == 1) | (orig_barrier_data >= 1), 1, 0).astype(orig_dest_data.dtype) 
                
            # Save barrier mask using GDAL
            if working_barrier_data is not None:
                working_barrier_raster_path = os.path.join(self.temp_directory, f"barrier_iter_{current_iteration}.tif")
                
                # Create barrier raster using GDAL
                driver = gdal.GetDriverByName('GTiff')
                if driver is None:
                    raise RuntimeError(f"GTiff driver not available.{self._get_gdal_error_message()}")
                    
                creation_options = ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
                barrier_out_ds = driver.Create(working_barrier_raster_path, barrier_cols, barrier_rows, 1, gdal.GDT_Byte, 
                                              options=creation_options)
                if barrier_out_ds is None:
                    raise RuntimeError(f"Failed to create barrier raster: {working_barrier_raster_path}.{self._get_gdal_error_message()}")
                    
                try:
                    barrier_out_ds.SetGeoTransform(barrier_geotransform)
                    if barrier_srs is not None:
                        barrier_out_ds.SetSpatialRef(barrier_srs)
                    elif barrier_projection:
                        barrier_out_ds.SetProjection(barrier_projection)
                        
                    barrier_out_band = barrier_out_ds.GetRasterBand(1)
                    barrier_out_band.WriteArray(working_barrier_data.astype(np.uint8))
                    barrier_out_band.SetNoDataValue(0)
                    
                finally:
                    barrier_out_ds = None  # Close dataset
                    
                print(f"[IterativeConstantSlopeLine] Working barrier raster created at {working_barrier_raster_path}")
            else:
                working_barrier_raster_path = None
                print("[IterativeConstantSlopeLine] No working barrier raster created (all barriers act as temporary destinations).")
                        
            working_destination_raster_path = os.path.join(self.temp_directory, f"destination_iter_{current_iteration}.tif")
            
            # Save destination mask using GDAL
            driver = gdal.GetDriverByName('GTiff')
            if driver is None:
                raise RuntimeError(f"GTiff driver not available.{self._get_gdal_error_message()}")
                
            creation_options = ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
            dest_out_ds = driver.Create(working_destination_raster_path, dest_cols, dest_rows, 1, gdal.GDT_Byte, 
                                       options=creation_options)
            if dest_out_ds is None:
                raise RuntimeError(f"Failed to create destination raster: {working_destination_raster_path}.{self._get_gdal_error_message()}")
                
            try:
                dest_out_ds.SetGeoTransform(dest_geotransform)
                if dest_srs is not None:
                    dest_out_ds.SetSpatialRef(dest_srs)
                elif dest_projection:
                    dest_out_ds.SetProjection(dest_projection)
                    
                dest_out_band = dest_out_ds.GetRasterBand(1)
                dest_out_band.WriteArray(working_destination_data.astype(np.uint8))
                dest_out_band.SetNoDataValue(0)
                
            finally:
                dest_out_ds = None  # Close dataset
                
            print(f"[IterativeConstantSlopeLine] Working destination raster created at {working_destination_raster_path}")

            # Call _get_constant_slope_line with current parameters
            print(f"[IterativeConstantSlopeLine] Tracing from point {current_start_point}")
                
            line_segment = self._get_constant_slope_line(
                dtm_path=dtm_path,
                start_point=current_start_point,
                destination_raster_path=working_destination_raster_path,
                slope=slope,
                barrier_raster_path=working_barrier_raster_path,
                slope_deviation_threshold=slope_deviation_threshold,
                max_iterations_slope=max_iterations_slope,
                feedback=feedback
            )

            # Check if a line segment was found
            if line_segment is None:
                warnings.warn(f"[IterativeConstantSlopeLine] Warning: No line found in iteration {current_iteration + 1}")
                if feedback:
                    feedback.pushWarning(f"[IterativeConstantSlopeLine] Warning: No line found in iteration {current_iteration + 1}")
                break

            line_coords = list(line_segment.coords)
            # Check if endpoint is on original (final) destination
            endpoint = Point(line_coords[-1])
            print(f"[IterativeConstantSlopeLine] Endpoint iteration {current_iteration + 1}: {endpoint.wkt}")
            final_destination_found = False
            # Convert endpoint coordinates to pixel indices using GDAL geotransform
            pixel_coords = TopoDrainCore._coords_to_pixel_indices([endpoint.coords[0]], dest_geotransform)
            end_col, end_row = pixel_coords[0]
            
            if 0 <= end_row < orig_dest_data.shape[0] and 0 <= end_col < orig_dest_data.shape[1]:
                if orig_dest_data[end_row, end_col] == 1:
                    print(f"[IterativeConstantSlopeLine] Reached final destination in iteration {current_iteration + 1}")
                    final_destination_found = True
                else:
                    print(f"[IterativeConstantSlopeLine] Endpoint not on destination in iteration {current_iteration + 1}, checking barriers.")
            
            last_iteration = (current_iteration == max_iterations_barrier - 1)
            if not final_destination_found and last_iteration:
                warnings.warn("[IterativeConstantSlopeLine] Warning: Maximum iterations reached without finding a fully valid line")
                if feedback:
                    feedback.pushWarning("[IterativeConstantSlopeLine] Warning: Maximum iterations reached without finding a fully valid line")                    

            if final_destination_found or last_iteration: 
                # If we reached the last iteration, add this final segment to avoid cutting in the last iteration:
                # If we reached the final destination, add this final segment and stop
                if accumulated_line_coords:
                    # Skip first coordinate to avoid duplication
                    accumulated_line_coords.extend(line_segment.coords[1:])
                else:
                    accumulated_line_coords.extend(line_segment.coords)
                
                print(f"[IterativeConstantSlopeLine] *** End of Barrier iteration {current_iteration + 1} ***") 
                if feedback:
                    feedback.pushInfo(f"[IterativeConstantSlopeLine] *** End of Barrier iteration {current_iteration + 1} ***")
                break

            else: 
                # Get barrier value at endpoint for next iteration using GDAL geotransform
                pixel_coords = TopoDrainCore._coords_to_pixel_indices([endpoint.coords[0]], barrier_geotransform)
                barrier_end_col, barrier_end_row = pixel_coords[0]
                
                if 0 <= barrier_end_row < barrier_rows and 0 <= barrier_end_col < barrier_cols:
                    end_barrier_value = int(orig_barrier_data[barrier_end_row, barrier_end_col])
                else:
                    end_barrier_value = None
                print(f"[IterativeConstantSlopeLine] Iteration reached barrier: {end_barrier_value}")

                if end_barrier_value is not None:
                    # Get start point for next iteration next to the barrier (back where the line came from)
                    next_start_point = TopoDrainCore._get_linedirection_start_point(
                        barrier_raster_path=barrier_raster_path,
                        line_geom=line_segment,
                        max_offset=10,  # adjust as needed
                        reverse=True  # always go backward were the line came from
                    )
                else:
                    next_start_point = endpoint  # if no barrier, continue from endpoint (should actually never happen in this case)

                print(f"[IterativeConstantSlopeLine] Start point for next iteration: {next_start_point.wkt}")

                print(f"[IterativeConstantSlopeLine] Adjusting line segment to new start point")
                # Adjust line_segment to only go up to the new next_start_point (not to the endpoint)
                if next_start_point != endpoint:
                    line_segment = self._cut_line_at_point(line_segment, next_start_point)

                # Add line segment to accumulated coordinates for continuing iterations
                if accumulated_line_coords:
                    # Skip first coordinate to avoid duplication
                    accumulated_line_coords.extend(line_segment.coords[1:])
                else:
                    accumulated_line_coords.extend(line_segment.coords)

            # Prepare for next iteration
            current_barrier_value = end_barrier_value if end_barrier_value is not None else None  # Update barrier value for next iteration
            current_start_point = next_start_point  # Update start point for next iteration
            print(f"[IterativeConstantSlopeLine] *** End of Barrier iteration {current_iteration + 1} ***")
            if feedback:
                feedback.pushInfo(f"[IterativeConstantSlopeLine] *** End of Barrier iteration {current_iteration + 1} ***")
            current_iteration += 1

        if len(accumulated_line_coords) >= 2:
            line = LineString(accumulated_line_coords)
            print(f"[IterativeConstantSlopeLine] Completed after {current_iteration + 1} iterations")
            print(f"[IterativeConstantSlopeLine] Finished processing")
            if feedback:
                feedback.pushInfo(f"[IterativeConstantSlopeLine] Completed after {current_iteration + 1} iterations")
                feedback.pushInfo(f"[IterativeConstantSlopeLine] Finished processing")
            return line
        else:
            warnings.warn("[IterativeConstantSlopeLine] Warning: No valid line could be created")
            if feedback:
                feedback.pushWarning("[IterativeConstantSlopeLine] Warning: No valid line could be created")
            return None


    def _adjust_constant_slope_after(
        self,
        dtm_path: str,
        input_line: LineString,
        change_after: float,
        slope_after: float,
        destination_raster_path: str,
        barrier_raster_path: str,
        allow_barriers_as_temp_destination: bool = False,
        max_iterations_barrier: int = 30,
        slope_deviation_threshold: float = 0.2,
        max_iterations_slope: int = 20,
        feedback=None,
    ) -> gpd.GeoDataFrame:
        """
        Modify constant slope lines by changing to a secondary slope after a specified distance.
        
        This function splits each input line at a specified fraction of its length and continues 
        with a new slope from that point using the get_constant_slope_lines function.
        
        Args:
            dtm_path (str): Path to the digital terrain model (GeoTIFF).
            input_line (LineString): Input constant slope line to modify.
            change_after (float): Fraction of line length where slope changes (0.0-1.0, e.g., 0.5 = from halfway).
            slope_after (float): New slope to apply after the change point (e.g., 0.01 for 1% downhill).
            destination_raster_path (str): Path to the destination raster for the new slope sections.
            barrier_raster_path (str): Path to the barrier raster to avoid. Use barrier (unique) value raster for iterative tracing
            allow_barriers_as_temp_destination (bool): If True, barriers are included as temporary destinations for iterative tracing.
            max_iterations_barrier (int): Maximum number of iterations when using barriers as temporary destinations.
            slope_deviation_threshold (float): Maximum allowed relative deviation from expected slope (0.0-1.0, e.g., 0.2 for 20% deviation before line cutting).
            max_iterations_slope (int): Maximum number of iterations for line refinement.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting.
            
        Returns:
            LineString: Modified line with secondary slopes applied.
        """
        print(f"[AdjustConstantSlopeAfter] Starting adjustment of input line...")
        if feedback:
            feedback.pushInfo(f"[AdjustConstantSlopeAfter] Starting adjustment of input line...")
            feedback.pushInfo(f"*for more information see in Python Console")
            
        # Validate change_after parameter
        if not (0.0 < change_after < 1.0):
            raise ValueError("change_after must be between 0.0 and 1.0")
        
        # Check for potential configuration issues
        if allow_barriers_as_temp_destination and not barrier_raster_path:
            warnings.warn("[AdjustConstantSlopeAfter] Warning: allow_barriers_as_temp_destination=True but no barrier_raster_path provided. This setting will have no effect.")

        # Phase 1: Process all lines to create first parts and collect start points for second parts
        if feedback:
            if feedback.isCanceled():
                feedback.reportError("Slope adjustment was cancelled by user")
                raise RuntimeError("Operation cancelled by user")

        print(f"[AdjustConstantSlopeAfter] Phase 1: Create first part of line...")

        # Handle MultiLineString by stitching into a single LineString
        if isinstance(input_line, LineString):
            line_geom = input_line
        elif isinstance(input_line, MultiLineString):
            # Converting MultiLineString to LineString (reduced logging)
            line_geom = TopoDrainCore._merge_lines_by_distance(input_line)
        else:
            warnings.warn(f"[AdjustConstantSlopeAfter] Skipping unsupported geometry type: {type(input_line)}")
            # Keep original line for unsupported geometry types
            return input_line
        
        # Calculate split point at specified fraction of line length
        line_length = line_geom.length
        split_distance = line_length * change_after
        
        # Check if split distance is valid
        if split_distance <= 0 or split_distance >= line_length:
            warnings.warn(f"[AdjustConstantSlopeAfter] Invalid split distance for input line, keeping original line")
            return input_line

        # Create first part of line (from start to split point)
        # Get coordinates up to split point
        coords = list(line_geom.coords)
        first_part_coords = []
        remaining_distance = split_distance
        
        for i in range(len(coords) - 1):
            start_pt = Point(coords[i])
            end_pt = Point(coords[i + 1])
            segment_length = start_pt.distance(end_pt)
            first_part_coords.append(coords[i])
            if remaining_distance <= segment_length:
                # Interpolate the exact split point on this segment
                if remaining_distance > 0:
                    fraction = remaining_distance / segment_length
                    split_x = coords[i][0] + fraction * (coords[i + 1][0] - coords[i][0])
                    split_y = coords[i][1] + fraction * (coords[i + 1][1] - coords[i][1])
                    first_part_coords.append((split_x, split_y))
                break
            else:
                remaining_distance -= segment_length
        
        # Create first part of the line
        if len(first_part_coords) >= 2:
            first_part_line = LineString(first_part_coords)
            # Create start point for second part (the split point)
            start_point_second = Point(first_part_coords[-1])
        else:
            warnings.warn(f"[AdjustConstantSlopeAfter] Not enough coordinates to form first part of line, keeping original line")
            return  input_line  # Not enough coordinates to form a line, keep original
                
        # Phase 2: Trace second part with new slope
        if allow_barriers_as_temp_destination and barrier_raster_path:
            second_part_line = self._get_iterative_constant_slope_line(
                dtm_path=dtm_path,
                start_point=start_point_second,
                destination_raster_path=destination_raster_path,
                slope=slope_after,
                barrier_raster_path=barrier_raster_path,  # Use barrier value raster for iterative tracing
                initial_barrier_value=None, # Assume split point is not on a barrier, so start without barrier value
                max_iterations_barrier=max_iterations_barrier,
                slope_deviation_threshold=slope_deviation_threshold,
                max_iterations_slope=max_iterations_slope,
                feedback=feedback
            )
        else:
            second_part_line = self._get_constant_slope_line(
                dtm_path=dtm_path,
                start_point=start_point_second,
                destination_raster_path=destination_raster_path,
                slope=slope_after,
                barrier_raster_path=barrier_raster_path,  # Use binary barrier raster for simple tracing
                slope_deviation_threshold=slope_deviation_threshold,
                max_iterations_slope=max_iterations_slope,
                feedback=feedback
                )
        
        if second_part_line is None:
            # If no second part could be traced, return only the first part (reduced logging)
            warnings.warn(f"[AdjustConstantSlopeAfter] Warning: No second part could be traced, keeping original line")
            return input_line  # Not enough coordinates to form a line, keep original

        if feedback:
            if feedback.isCanceled():
                feedback.reportError("Slope adjustment was cancelled by user")
                raise RuntimeError("Operation cancelled by user")
            
        # Phase 3: Combine first and second part
        print(f"[AdjustConstantSlopeAfter] Phase 3: Combining first and second part...")
        
        # Merge coordinates, avoiding duplication of the split point
        first_coords = list(first_part_line.coords)
        second_coords = list(second_part_line.coords)
                    
        # Remove duplicate split point if it exists
        if len(second_coords) > 0 and first_coords[-1] == second_coords[0]:
            combined_coords = first_coords + second_coords[1:]
        else:
            combined_coords = first_coords + second_coords
                    
        combined_line = LineString(combined_coords)

        return combined_line



    def get_constant_slope_lines(
        self,
        dtm_path: str,
        start_points: gpd.GeoDataFrame,
        destination_features: list[gpd.GeoDataFrame],
        slope: float = 0.01,
        perimeter: gpd.GeoDataFrame = None,
        barrier_features: list[gpd.GeoDataFrame] = None,
        allow_barriers_as_temp_destination: bool = False,
        max_iterations_barrier: int = 30,
        slope_deviation_threshold: float = 0.2,
        max_iterations_slope: int = 30,
        feedback=None,
    ) -> gpd.GeoDataFrame:
        """
        Trace lines with constant slope starting from given points using a cost-distance approach
        based on slope deviation. This function handles point classification and creates offset points
        when start points overlap with barriers or perimeter features.

        Process:
        1. Classify start points based on their location (barrier features, perimeter features, neutral)
        2. Create orthogonal offset points for points that are on barriers or perimeter lines
        3. Trace constant slope lines from each processed start point to destination features
        4. Support barriers as temporary destinations for iterative tracing if desired

        Args:
            dtm_path (str): Path to the digital terrain model (GeoTIFF).
            start_points (gpd.GeoDataFrame): Starting points for slope line tracing.
            destination_features (list[gpd.GeoDataFrame]): List of destination features to trace toward.
            slope (float): Desired slope for the lines (e.g., 0.01 for 1% downhill). Default 0.01.
            perimeter (gpd.GeoDataFrame, optional): Optional perimeter polygon to limit the extent of traced lines. Acts as barriers.
            barrier_features (list[gpd.GeoDataFrame], optional): List of barrier features to avoid during tracing.
            allow_barriers_as_temp_destination (bool): If True, barriers are included as temporary destinations for iterative tracing. Default False.
            max_iterations_barrier (int): Maximum number of iterations for iterative tracing when allowing barriers as temporary destinations. Default 30.
            max_iterations_slope (int): Maximum number of iterations for line refinement (1-50, higher values allow more complex paths). Default 30.
            slope_deviation_threshold (float): Maximum allowed relative deviation from expected slope (0.0-1.0, e.g., 0.2 for 20% deviation before line cutting). Default 0.2.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting and logging.
            
        Returns:
            gpd.GeoDataFrame: Traced constant slope lines with geometry column containing LineString objects.
        """

        if feedback:
            feedback.pushInfo(f"[GetConstantSlopeLines] Starting constant slope tracing from {len(start_points)} start points...")
            feedback.setProgress(0)
            if feedback.isCanceled():
                feedback.reportError("[GetConstantSlopeLines] Constant slope line creation was cancelled by user")
                raise RuntimeError("Operation cancelled by user")
        else:
            print(f"[GetConstantSlopeLines] Starting constant slope tracing from {len(start_points)} start points...")
            print("[GetConstantSlopeLines] Progress: 0%")

        # Read DTM raster metadata information (used multiple times below)
        dtm_ds = gdal.Open(dtm_path, gdal.GA_ReadOnly)
        if dtm_ds is None:
            raise RuntimeError(f"Cannot open DTM raster: {dtm_path}.{self._get_gdal_error_message()}")
        try:
            dtm_geotransform = dtm_ds.GetGeoTransform()
            if dtm_geotransform is None:
                raise RuntimeError(f"Cannot get geotransform from DTM raster: {dtm_path}.{self._get_gdal_error_message()}")
            dtm_projection = dtm_ds.GetProjection()
            dtm_rows = dtm_ds.RasterYSize
            dtm_cols = dtm_ds.RasterXSize
        finally:
            dtm_ds = None

        # Prepare destination, barrier and perimeter features for rasterization
        # If perimeter is a polygon, use its boundary for rasterization
        if perimeter is not None and not perimeter.empty:
            if perimeter.geom_type.isin(["Polygon", "MultiPolygon"]).any():
                perimeter_lines = perimeter.copy()
                perimeter_lines["geometry"] = perimeter_lines.boundary
            else:
                if feedback:
                    feedback.reportError("Perimeter must be a GeoDataFrame with Polygon or MultiPolygon geometry")  
                raise ValueError("Perimeter must be a GeoDataFrame with Polygon or MultiPolygon geometry")

        # Create unique value raster masks for barrier features to classify start points
        barrier_unique_raster_path = os.path.join(self.temp_directory, "barrier_unique.tif")
        if barrier_features:
            barrier_processed_for_classification = self._features_to_single_linestring(barrier_features)
            barrier_unique_raster_path, barrier_id_to_geom = self._vector_to_mask_raster([barrier_processed_for_classification], dtm_path, output_path=barrier_unique_raster_path, unique_values=True, flatten_lines=False, buffer_lines=True)
        else:
            barrier_unique_raster_path, barrier_id_to_geom = self._vector_to_mask_raster([], dtm_path, output_path=barrier_unique_raster_path, unique_values=True, flatten_lines=False, buffer_lines=True)
        barrier_unique_ds = gdal.Open(barrier_unique_raster_path, gdal.GA_ReadOnly)
        if barrier_unique_ds is None:
            raise RuntimeError(f"Cannot open barrier lines raster: {barrier_unique_raster_path}.{self._get_gdal_error_message()}")
        try:
            barrier_unique_band = barrier_unique_ds.GetRasterBand(1)
            barrier_unique_mask = barrier_unique_band.ReadAsArray()
            if barrier_unique_mask is None:
                raise RuntimeError(f"Cannot read barrier mask data from: {barrier_unique_raster_path}.{self._get_gdal_error_message()}")
        finally:
            barrier_unique_ds = None
        
        # Combine perimeter features with barrier mask if perimeter is provided
        if perimeter is not None and not perimeter.empty:
            perimeter_unique_raster_path = os.path.join(self.temp_directory, "perimeter_unique.tif")
            perimeter_unique_raster_path, perimeter_id_to_geom = self._vector_to_mask_raster([perimeter_lines], dtm_path, output_path=perimeter_unique_raster_path, unique_values=True, flatten_lines=True, buffer_lines=True)
            perimeter_unique_ds = gdal.Open(perimeter_unique_raster_path, gdal.GA_ReadOnly)
            if perimeter_unique_ds is None:
                raise RuntimeError(f"Cannot open perimeter raster: {perimeter_unique_raster_path}.{self._get_gdal_error_message()}")
            try:
                perimeter_unique_band = perimeter_unique_ds.GetRasterBand(1)
                perimeter_unique_mask = perimeter_unique_band.ReadAsArray()
                if perimeter_unique_mask is None:
                    raise RuntimeError(f"Cannot read perimeter mask data from: {perimeter_unique_raster_path}.{self._get_gdal_error_message()}")
                
                # Combine perimeter with barrier mask using offset values to avoid conflicts
                barrier_max_value = np.max(barrier_unique_mask) # get max value in barrier mask for offsetting perimeter IDs
                perimeter_nonzero = perimeter_unique_mask > 0
                if np.any(perimeter_nonzero):
                    # Offset perimeter values and add them to barrier mask
                    offset_perimeter_values = perimeter_unique_mask[perimeter_nonzero] + barrier_max_value
                    barrier_unique_mask[perimeter_nonzero] = offset_perimeter_values
                    
                    # Update the ID mapping dictionary with offset values and combine with barrier mapping
                    for orig_id, geom in perimeter_id_to_geom.items():
                        barrier_id_to_geom[orig_id + barrier_max_value] = geom
            finally:
                perimeter_unique_ds = None

        # Create binary masks here
        barrier_binary_mask = (barrier_unique_mask > 0).astype(np.uint8) # perimeter included in barrier unique mask
        barrier_binary_raster_path = os.path.join(self.temp_directory, f"barrier_binary.tif")
        driver = gdal.GetDriverByName('GTiff')
        barrier_binary_ds = driver.Create(barrier_binary_raster_path, dtm_cols, dtm_rows, 1, gdal.GDT_Byte)
        try:
            barrier_binary_ds.SetGeoTransform(dtm_geotransform)
            barrier_binary_ds.SetProjection(dtm_projection)
            barrier_binary_band = barrier_binary_ds.GetRasterBand(1)
            barrier_binary_band.WriteArray(barrier_binary_mask)
        finally:
            barrier_binary_ds = None
        # Process destination features and create binary destination mask
        destination_processed = self._features_to_single_linestring(destination_features)
        destination_binary_raster_path = os.path.join(self.temp_directory, "destination_binary.tif")
        destination_binary_raster_path = self._vector_to_mask_raster([destination_processed], dtm_path, output_path=destination_binary_raster_path, unique_values=False, flatten_lines=False, buffer_lines=True)
        destination_binary_ds = gdal.Open(destination_binary_raster_path, gdal.GA_ReadOnly)
        if destination_binary_ds is None:
            raise RuntimeError(f"Cannot open destination lines raster: {destination_binary_raster_path}.{self._get_gdal_error_message()}")
        try:
            destination_binary_band = destination_binary_ds.GetRasterBand(1)
            destination_binary_mask = destination_binary_band.ReadAsArray()
            if destination_binary_mask is None:
                raise RuntimeError(f"Cannot read destination mask data from: {destination_binary_raster_path}.{self._get_gdal_error_message()}")
        finally:
            destination_binary_ds = None

        # Create perimeter polygon binary mask for efficient point-in-polygon checking
        # Create binary mask for perimeter polygons (not boundaries)
        if perimeter is not None and not perimeter.empty:
            perimeter_polygon_raster_path = os.path.join(self.temp_directory, "perimeter_polygon_mask.tif")
            perimeter_polygon_raster_path = self._vector_to_mask_raster([perimeter], dtm_path, output_path=perimeter_polygon_raster_path, unique_values=False, flatten_lines=False, buffer_lines=False)
            perimeter_polygon_ds = gdal.Open(perimeter_polygon_raster_path, gdal.GA_ReadOnly)
            if perimeter_polygon_ds is None:
                raise RuntimeError(f"Cannot open perimeter polygon raster: {perimeter_polygon_raster_path}.{self._get_gdal_error_message()}")
            try:
                perimeter_polygon_band = perimeter_polygon_ds.GetRasterBand(1)
                perimeter_polygon_mask = perimeter_polygon_band.ReadAsArray()
                if perimeter_polygon_mask is None:
                        raise RuntimeError(f"Cannot read perimeter polygon mask data from: {perimeter_polygon_raster_path}.{self._get_gdal_error_message()}")
            finally:
                perimeter_polygon_ds = None
        else:
            perimeter_polygon_mask = None  # No perimeter polygon provided

        if feedback:
            feedback.pushInfo(f"[GetConstantSlopeLines] Raster masks created for barrier, destination and perimeter")
            feedback.setProgress(5)
        else:
            print(f"[GetConstantSlopeLines] Raster masks created for barrier, destination and perimeter")
            print("[GetConstantSlopeLines] Progress: 5%")

        # Classify start points and create updated start points with barrier_id attributes
        updated_start_points = []
        if feedback:
            feedback.pushInfo(f"[GetConstantSlopeLines] Classifying {len(start_points)} start points...")
        else:
            print(f"[GetConstantSlopeLines] Classifying {len(start_points)} start points...")
        
        for idx, row in start_points.iterrows():
            point = row.geometry

            # Get raster coordinates for the point using TopoDrainCore utility function
            pixel_coords = TopoDrainCore._coords_to_pixel_indices([point.coords[0]], dtm_geotransform)
            point_c, point_r = pixel_coords[0]
            
            # Check if point is within raster bounds
            if not (0 <= point_r < dtm_rows and 0 <= point_c < dtm_cols):
                if feedback:
                    feedback.pushWarning(f"[GetConstantSlopeLines] Warning: Start point {idx} is outside raster bounds, skipping")
                else:
                    warnings.warn(f"[GetConstantSlopeLines] Warning: Start point {idx} is outside raster bounds, skipping")
                continue
            
            # Check if point is on barrier/perimeter or destination mask
            barrier_value = int(barrier_unique_mask[point_r, point_c]) # perimeter included in barrier unique mask
            destination_value = int(destination_binary_mask[point_r, point_c])
                            
            if destination_value > 0:
                # Point is on destination - not allowed
                if feedback:
                    feedback.pushWarning(f"[GetConstantSlopeLines] Warning: Start point {idx} is on destination, skipping")
                else:
                    warnings.warn(f"[GetConstantSlopeLines] Warning: Start point {idx} is on destination, skipping")
                continue

            elif barrier_value > 0:
                # Point is on barrier line - create orthogonal offset points
                barrier_geom = barrier_id_to_geom.get(barrier_value)
                if barrier_geom is None:
                    if feedback:
                        feedback.pushWarning(f"[GetConstantSlopeLines] Warning: No barrier geometry found for ID {barrier_value}, skipping point {idx}")
                    else:
                        warnings.warn(f"[GetConstantSlopeLines] Warning: No barrier geometry found for ID {barrier_value}, skipping point {idx}")
                    continue
                # Get orthogonal offset points
                left_pt, right_pt = TopoDrainCore._get_orthogonal_directions_start_points(
                    barrier_raster_path=barrier_binary_raster_path,
                    point=point,
                    line_geom=barrier_geom
                )
                
                # Check if offset points are inside the perimeter polygon using raster mask
                if left_pt is not None:
                    # Get raster coordinates for the left point
                    left_pixel_coords = TopoDrainCore._coords_to_pixel_indices([left_pt.coords[0]], dtm_geotransform)
                    left_c, left_r = left_pixel_coords[0]
                    # Check if left point is within raster bounds and inside perimeter polygon
                    is_inside_perimeter = False
                    if 0 <= left_r < dtm_rows and 0 <= left_c < dtm_cols:
                        if perimeter_polygon_mask is not None:
                            is_inside_perimeter = perimeter_polygon_mask[left_r, left_c] > 0
                        else:
                            is_inside_perimeter = True  # Assume inside perimeter when no perimeter is provided
                    # Only add left point if it's inside perimeter
                    if is_inside_perimeter:
                        left_row = row.copy()
                        left_row.geometry = left_pt
                        left_row['barrier_id_key'] = barrier_value
                        updated_start_points.append(left_row)
                        if feedback:
                            feedback.pushInfo(f"[GetConstantSlopeLines] Left offset point ({left_pt}) for barrier point {idx} is inside perimeter, adding")
                        else:
                            print(f"[GetConstantSlopeLines] Left offset point ({left_pt}) for barrier point {idx} is inside perimeter, adding")
                    else:
                        if feedback:
                            feedback.pushInfo(f"[GetConstantSlopeLines] Left offset point ({left_pt}) for barrier point {idx} is outside perimeter, skipping")
                        else:
                            print(f"[GetConstantSlopeLines] Left offset point ({left_pt}) for barrier point {idx} is outside perimeter, skipping")
                
                if right_pt is not None:
                    # Get raster coordinates for the right point
                    right_pixel_coords = TopoDrainCore._coords_to_pixel_indices([right_pt.coords[0]], dtm_geotransform)
                    right_c, right_r = right_pixel_coords[0]
                    # Check if right point is within raster bounds and inside perimeter polygon
                    is_inside_perimeter = False
                    if 0 <= right_r < dtm_rows and 0 <= right_c < dtm_cols:
                        if perimeter_polygon_mask is not None:
                            is_inside_perimeter = perimeter_polygon_mask[right_r, right_c] > 0
                        else:
                            is_inside_perimeter = True  # Assume inside perimeter when no perimeter is provided
                    # Only add right point if it's inside perimeter
                    if is_inside_perimeter:
                        right_row = row.copy()
                        right_row.geometry = right_pt
                        right_row['barrier_id_key'] = barrier_value
                        updated_start_points.append(right_row)
                        if feedback:
                            feedback.pushInfo(f"[GetConstantSlopeLines] Right offset point ({right_pt}) for barrier point {idx} is inside perimeter, adding")
                        else:
                            print(f"[GetConstantSlopeLines] Right offset point ({right_pt}) for barrier point {idx} is inside perimeter, adding")
                    else:
                        if feedback:
                            feedback.pushInfo(f"[GetConstantSlopeLines] Right offset point ({right_pt}) for barrier point {idx} is outside perimeter, skipping")
                        else:
                            print(f"[GetConstantSlopeLines] Right offset point ({right_pt}) for barrier point {idx} is outside perimeter, skipping")
            
            else:
                # Point is on neither barrier nor destination - neutral point: keep it (do not skip)
                neutral_row = row.copy()
                neutral_row['barrier_id_key'] = -1
                updated_start_points.append(neutral_row)
                if feedback:
                    feedback.pushInfo(f"[GetConstantSlopeLines] Neutral point {idx} (not on barrier or destination), keeping as is")
                else:
                    print(f"[GetConstantSlopeLines] Neutral point {idx} (not on barrier or destination), keeping as is")

        # Create GeoDataFrame with updated start points
        if not updated_start_points:
            if feedback:
                feedback.reportError("[GetConstantSlopeLines] No valid start points found after classification")
            raise RuntimeError("No valid start points found after classification")

        updated_start_points_gdf = gpd.GeoDataFrame(updated_start_points, crs=self.crs)

        # Report classification results
        barrier_count = len([pt for pt in updated_start_points if pt.get('barrier_id_key', -1) > 0])
        neutral_count = len(updated_start_points) - barrier_count
        if feedback:
            feedback.pushInfo(f"[GetConstantSlopeLines] Created {len(updated_start_points_gdf)} updated start points: {barrier_count} from barriers, {neutral_count} neutral")
        else:
            print(f"[GetConstantSlopeLines] Created {len(updated_start_points_gdf)} updated start points: {barrier_count} from barriers, {neutral_count} neutral")
        
        if feedback:
            feedback.pushInfo(f"[GetConstantSlopeLines] Starting processing of {len(updated_start_points_gdf)} start points...")
            feedback.setProgress(10)
            if feedback.isCanceled():
                feedback.reportError("[GetConstantSlopeLines] Constant slope line creation was cancelled by user")
                raise RuntimeError("Operation cancelled by user")
        else:
            print(f"[GetConstantSlopeLines] Starting processing of {len(updated_start_points_gdf)} start points...")
            print("[GetConstantSlopeLines] Progress: 10%")

        # Process each start point individually based on barrier classification
        constant_slope_lines = []
        start_point_attributes = []  # Store attributes for each successfully traced line
        total_points = len(updated_start_points_gdf)

        # Iterate over updated start points
        for pt_idx, pt_row in updated_start_points_gdf.iterrows():
            start_point = pt_row.geometry
            barrier_id = pt_row.get('barrier_id_key', -1)
            
            # Calculate progress (10% already used for initialization, 90% for processing points)
            point_progress = 10 + int((pt_idx / total_points) * 85)
            
            if feedback:
                feedback.setProgress(point_progress)
                if feedback.isCanceled():
                    feedback.reportError("[GetConstantSlopeLines] Constant slope line creation was cancelled by user")
                    raise RuntimeError("Operation cancelled by user")
                feedback.pushInfo(f"[GetConstantSlopeLines] Processing point {pt_idx + 1}/{total_points} (barrier_id_key={barrier_id})")
            else:
                print(f"[GetConstantSlopeLines] Processing point {pt_idx + 1}/{total_points} (barrier_id_key={barrier_id})")
                print(f"[GetConstantSlopeLines] Progress: {point_progress}%")

            # Create barrier and destination masks by combining the unique masks for this specific point
            # Initialize barrier and destination masks
            if allow_barriers_as_temp_destination:
                # use unique barrier mask for iterative tracing with barriers as temporary destinations
                barrier_mask = barrier_unique_mask.copy()
            else:
                # use binary barrier mask for simple tracing without barriers as temporary destinations
                barrier_mask = barrier_binary_mask.copy()

            # destination mask is always binary in case of constant slope tracing (compared to create_keylines)
            destination_mask = destination_binary_mask.copy()

            # Handle overlapping barrier and destination cells --> adjust destination mask
            # Set destination_mask to 0 at overlapping cells, because not possible to be barrier and destination at the same time
            barrier_destination_overlap = (barrier_mask == 1) & (destination_mask == 1)
            if np.any(barrier_destination_overlap):
                destination_mask[barrier_destination_overlap] = 0
                overlap_count = np.sum(barrier_destination_overlap)
                if feedback:
                    feedback.pushInfo(f"[GetConstantSlopeLines] Adjusted {overlap_count} overlapping barrier/destination cells for point {pt_idx}")
                else:
                    print(f"[GetConstantSlopeLines] Adjusted {overlap_count} overlapping barrier/destination cells for point {pt_idx}")
            
            # Save barrier and destination masks as raster files
            barrier_raster_path = os.path.join(self.temp_directory, f"barrier_pt{pt_idx}.tif")
            destination_raster_path = os.path.join(self.temp_directory, f"destination_pt{pt_idx}.tif")

            # Write barrier mask
            if np.any(barrier_mask):
                barrier_ds = driver.Create(barrier_raster_path, dtm_cols, dtm_rows, 1, gdal.GDT_Byte)
                try:
                    barrier_ds.SetGeoTransform(dtm_geotransform)
                    barrier_ds.SetProjection(dtm_projection)
                    barrier_band = barrier_ds.GetRasterBand(1)
                    barrier_band.WriteArray(barrier_mask)
                finally:
                    barrier_ds = None
            else:
                barrier_raster_path = None
            
            # Write destination mask  
            if np.any(destination_mask):
                destination_ds = driver.Create(destination_raster_path, dtm_cols, dtm_rows, 1, gdal.GDT_Byte)
                try:
                    destination_ds.SetGeoTransform(dtm_geotransform)
                    destination_ds.SetProjection(dtm_projection)
                    destination_band = destination_ds.GetRasterBand(1)
                    destination_band.WriteArray(destination_mask)
                finally:
                    destination_ds = None
            else:
                if feedback:
                    feedback.pushWarning(f"[GetConstantSlopeLines] No destination mask created for point {pt_idx}, skipping")
                else:
                    warnings.warn(f"[GetConstantSlopeLines] No destination mask created for point {pt_idx}, skipping")
                continue
            
            if feedback:
                feedback.pushInfo(f"[GetConstantSlopeLines] Parameters for point {pt_idx}:")
                feedback.pushInfo(f"[GetConstantSlopeLines] Start point: {start_point}")
                feedback.pushInfo(f"[GetConstantSlopeLines] Destination raster: {destination_raster_path}")
                feedback.pushInfo(f"[GetConstantSlopeLines] Barrier raster: {barrier_raster_path}")
                feedback.pushInfo(f"[GetConstantSlopeLines] slope: {slope}, slope_deviation_threshold: {slope_deviation_threshold}, max_iterations_slope: {max_iterations_slope}")
                if allow_barriers_as_temp_destination:
                    feedback.pushInfo(f"[GetConstantSlopeLines] allow_barriers_as_temp_destination: True, max_iterations_barrier: {max_iterations_barrier}")
            else:
                print(f"[GetConstantSlopeLines] Parameters for point {pt_idx}:")
                print(f"[GetConstantSlopeLines] Start point: {start_point}")
                print(f"[GetConstantSlopeLines] Destination raster: {destination_raster_path}")
                print(f"[GetConstantSlopeLines] Barrier raster: {barrier_raster_path}")
                print(f"[GetConstantSlopeLines] slope: {slope}, slope_deviation_threshold: {slope_deviation_threshold}, max_iterations_slope: {max_iterations_slope}")
                if allow_barriers_as_temp_destination:
                    print(f"[GetConstantSlopeLines] allow_barriers_as_temp_destination: True, max_iterations_barrier: {max_iterations_barrier}")

            # Trace the constant slope line for this point
            if feedback:
                if allow_barriers_as_temp_destination:
                    feedback.pushInfo(f"[GetConstantSlopeLines] Starting iterative tracing for point {pt_idx + 1}/{total_points}...")
                else:
                    feedback.pushInfo(f"[GetConstantSlopeLines] Starting tracing for point {pt_idx + 1}/{total_points}...")
            else:
                if allow_barriers_as_temp_destination:
                    print(f"[GetConstantSlopeLines] Starting iterative tracing for point {pt_idx + 1}/{total_points}...")
                else:
                    print(f"[GetConstantSlopeLines] Starting tracing for point {pt_idx + 1}/{total_points}...")
            
            if allow_barriers_as_temp_destination:
                try:
                    traced_line = self._get_iterative_constant_slope_line(
                        dtm_path=dtm_path,
                        start_point=start_point,
                        destination_raster_path=destination_raster_path,
                        slope=slope,
                        barrier_raster_path=barrier_raster_path,
                        initial_barrier_value=barrier_id if barrier_id > 0 else None,
                        max_iterations_barrier=max_iterations_barrier,
                        slope_deviation_threshold=slope_deviation_threshold,
                        max_iterations_slope=max_iterations_slope,
                        feedback=feedback
                    )

                    if traced_line is not None and not traced_line.is_empty:
                        constant_slope_lines.append(traced_line)
                        # Store attributes from the original start point plus input parameters
                        point_attrs = pt_row.drop('geometry').to_dict()  # Get all attributes except geometry
                        point_attrs['slope'] = slope  # Add input parameter
                        start_point_attributes.append(point_attrs)
                        if feedback:
                            feedback.pushInfo(f"[GetConstantSlopeLines] Successfully traced iterative line for point {pt_idx + 1}/{total_points} (total lines: {len(constant_slope_lines)})")
                        else:
                            print(f"[GetConstantSlopeLines] Successfully traced iterative line for point {pt_idx + 1}/{total_points} (total lines: {len(constant_slope_lines)})")
                    else:
                        raise RuntimeError("No line traced")
                
                except Exception as e:
                    if feedback:
                        feedback.reportError(f"[GetConstantSlopeLines] Error tracing iterative line for point {pt_idx + 1}: {str(e)}")
                    else:
                        print(f"[GetConstantSlopeLines] Error tracing iterative line for point {pt_idx + 1}: {str(e)}")

            else:
                try:
                    traced_line = self._get_constant_slope_line(
                        dtm_path=dtm_path,
                        start_point=start_point,
                        destination_raster_path=destination_raster_path,
                        slope=slope,
                        barrier_raster_path=barrier_raster_path,
                        slope_deviation_threshold=slope_deviation_threshold,
                        max_iterations_slope=max_iterations_slope,
                        feedback=feedback
                    )

                    if traced_line is not None and not traced_line.is_empty:
                        constant_slope_lines.append(traced_line)
                        # Store attributes from the original start point plus input parameters
                        point_attrs = pt_row.drop('geometry').to_dict()  # Get all attributes except geometry
                        point_attrs['slope'] = slope  # Add input parameters
                        start_point_attributes.append(point_attrs)
                        if feedback:
                            feedback.pushInfo(f"[GetConstantSlopeLines] Successfully traced line for point {pt_idx + 1}/{total_points} (total lines: {len(constant_slope_lines)})")
                        else:
                            print(f"[GetConstantSlopeLines] Successfully traced line for point {pt_idx + 1}/{total_points} (total lines: {len(constant_slope_lines)})")
                    else:
                        raise RuntimeError("No line traced")
                
                except Exception as e:
                    if feedback:
                        feedback.reportError(f"[GetConstantSlopeLines] Error tracing line for point {pt_idx + 1}: {str(e)}")
                    else:
                        print(f"[GetConstantSlopeLines] Error tracing line for point {pt_idx + 1}: {str(e)}")

        # Create result GeoDataFrame
        if constant_slope_lines:
            # Create GeoDataFrame with geometries and preserved attributes
            result_gdf = gpd.GeoDataFrame(start_point_attributes, geometry=constant_slope_lines, crs=self.crs)
        else:
            result_gdf = gpd.GeoDataFrame(crs=self.crs)
        
        if feedback:
            feedback.setProgress(100)
            if feedback.isCanceled():
                feedback.reportError("[GetConstantSlopeLines] Constant slope line creation was cancelled by user")
                raise RuntimeError("Operation cancelled by user")
            feedback.pushInfo(f"[GetConstantSlopeLines] Constant slope line creation complete: {len(result_gdf)}/{total_points} lines traced")
        else:
            print(f"[GetConstantSlopeLines] Constant slope line creation complete: {len(result_gdf)}/{total_points} lines traced")
            print("[GetConstantSlopeLines] Progress: 100%")
        
        return result_gdf

    def adjust_constant_slope_after(
        self,
        dtm_path: str,
        input_lines: gpd.GeoDataFrame,
        change_after: float,
        slope_after: float,
        destination_features: list[gpd.GeoDataFrame],
        perimeter: gpd.GeoDataFrame = None,
        barrier_features: list[gpd.GeoDataFrame] = None,
        allow_barriers_as_temp_destination: bool = False,
        max_iterations_barrier: int = 30,
        slope_deviation_threshold: float = 0.2,
        max_iterations_slope: int = 20,
        feedback=None
        ) -> gpd.GeoDataFrame:
        """
        Modify constant slope lines by changing to a secondary slope after a specified distance.
        
        This function splits each input line at a specified fraction of its length and continues 
        with a new slope from that point using the get_constant_slope_lines function.
        
        Args:
            dtm_path (str): Path to the digital terrain model (GeoTIFF).
            input_lines (gpd.GeoDataFrame): Input constant slope lines to modify.
            change_after (float): Fraction of line length where slope changes (0.0-1.0, e.g., 0.5 = from halfway).
            slope_after (float): New slope to apply after the change point (e.g., 0.01 for 1% downhill).
            destination_features (list[gpd.GeoDataFrame]): Destination features for the new slope sections, e.g. ridge lines in case of keylines.
            perimeter (gpd.GeoDataFrame, optional): Polygon features defining area of interest. Acts as both barrier (boundary cannot be crossed) and is used to check if points are inside the perimeter area.
            barrier_features (list[gpd.GeoDataFrame], optional): Barrier features to avoid, e.g. valley lines in case of keylines.
            allow_barriers_as_temp_destination (bool): If True, barriers are included as temporary destinations for iterative tracing.
            max_iterations_barrier (int): Maximum number of iterations when using barriers as temporary destinations.
            slope_deviation_threshold (float): Maximum allowed relative deviation from expected slope (0.0-1.0, e.g., 0.2 for 20% deviation before line cutting).
            max_iterations_slope (int): Maximum number of iterations for line refinement.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting.
            
        Returns:
            gpd.GeoDataFrame: Modified lines with secondary slopes applied.
        """
        if feedback:
            if feedback.isCanceled():
                feedback.reportError("Slope adjustment was cancelled by user")
                raise RuntimeError("Operation cancelled by user")
            
        if feedback:
            feedback.pushInfo(f"[AdjustConstantSlopeAfter] Starting adjustment of {len(input_lines)} lines...")
            feedback.setProgress(0)
        else:
            print(f"[AdjustConstantSlopeAfter] Starting adjustment of {len(input_lines)} lines...")
            print("[AdjustConstantSlopeAfter] Progress: 0%")

        # Validate change_after parameter
        if not (0.0 < change_after < 1.0):
            raise ValueError("change_after must be between 0.0 and 1.0")
        
        # Check for potential configuration issues
        if allow_barriers_as_temp_destination and not barrier_features:
            warning_msg = "[AdjustConstantSlopeAfter] Warning: allow_barriers_as_temp_destination=True but no barrier_features provided. This setting will have no effect."
            if feedback:
                feedback.pushWarning(warning_msg)
            else:
                warnings.warn(warning_msg)
        
        # Phase 1: Process all lines to create first parts and collect start points for second parts
        if feedback:
            if feedback.isCanceled():
                feedback.reportError("Slope adjustment was cancelled by user")
                raise RuntimeError("Operation cancelled by user")

        if feedback:
            feedback.pushInfo(f"[AdjustConstantSlopeAfter] Phase 1: Processing {len(input_lines)} lines to create first parts...")
            feedback.setProgress(10)
        else:
            print(f"[AdjustConstantSlopeAfter] Phase 1: Processing {len(input_lines)} lines to create first parts...")

        first_parts_data = []  # Store first part data with mapping info
        all_start_points = []  # Collect all start points for second parts
        line_mapping = {}      # Map start point index to original line index
        total_lines = len(input_lines)
        
        for idx, row in input_lines.iterrows():
            line_geom = row.geometry
            
            # Progress reporting for Phase 1 (10-30% range)
            if feedback:
                phase1_progress = int(10 + ((idx + 1) / total_lines) * 20)
                feedback.setProgress(phase1_progress)
            
            # Handle MultiLineString by stitching into a single LineString
            if isinstance(line_geom, MultiLineString):
                # Converting MultiLineString to LineString (reduced logging)
                line_geom = TopoDrainCore._merge_lines_by_distance(line_geom)
            elif not isinstance(line_geom, LineString):
                if feedback:
                    feedback.pushInfo(f"[AdjustConstantSlopeAfter] Skipping unsupported geometry type: {type(line_geom)} at index {idx}")
                else:
                    print(f"[AdjustConstantSlopeAfter] Skipping unsupported geometry type: {type(line_geom)} at index {idx}")   
                # Keep original line for unsupported geometry types
                first_parts_data.append({
                    'original_row': row,
                    'first_part_line': None,
                    'needs_second_part': False,
                    'start_point_index': None
                })
                continue
            
            # Calculate split point at specified fraction of line length
            line_length = line_geom.length
            split_distance = line_length * change_after
            
            # Check if split distance is valid
            if split_distance <= 0 or split_distance >= line_length:
                if feedback:
                    feedback.pushInfo(f"[AdjustConstantSlopeAfter] Invalid split distance for line {idx}, keeping original line")
                else:
                    print(f"[AdjustConstantSlopeAfter] Invalid split distance for line {idx}, keeping original line")   
                # Keep original line if split point is invalid
                first_parts_data.append({
                    'original_row': row,
                    'first_part_line': None,
                    'needs_second_part': False,
                    'start_point_index': None
                })
                continue
            
            # Create first part of line (from start to split point)
            try:
                # Get coordinates up to split point
                coords = list(line_geom.coords)
                first_part_coords = []
                remaining_distance = split_distance
                
                for i in range(len(coords) - 1):
                    start_pt = Point(coords[i])
                    end_pt = Point(coords[i + 1])
                    segment_length = start_pt.distance(end_pt)
                    
                    first_part_coords.append(coords[i])
                    
                    if remaining_distance <= segment_length:
                        # Interpolate the exact split point on this segment
                        if remaining_distance > 0:
                            fraction = remaining_distance / segment_length
                            split_x = coords[i][0] + fraction * (coords[i + 1][0] - coords[i][0])
                            split_y = coords[i][1] + fraction * (coords[i + 1][1] - coords[i][1])
                            first_part_coords.append((split_x, split_y))
                        break
                    else:
                        remaining_distance -= segment_length
                
                # Create first part of the line
                if len(first_part_coords) >= 2:
                    first_part_line = LineString(first_part_coords)
                    
                    # Create start point for second part (the split point)
                    start_point_second = Point(first_part_coords[-1])
                    start_point_index = len(all_start_points)
                    all_start_points.append(start_point_second)
                    line_mapping[start_point_index] = len(first_parts_data)
                    
                    first_parts_data.append({
                        'original_row': row,
                        'first_part_line': first_part_line,
                        'needs_second_part': True,
                        'start_point_index': start_point_index
                    })
                else:
                    # Fallback: use original line if we can't create valid first part
                    if feedback:
                        feedback.pushInfo(f"[AdjustConstantSlopeAfter] Could not create valid first part for line {idx}, keeping original line")
                    else:
                        print(f"[AdjustConstantSlopeAfter] Could not create valid first part for line {idx}, keeping original line")
                    first_parts_data.append({
                        'original_row': row,
                        'first_part_line': None,
                        'needs_second_part': False,
                        'start_point_index': None
                    })
                    
            except Exception as e:
                if feedback:
                    feedback.pushInfo(f"[AdjustConstantSlopeAfter] Error processing line {idx}: {str(e)}, keeping original")
                else:
                    print(f"[AdjustConstantSlopeAfter] Error processing line {idx}: {str(e)}, keeping original")
                first_parts_data.append({
                    'original_row': row,
                    'first_part_line': None,
                    'needs_second_part': False,
                    'start_point_index': None
                })
        
        # Phase 2: Trace all second parts in a single call if we have start points
        second_part_lines = gpd.GeoDataFrame(geometry=[])  # Empty GeoDataFrame with geometry column
        if all_start_points:
            if feedback:
                if feedback.isCanceled():
                    feedback.reportError("Slope adjustment was cancelled by user")
                    raise RuntimeError("Operation cancelled by user")
                
            if feedback:
                feedback.pushInfo(f"[AdjustConstantSlopeAfter] Phase 2: Tracing {len(all_start_points)} second parts...")
                feedback.setProgress(40)
            else:
                print(f"[AdjustConstantSlopeAfter] Phase 2: Tracing {len(all_start_points)} second parts..")

            # Set CRS after creation when we have geometry column
            second_part_lines = second_part_lines.set_crs(self.crs)
            
            # Create GeoDataFrame with all start points and add mapping information
            start_points_gdf = gpd.GeoDataFrame(geometry=all_start_points, crs=self.crs)
            start_points_gdf['original_line_idx'] = [line_mapping[i] for i in range(len(all_start_points))]
            
            try:
                # Trace all second parts with new slope in a single call
                second_part_lines = self.get_constant_slope_lines(
                    dtm_path=dtm_path,
                    start_points=start_points_gdf,
                    destination_features=destination_features,
                    slope=slope_after,
                    perimeter=perimeter,
                    barrier_features=barrier_features,
                    allow_barriers_as_temp_destination=allow_barriers_as_temp_destination,
                    max_iterations_barrier=max_iterations_barrier,
                    slope_deviation_threshold=slope_deviation_threshold,
                    max_iterations_slope=max_iterations_slope,
                    feedback=feedback
            )

                if feedback:
                    feedback.pushInfo(f"[AdjustConstantSlopeAfter] Successfully traced {len(second_part_lines)} second parts")
                else:
                    print(f"[AdjustConstantSlopeAfter] Successfully traced {len(second_part_lines)} second parts")
            except Exception as e:
                if feedback:
                    feedback.reportError(f"[AdjustConstantSlopeAfter] Error tracing second parts: {str(e)}")
                    raise RuntimeError(f"[AdjustConstantSlopeAfter] Error tracing second parts: {str(e)}")
                else:
                    raise RuntimeError(f"[AdjustConstantSlopeAfter] Error tracing second parts: {str(e)}")
                # Continue with empty second_part_lines
        else:
            # No second parts to trace (reduced logging)
            pass
        
        # Phase 3: Combine first and second parts
        if feedback:
            if feedback.isCanceled():
                feedback.reportError("Slope adjustment was cancelled by user")
                raise RuntimeError("Operation cancelled by user")

        if feedback:
            feedback.pushInfo(f"[AdjustConstantSlopeAfter] Phase 3: Combining first and second parts...")
            feedback.setProgress(80)
        else:
            print(f"[AdjustConstantSlopeAfter] Phase 3: Combining first and second parts...")
                 
        adjusted_lines = []
        total_parts = len(first_parts_data)
        
        for data_idx, part_data in enumerate(first_parts_data):
            # Progress reporting for Phase 3 (80-90% range)
            if feedback and total_parts > 0:
                phase3_progress = int(80 + ((data_idx + 1) / total_parts) * 10)
                feedback.setProgress(phase3_progress)
                
            original_row = part_data['original_row']
            first_part_line = part_data['first_part_line']
            needs_second_part = part_data['needs_second_part']
            start_point_index = part_data['start_point_index']
            
            if not needs_second_part or first_part_line is None:
                # Keep original line with input parameters
                new_row = original_row.copy()
                new_row['change_after'] = change_after
                new_row['slope_after'] = slope_after
                adjusted_lines.append(new_row)
                continue
            
            # Find corresponding second part line(s)
            if not second_part_lines.empty and 'orig_index' in second_part_lines.columns:
                # Filter second parts that belong to this original line
                matching_second_parts = second_part_lines[second_part_lines['orig_index'] == start_point_index]
                
                if not matching_second_parts.empty:
                    # Get the first (and should be only) matching second part
                    second_part_line = matching_second_parts.iloc[0].geometry
                    
                    # Combine first and second parts
                    if isinstance(second_part_line, LineString):
                        # Merge coordinates, avoiding duplication of the split point
                        first_coords = list(first_part_line.coords)
                        second_coords = list(second_part_line.coords)
                        
                        # Remove duplicate split point if it exists
                        if len(second_coords) > 0 and first_coords[-1] == second_coords[0]:
                            combined_coords = first_coords + second_coords[1:]
                        else:
                            combined_coords = first_coords + second_coords
                        
                        combined_line = LineString(combined_coords)
                        
                        # Create new row with combined geometry and original attributes plus input parameters
                        new_row = original_row.copy()
                        new_row['geometry'] = combined_line
                        new_row['change_after'] = change_after
                        new_row['slope_after'] = slope_after
                        adjusted_lines.append(new_row)
                        
                        # Successfully combined line parts (reduced logging)
                    else:
                        # Second part is not LineString, keeping first part only (reduced logging)
                        new_row = original_row.copy()
                        new_row['geometry'] = first_part_line
                        new_row['change_after'] = change_after
                        new_row['slope_after'] = slope_after
                        adjusted_lines.append(new_row)
                else:
                    # No matching second part found, keeping first part only (reduced logging)
                    new_row = original_row.copy()
                    new_row['geometry'] = first_part_line
                    new_row['change_after'] = change_after
                    new_row['slope_after'] = slope_after
                    adjusted_lines.append(new_row)
            else:
                # No second parts available, keeping first part only (reduced logging)
                new_row = original_row.copy()
                new_row['geometry'] = first_part_line
                new_row['change_after'] = change_after
                new_row['slope_after'] = slope_after
                adjusted_lines.append(new_row)
        
        # Create result GeoDataFrame
        if adjusted_lines:
            result_gdf = gpd.GeoDataFrame(adjusted_lines, crs=self.crs).reset_index(drop=True)
        else:
            result_gdf = gpd.GeoDataFrame(crs=self.crs)

        if feedback:
            if feedback.isCanceled():
                feedback.reportError("Slope adjustment was cancelled by user")
                raise RuntimeError("Operation cancelled by user")
        

        if feedback:
            feedback.pushInfo(f"[AdjustConstantSlopeAfter] Adjustment complete: {len(result_gdf)} adjusted lines")
            feedback.setProgress(100)
        else:
            print(f"[AdjustConstantSlopeAfter] Adjustment complete: {len(result_gdf)} adjusted lines")

        return result_gdf

    def create_keylines(
            self, 
            dtm_path, 
            start_points, 
            valley_lines, 
            ridge_lines, 
            slope, 
            perimeter=None, 
            change_after=None,
            slope_after=None,
            slope_deviation_threshold=0.2,
            max_iterations_slope=30,
            feedback=None):
        """
        Create keylines using an iterative process with flexible valley-to-valley and ridge-to-ridge tracing:
        1. Classify start points based on their location (valley lines, ridge lines) with unique IDs
        2. Create orthogonal offset points for points on barriers
        3. Process all start points together with individual parameter adjustment based on classification
        4. Support valley→valley, ridge→ridge, valley→ridge, and ridge→valley tracing

        All output keylines will be oriented from valley to ridge (valley → ridge direction).

        Parameters:
        -----------
        dtm_path : str
            Path to the digital terrain model (GeoTIFF)
        start_points : GeoDataFrame
            Input keypoints to start keyline creation from (can be on valley lines, ridge lines, or neutral)
        valley_lines : GeoDataFrame
            Valley line features to use as barriers/destinations
        ridge_lines : GeoDataFrame
            Ridge line features to use as barriers/destinations
        slope : float
            Target slope for the constant slope lines (e.g., 0.01 for 1%)
        perimeter : GeoDataFrame
            Area of interest (perimeter) that always acts as destination feature (e.g. watershed, parcel polygon). If None, bounding box of valley and ridge lines is used.
        change_after : float, optional
            Fraction of line length where slope changes (0.0-1.0, e.g., 0.5 = from halfway). If None, no slope adjustment is applied.
        slope_after : float, optional
            New slope to apply after the change point (e.g., 0.005 for 0.5% downhill). Required if change_after is provided.
        slope_deviation_threshold : float, optional
            Maximum allowed relative deviation from expected slope (0.0-1.0, e.g., 0.2 for 20% deviation before line cutting). Default 0.2.
        max_iterations_slope : int, optional
            Maximum number of iterations for line refinement. Default 30.
        feedback : QgsProcessingFeedback
            Feedback object for progress reporting

        Returns:
        --------
        GeoDataFrame
            Combined keylines from all stages, all oriented from valley to ridge.
        """
        if feedback:
            feedback.pushInfo("[CreateKeylines] Starting keyline creation with flexible tracing...")
            feedback.setProgress(0)
        else:
            print("[CreateKeylines] Starting keyline creation with flexible tracing...")
            print("[CreateKeylines] Progress: 0%")


        if perimeter is None or perimeter.empty:
            # Get bounding box of valley and ridge lines as perimeter if none provided
            try:
                perimeter = self._perimeter_from_features([valley_lines, ridge_lines], buffer_distance=10) # 10 m bufffer
                if feedback:
                    feedback.pushInfo("[CreateKeylines] Created bounding box perimeter from valley and ridge lines")
                else:
                    print("[CreateKeylines] Created bounding box perimeter from valley and ridge lines")
            except ValueError as e:
                if feedback:
                    feedback.pushWarning(f"[CreateKeylines] No perimeter provided and no valley/ridge lines available to create bounding box: {e}")
                else:
                    print(f"[CreateKeylines] Warning: No perimeter provided and no valley/ridge lines available to create bounding box: {e}")
                
        # Read dtm with raster metadata information here, we use it multiple times below
        dtm_ds = gdal.Open(dtm_path, gdal.GA_ReadOnly)
        if dtm_ds is None:
            raise RuntimeError(f"Cannot open DTM raster: {dtm_path}.{self._get_gdal_error_message()}")
        try:
            dtm_geotransform = dtm_ds.GetGeoTransform()
            if dtm_geotransform is None:
                raise RuntimeError(f"Cannot get geotransform from DTM raster: {dtm_path}.{self._get_gdal_error_message()}")
            dtm_projection = dtm_ds.GetProjection()
            dtm_rows = dtm_ds.RasterYSize
            dtm_cols = dtm_ds.RasterXSize
        finally:
            dtm_ds = None

        # Prepare perimeter for rasterization
        # If perimeter is a polygon, use its boundary for rasterization
        if perimeter.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            perimeter_lines = perimeter.copy()
            perimeter_lines["geometry"] = perimeter_lines.boundary
        else:
            if feedback:
                feedback.reportError("Perimeter must be a GeoDataFrame with Polygon or MultiPolygon geometry")  
            raise ValueError("Perimeter must be a GeoDataFrame with Polygon or MultiPolygon geometry")
        
        # Create unique value raster masks for valley, ridge and perimeter lines to classify start points
        valley_unique_raster_path = os.path.join(self.temp_directory, "valley_unique.tif")
        ridge_unique_raster_path = os.path.join(self.temp_directory, "ridge_unique.tif")
        perimeter_unique_raster_path = os.path.join(self.temp_directory, "perimeter_unique.tif")
        
        # Rasterize valley, ridge and perimeter lines with unique values for each feature
        valley_unique_raster_path, valley_id_to_geom = self._vector_to_mask_raster([valley_lines], dtm_path, output_path=valley_unique_raster_path, unique_values=True, flatten_lines=True, buffer_lines=True)
        ridge_unique_raster_path, ridge_id_to_geom = self._vector_to_mask_raster([ridge_lines], dtm_path, output_path=ridge_unique_raster_path, unique_values=True, flatten_lines=True, buffer_lines=True)
        perimeter_unique_raster_path, perimeter_id_to_geom = self._vector_to_mask_raster([perimeter_lines], dtm_path, output_path=perimeter_unique_raster_path, unique_values=True, flatten_lines=True, buffer_lines=True)

        # Read unique masks for start point classification
        valley_unique_ds = gdal.Open(valley_unique_raster_path, gdal.GA_ReadOnly)
        if valley_unique_ds is None:
            raise RuntimeError(f"Cannot open valley lines raster: {valley_unique_raster_path}.{self._get_gdal_error_message()}")
        try:
            valley_unique_band = valley_unique_ds.GetRasterBand(1)
            valley_unique_mask = valley_unique_band.ReadAsArray()
            if valley_unique_mask is None:
                raise RuntimeError(f"Cannot read valley mask data from: {valley_unique_raster_path}.{self._get_gdal_error_message()}")
        finally:
            valley_unique_ds = None
        ridge_unique_ds = gdal.Open(ridge_unique_raster_path, gdal.GA_ReadOnly)
        if ridge_unique_ds is None:
            raise RuntimeError(f"Cannot open ridge lines raster: {ridge_unique_raster_path}.{self._get_gdal_error_message()}")
        try:
            ridge_unique_band = ridge_unique_ds.GetRasterBand(1)
            ridge_unique_mask = ridge_unique_band.ReadAsArray()
            if ridge_unique_mask is None:
                raise RuntimeError(f"Cannot read ridge mask data from: {ridge_unique_raster_path}.{self._get_gdal_error_message()}")
        finally:
            ridge_unique_ds = None
        perimeter_unique_ds = gdal.Open(perimeter_unique_raster_path, gdal.GA_ReadOnly)
        if perimeter_unique_ds is None:
            raise RuntimeError(f"Cannot open perimeter raster: {perimeter_unique_raster_path}.{self._get_gdal_error_message()}")
        try:
            perimeter_unique_band = perimeter_unique_ds.GetRasterBand(1)
            perimeter_unique_mask = perimeter_unique_band.ReadAsArray()
            if perimeter_unique_mask is None:
                raise RuntimeError(f"Cannot read perimeter mask data from: {perimeter_unique_raster_path}.{self._get_gdal_error_message()}")
        finally:
            perimeter_unique_ds = None


        # Create binary masks here, we use it multiple times below
        valley_binary_mask = (valley_unique_mask > 0).astype(np.uint8)
        valley_binary_raster_path = os.path.join(self.temp_directory, f"valley_binary.tif")
        driver = gdal.GetDriverByName('GTiff')
        valley_binary_ds = driver.Create(valley_binary_raster_path, dtm_cols, dtm_rows, 1, gdal.GDT_Byte)
        try:
            valley_binary_ds.SetGeoTransform(dtm_geotransform)
            valley_binary_ds.SetProjection(dtm_projection)
            valley_binary_band = valley_binary_ds.GetRasterBand(1)
            valley_binary_band.WriteArray(valley_binary_mask)
        finally:
            valley_binary_ds = None
        ridge_binary_mask = (ridge_unique_mask > 0).astype(np.uint8)
        ridge_binary_raster_path = os.path.join(self.temp_directory, f"ridge_binary.tif")
        ridge_binary_ds = driver.Create(ridge_binary_raster_path, dtm_cols, dtm_rows, 1, gdal.GDT_Byte)
        try:
            ridge_binary_ds.SetGeoTransform(dtm_geotransform)
            ridge_binary_ds.SetProjection(dtm_projection)
            ridge_binary_band = ridge_binary_ds.GetRasterBand(1)
            ridge_binary_band.WriteArray(ridge_binary_mask)
        finally:
            ridge_binary_ds = None
        perimeter_binary_mask = (perimeter_unique_mask > 0).astype(np.uint8)
        perimeter_binary_raster_path = os.path.join(self.temp_directory, f"perimeter_binary.tif")
        perimeter_binary_ds = driver.Create(perimeter_binary_raster_path, dtm_cols, dtm_rows, 1, gdal.GDT_Byte)
        try:
            perimeter_binary_ds.SetGeoTransform(dtm_geotransform)
            perimeter_binary_ds.SetProjection(dtm_projection)
            perimeter_binary_band = perimeter_binary_ds.GetRasterBand(1)
            perimeter_binary_band.WriteArray(perimeter_binary_mask)
        finally:
            perimeter_binary_ds = None

        # Create perimeter polygon binary mask for efficient point-in-polygon checking
        # Create binary mask for perimeter polygons (not boundaries)
        perimeter_polygon_raster_path = os.path.join(self.temp_directory, "perimeter_polygon_mask.tif")
        perimeter_polygon_raster_path = self._vector_to_mask_raster([perimeter], dtm_path, output_path=perimeter_polygon_raster_path, unique_values=False, flatten_lines=False, buffer_lines=False)
        perimeter_polygon_ds = gdal.Open(perimeter_polygon_raster_path, gdal.GA_ReadOnly)
        if perimeter_polygon_ds is None:
            raise RuntimeError(f"Cannot open perimeter polygon raster: {perimeter_polygon_raster_path}.{self._get_gdal_error_message()}")
        try:
            perimeter_polygon_band = perimeter_polygon_ds.GetRasterBand(1)
            perimeter_polygon_mask = perimeter_polygon_band.ReadAsArray()
            if perimeter_polygon_mask is None:
                    raise RuntimeError(f"Cannot read perimeter polygon mask data from: {perimeter_polygon_raster_path}.{self._get_gdal_error_message()}")
        finally:
            perimeter_polygon_ds = None

        if feedback:
            feedback.pushInfo(f"[CreateKeylines] Raster masks created for valley, ridge, and perimeter")
            feedback.setProgress(5)
        else:
            print(f"[CreateKeylines] Raster masks created for valley, ridge, and perimeter")
            print("[CreateKeylines] Progress: 5%")

        # Classify start points and create updated start points with valley_id_key, ridge_id_key, perimeter_id_key attributes
        updated_start_points = []
        if feedback:
            feedback.pushInfo(f"[CreateKeylines] Classifying {len(start_points)} start points...")
        else:
            print(f"[CreateKeylines] Classifying {len(start_points)} start points...")
        
        for idx, row in start_points.iterrows():
            point = row.geometry

            # Get raster coordinates for the point using TopoDrainCore utility function
            pixel_coords = TopoDrainCore._coords_to_pixel_indices([point.coords[0]], dtm_geotransform)
            point_c, point_r = pixel_coords[0]
            
            # Check if point is within raster bounds
            if not (0 <= point_r < dtm_rows and 0 <= point_c < dtm_cols):
                if feedback:
                    feedback.pushWarning(f"[CreateKeylines] Warning: Start point {idx} is outside raster bounds, skipping")
                else:
                    warnings.warn(f"[CreateKeylines] Warning: Start point {idx} is outside raster bounds, skipping")
                continue
            
            # Check if point is on valley, ridge or perimeter mask
            valley_value = int(valley_unique_mask[point_r, point_c])
            ridge_value = int(ridge_unique_mask[point_r, point_c])
            perimeter_value = int(perimeter_unique_mask[point_r, point_c])
            
            if valley_value > 0 and ridge_value > 0:
                # Point is on both valley and ridge - skip this point
                if feedback:
                    feedback.pushWarning(f"[CreateKeylines] Warning: Start point {idx} is on both valley and ridge lines, skipping")
                else:
                    warnings.warn(f"[CreateKeylines] Warning: Start point {idx} is on both valley and ridge lines, skipping")
                continue

            elif valley_value > 0:
                # Point is on valley line - create orthogonal offset points
                valley_geom = valley_id_to_geom.get(valley_value)
                if valley_geom is None:
                    if feedback:
                        feedback.pushWarning(f"[CreateKeylines] Warning: No valley geometry found for ID {valley_value}, skipping point {idx}")
                    else:
                        warnings.warn(f"[CreateKeylines] Warning: No valley geometry found for ID {valley_value}, skipping point {idx}")
                    continue
                # Get orthogonal offset points
                left_pt, right_pt = TopoDrainCore._get_orthogonal_directions_start_points(
                    barrier_raster_path=valley_binary_raster_path,
                    point=point,
                    line_geom=valley_geom
                )
                # Add left point if it exists
                if left_pt is not None:
                    left_row = row.copy()
                    left_row.geometry = left_pt
                    left_row['valley_id_key'] = valley_value
                    left_row['ridge_id_key'] = -1
                    left_row['perimeter_id_key'] = -1
                    updated_start_points.append(left_row)
                    if feedback:
                        feedback.pushInfo(f"[CreateKeylines] Created left offset point ({left_pt}) for valley point {idx}")
                    else:
                        print(f"[CreateKeylines] Created left offset point ({left_pt}) for valley point {idx}")
                # Add right point if it exists
                if right_pt is not None:
                    right_row = row.copy()
                    right_row.geometry = right_pt
                    right_row['valley_id_key'] = valley_value
                    right_row['ridge_id_key'] = -1
                    right_row['perimeter_id_key'] = -1
                    updated_start_points.append(right_row)
                    if feedback:
                        feedback.pushInfo(f"[CreateKeylines] Created right offset point ({right_pt}) for valley point {idx}")
                    else:
                        print(f"[CreateKeylines] Created right offset point ({right_pt}) for valley point {idx}")

            elif ridge_value > 0:
                # Point is on ridge line - create orthogonal offset points
                ridge_geom = ridge_id_to_geom.get(ridge_value)
                if ridge_geom is None:
                    if feedback:
                        feedback.pushWarning(f"[CreateKeylines] Warning: No ridge geometry found for ID {ridge_value}, skipping point {idx}")
                    else:
                        warnings.warn(f"[CreateKeylines] Warning: No ridge geometry found for ID {ridge_value}, skipping point {idx}")
                    continue
                # Get orthogonal offset points
                left_pt, right_pt = TopoDrainCore._get_orthogonal_directions_start_points(
                    barrier_raster_path=ridge_binary_raster_path,
                    point=point,
                    line_geom=ridge_geom
                )
                # Add left point if it exists
                if left_pt is not None:
                    left_row = row.copy()
                    left_row.geometry = left_pt
                    left_row['valley_id_key'] = -1
                    left_row['ridge_id_key'] = ridge_value
                    left_row['perimeter_id_key'] = -1
                    updated_start_points.append(left_row)
                    if feedback:
                        feedback.pushInfo(f"[CreateKeylines] Created left offset point ({left_pt}) for ridge point {idx}")
                    else:
                        print(f"[CreateKeylines] Created left offset point ({left_pt}) for ridge point {idx}")

                # Add right point if it exists
                if right_pt is not None:
                    right_row = row.copy()
                    right_row.geometry = right_pt
                    right_row['valley_id_key'] = -1
                    right_row['ridge_id_key'] = ridge_value
                    right_row['perimeter_id_key'] = -1
                    updated_start_points.append(right_row)
                    if feedback:
                        feedback.pushInfo(f"[CreateKeylines] Created right offset point ({right_pt}) for ridge point {idx}")
                    else:       
                        print(f"[CreateKeylines] Created right offset point ({right_pt}) for ridge point {idx}")

            elif perimeter_value > 0:
                # Point is on perimeter line - create orthogonal offset points
                perimeter_geom = perimeter_id_to_geom.get(perimeter_value)
                if perimeter_geom is None:
                    if feedback:
                        feedback.pushWarning(f"[CreateKeylines] Warning: No perimeter geometry found for ID {perimeter_value}, skipping point {idx}")
                    else:
                        warnings.warn(f"[CreateKeylines] Warning: No perimeter geometry found for ID {perimeter_value}, skipping point {idx}")
                    continue
                # Get orthogonal offset points
                left_pt, right_pt = TopoDrainCore._get_orthogonal_directions_start_points(
                    barrier_raster_path=perimeter_binary_raster_path,
                    point=point,
                    line_geom=perimeter_geom
                )
                
                # Check if offset points are inside the perimeter polygon using raster mask
                if left_pt is not None:
                    # Get raster coordinates for the left point
                    left_pixel_coords = TopoDrainCore._coords_to_pixel_indices([left_pt.coords[0]], dtm_geotransform)
                    left_c, left_r = left_pixel_coords[0]
                    # Check if left point is within raster bounds and inside perimeter polygon
                    is_inside_perimeter = False
                    if 0 <= left_r < dtm_rows and 0 <= left_c < dtm_cols:
                        if perimeter_polygon_mask is not None:
                            is_inside_perimeter = perimeter_polygon_mask[left_r, left_c] > 0
                        else:
                            is_inside_perimeter = True  # Assume inside perimeter when no perimeter is provided
                    # Only add left point if it's inside perimeter
                    if is_inside_perimeter:
                        left_row = row.copy()
                        left_row.geometry = left_pt
                        left_row['valley_id_key'] = -1
                        left_row['ridge_id_key'] = -1
                        left_row['perimeter_id_key'] = perimeter_value
                        updated_start_points.append(left_row)
                        if feedback:
                            feedback.pushInfo(f"[CreateKeylines] Left offset point ({left_pt}) for perimeter point {idx} is inside perimeter, adding")
                        else:
                            print(f"[CreateKeylines] Left offset point ({left_pt}) for perimeter point {idx} is inside perimeter, adding")
                    else:
                        if feedback:
                            feedback.pushInfo(f"[CreateKeylines] Left offset point ({left_pt}) for perimeter point {idx} is outside perimeter, skipping")
                
                if right_pt is not None:
                    # Get raster coordinates for the right point
                    right_pixel_coords = TopoDrainCore._coords_to_pixel_indices([right_pt.coords[0]], dtm_geotransform)
                    right_c, right_r = right_pixel_coords[0]
                    # Check if right point is within raster bounds and inside perimeter polygon
                    is_inside_perimeter = False
                    if 0 <= right_r < dtm_rows and 0 <= right_c < dtm_cols:
                        if perimeter_polygon_mask is not None:
                            is_inside_perimeter = perimeter_polygon_mask[right_r, right_c] > 0
                        else:
                            is_inside_perimeter = True  # Assume inside perimeter when no perimeter is provided
                    # Only add right point if it's inside perimeter
                    if is_inside_perimeter:
                        right_row = row.copy()
                        right_row.geometry = right_pt
                        right_row['valley_id_key'] = -1
                        right_row['ridge_id_key'] = -1
                        right_row['perimeter_id_key'] = perimeter_value
                        updated_start_points.append(right_row)
                        if feedback:
                            feedback.pushInfo(f"[CreateKeylines] Right offset point ({right_pt}) for perimeter point {idx} is inside perimeter, adding")
                        else:
                            print(f"[CreateKeylines] Right offset point ({right_pt}) for perimeter point {idx} is inside perimeter, adding")
                    else:
                        if feedback:
                            feedback.pushInfo(f"[CreateKeylines] Right offset point ({right_pt}) for perimeter point {idx} is outside perimeter, skipping")
            
            else:
                # Point is on neither valley, ridge or perimeter - neutral point: keep it (do not skip)
                neutral_row = row.copy()
                neutral_row['valley_id_key'] = -1
                neutral_row['ridge_id_key'] = -1
                neutral_row['perimeter_id_key'] = -1
                updated_start_points.append(neutral_row)
                if feedback:
                    feedback.pushInfo(f"[CreateKeylines] Neutral point {idx} (not on valley, ridge or perimeter), keeping as is")
                else:
                    print(f"[CreateKeylines] Neutral point {idx} (not on valley, ridge or perimeter), keeping as is")

        # Create GeoDataFrame with updated start points
        if not updated_start_points:
            if feedback:
                feedback.reportError("[CreateKeylines] No valid start points found after classification")
            raise RuntimeError("No valid start points found after classification")

        updated_start_points_gdf = gpd.GeoDataFrame(updated_start_points, crs=self.crs)

        # Report classification results
        valley_count = len([pt for pt in updated_start_points if pt.get('valley_id_key', -1) > 0])
        ridge_count = len([pt for pt in updated_start_points if pt.get('ridge_id_key', -1) > 0])
        perimeter_count = len([pt for pt in updated_start_points if pt.get('perimeter_id_key', -1) > 0])
        neutral_count = len(updated_start_points) - valley_count - ridge_count - perimeter_count
        if feedback:
            feedback.pushInfo(f"[CreateKeylines] Created {len(updated_start_points_gdf)} updated start points: {valley_count} from valleys, {ridge_count} from ridges, {perimeter_count} from perimeters, {neutral_count} neutral")
        else:
            print(f"[CreateKeylines] Created {len(updated_start_points_gdf)} updated start points: {valley_count} from valleys, {ridge_count} from ridges, {perimeter_count} from perimeters, {neutral_count} neutral")

        # Start iteration, keep iterating until no new start points are found
        # Initialize variables
        all_keylines = []
        current_start_points = updated_start_points_gdf.copy()
        iteration = 1
        
        # Set a maximum number of iterations to prevent infinite loops
        # Dynamic iteration limit: number of valley lines + number of ridge lines
        expected_iterations_keyline = (len(valley_lines) if valley_lines is not None else 0) + (len(ridge_lines) if ridge_lines is not None else 0)
        max_iterations_keyline = expected_iterations_keyline + 10  # Add some buffer

        # Iterate until no new start points are found or max iterations reached
        while not current_start_points.empty and iteration <= max_iterations_keyline:
            # Progress: 10% at start, 90% spread over iterations
            progress = 10 + int((iteration - 1) * (90 / expected_iterations_keyline))
            if feedback:
                feedback.pushInfo(f"[CreateKeylines] **** Ridge/Valley Iteration {iteration}/ca. {expected_iterations_keyline}-max. {max_iterations_keyline}: Processing {len(current_start_points)} start points ****")
                feedback.setProgress(min(progress, 99))
                if feedback.isCanceled():
                    feedback.reportError("[CreateKeylines] Keyline creation was cancelled by user")
                    raise RuntimeError("Operation cancelled by user")
            else:
                print(f"[CreateKeylines] **** Ridge/Valley Iteration {iteration}/ca. {expected_iterations_keyline}-max. {max_iterations_keyline}: Processing {len(current_start_points)} start points ****")
                print(f"[CreateKeylines] Progress: {progress}%")

            # Process each start point individually based on valley_id_key, ridge_id_key, perimeter_id_key
            iteration_keylines = []
            new_start_points = []

            # Iterate over current start points
            for pt_idx, pt_row in current_start_points.iterrows():
                start_point = pt_row.geometry
                valley_id = pt_row.get('valley_id_key', -1)
                ridge_id = pt_row.get('ridge_id_key', -1)
                perimeter_id = pt_row.get('perimeter_id_key', -1)

                if feedback:
                    feedback.pushInfo(f"[CreateKeylines] Iteration {iteration}: Processing point {pt_idx} (valley_id_key={valley_id}, ridge_id_key={ridge_id})")
                else:
                    print(f"[CreateKeylines] Iteration {iteration}: Processing point {pt_idx} (valley_id_key={valley_id}, ridge_id_key={ridge_id})")

                # Determine slope parameters based on start point classification
                if valley_id > 0:
                    # Point is from valley - use standard valley→ridge parameters
                    use_slope = slope
                    use_change_after = change_after
                    use_slope_after = slope_after
                    direction_type = "valley→ridge"

                elif ridge_id > 0:
                    # Point is from ridge - use swapped ridge→valley parameters (because they are always defined in valley to ridge perspective)
                    use_slope = -slope_after if slope_after is not None else -slope
                    # For slope adjustment: if both change_after and slope_after are provided, swap them
                    if change_after is not None and slope_after is not None:
                        use_change_after = (1-change_after)
                        use_slope_after = -slope
                    else:
                        use_change_after = None
                        use_slope_after = None
                    direction_type = "ridge→valley"
                    
                elif perimeter_id > 0:
                    # Point is from perimeter - determine direction based on proximity to valleys vs ridges
                    # Calculate minimum distance to valley features
                    if feedback:
                        feedback.pushWarning(f"[CreateKeylines] Start point {pt_idx} is on perimeter line, determining direction based on proximity to valleys vs ridges, which is not guaranteed to be accurate! Better start from valley or ridge line.")
                    else:
                        warnings.warn(f"[CreateKeylines] Start point {pt_idx} is on perimeter line, determining direction based on proximity to valleys vs ridges, which is not guaranteed to be accurate! Better start from valley or ridge line.")

                    min_valley_distance = float('inf')
                    if valley_lines is not None and not valley_lines.empty:
                        for _, valley_row in valley_lines.iterrows():
                            distance = start_point.distance(valley_row.geometry)
                            min_valley_distance = min(min_valley_distance, distance)
                    # Calculate minimum distance to ridge features
                    min_ridge_distance = float('inf')
                    if ridge_lines is not None and not ridge_lines.empty:
                        for _, ridge_row in ridge_lines.iterrows():
                            distance = start_point.distance(ridge_row.geometry)
                            min_ridge_distance = min(min_ridge_distance, distance)
                    
                    # Choose parameters based on which type of feature is closer
                    if min_ridge_distance <= min_valley_distance:
                        # Ridge is closer - assume valley to ridge direction (standard parameters)
                        use_slope = slope
                        use_change_after = change_after
                        use_slope_after = slope_after
                        direction_type = "valley→ridge"
                    else:
                        # Valley is closer - assume ridge to valley direction (swapped parameters)
                        use_slope = -slope_after if slope_after is not None else -slope
                        # For slope adjustment: if both change_after and slope_after are provided, swap them
                        if change_after is not None and slope_after is not None:
                            use_change_after = (1-change_after)
                            use_slope_after = -slope
                        else:
                            use_change_after = None
                            use_slope_after = None
                        direction_type = "ridge→valley"
                    
                    if feedback:
                        feedback.pushInfo(f"[CreateKeylines] Perimeter point {pt_idx}: min_valley_dist={min_valley_distance:.2f}m, min_ridge_dist={min_ridge_distance:.2f}m, assuming {direction_type} parameters")
                    else:
                        print(f"[CreateKeylines] Perimeter point {pt_idx}: min_valley_dist={min_valley_distance:.2f}m, min_ridge_dist={min_ridge_distance:.2f}m, assuming {direction_type} parameters")
                else:
                    # Neutral point - use standard valley→ridge parameters  
                    use_slope = slope
                    use_change_after = change_after
                    use_slope_after = slope_after
                    direction_type = "valley→ridge"

                # Create barrier and destination masks by combining the unique masks for this specific point
                # Initialize barrier and destination masks
                barrier_mask = np.zeros((dtm_rows, dtm_cols), dtype=np.uint8)
                destination_mask = np.zeros((dtm_rows, dtm_cols), dtype=np.uint8)
                
                if valley_id > 0:
                    # Point is from valley - ONLY the specific valley as barrier, ALL others as destination
                    # Barrier: ONLY the specific valley
                    barrier_mask[valley_unique_mask == valley_id] = 1
                    # Destination: ALL ridges + other valleys + perimeter lines
                    destination_mask[ridge_unique_mask > 0] = 1  # All ridges
                    destination_mask[(valley_unique_mask > 0) & (valley_unique_mask != valley_id)] = 1  # Other valleys
                    destination_mask[perimeter_unique_mask > 0] = 1  # All perimeter
                    
                elif ridge_id > 0:
                    # Point is from ridge - ONLY the specific ridge as barrier, ALL others as destination
                    # Barrier: ONLY the specific ridge
                    barrier_mask[ridge_unique_mask == ridge_id] = 1
                    # Destination: ALL valleys + other ridges + perimeter lines
                    destination_mask[valley_unique_mask > 0] = 1  # All valleys
                    destination_mask[(ridge_unique_mask > 0) & (ridge_unique_mask != ridge_id)] = 1  # Other ridges
                    destination_mask[perimeter_unique_mask > 0] = 1  # All perimeter
                    
                elif perimeter_id > 0:
                    # Point is from perimeter - ONLY the specific perimeter as barrier, ALL others as destination
                    # Barrier: ONLY the specific perimeter
                    barrier_mask[perimeter_unique_mask == perimeter_id] = 1
                    # Destination: ALL valleys + ALL ridges + other perimeter lines
                    destination_mask[valley_unique_mask > 0] = 1  # All valleys
                    destination_mask[ridge_unique_mask > 0] = 1  # All ridges
                    destination_mask[(perimeter_unique_mask > 0) & (perimeter_unique_mask != perimeter_id)] = 1  # Other perimeters
                    
                else:
                    # Neutral point - no barriers, trace to all features ############ maybe later porximity check as done for perimeter points
                    barrier_mask[valley_unique_mask > 0] = 1  # All valleys as barrier
                    destination_mask[ridge_unique_mask > 0] = 1  # All ridges as destination
                    destination_mask[perimeter_unique_mask > 0] = 1  # All perimeter as destination
                
                # Handle overlapping barrier and destination cells --> adjust destination mask
                # Set destination_mask to 0 at overlapping cells, because not possible to be barrier and destination at the same time
                barrier_destination_overlap = (barrier_mask == 1) & (destination_mask == 1)
                if np.any(barrier_destination_overlap):
                    destination_mask[barrier_destination_overlap] = 0
                    if feedback:
                        overlap_count = np.sum(barrier_destination_overlap)
                        feedback.pushInfo(f"[CreateKeylines] Adjusted {overlap_count} overlapping barrier/destination cells for point {pt_idx}")
                    else:
                        overlap_count = np.sum(barrier_destination_overlap)
                        print(f"[CreateKeylines] Adjusted {overlap_count} overlapping barrier/destination cells for point {pt_idx}")
                
                # Save barrier and destination masks as raster files
                barrier_raster_path = os.path.join(self.temp_directory, f"barrier_iter{iteration}_pt{pt_idx}.tif")
                destination_raster_path = os.path.join(self.temp_directory, f"destination_iter{iteration}_pt{pt_idx}.tif")
                
                # Write barrier mask
                if np.any(barrier_mask):
                    barrier_ds = driver.Create(barrier_raster_path, dtm_cols, dtm_rows, 1, gdal.GDT_Byte)
                    try:
                        barrier_ds.SetGeoTransform(dtm_geotransform)
                        barrier_ds.SetProjection(dtm_projection)
                        barrier_band = barrier_ds.GetRasterBand(1)
                        barrier_band.WriteArray(barrier_mask)
                    finally:
                        barrier_ds = None
                else:
                    barrier_raster_path = None
                
                # Write destination mask  
                if np.any(destination_mask):
                    destination_ds = driver.Create(destination_raster_path, dtm_cols, dtm_rows, 1, gdal.GDT_Byte)
                    try:
                        destination_ds.SetGeoTransform(dtm_geotransform)
                        destination_ds.SetProjection(dtm_projection)
                        destination_band = destination_ds.GetRasterBand(1)
                        destination_band.WriteArray(destination_mask)
                    finally:
                        destination_ds = None
                else:
                    if feedback:
                        feedback.pushWarning(f"[CreateKeylines] No destination mask created for point {pt_idx}, skipping")
                    else:
                        warnings.warn(f"[CreateKeylines] No destination mask created for point {pt_idx}, skipping")
                    continue
                
                if feedback:
                    feedback.pushInfo(f"[CreateKeylines] Parameters for point {pt_idx}:")
                    feedback.pushInfo(f"[CreateKeylines] Direction type: {direction_type}")
                    feedback.pushInfo(f"[CreateKeylines] Start point: {start_point}")
                    feedback.pushInfo(f"[CreateKeylines] Destination raster: {destination_raster_path}")
                    feedback.pushInfo(f"[CreateKeylines] Barrier raster: {barrier_raster_path}")
                    feedback.pushInfo(f"[CreateKeylines] slope={use_slope}, change_after={use_change_after}, slope_after={use_slope_after}")
                else:
                    print(f"[CreateKeylines] Parameters for point {pt_idx}:")
                    print(f"[CreateKeylines] Direction type: {direction_type}")
                    print(f"[CreateKeylines] Start point: {start_point}")
                    print(f"[CreateKeylines] Destination raster: {destination_raster_path}")
                    print(f"[CreateKeylines] Barrier raster: {barrier_raster_path}")
                    print(f"[CreateKeylines] slope={use_slope}, change_after={use_change_after}, slope_after={use_slope_after}")

                # Trace the constant slope line for this point
                try:
                    traced_line = self._get_constant_slope_line(
                        dtm_path=dtm_path,
                        start_point=start_point,
                        destination_raster_path=destination_raster_path,
                        slope=use_slope,
                        barrier_raster_path=barrier_raster_path,
                        slope_deviation_threshold=slope_deviation_threshold,
                        max_iterations_slope=max_iterations_slope,
                        feedback=feedback
                    )

                    if traced_line is not None and not traced_line.is_empty:
                        if feedback:
                            feedback.pushInfo(f"[CreateKeylines] Successfully traced line for point {pt_idx}")
                        else:
                            print(f"[CreateKeylines] Successfully traced line for point {pt_idx}")
                    else:
                        raise RuntimeError("No line traced")
                    

                except Exception as e:
                    if feedback:
                        feedback.reportError(f"[CreateKeylines] Error tracing line for point {pt_idx}: {str(e)}")
                    else:
                        print(f"[CreateKeylines] Error tracing line for point {pt_idx}: {str(e)}")
                    #continue
    
                # Apply slope adjustment if needed
                if use_change_after is not None and use_slope_after is not None:
                    try:
                        adjusted_line = self._adjust_constant_slope_after(
                            dtm_path=dtm_path,
                            input_line=traced_line,
                            change_after=use_change_after,
                            slope_after=use_slope_after,
                            destination_raster_path=destination_raster_path,
                            barrier_raster_path=barrier_raster_path,
                            slope_deviation_threshold=slope_deviation_threshold,
                            max_iterations_slope=max_iterations_slope,
                            feedback=feedback,
                        )

                        if adjusted_line is not None and not adjusted_line.is_empty:
                            if feedback:
                                feedback.pushInfo(f"[CreateKeylines] Successfully adjusted line for point {pt_idx}")
                            else:
                                print(f"[CreateKeylines] Successfully adjusted line for point {pt_idx}")
                        else:
                            raise RuntimeError("No line after adjustment")
                            
                    except Exception as e:
                        if feedback:
                            feedback.reportError(f"[CreateKeylines] Error adjusting line for point {pt_idx}: {str(e)}")
                        else:
                            print(f"[CreateKeylines] Error adjusting line for point {pt_idx}: {str(e)}")
                        continue

                    iteration_line = adjusted_line
                else:
                    iteration_line = traced_line

                # Store the line with its metadata for later direction correction
                line_data = {
                    'line': iteration_line,
                    'ridge_id': ridge_id,
                    'perimeter_id': perimeter_id,
                    'direction_type': direction_type,
                    'pt_idx': pt_idx,
                    'valley_id': valley_id,
                    'slope': use_slope,
                    'change_after': use_change_after,
                    'slope_after': use_slope_after,
                    # Add any additional point attributes from the original point
                    #'point_attributes': pt_row.drop('geometry').to_dict() # debug
                }
                iteration_keylines.append(line_data)
                    
                # Determine if new start points need to be created from line endpoint
                current_iteration_line = line_data['line']
                end_point = Point(current_iteration_line.coords[-1])
                # Get raster coordinates for endpoint
                pixel_coords = TopoDrainCore._coords_to_pixel_indices([end_point.coords[0]], dtm_geotransform)
                end_c, end_r = pixel_coords[0]
                    
                # Check if endpoint is within bounds
                if not (0 <= end_r < dtm_rows and 0 <= end_c < dtm_cols):
                    continue
                # Check if endpoint is inside perimeter polygon - if not, skip creating new start point
                if perimeter_polygon_mask[end_r, end_c] == 0:
                    continue  # Don't create new start point for points outside perimeter polygon
                # Check if endpoint reached perimeter (stop tracing)
                perimeter_binary_value = int(perimeter_binary_mask[end_r, end_c])
                if perimeter_binary_value == 1:
                    continue  # Don't create new start point if perimeter is reached
                
                # Check if endpoint is on valley or ridge features
                valley_value = int(valley_unique_mask[end_r, end_c])
                ridge_value = int(ridge_unique_mask[end_r, end_c])
            
                # Create new start point with proper classification
                new_pt_data = pt_row.copy()
                # Set valley_id_key and ridge_id_key based on endpoint location
                if valley_value > 0:
                    new_pt_data['valley_id_key'] = valley_value
                    new_pt_data['ridge_id_key'] = -1
                    new_pt_data['perimeter_id_key'] = -1
                    # Get offset point away from valley
                    offset_point = TopoDrainCore._get_linedirection_start_point(
                        valley_binary_raster_path, current_iteration_line, max_offset=10
                    )
                    if offset_point:
                        new_pt_data['geometry'] = offset_point
                        new_start_points.append(new_pt_data)
                        if feedback:
                            feedback.pushInfo(f"[CreateKeylines] Created new start point from valley endpoint of point {pt_idx}")
                        else:
                            print(f"[CreateKeylines] Created new start point from valley endpoint of point {pt_idx}")
                    else:
                        if feedback:
                            feedback.pushWarning(f"[CreateKeylines] Warning: Could not find offset point away from valley for endpoint of point {pt_idx}, skipping new start point")
                        else:
                            warnings.warn(f"[CreateKeylines] Warning: Could not find offset point away from valley for endpoint of point {pt_idx}, skipping new start point")

                elif ridge_value > 0:
                    new_pt_data['valley_id_key'] = -1
                    new_pt_data['ridge_id_key'] = ridge_value
                    new_pt_data['perimeter_id_key'] = -1
                    # Get offset point away from ridge                    
                    offset_point = TopoDrainCore._get_linedirection_start_point(
                        ridge_binary_raster_path, current_iteration_line, max_offset=10
                    )
                    if offset_point:
                        new_pt_data['geometry'] = offset_point
                        new_start_points.append(new_pt_data)
                        if feedback:
                            feedback.pushInfo(f"[CreateKeylines] Created new start point from ridge endpoint of point {pt_idx}")
                        else:
                            print(f"[CreateKeylines] Created new start point from ridge endpoint of point {pt_idx}")
                            
                    else:
                        warnings.warn(f"[CreateKeylines] Warning: Could not find offset point away from ridge for endpoint of point {pt_idx}, skipping new start point")
                        if feedback:
                            feedback.pushWarning(f"[CreateKeylines] Warning: Could not find offset point away from valley for endpoint of point {pt_idx}, skipping new start point")

                else:
                    warnings.warn(f"[CreateKeylines] Endpoint of point {pt_idx} is neutral, no new start point created")
                    if feedback:
                        feedback.pushWarning(f"[CreateKeylines] Endpoint of point {pt_idx} is neutral, no new start point created")
                    continue  # Don't create new start point if endpoint is neutral (should not occur due to perimeter check above)

            # After processing all points in this iteration
            if not iteration_keylines:
                if feedback:
                    feedback.pushInfo(f"[CreateKeylines] Iteration {iteration}: No keylines generated, stopping iteration")
                else:
                    print(f"[CreateKeylines] Iteration {iteration}: No keylines generated, stopping iteration")
                break
            
            # Apply line direction correction to ensure all lines are valley→ridge oriented
            corrected_lines_with_attributes = []
            for line_data in iteration_keylines:
                line = line_data['line']
                ridge_id = line_data['ridge_id']
                perimeter_id = line_data['perimeter_id']
                direction_type = line_data['direction_type']
                pt_idx = line_data['pt_idx']
                
                # Reverse lines that were traced ridge→valley to ensure valley→ridge direction
                if ridge_id > 0 or (perimeter_id > 0 and direction_type == "ridge→valley"):
                    # Line was traced from ridge or perimeter point in ridge→valley direction
                    # Reverse to ensure valley→ridge direction for all output keylines
                    corrected_line = self._reverse_line_direction(line)
                    if feedback:
                        feedback.pushInfo(f"[CreateKeylines] Reversed line direction for ridge→valley traced line from point {pt_idx}")
                    else:
                        print(f"[CreateKeylines] Reversed line direction for ridge→valley traced line from point {pt_idx}")
                else:
                    # Keep original direction for valley→ridge traced lines
                    corrected_line = line
                
                # Create line attributes dictionary with input parameters
                line_attributes = {
                    'geometry': corrected_line,
                    'slope': slope,
                }
                
                # Add change_after and slope_after if they were used
                if line_data['change_after'] is not None:
                    line_attributes['change_after'] = change_after
                if line_data['slope_after'] is not None:
                    line_attributes['slope_after'] = slope_after
                
                # Add any additional point attributes
                #line_attributes.update(line_data['point_attributes']) #debug
                
                corrected_lines_with_attributes.append(line_attributes)
            
            # Add all corrected lines from this iteration to all_keylines
            all_keylines.extend(corrected_lines_with_attributes)
            
            if feedback:
                feedback.pushInfo(f"[CreateKeylines] Iteration {iteration}: Generated {len(iteration_keylines)} keylines")
            else:
                print(f"[CreateKeylines] Iteration {iteration}: Generated {len(iteration_keylines)} keylines")

            if not new_start_points:
                if feedback:
                    feedback.pushInfo(f"[CreateKeylines] Iteration {iteration}: No new start points generated, stopping iteration")
                    feedback.pushInfo(f"[CreateKeylines] **** End of Ridge/Valley iteration {iteration} ****")
                else:
                    print(f"[CreateKeylines] Iteration {iteration}: No new start points generated, stopping iteration")
                    print(f"[CreateKeylines] **** End of Ridge/Valley iteration {iteration} ****")
                break
            
            # Create GeoDataFrame from new start points for next iteration
            current_start_points = gpd.GeoDataFrame(new_start_points, crs=self.crs)
            
            if feedback:
                feedback.pushInfo(f"[CreateKeylines] Iteration {iteration}: Generated {len(new_start_points)} new start points for next iteration")
                feedback.pushInfo(f"[CreateKeylines] **** End of Ridge/Valley iteration {iteration} ****")
            else:
                print(f"[CreateKeylines] Iteration {iteration}: Generated {len(new_start_points)} new start points for next iteration")
                print(f"[CreateKeylines] **** End of Ridge/Valley iteration {iteration} ****")

            # Increment iteration counter
            iteration += 1

        # End of iteration loop
        if iteration > max_iterations_keyline:
            if feedback:
                feedback.pushWarning(f"[CreateKeylines] Maximum iterations ({max_iterations_keyline}) reached, stopping iteration...")
            else:
                warnings.warn(f"[CreateKeylines] Warning: Maximum iterations ({max_iterations_keyline}) reached, stopping iteration...")

        # Create result GeoDataFrame
        if all_keylines:
            # Extract geometries and attributes separately
            geometries = [line_data['geometry'] for line_data in all_keylines]
            attributes_list = []
            
            for line_data in all_keylines:
                # Create attributes dict without geometry
                attrs = {key: value for key, value in line_data.items() if key != 'geometry'}
                attributes_list.append(attrs)
            
            result_gdf = gpd.GeoDataFrame(attributes_list, geometry=geometries, crs=self.crs)
        else:
            result_gdf = gpd.GeoDataFrame(crs=self.crs)
        
        if feedback:
            feedback.setProgress(100)
            if feedback.isCanceled():
                feedback.reportError("Keyline creation was cancelled by user")
                raise RuntimeError("Operation cancelled by user")
            feedback.pushInfo(f"[CreateKeylines] Keyline creation complete: {len(result_gdf)} total keylines from {iteration-1} iterations")
        else:
            print(f"[CreateKeylines] Keyline creation complete: {len(result_gdf)} total keylines from {iteration-1} iterations")
            print("[CreateKeylines] Progress: 100%")
        
        return result_gdf



if __name__ == "__main__":
    print("No main part")
