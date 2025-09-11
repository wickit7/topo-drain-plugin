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
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.sample import sample_gen
from rasterio.features import rasterize
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, nearest_points, substring
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import re
import subprocess

# Import QGIS dependencies only if available
try:
    from qgis.core import (
        QgsRunProcess,
        QgsBlockingProcess,
        QgsProcessingFeedback,
        QgsProcessingException,
    )
    from qgis.PyQt.QtCore import QProcess
    try:
        from qgis.utils import iface
    except ImportError:
        iface = None
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False
    QgsProcessingFeedback = None
    QgsProcessingException = Exception

# Progress regex for parsing WhiteboxTools output
progress_regex = re.compile(r'\d+%')

# ---  Class TopoDrainCore ---
class TopoDrainCore:
    def __init__(self, whitebox_directory=None, nodata=None, crs=None, temp_directory=None, working_directory=None):
        print("[TopoDrainCore] Initializing TopoDrainCore...")
        self._thisdir = os.path.dirname(__file__)
        print(f"[TopoDrainCore] Module directory: {self._thisdir}")
        
        # Handle None whitebox_directory gracefully
        if whitebox_directory is None:
            print("[TopoDrainCore] No WhiteboxTools directory provided - will configure later")
            self.whitebox_directory = None  # Keep as None to trigger lazy loading
        else:
            self.whitebox_directory = whitebox_directory
        print(f"[TopoDrainCore] WhiteboxTools directory: {self.whitebox_directory if self.whitebox_directory else 'Not set (will configure later)'}")
        
        self.nodata = nodata if nodata is not None else -32768
        print(f"[TopoDrainCore] NoData value set to: {self.nodata}")
        self.crs = crs if crs is not None else "EPSG:2056"
        print(f"[TopoDrainCore] crs value set to: {self.crs}")
        self.temp_directory = temp_directory if temp_directory is not None else None
        print(f"[TopoDrainCore] Temp directory set to: {self.temp_directory if self.temp_directory else 'Not set'}")
        self.working_directory = working_directory if working_directory is not None else None
        print(f"[TopoDrainCore] Working directory set to: {self.working_directory if self.working_directory else 'Not set'}")
        
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
            if whitebox_directory not in sys.path:
                sys.path.insert(0, whitebox_directory)
                
            # Try to import from the provided location
            wbt_path = os.path.join(whitebox_directory, "whitebox_tools.py")
            if not os.path.exists(wbt_path):
                print(f"[TopoDrainCore] WhiteboxTools not found at {wbt_path}")
                print("[TopoDrainCore] WhiteboxTools will be configured when available")
                return None
                
            spec = importlib.util.spec_from_file_location("whitebox_tools", wbt_path)
            if spec is None or spec.loader is None:
                print(f"[TopoDrainCore] Could not create spec for WhiteboxTools from {wbt_path}")
                return None
                
            whitebox_tools_mod = importlib.util.module_from_spec(spec)
            sys.modules["whitebox_tools"] = whitebox_tools_mod
            spec.loader.exec_module(whitebox_tools_mod)
            WhiteboxTools = whitebox_tools_mod.WhiteboxTools
            print(f"[TopoDrainCore] Using WhiteboxTools from directory: {whitebox_directory}")
                
            wbt = WhiteboxTools()
            if self.working_directory:
                wbt.set_working_dir(self.working_directory)
            print("[TopoDrainCore] WhiteboxTools successfully initialized")
            return wbt
            
        except Exception as e:
            print(f"[TopoDrainCore] WhiteboxTools initialization failed: {e}")
            print("[TopoDrainCore] WhiteboxTools will be configured when available")
            return None

    ## Setters for configuration
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

    def _execute_wbt(self, tool_name, feedback=None, **kwargs):
        """
        Execute a WhiteboxTools command with progress monitoring.
        Similar to the WBT for QGIS plugin's execute() function.

        Args:
            tool_name (str): Name of the WhiteboxTools command
            feedback (QgsProcessingFeedback, optional): Feedback object for progress reporting
            **kwargs: Tool parameters as keyword arguments

        Returns:
            int: Return code (0 = success)
            
        Raises:
            RuntimeError: If WhiteboxTools is not properly configured
        """
        # Check if WhiteboxTools is available
        if self.wbt is None or self.whitebox_directory is None:
            error_msg = "WhiteboxTools is not properly configured. Please ensure WhiteboxTools plugin is loaded and configured."
            if feedback:
                feedback.reportError(error_msg)
            raise RuntimeError(error_msg)
            
        if feedback is None and QGIS_AVAILABLE:
            feedback = QgsProcessingFeedback()

        # Build command line arguments
        wbt_executable = os.path.join(self.whitebox_directory, "whitebox_tools")
        if os.name == 'nt':  # Windows
            wbt_executable += ".exe"
        
        # Normalize path separators for consistency
        wbt_executable = os.path.normpath(wbt_executable)

        arguments = [wbt_executable, f'--run={tool_name}']

        # Add parameters
        for param, value in kwargs.items():
            if value is not None:
                arguments.append(f'--{param}="{value}"')

        # Create display command with proper quoting for paths with spaces
        display_args = []
        for i, arg in enumerate(arguments):
            if i == 0 and ' ' in arg:  # Quote the executable if it contains spaces
                display_args.append(f'"{arg}"')
            else:
                display_args.append(arg)
        fused_command = ' '.join(display_args)
        
        if feedback:
            feedback.pushInfo('WhiteboxTools command:')
            feedback.pushCommandInfo(fused_command)
            feedback.pushInfo('WhiteboxTools output:')
        else:
            print(f"[TopoDrainCore] Executing WhiteboxTools command: {fused_command}")
            print("[TopoDrainCore] WhiteboxTools output: ")

        if QGIS_AVAILABLE and feedback:
            # Use QGIS process handling with progress monitoring
            return self._execute_with_qgis_process(arguments, feedback)
        else:
            # Fallback to subprocess
            return self._execute_with_subprocess(arguments)

    def _execute_with_qgis_process(self, arguments, feedback=None):
        """Execute using QGIS QgsBlockingProcess with progress monitoring."""
        
        def on_stdout(ba):
            val = ba.data().decode('utf-8')
            if '%' in val:
                # Extract progress percentage
                match = progress_regex.search(val)
                if match:
                    progress_str = match.group(0).rstrip('%')
                    try:
                        progress = int(progress_str)
                        if feedback:    
                            feedback.setProgress(progress)
                        else:
                            print(f"[QWhiteboxTools] Progress: {progress}%")
                    except ValueError:
                        pass
                else:
                    on_stdout.buffer += val
            else:
                on_stdout.buffer += val

            if on_stdout.buffer.endswith(('\n', '\r')):
                if feedback:
                    feedback.pushConsoleInfo(on_stdout.buffer.rstrip())
                else:
                    print(on_stdout.buffer.rstrip())
                on_stdout.buffer = ''

        on_stdout.buffer = ''

        def on_stderr(ba):
            val = ba.data().decode('utf-8')
            on_stderr.buffer += val

            if on_stderr.buffer.endswith(('\n', '\r')):
                if feedback:
                    feedback.reportError(on_stderr.buffer.rstrip())
                else:
                    print(on_stderr.buffer.rstrip())
                on_stderr.buffer = ''

        on_stderr.buffer = ''

        # Use the first argument as command and rest as arguments
        # This properly handles paths with spaces without needing to join/split
        command = arguments[0]
        args = arguments[1:]
        proc = QgsBlockingProcess(command, args)
        proc.setStdOutHandler(on_stdout)
        proc.setStdErrHandler(on_stderr)

        res = proc.run(feedback)
        
        if feedback.isCanceled() and res != 0:
            feedback.pushInfo('Process was canceled and did not complete.')
        elif not feedback.isCanceled() and proc.exitStatus() == QProcess.CrashExit:
            raise QgsProcessingException('Process was unexpectedly terminated.')
        elif res == 0:
            if feedback:
                feedback.pushInfo('Process completed successfully.')
            else:
                print('Process completed successfully.')
        elif proc.processError() == QProcess.FailedToStart:
            raise QgsProcessingException(f'Process "{command}" failed to start. Either "{command}" is missing, or you may have insufficient permissions to run the program.')
        else:
            if feedback:
                feedback.reportError(f'Process returned error code {res}')
            else:
                print(f'Process returned error code {res}') 

        return res

    def _execute_with_subprocess(self, arguments):
        """Fallback execution using subprocess."""
        try:
            result = subprocess.run(arguments, capture_output=True, text=True, check=False)
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                warnings.warn(result.stderr)
                    
            return result.returncode
        except Exception as e:
            warnings.warn(f"Error executing WhiteboxTools: {e}")
            return 1
    
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
            warnings.warn("Cannot snap endpoint for MultiLineString geometry")
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
        with rasterio.open(reference_raster_path) as src:
            out_shape = (src.height, src.width)
            transform = src.transform
            res = src.res[0]

        buffer_distance = res + 0.01  # Small buffer to ensure rasterization of lines has no diagonal gaps

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

        # Create Shapes for rasterization
        all_shapes = []
        geometry_mapping = {}  # Track mapping from raster value to geometry when unique_values=True
        
        for i, geom in enumerate(all_geometries):
            if unique_values:
                mask_value = i + 1  # Assign unique values to each individual geometry. Start from 1
                geometry_mapping[mask_value] = geom  # Store the mapping
            else:
                mask_value = 1  # Default value for all geometries (binary mask)
            if geom.geom_type in ("LineString", "MultiLineString"):
                # Buffer by a small distance (e.g., 1 pixel width + tolerance), so that there are actually no gaps in the rasterization where least cost path will find a way
                buffered_geom = geom.buffer(buffer_distance) if buffer_lines else geom
                all_shapes.append((buffered_geom, mask_value))
            else:
                all_shapes.append((geom, mask_value)) # Not buffering Points und Polygons

        # Rasterize the shapes to create the mask
        # Always use uint32 for simplicity and compatibility
        raster_dtype = np.uint32
        
        mask = rasterize(
            shapes=all_shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=raster_dtype
        )

        # Generate output path if not provided
        if output_path is None:
            output_path = os.path.join(self.temp_directory, f"mask_{uuid.uuid4().hex[:8]}.tif")

        with rasterio.open(reference_raster_path) as src:
            profile = src.profile.copy()
            profile.update(
                dtype=raster_dtype,
                count=1,
                compress='lzw',
                nodata=0  # Use 0 as nodata for uint32 (works for both binary and unique value masks)
            )

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mask, 1)

        # Return path only for binary mask, or tuple with geometry mapping for unique values
        if unique_values:
            return output_path, geometry_mapping
        else:
            return output_path

    ## Raster functions
    @staticmethod
    def _mask_raster(raster_path: str, mask: gpd.GeoDataFrame, out_path: str) -> str:
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

    def _invert_dtm(self, dtm_path: str, output_path: str, feedback=None) -> str:
        """
        Create an inverted DTM (multiply by -1) to extract ridges.

        Args:
            dtm_path (str): Path to original DTM.
            output_path (str): Path to output inverted DTM.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting.

        Returns:
            str: Path to inverted DTM.
        """
        if self.wbt is None:
            raise RuntimeError("WhiteboxTools not initialized. Check WhiteboxTools configuration: QGIS settings -> Options -> Processing -> Provider -> WhiteboxTools -> WhiteboxTools executable.")

        ret = self._execute_wbt(
            'multiply',
            feedback=feedback,
            input1=dtm_path,
            input2=-1.0,
            output=output_path
        )
        
        if ret != 0 or not os.path.exists(output_path):
            raise RuntimeError(f"DTM inversion failed: WhiteboxTools returned {ret}, output not found at {output_path}")

        return output_path

    @staticmethod
    def _log_raster(
        input_raster: str,
        output_path: str,
        overwrite: bool = True,
        val_band: int = 1
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
            nodata = profile.get('nodata', None)

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

    # Alternatively define constant value of absolute elevation at masked cells or input raster contains absolute elevation values
    @staticmethod
    def _modify_dtm_with_mask(
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

    def _raster_to_linestring_wbt(self, raster_path: str, snap_to_start_point: Point = None, snap_to_endpoint: Point = None, feedback=None) -> LineString:
        """
        Uses WhiteboxTools to vectorize a raster and return a merged LineString or MultiLineString.
        Optionally snaps the endpoint to the center of a destination cell.

        Args:
            raster_path (str): Path to a raster where 1-valued pixels form your keyline.
            snap_to_start_point (Point, optional): Point to snap the start of the line to.
            snap_to_endpoint (Point, optional): Point to snap the endpoint of the line to.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting.

        Returns:
            LineString or MultiLineString, or None if empty.
        """
        if self.wbt is None:
            raise RuntimeError("WhiteboxTools not initialized. Check WhiteboxTools configuration: QGIS settings -> Options -> Processing -> Provider -> WhiteboxTools -> WhiteboxTools executable.")

        vector_path = raster_path.replace(".tif", ".shp")
        ret = self._execute_wbt(
            'raster_to_vector_lines',
            feedback=feedback,
            i=raster_path,
            output=vector_path
        )
        
        if ret != 0 or not os.path.exists(vector_path):
            raise RuntimeError(f"Raster to vector lines failed: WhiteboxTools returned {ret}, output not found at {vector_path}")

        gdf = gpd.read_file(vector_path)
        
        if gdf.empty:
            warnings.warn(f"No vector features found in {vector_path}.")
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
            warnings.warn("No valid LineString geometries found after vectorization.")
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
            warnings.warn("No valid line segments found after vectorization.")
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
                Path to input DTM GeoTIFF (e.g. "/path/to/dtm.tif").
            filled_output_path (str, optional):
                Path to save the depression-filled DTM GeoTIFF (".tif").
            fdir_output_path (str, optional):
                Path to save the flow-direction raster GeoTIFF (".tif").
            facc_output_path (str, optional):
                Path to save the flow-accumulation raster GeoTIFF (".tif").
            facc_log_output_path (str, optional):
                Path to save the log-scaled accumulation raster GeoTIFF (".tif").
            streams_output_path (str, optional):
                Path to save the extracted stream raster GeoTIFF (".tif").
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
                feedback.pushInfo(f"[ExtractValleys] Filling depressions → {filled_output_path}")
            else:
                print(f"[ExtractValleys] Filling depressions → {filled_output_path}")
            try:
                ret = self._execute_wbt(
                    'breach_depressions_least_cost',
                    feedback=feedback,
                    dem=dtm_path,
                    output=filled_output_path,
                    dist=int(dist_facc),
                    fill=True,
                    min_dist=True
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
                ret = self._execute_wbt(
                    'd8_pointer',
                    feedback=feedback,
                    dem=filled_output_path,
                    output=fdir_output_path
                )
                if ret != 0 or not os.path.exists(fdir_output_path):
                    raise RuntimeError(f"[ExtractValleys] Flow direction failed: WhiteboxTools returned {ret}, output not found at {fdir_output_path}")
            except Exception as e:
                raise RuntimeError(f"[ExtractValleys] Flow direction failed: {e}")

            if feedback:
                feedback.pushInfo(f"[ExtractValleys] Flow accumulation → {facc_output_path}")
            else:
                print(f"[ExtractValleys] Flow accumulation → {facc_output_path}")
            try:
                ret = self._execute_wbt(
                    'd8_flow_accumulation',
                    feedback=feedback,
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
                TopoDrainCore._log_raster(input_raster=facc_output_path, output_path=facc_log_output_path)
                if not os.path.exists(facc_log_output_path):
                    raise RuntimeError(f"[ExtractValleys] Log-scaled accumulation output not found at {facc_log_output_path}")
            except Exception as e:
                raise RuntimeError(f"[ExtractValleys] Log-scaled accumulation failed: {e}")

            if feedback:
                feedback.pushInfo(f"[ExtractValleys] Extracting streams (threshold={accumulation_threshold})")
            else:
                print(f"[ExtractValleys] Extracting streams (threshold={accumulation_threshold})")
            try:
                ret = self._execute_wbt(
                    'extract_streams',
                    feedback=feedback,
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
                ret = self._execute_wbt(
                    'raster_streams_to_vector',
                    feedback=feedback,
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
                ret = self._execute_wbt(
                    'stream_link_identifier',
                    feedback=feedback,
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
                ret = self._execute_wbt(
                    'raster_streams_to_vector',
                    feedback=feedback,
                    streams=streams_vec_id,
                    d8_pntr=fdir_output_path,
                    output=streams_linked_output_path
                )
                if ret != 0 or not os.path.exists(streams_linked_output_path):
                    raise RuntimeError(f"[ExtractValleys] Converting linked streams failed: WhiteboxTools returned {ret}, output not found at {streams_linked_output_path}")
            except Exception as e:
                raise RuntimeError(f"[ExtractValleys] Converting linked streams failed: {e}")

            try:
                if feedback:
                    feedback.pushInfo("[ExtractValleys] Network analysis")
                else:
                    print("[ExtractValleys] Network analysis")
                ret = self._execute_wbt(
                    'VectorStreamNetworkAnalysis',
                    feedback=feedback,
                    streams=streams_linked_output_path,
                    dem=filled_output_path,
                    output=stream_network_output_path
                    )
                if ret != 0 or not os.path.exists(stream_network_output_path):
                    raise RuntimeError(f"[ExtractValleys] Network analysis failed: WhiteboxTools returned {ret}, output not found at {stream_network_output_path}")
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
            
            # Always create a reliable LINK_ID field as primary identifier
            # This addresses Windows case-sensitivity and temporary layer issues
            if 'FID' in gdf.columns or 'fid' in gdf.columns:
                fid_col = 'FID' if 'FID' in gdf.columns else 'fid'
                gdf['LINK_ID'] = gdf[fid_col]
            else:
                gdf['LINK_ID'] = range(1, len(gdf) + 1)
            
            if feedback:
                feedback.pushInfo("[ExtractValleys] Created reliable 'LINK_ID' field for cross-platform compatibility")
            else:
                print("[ExtractValleys] Created reliable 'LINK_ID' field for cross-platform compatibility")
            
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
                Path to the input DTM GeoTIFF.
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

        # 1) Invert the DTM
        print("[ExtractRidges] Inverting DTM…")
        inverted_dtm = os.path.join(self.temp_directory, f"inverted_dtm_{postfix}.tif")
        inverted_dtm = self._invert_dtm(dtm_path, inverted_dtm, feedback=feedback)
        print(f"[ExtractRidges] Inversion complete: {inverted_dtm}")

        # 2) Compute defaults for the four inverted outputs
        #    We leverage extract_valleys’ own default logic by passing these params through.
        print("[ExtractRidges] Preparing inverted-output paths…")
        # If the user did not supply, leave as None—extract_valleys will pick its defaults (which include postfix).
        inv_filled = inverted_filled_output_path
        inv_fdir   = inverted_fdir_output_path
        inv_facc   = inverted_facc_output_path
        inv_facc_log = inverted_facc_log_output_path
        inv_streams = inverted_streams_output_path

        # 3) Call extract_valleys on the inverted DTM
        print("[ExtractRidges] Running valley-extraction on inverted DTM…")
        ridges = self.extract_valleys(
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

        print(f"[ExtractRidges] Ridge extraction complete: {len(ridges)} features")
        return ridges


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
        Now processes each polygon in the perimeter separately, selecting nr_main features for each.
        If perimeter is not provided, uses the extent of valley_lines as perimeter.
        """
        if feedback:
            feedback.pushInfo("[ExtractMainValleys] Starting main valley extraction...")
        else:
            print("[ExtractMainValleys] Starting main valley extraction...")

        # Ensure LINK_ID field is present (required)
        if 'LINK_ID' not in valley_lines.columns:
            raise RuntimeError("[ExtractMainValleys] Input valley_lines must have a 'LINK_ID' attribute. Please use valley lines generated by the Create Valleys algorithm.")
        
        # Ensure TRIB_ID field is present in valley_lines
        if 'TRIB_ID' not in valley_lines.columns:
            # Check for case-insensitive variations
            trib_id_variations = [col for col in valley_lines.columns if col.lower() == 'trib_id']
            if trib_id_variations:
                valley_lines = valley_lines.copy()  # Avoid modifying original
                valley_lines = valley_lines.rename(columns={trib_id_variations[0]: 'TRIB_ID'})
                if feedback:
                    feedback.pushInfo(f"[ExtractMainValleys] Renamed '{trib_id_variations[0]}' column to 'TRIB_ID'")
                else:
                    print(f"[ExtractMainValleys] Renamed '{trib_id_variations[0]}' column to 'TRIB_ID'")
            else:
                # Create TRIB_ID field using LINK_ID values as fallback
                if valley_lines.empty:
                    raise RuntimeError("[ExtractMainValleys] Input valley_lines is empty")
                valley_lines = valley_lines.copy()  # Avoid modifying original
                valley_lines['TRIB_ID'] = valley_lines['LINK_ID']
                if feedback:
                    feedback.pushInfo("[ExtractMainValleys] Created 'TRIB_ID' column using LINK_ID values")
                else:
                    print("[ExtractMainValleys] Created 'TRIB_ID' column using LINK_ID values")

        # Ensure DS_LINK_ID field is present in valley_lines (optional field, can be null)
        if 'DS_LINK_ID' not in valley_lines.columns:
            # Check for case-insensitive variations
            ds_link_variations = [col for col in valley_lines.columns if col.lower() == 'ds_link_id']
            if ds_link_variations:
                valley_lines = valley_lines.copy()  # Avoid modifying original if not already copied
                valley_lines = valley_lines.rename(columns={ds_link_variations[0]: 'DS_LINK_ID'})
                if feedback:
                    feedback.pushInfo(f"[ExtractMainValleys] Renamed '{ds_link_variations[0]}' column to 'DS_LINK_ID'")
                else:
                    print(f"[ExtractMainValleys] Renamed '{ds_link_variations[0]}' column to 'DS_LINK_ID'")
            else:
                # Create DS_LINK_ID field with null values (not critical for main valley extraction)
                if not hasattr(valley_lines, '_is_copy') or valley_lines._is_copy is None:
                    valley_lines = valley_lines.copy()  # Avoid modifying original
                valley_lines['DS_LINK_ID'] = None
                if feedback:
                    feedback.pushInfo("[ExtractMainValleys] Created 'DS_LINK_ID' column with null values")
                else:
                    print("[ExtractMainValleys] Created 'DS_LINK_ID' column with null values")

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
        else:
            print("[ExtractMainValleys] Reading flow accumulation raster...")
        with rasterio.open(facc_path) as src:
            facc = src.read(1)
            transform = src.transform

        # Process each polygon in the perimeter separately
        all_merged_records = []
        global_fid_counter = 1
        
        for poly_idx, poly_row in perimeter.iterrows():
            single_polygon = gpd.GeoDataFrame([poly_row], crs=perimeter.crs)
            
            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Processing polygon {poly_idx + 1}/{len(perimeter)}...")
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
            valley_mask_path = self._vector_to_mask_raster(
                features=[valley_clipped],
                reference_raster_path=facc_path,
                output_path=valley_raster_path,
                flatten_lines=False,
                buffer_lines=False
            )
            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Valley mask created at {valley_mask_path}")
            else:
                print(f"[ExtractMainValleys] Valley mask created at {valley_mask_path}")

            # Read the valley mask data from the saved raster file
            with rasterio.open(valley_mask_path) as valley_src:
                valley_mask = valley_src.read(1)

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

            coords = [rasterio.transform.xy(transform, row, col) for row, col in zip(rows, cols)]
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
            
            points_joined = gpd.sjoin(
                points,
                valley_clipped[join_columns],
                how="inner"
            ).drop(columns="index_right")

            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Filtering ambiguous facc points for polygon {poly_idx + 1}...")
            else:
                print(f"[ExtractMainValleys] Filtering ambiguous facc points for polygon {poly_idx + 1}...")
            points_joined["geom_wkt"] = points_joined.geometry.apply(lambda geom: geom.wkt)
            geom_counts = points_joined.groupby("geom_wkt")["TRIB_ID"].nunique()
            valid_geoms = geom_counts[geom_counts == 1].index
            points_unique = points_joined[points_joined["geom_wkt"].isin(valid_geoms)].copy()

            if points_unique.empty:
                if feedback:
                    feedback.pushInfo(f"[ExtractMainValleys] No unique valley points found in polygon {poly_idx + 1}, skipping...")
                else:
                    print(f"[ExtractMainValleys] No unique valley points found in polygon {poly_idx + 1}, skipping...")
                continue

            if feedback:
                feedback.pushInfo(f"[ExtractMainValleys] Selecting top {nr_main} TRIB_IDs for polygon {poly_idx + 1}...")
            else:
                print(f"[ExtractMainValleys] Selecting top {nr_main} TRIB_IDs for polygon {poly_idx + 1}...")
            points_sorted = points_unique.sort_values("facc", ascending=False)
            points_top = points_sorted.drop_duplicates(subset="TRIB_ID").head(nr_main)

            if points_top.empty:
                if feedback:
                    feedback.pushInfo(f"[ExtractMainValleys] No main valley lines could be selected for polygon {poly_idx + 1}, skipping...")
                else:
                    print(f"[ExtractMainValleys] No main valley lines could be selected for polygon {poly_idx + 1}, skipping...")
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
                print(f"[ExtractMainValleys] Merging valley line segments for polygon {poly_idx + 1}...")
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
            else:
                print("[ExtractMainValleys] Clipping final valley lines to perimeter...")
            gdf = gpd.overlay(gdf, perimeter, how="intersection")

        if feedback:
            feedback.pushInfo(f"[ExtractMainValleys] Main valley extraction complete. {len(gdf)} valleys extracted from {len(perimeter)} polygons.")
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
            feedback.pushInfo("[ExtractMainRidges] Starting main ridge extraction using main valleys logic...")
        else:
            print("[ExtractMainRidges] Starting main ridge extraction using main valleys logic...")

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
        (second derivative). Keypoints are locations with high convexity, typically
        indicating a local change from concave to convex profile shape.

        The elevation profile is extracted along each valley line using the DTM at
        pixel resolution (all values along the line) and smoothed using a Savitzky-Golay 
        filter. The second derivative is then computed, and the top N points with the 
        strongest convex curvature are selected as keypoints.

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

        if feedback:
            feedback.pushInfo(f"[GetKeypoints] Starting keypoint detection on {len(valley_lines)} valley lines...")
        else:
            print(f"[GetKeypoints] Starting keypoint detection on {len(valley_lines)} valley lines...")

        with rasterio.open(dtm_path) as src:
            pixel_size = src.res[0]
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
                sampling_distance = pixel_size
                num_samples = max(int(length / sampling_distance), 2)  # At least 2 samples

                # Progress reporting for every line (or every 5 lines for large datasets)
                current_line = idx + 1
                if feedback:
                    if total_lines <= 20 or current_line % 5 == 0 or current_line == total_lines:
                        progress_pct = int((current_line / total_lines) * 100)
                        feedback.pushInfo(f"[GetKeypoints] Processing line {current_line}/{total_lines} ({progress_pct}%) - Line ID: {line_id}, Length: {length:.1f}m, Samples: {num_samples}")

                processed_lines += 1

                distances = np.linspace(0, length, num=num_samples)
                sample_points = [line.interpolate(d) for d in distances]
                coords = [(pt.x, pt.y) for pt in sample_points]
                elevations = [val[0] for val in sample_gen(src, coords)]

                # Smooth elevation profile
                elev_smooth = savgol_filter(elevations, smoothing_window, polyorder) ###### Maybe plot later?

                # Second derivative = curvature
                curvature = savgol_filter(elevations, smoothing_window, polyorder, deriv=2) ###### or better derivative of elev_smooth?

                # Find concave→convex transitions
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

        gdf = gpd.GeoDataFrame(results, geometry="geometry", crs=self.crs)

        if feedback:
            feedback.pushInfo(f"[GetKeypoints] Keypoint detection complete:")
            feedback.pushInfo(f"[GetKeypoints] - Total valley lines: {total_lines}")
            feedback.pushInfo(f"[GetKeypoints] - Processed lines: {processed_lines}")
            feedback.pushInfo(f"[GetKeypoints] - Skipped lines: {skipped_lines}")
            feedback.pushInfo(f"[GetKeypoints] - Total keypoints found: {len(gdf)}")
        else:
            print(f"[GetKeypoints] Keypoint detection complete: {len(gdf)} keypoints found from {processed_lines}/{total_lines} valley lines (skipped: {skipped_lines})")

        return gdf

    @staticmethod
    def _get_orthogonal_directions_start_points(
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
            line_geom (LineString): Reference line geometry used to determine orientation, e.g. valley line.
            max_offset (int): Maximum number of cells to move outward when searching.

        Returns:
            tuple: (left_point, right_point), or (None, None) if no valid points found.
        """
        print(f"[GetOrthogonalDirectionsStartPoints] Checking point {point}, max_offset={max_offset}")
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

                    # barrier >= 1 means forbidden
                    if barrier_mask[row_idx, col_idx] >= 1:
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
    def _get_linedirection_start_point( ### maybe adjust line_geom
        barrier_raster_path: str,
        line_geom: LineString,
        max_offset: int = 5,
        reverse: bool = False
    ) -> Point:
        """
        Determine a start point in the proceeding direction of a given input line.

        The function searches from the endpoint of the line along its direction until it finds
        a non-barrier cell. If reverse=True, follows the line geometry backwards from the endpoint.

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

        with rasterio.open(barrier_raster_path) as src:
            barrier_mask = src.read(1)
            rows, cols = barrier_mask.shape
            res = src.res[0]  # assumes square pixels

            row_ep, col_ep = src.index(end_point[0], end_point[1])
            print(f"[GetLinedirectionStartPoint] Endpoint raster index: row={row_ep}, col={col_ep}")
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
                            
                            row_idx, col_idx = src.index(test_x, test_y)
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
                # For forward, use tangent vector approach as before
                print("[GetLinedirectionStartPoint] Forward mode: using tangent vector extrapolation")
                
                # Create tangent vector based on line direction
                dx = end_point[0] - ref_point[0]
                dy = end_point[1] - ref_point[1]
                
                norm = np.linalg.norm([dx, dy])
                if norm > 0:
                    tangent = np.array([dx, dy]) / norm
                    print(f"[GetLinedirectionStartPoint] Tangent vector: {tangent}")
                else:
                    print("[GetLinedirectionStartPoint] Zero-length tangent vector, cannot proceed.")
                    return None

                def find_valid_point_forward():
                    for i in range(1, max_offset + 1):
                        offset = res * i
                        test_x = end_point[0] + tangent[0] * offset
                        test_y = end_point[1] + tangent[1] * offset

                        row_idx, col_idx = src.index(test_x, test_y)
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
    def _create_slope_cost_raster(
        dtm_path: str,
        start_point: Point,
        output_cost_raster_path: str,
        slope: float = 0.01,
        barrier_raster_path: str = None,
        penalty_exp: float = 2.0
    ) -> str:
        """
        Create a raster with cost values based on deviation from desired slope.
        You can now set penalty_exp>1 to punish larger deviations more heavily.

        Args:
            dtm_path (str): Path to the digital terrain model (GeoTIFF).
            start_point (Point): Starting point of the constant slope line.
            output_cost_raster_path (str): Path to output cost raster.
            slope (float): Desired slope (1% downhill = 0.01).
            barrier_raster_path (str): Path to a binary raster of barriers (1=barrier).
            penalty_exp (float): Exponent on the absolute deviation (>=1) of slope. 2.0 => quadratic penalty --> as higher the exponent as stronger penalty for larger deviations.

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
            dist = np.hypot(rr - key_row, cc - key_col) * src.res[0] ## maybe later with .distance()?
            expected_dz = -dist * slope

            # linear deviation
            deviation = np.abs(dz - expected_dz)

            # apply exponentiation for stronger penalty
            cost = deviation ** penalty_exp

            # enforce NoData
            cost[np.isnan(dtm)] = 1e6

            # Read barrier mask from raster if provided
            if barrier_raster_path is not None:
                with rasterio.open(barrier_raster_path) as bsrc:
                    barrier_mask = bsrc.read(1)
                    if barrier_mask.shape != cost.shape:
                        raise ValueError("Barrier raster shape does not match DTM shape.")
                    barrier_mask = barrier_mask.astype(bool)
                    if np.any(barrier_mask):
                        print("[TopoDrainCore] Applying barrier mask to cost raster.")
                        # Set cost to a very high value where barriers are present
                        cost[barrier_mask.astype(bool)] = 1e6

            # zero‐cost at the true start
            cost[key_row, key_col] = 0

            profile = src.profile
            profile.update(dtype=rasterio.float32, nodata=1e6)

        # write out
        with rasterio.open(output_cost_raster_path, "w", **profile) as dst:
            dst.write(cost.astype('float32'), 1)

        return output_cost_raster_path

    @staticmethod
    def _create_source_raster(
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

    @staticmethod
    def _select_best_destination_cell(
        accum_raster_path: str,
        destination_raster_path: str,
        best_destination_raster_path: str
    ) -> str:
        """
        Select the best destination cell from a binary destination raster based on
        minimum accumulated cost, and write it as a single-cell binary raster.

        Args:
            accum_raster_path (str): Path to the cost accumulation raster (GeoTIFF).
            destination_raster_path (str): Path to the binary destination raster (1 = destination, 0 = background).
            best_destination_raster_path (str): Path to output raster with only the best cell marked (GeoTIFF).

        Returns:
            str: Path to the output best destination raster (GeoTIFF).
            Point: The spatial coordinates of the best destination cell.
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

            # Find the minimum value and its index
            min_val = np.nanmin(acc_masked)
            min_indices = np.where(acc_masked == min_val)
            # Take the first occurrence if multiple
            row, col = min_indices[0][0], min_indices[1][0]

            # If you want the spatial coordinates:
            x, y = acc_src.xy(row, col)
            best_destination_point = Point(x, y)

            # Create output raster marking only the best cell
            best_dest = np.zeros_like(dest_data, dtype=np.uint8)
            best_dest[row, col] = 1

            profile = dest_src.profile
            profile.update(dtype=rasterio.uint8, nodata=0)
 
            # Write the best destination raster
            with rasterio.open(best_destination_raster_path, "w", **profile) as dst:
                dst.write(best_dest, 1)

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
        max_iterations_slope: int = 20,
        feedback=None
    ) -> LineString:
        """
        Trace lines with constant slope starting from a given point using an iterative approach.
        
        This function traces a line, checks where the line distance deviates too much from 
        the Euclidean distance from start point, cuts the line at that point, and continues 
        from there in subsequent iterations.

        Args:
            dtm_path (str): Path to the digital terrain model (GeoTIFF).
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
        print(f"[GetConstantSlopeLine] destination raster path: {destination_raster_path}")
        print(f"[GetConstantSlopeLine] barrier raster path: {barrier_raster_path}")
        print(f"[GetConstantSlopeLine] slope: {slope}, max_iterations_slope: {max_iterations_slope}, slope_deviation_threshold: {slope_deviation_threshold}")

        current_start_point = start_point
        accumulated_line_coords = []
        
        for iteration in range(max_iterations_slope):
            print(f"[GetConstantSlopeLine] Iteration {iteration + 1}/{max_iterations_slope}")
            
            # --- Temporary file paths ---
            cost_raster_path = os.path.join(self.temp_directory, f"cost_iter_{iteration}.tif")
            source_raster_path = os.path.join(self.temp_directory, f"source_iter_{iteration}.tif")
            accum_raster_path = os.path.join(self.temp_directory, f"accum_iter_{iteration}.tif")
            backlink_raster_path = os.path.join(self.temp_directory, f"backlink_iter_{iteration}.tif")
            best_destination_raster_path = os.path.join(self.temp_directory, f"destination_best_iter_{iteration}.tif")
            pathway_raster_path = os.path.join(self.temp_directory, f"pathway_iter_{iteration}.tif")

            print(f"[GetConstantSlopeLine] Create cost slope raster for iteration {iteration + 1}")
            # --- Create cost raster ---
            cost_raster_path = TopoDrainCore._create_slope_cost_raster(
                dtm_path=dtm_path,
                start_point=current_start_point,
                output_cost_raster_path=cost_raster_path,
                slope=slope,
                barrier_raster_path=barrier_raster_path
            )

            print(f"[GetConstantSlopeLine] Create source raster for iteration {iteration + 1}")
            # --- Create source raster ---
            source_raster_path = TopoDrainCore._create_source_raster(
                reference_raster_path=dtm_path,
                source_point=current_start_point,
                output_source_raster_path=source_raster_path
            )

            print(f"[GetConstantSlopeLine] Starting cost-distance analysis for iteration {iteration + 1}")
            # --- Run cost-distance analysis ---
            ret = self._execute_wbt(
                'cost_distance',
                feedback=feedback,
                source=source_raster_path,
                cost=cost_raster_path,
                out_accum=accum_raster_path,
                out_backlink=backlink_raster_path
            )
            
            if ret != 0 or not os.path.exists(accum_raster_path) or not os.path.exists(backlink_raster_path):
                raise RuntimeError(f"Cost distance analysis failed in iteration {iteration + 1}: WhiteboxTools returned {ret}")

            print(f"[GetConstantSlopeLine] Selecting best destination cell for iteration {iteration + 1}")
            # --- Select best destination cell ---
            best_destination_raster_path, best_destination_point = TopoDrainCore._select_best_destination_cell(
                accum_raster_path=accum_raster_path,
                destination_raster_path=destination_raster_path,
                best_destination_raster_path=best_destination_raster_path
            )

            print(f"[GetConstantSlopeLine] Tracing least-cost pathway for iteration {iteration + 1}")
            # --- Trace least-cost pathway ---
            ret = self._execute_wbt(
                'cost_pathway',
                feedback=feedback,
                destination=best_destination_raster_path,
                backlink=backlink_raster_path,
                output=pathway_raster_path
            )
            
            if ret != 0 or not os.path.exists(pathway_raster_path):
                raise RuntimeError(f"Cost pathway analysis failed in iteration {iteration + 1}")

            # --- Set correct NoData value for pathway raster --- ## Noch prüfen, ob das notwendig ist
            with rasterio.open(backlink_raster_path) as src:
                nodata_value = src.nodata
            with rasterio.open(pathway_raster_path, 'r+') as dst:
                dst.nodata = nodata_value

            print(f"[GetConstantSlopeLine] Converting pathway raster to LineString for iteration {iteration + 1}")
            # --- Convert to LineString ---
            line_segment = self._raster_to_linestring_wbt(
                pathway_raster_path, 
                snap_to_start_point=current_start_point, 
                snap_to_endpoint=best_destination_point, 
                feedback=feedback
            )

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
                # Read the destination raster and get the value at the cut point
                with rasterio.open(destination_raster_path) as dest_src:
                    row, col = dest_src.index(cut_point.x, cut_point.y)
                    if 0 <= row < dest_src.height and 0 <= col < dest_src.width:
                        dest_data = dest_src.read(1)
                        dest_value = dest_data[row, col]
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
                print(f"[GetConstantSlopeLine] Last iteration reached without finding fully valid line segment")

            if last_iteration or cut_point is None or reached_destination:
                # If we are at the last iteration or reached destination, we can wan to add fully line segement instead of doing another iteration
                if accumulated_line_coords:
                    # Skip first coordinate to avoid duplication
                    accumulated_line_coords.extend(line_segment.coords[1:])
                else:
                    accumulated_line_coords.extend(line_segment.coords)
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
                    iteration += 1
                    print(f"[GetConstantSlopeLine] Continuing from cut point: {cut_point}")
                else:
                    # If cutting failed, use the whole segment
                    print(f"[GetConstantSlopeLine] Cutting failed, using whole line segment")
                    if accumulated_line_coords:
                        # Skip first coordinate to avoid duplication
                        accumulated_line_coords.extend(line_segment.coords[1:])
                    else:
                        accumulated_line_coords.extend(line_segment.coords)
                    break

        # --- Combine all segments into final line ---
        if not accumulated_line_coords or len(accumulated_line_coords) < 2:
            warnings.warn("[GetConstantSlopeLine] No valid line segments could be extracted.")
            return None

        final_line = LineString(accumulated_line_coords)

        print("[GetConstantSlopeLine] Smoothing the resulting line")
        # --- Optional smoothing ---
        final_line = TopoDrainCore._smooth_linestring(final_line, sigma=1.0)

        return final_line

    def _get_iterative_constant_slope_line(
        self,
        dtm_path: str,
        start_point: Point,
        destination_raster_path: str,
        slope: float,
        barrier_raster_path: str,
        initial_barrier_value: int = None,
        max_iterations_barrier: int = 10,
        slope_deviation_threshold: float = 0.2,
        max_iterations_slope: int = 20,
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
    
        # Read destination raster data
        with rasterio.open(destination_raster_path) as dest_src:
            destination_profile = dest_src.profile.copy()
            orig_dest_data = dest_src.read(1).copy() # create a copy to avoid modifying the original raster
        
        # Read barrier raster data
        with rasterio.open(barrier_raster_path) as src:
            barrier_profile = src.profile.copy() 
            orig_barrier_data = src.read(1).copy() # create a copy to avoid modifying the original raster
        
        while current_iteration < max_iterations_barrier:
            if feedback:
                feedback.pushInfo(f"[IterativeConstantSlopeLine] Iteration {current_iteration + 1}/{max_iterations_barrier}")
            else:
                print(f"[IterativeConstantSlopeLine] Iteration {current_iteration + 1}/{max_iterations_barrier}")

            # Debug: Print start point and extracted value
            if feedback:
                feedback.pushInfo(f"[IterativeConstantSlopeLine] Start point: {current_start_point.wkt}")
                feedback.pushInfo(f"[IterativeConstantSlopeLine] Start barrier value: {current_barrier_value}")
            else:
                print(f"[IterativeConstantSlopeLine] Start point: {current_start_point.wkt}")
                print(f"[IterativeConstantSlopeLine] Start barrier value: {current_barrier_value}")

            print(f"[IterativeConstantSlopeLine] Creating working rasters for _get_constant_slope_line...")
            # --- Create barrier raster for _get_constant_slope_line ---
            # Get dtype from profiles to ensure consistency
            barrier_dtype = barrier_profile['dtype']
            dest_dtype = destination_profile['dtype']
            
            if current_barrier_value:
                working_barrier_data = np.where(orig_barrier_data == current_barrier_value, 1, 0).astype(barrier_dtype) # current barrier act as barrier and not as destination
                working_destination_data = np.where((orig_dest_data == 1) | ((orig_barrier_data >= 1) & (orig_barrier_data != current_barrier_value)), 1, 0).astype(dest_dtype)  # all other barriers act as temporary destinations except the current one
            else:
                working_barrier_data = None # all barriers acting as temporary destinations and not as barriers
                working_destination_data = np.where((orig_dest_data == 1) | (orig_barrier_data >= 1), 1, 0).astype(dest_dtype) 
                
            # Save barrier mask
            if working_barrier_data is not None:
                working_barrier_raster_path = os.path.join(self.temp_directory, f"barrier_iter_{current_iteration}.tif")
                with rasterio.open(working_barrier_raster_path, "w", **barrier_profile) as dst:
                    dst.write(working_barrier_data, 1)
                if feedback:
                    feedback.pushInfo(f"[IterativeConstantSlopeLine] Working barrier raster created at {working_barrier_raster_path}") 
                else:
                    print(f"[IterativeConstantSlopeLine] Working barrier raster created at {working_barrier_raster_path}")
            else:
                working_barrier_raster_path = None
                if feedback:
                    feedback.pushInfo("[IterativeConstantSlopeLine] No working barrier raster created (all barriers act as temporary destinations).")
                else:
                    print("[IterativeConstantSlopeLine] No working barrier raster created (all barriers act as temporary destinations).")
                        
            working_destination_raster_path = os.path.join(self.temp_directory, f"destination_iter_{current_iteration}.tif")
            # Save destination mask
            with rasterio.open(working_destination_raster_path, "w", **destination_profile) as dst:
                dst.write(working_destination_data, 1)
            if feedback:
                feedback.pushInfo(f"[IterativeConstantSlopeLine] Working destination raster created at {working_destination_raster_path}")
            else:
                print(f"[IterativeConstantSlopeLine] Working destination raster created at {working_destination_raster_path}")

            # Call _get_constant_slope_line with current parameters
            if feedback:
                feedback.pushInfo(f"[IterativeConstantSlopeLine] Tracing from point {current_start_point}")
            else:
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
                if feedback:
                    feedback.pushWarning(f"[IterativeConstantSlopeLine] No line found in iteration {current_iteration + 1}")
                else:
                    warnings.warn(f"[IterativeConstantSlopeLine] No line found in iteration {current_iteration + 1}")
                break

            line_coords = list(line_segment.coords)
            # Check if endpoint is on original (final) destination
            endpoint = Point(line_coords[-1])
            print(f"[IterativeConstantSlopeLine] Endpoint iteration {current_iteration + 1}: {endpoint.wkt}")
            final_destination_found = False
            with rasterio.open(destination_raster_path) as dest_src:
                end_row, end_col = dest_src.index(endpoint.x, endpoint.y)
                if 0 <= end_row < orig_dest_data.shape[0] and 0 <= end_col < orig_dest_data.shape[1]:
                    if orig_dest_data[end_row, end_col] == 1:
                        if feedback:
                            feedback.pushInfo(f"[IterativeConstantSlopeLine] Reached final destination in iteration {current_iteration + 1}")
                        else:
                            print(f"[IterativeConstantSlopeLine] Reached final destination in iteration {current_iteration + 1}")
                        final_destination_found = True
                    else:
                        print(f"[IterativeConstantSlopeLine] Endpoint not on destination in iteration {current_iteration + 1}, checking barriers.")
            
            last_iteration = (current_iteration == max_iterations_barrier - 1)
            if not final_destination_found and last_iteration:
                if feedback:
                    feedback.pushWarning("[IterativeConstantSlopeLine] Maximum iterations reached without finding a fully valid line")
                else:
                    print("[IterativeConstantSlopeLine] Maximum iterations reached without finding a fully valid line")

            if final_destination_found or last_iteration: 
                # If we reached the last iteration, add this final segment to avoid cutting in the last iteration:
                # If we reached the final destination, add this final segment and stop
                if accumulated_line_coords:
                    # Skip first coordinate to avoid duplication
                    accumulated_line_coords.extend(line_segment.coords[1:])
                else:
                    accumulated_line_coords.extend(line_segment.coords)
                break

            else: 
                # Get barrier value at endpoint for next iteration
                with rasterio.open(barrier_raster_path) as barrier_src:
                    end_row, end_col = barrier_src.index(endpoint.x, endpoint.y)
                    if 0 <= end_row < barrier_src.height and 0 <= end_col < barrier_src.width:
                        end_barrier_value = int(barrier_src.read(1)[end_row, end_col])
                    else:
                        end_barrier_value = None
                    print(f"[IterativeConstantSlopeLine] Iteration reached barrier: {end_barrier_value}")

                if end_barrier_value:
                    # Get start point for next iteration next to the barrier (back where the line came from)
                    next_start_point = TopoDrainCore._get_linedirection_start_point(
                        barrier_raster_path=barrier_raster_path,
                        line_geom=line_segment,
                        max_offset=5,  # adjust as needed
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
            current_barrier_value = end_barrier_value if end_barrier_value else None  # Update barrier value for next iteration
            current_start_point = next_start_point  # Update start point for next iteration
            current_iteration += 1


        if len(accumulated_line_coords) >= 2:
            line = LineString(accumulated_line_coords)
            if feedback:
                feedback.pushInfo(f"[IterativeConstantSlopeLine] Completed after {current_iteration + 1} iterations")
            else:
                print(f"[IterativeConstantSlopeLine] Completed after {current_iteration + 1} iterations")
            return line
        else:
            if feedback:
                feedback.pushWarning("[IterativeConstantSlopeLine] No valid line could be created")
            else:
                print("[IterativeConstantSlopeLine] No valid line could be created")
            return None

    def get_constant_slope_lines(
        self,
        dtm_path: str,
        start_points: gpd.GeoDataFrame,
        destination_features: list[gpd.GeoDataFrame],
        slope: float = 0.01,
        barrier_features: list[gpd.GeoDataFrame] = None,
        allow_barriers_as_temp_destination: bool = False,
        max_iterations_barrier: int = 30,
        slope_deviation_threshold: float = 0.2,
        max_iterations_slope: int = 20,
        feedback=None
    ) -> gpd.GeoDataFrame:
        """
        Trace lines with constant slope starting from given points using a cost-distance approach
        based on slope deviation, snapping true original start-points only when they overlapped barrier lines.
        All barrier_features (lines, polygons, points) are rasterized into barrier_mask,
        but only the line geometries are used for splitting and offsetting start points.
        
        Args:
            dtm_path (str): Path to the digital terrain model (GeoTIFF).
            start_points (gpd.GeoDataFrame): Starting points for slope line tracing (e.g. Keypoints).
            destination_features (list[gpd.GeoDataFrame]): List of destination features (e.g. main ridge lines, area of interest).
            slope (float): Desired slope for the lines (e.g., 0.01 for 1% downhill).
            barrier_features (list[gpd.GeoDataFrame], optional): List of barrier features to avoid (e.g. main valley lines).
            allow_barriers_as_temp_destination (bool): If True, barriers are included as temporary destinations for iterative tracing.
            max_iterations_barrier (int): Maximum number of iterations for iterative tracing when allowing barriers as temporary destinations. Default 30.
            max_iterations_slope (int): Maximum number of iterations for line refinement (1-50, higher values allow more complex paths). Default 20.
            slope_deviation_threshold (float): Maximum allowed relative deviation from expected slope (0.0-1.0, e.g., 0.2 for 20% deviation before line cutting). Default 0.2.
            feedback (QgsProcessingFeedback, optional): Optional feedback object for progress reporting/logging.
            
        Returns:
            gpd.GeoDataFrame: Traced constant slope lines.
        """
        if feedback:
            feedback.pushInfo("[ConstantSlopeLines] Starting tracing")
        else:
            print("[ConstantSlopeLines] Starting tracing")

        # store original start points for adding later to the traced line geometry if starting from adjusted start point
        original_pts = start_points.copy()

        # --- Destination mask ---
        if feedback:
            feedback.pushInfo("[ConstantSlopeLines] Building destination mask…")
        else:
            print("[ConstantSlopeLines] Building destination mask…")
        
        # Process destination features - convert polygons to boundaries and flatten to LineStrings
        destination_processed = TopoDrainCore._features_to_single_linestring(destination_features)

        # Create binary destination mask
        destination_raster_path = self._vector_to_mask_raster([destination_processed], dtm_path, unique_values=False, flatten_lines=False, buffer_lines=True) # destination_processed are already flattened. Binary mask raster, no need of unique values

        # Read destination mask for later processing
        with rasterio.open(destination_raster_path) as src:
            destination_mask = src.read(1)
            
        if feedback:
            feedback.pushInfo("[ConstantSlopeLines] Destination mask ready")
        else:
            print("[ConstantSlopeLines] Destination mask ready")

        # --- Barrier mask ---
        if barrier_features:
            if feedback:
                feedback.pushInfo("[ConstantSlopeLines] Preparing barrier mask...")
            else:
                print("[ConstantSlopeLines] Preparing barrier mask...")

            # Process barrier features - convert polygons to boundaries like destinations
            barrier_processed = TopoDrainCore._features_to_single_linestring(barrier_features)

            # Create binary barrier mask (always binary: 0=free, 1=barrier)
            barrier_raster_path = self._vector_to_mask_raster([barrier_processed], dtm_path, unique_values=False, flatten_lines=False, buffer_lines=True)
            
            # Create barrier value raster with unique values to identify which barrier feature each cell belongs to (needed for overlap analysis and allow barrier as destination option)
            barrier_value_raster_path, barrier_id_to_geom = self._vector_to_mask_raster([barrier_processed], dtm_path, unique_values=True, flatten_lines=False, buffer_lines=True)

            # Read barrier masks for processing
            with rasterio.open(barrier_raster_path) as src:
                barrier_mask = src.read(1)
            with rasterio.open(barrier_value_raster_path) as src:
                barrier_value_mask = src.read(1)

            if feedback:
                feedback.pushInfo(f"[ConstantSlopeLines] Barrier raster saved to {barrier_raster_path}")
            else:
                print(f"[ConstantSlopeLines] Barrier raster saved to {barrier_raster_path}")

            # Handle overlapping barrier and original destination cells --> adjust destination mask
            dest_barrier_overlap = (barrier_mask == 1) & (destination_mask == 1)
            if np.any(dest_barrier_overlap):
                num_overlaps = np.sum(dest_barrier_overlap)
                if feedback:
                    feedback.pushInfo(f"[ConstantSlopeLines] Found {num_overlaps} overlapping barrier/original destination cells")
                else:
                    print(f"[ConstantSlopeLines] Found {num_overlaps} overlapping barrier/original destination cells")
                # Set destination_mask to 0 at overlapping cells, because not possible to be barrier and destination at the same time
                destination_mask[dest_barrier_overlap] = 0
                # update desination raster
                with rasterio.open(destination_raster_path, 'r+') as dst:
                    dst.write(destination_mask, 1)
            else:
                if feedback:
                    feedback.pushInfo("[ConstantSlopeLines] No overlapping barrier/original destination cells found")
                else:
                    print("[ConstantSlopeLines] No overlapping barrier/original destination cells found")
            
            # Check if start points are on barrier raster cells using precise raster-based overlap detection
            if feedback:
                feedback.pushInfo("[ConstantSlopeLines] Check if start points lies on barrier features...")
            else:
                print("[ConstantSlopeLines] Check if start points lies on barrier features...")
            
            # Get barrier values at each start point location using raster lookup
            point_barrier_info = []
            with rasterio.open(dtm_path) as dtm_src:
                for orig_idx, row in original_pts.iterrows():
                    pt = row.geometry
                    # Get raster indices for the point
                    pt_r, pt_c = dtm_src.index(pt.x, pt.y)
                    
                    # Check if point is within raster bounds
                    if (0 <= pt_r < barrier_mask.shape[0] and 0 <= pt_c < barrier_mask.shape[1]):
                        barrier_binary_value = int(barrier_mask[pt_r, pt_c])
                        barrier_feature_id = int(barrier_value_mask[pt_r, pt_c])
                        
                        point_barrier_info.append({
                            'orig_idx': orig_idx,
                            'is_on_barrier': barrier_binary_value == 1,
                            'barrier_feature_id': barrier_feature_id if barrier_binary_value == 1 else None,
                            'row': row
                        })
                    else:
                        # Point outside raster bounds - treat as non-overlapping
                        point_barrier_info.append({
                            'orig_idx': orig_idx,
                            'is_on_barrier': False,
                            'barrier_feature_id': None,
                            'row': row
                        })
            
            # Count overlapping vs non-overlapping for reporting
            overlapping_count = sum(1 for info in point_barrier_info if info['is_on_barrier'])
            non_overlapping_count = len(point_barrier_info) - overlapping_count
            
            if feedback:
                feedback.pushInfo(f"[ConstantSlopeLines]  → {overlapping_count} overlapping on barrier cells, {non_overlapping_count} non-overlapping")
            else:
                print(f"[ConstantSlopeLines]  → {overlapping_count} overlapping on barrier cells, {non_overlapping_count} non-overlapping")
        else:
            if feedback:
                feedback.pushInfo("[ConstantSlopeLines] No barrier features provided")
            else:
                print("[ConstantSlopeLines] No barrier features provided")
            barrier_mask = None
            barrier_raster_path = None
            # Create point_barrier_info for all points as non-overlapping
            point_barrier_info = []
            for orig_idx, row in original_pts.iterrows():
                point_barrier_info.append({
                    'orig_idx': orig_idx,
                    'is_on_barrier': False,
                    'barrier_feature_id': None,
                    'row': row
                })

        # --- Trace slope lines directly from point_barrier_info ---
        results = []
        total_points = overlapping_count*2 + non_overlapping_count  # each overlapping point creates two lines (left and right offset)
        current_point = 0

        if feedback:
            feedback.pushInfo("[ConstantSlopeLines] Starting slope line tracing...")
        else:
            print("[ConstantSlopeLines] Starting slope line tracing...")

        for info in point_barrier_info:
            orig_idx = info['orig_idx']
            orig_pt = info['row'].geometry
            is_on_barrier = info['is_on_barrier']
            barrier_feature_id = info['barrier_feature_id']
            orig_attrs = original_pts.loc[orig_idx].drop(labels="geometry").to_dict()

            if is_on_barrier:
                # Point is on barrier - need to create adjusted start points
                if feedback:
                    feedback.pushInfo(f"[ConstantSlopeLines] Point {orig_idx} on barrier feature {barrier_feature_id} → creating offset points")
                else:
                    print(f"[ConstantSlopeLines] Point {orig_idx} on barrier feature {barrier_feature_id} → creating offset points")

                # Get the specific barrier geometry that this point overlaps with
                barrier_geom = barrier_id_to_geom.get(barrier_feature_id)
                if barrier_geom is None:
                    if feedback:
                        feedback.pushInfo(f"[ConstantSlopeLines] Warning: No barrier geometry found for feature ID {barrier_feature_id} - skipping point {orig_idx}")
                    else:
                        print(f"[ConstantSlopeLines] Warning: No barrier geometry found for feature ID {barrier_feature_id} - skipping point {orig_idx}")
                    continue

                # Get orthogonal offset points
                left_pt, right_pt = TopoDrainCore._get_orthogonal_directions_start_points(
                    barrier_raster_path=barrier_raster_path,
                    point=orig_pt,
                    line_geom=barrier_geom
                )

                # Trace from both offset points (if they exist)
                adj_idx = 0
                for offset_pt, offset_name in [(left_pt, "left"), (right_pt, "right")]:
                    current_point += 1
                    if offset_pt is None:
                        # Check if adjusted point is still on barrier (warn if so) ##### maybe not necessary
                        if offset_pt:
                            if feedback:
                                feedback.pushInfo(f"[ConstantSlopeLines] Warning: No {offset_name} offset point could be created for {orig_idx}")
                            else:
                                print(f"[ConstantSlopeLines] Warning: No {offset_name} offset point could be created for {orig_idx}")
                        continue

                    # Progress reporting
                    if feedback:
                        progress_pct = int((current_point / total_points) * 100)
                        feedback.setProgress(progress_pct)
                        feedback.pushInfo(f"[ConstantSlopeLines] Processing point {current_point}/{total_points} ({progress_pct}%) - Point {orig_idx}...")
                    else:
                        print(f"[ConstantSlopeLines] Processing point {current_point}/{total_points} - Point {orig_idx}...")

                    adj_idx += 1
                    if feedback:
                        feedback.pushInfo(f"[ConstantSlopeLines]   → Tracing from {offset_name} offset point for {orig_idx}")
                    else:
                        print(f"[ConstantSlopeLines]   → Tracing from {offset_name} offset point for {orig_idx}")


                    # Trace line from offset point
                    if allow_barriers_as_temp_destination and barrier_features:
                        raw_line = self._get_iterative_constant_slope_line(
                            dtm_path=dtm_path,
                            start_point=offset_pt,
                            destination_raster_path=destination_raster_path,
                            slope=slope,
                            barrier_raster_path=barrier_value_raster_path,  # Use barrier value raster for iterative tracing
                            initial_barrier_value=barrier_feature_id,
                            max_iterations_barrier=max_iterations_barrier,
                            slope_deviation_threshold=slope_deviation_threshold,
                            feedback=feedback
                        )
                    else:
                        raw_line = self._get_constant_slope_line(
                            dtm_path=dtm_path,
                            start_point=offset_pt,
                            destination_raster_path=destination_raster_path,
                            slope=slope,
                            barrier_raster_path=barrier_raster_path,  # Use binary barrier raster for simple tracing
                            max_iterations_slope=max_iterations_slope,
                            slope_deviation_threshold=slope_deviation_threshold,
                            feedback=None
                        )

                    if not raw_line:
                        if feedback:
                            feedback.pushInfo(f"[ConstantSlopeLines]   → No line from {offset_name} offset for {orig_idx}")
                        else:
                            print(f"[ConstantSlopeLines]   → No line from {offset_name} offset for {orig_idx}")
                        continue

                    # Snap line to original start point
                    raw_line = TopoDrainCore._snap_line_to_point(raw_line, orig_pt, "start")

                    # Append to results
                    results.append({
                        "geometry": raw_line,
                        "orig_index": orig_idx,
                        "adj_index": adj_idx,
                        **orig_attrs
                    })

                    if feedback:
                        feedback.pushInfo(f"[ConstantSlopeLines]   → Line created from {offset_name} offset for {orig_idx}")
                    else:
                        print(f"[ConstantSlopeLines]   → Line created from {offset_name} offset for {orig_idx}")

            else:
                # Point is not on barrier - trace directly from original point
                current_point += 1

                # Progress reporting
                if feedback:
                    progress_pct = int((current_point / total_points) * 100)
                    feedback.setProgress(progress_pct)
                    feedback.pushInfo(f"[ConstantSlopeLines] Processing point {current_point}/{total_points} ({progress_pct}%) - Point {orig_idx}...")
                else:
                    print(f"[ConstantSlopeLines] Processing point {current_point}/{total_points} - Point {orig_idx}...")
                
                if feedback:
                    feedback.pushInfo(f"[ConstantSlopeLines] Point {orig_idx} not on barrier → tracing directly")
                else:
                    print(f"[ConstantSlopeLines] Point {orig_idx} not on barrier → tracing directly")

                # Trace line from original point
                if allow_barriers_as_temp_destination:
                    raw_line = self._get_iterative_constant_slope_line(
                        dtm_path=dtm_path,
                        start_point=orig_pt,
                        destination_raster_path=destination_raster_path,
                        slope=slope,
                        barrier_raster_path=barrier_value_raster_path,  # Use barrier value raster for iterative tracing
                        initial_barrier_value=None,  # No initial barrier value for non-overlapping points
                        max_iterations_barrier=max_iterations_barrier,  # max iterations for iterative tracing (nr of times barriers used as temporary destinations)
                        slope_deviation_threshold=slope_deviation_threshold,
                        feedback=feedback
                    )
                else:
                    raw_line = self._get_constant_slope_line(
                        dtm_path=dtm_path,
                        start_point=orig_pt,
                        destination_raster_path=destination_raster_path,
                        slope=slope,
                        barrier_raster_path=barrier_raster_path,  # Use binary barrier raster for simple tracing
                        max_iterations_slope=max_iterations_slope,
                        slope_deviation_threshold=slope_deviation_threshold,
                        feedback=None
                    )

                if not raw_line:
                    if feedback:
                        feedback.pushInfo(f"[ConstantSlopeLines]   → No line for point {orig_idx}")
                    else:
                        print(f"[ConstantSlopeLines]   → No line for point {orig_idx}")
                    continue

                # Append to results (no snapping needed since we started from original point)
                results.append({
                    "geometry": raw_line,
                    "orig_index": orig_idx,
                    **orig_attrs
                })

                if feedback:
                    feedback.pushInfo(f"[ConstantSlopeLines]   → Line created for {orig_idx}")
                else:
                    print(f"[ConstantSlopeLines]   → Line created for {orig_idx}")

        if not results:
            if feedback:
                feedback.reportError("[ConstantSlopeLines] No slope lines could be created.")
            raise RuntimeError("No slope lines could be created.")
        
        if feedback:
            feedback.pushInfo(f"[ConstantSlopeLines] Done: generated {len(results)} lines")
            feedback.setProgress(100)
        else:
            print(f"[ConstantSlopeLines] Done: generated {len(results)} lines")

        # build GeoDataFrame including all original attributes
        out_gdf = gpd.GeoDataFrame(results, crs=self.crs)
        
        return out_gdf
    
    def create_keylines(self, dtm_path, start_points, valley_lines, ridge_lines, slope, perimeter, 
                        change_after=None, slope_after=None, slope_deviation_threshold=0.2, max_iterations_slope=20, feedback=None):
        """
        Create keylines using an iterative process:
        1. Trace from start points to ridges (using valleys as barriers)
        2. Check if endpoints are on ridges, create new start points beyond ridges
        3. Trace from new start points to valleys (using ridges as barriers)
        4. Continue iteratively while endpoints reach target features

        All output keylines will be oriented from valley to ridge (valley → ridge direction).

        Parameters:
        -----------
        dtm_path : str
            Path to the digital terrain model (GeoTIFF)
        start_points : GeoDataFrame
            Input keypoints to start keyline creation from
        valley_lines : GeoDataFrame
            Valley line features to use as barriers/destinations
        ridge_lines : GeoDataFrame
            Ridge line features to use as barriers/destinations
        slope : float
            Target slope for the constant slope lines (e.g., 0.01 for 1%)
        perimeter : GeoDataFrame
            Area of interest (perimeter) that always acts as destination feature (e.g. watershed, parcel polygon)
        change_after : float, optional
            Fraction of line length where slope changes (0.0-1.0, e.g., 0.5 = from halfway). If None, no slope adjustment is applied.
        slope_after : float, optional
            New slope to apply after the change point (e.g., 0.005 for 0.5% downhill). Required if change_after is provided.
        slope_deviation_threshold : float, optional
            Maximum allowed relative deviation from expected slope (0.0-1.0, e.g., 0.2 for 20% deviation before line cutting). Default 0.2.
        max_iterations_slope : int, optional
            Maximum number of iterations for line refinement (1-50). Default 20.
        feedback : QgsProcessingFeedback
            Feedback object for progress reporting

        Returns:
        --------
        GeoDataFrame
            Combined keylines from all stages, all oriented from valley to ridge.
        """
        if feedback:
            feedback.pushInfo("Starting iterative keyline creation process...")
            feedback.setProgress(5)
        else:
            print("Starting iterative keyline creation process...")
            print("Progress: 5%")
        # Create raster .tif files for valley_lines and ridge_lines to use in _get_linedirection_start_point
        valley_lines_raster_path = os.path.join(self.temp_directory, "valley_lines_mask.tif")
        ridge_lines_raster_path = os.path.join(self.temp_directory, "ridge_lines_mask.tif")

        # Rasterize valley_lines - now returns path directly
        valley_lines_raster_path = self._vector_to_mask_raster([valley_lines], dtm_path, output_path=valley_lines_raster_path, unique_values=False, flatten_lines=True, buffer_lines=True)

        # Rasterize ridge_lines - now returns path directly
        ridge_lines_raster_path = self._vector_to_mask_raster([ridge_lines], dtm_path, output_path=ridge_lines_raster_path, unique_values=False, flatten_lines=True, buffer_lines=True)
        
        # Rasterize perimeter for precise endpoint checking
        perimeter_raster_path = None
        perimeter_mask = None
        if perimeter is not None:
            # If perimeter is a polygon, use its boundary for rasterization
            if perimeter.geom_type.isin(["Polygon", "MultiPolygon"]).any():
                perimeter_is_polygon = True
                perimeter_line = perimeter.copy()
                perimeter_line["geometry"] = perimeter_line.boundary # maybe .unary_union?
            else:
                perimeter_line = perimeter

            perimeter_raster_path = os.path.join(self.temp_directory, "perimeter_mask.tif")
            perimeter_raster_path = self._vector_to_mask_raster(
                [perimeter_line],
                dtm_path,
                output_path=perimeter_raster_path,
                unique_values=False,
                flatten_lines=True,
                buffer_lines=True
            )
            # Read perimeter mask for endpoint checking
            with rasterio.open(perimeter_raster_path) as src:
                perimeter_mask = src.read(1)
        
        # Read barrier masks for endpoint checking
        with rasterio.open(valley_lines_raster_path) as src:
            valley_mask = src.read(1)
        with rasterio.open(ridge_lines_raster_path) as src:
            ridge_mask = src.read(1)
        # Initialize variables
        all_keylines = []
        current_start_points = start_points.copy()
        stage = 1
        
        # Set a maximum number of iterations to prevent infinite loops
        expected_stages = (len(valley_lines) + len(ridge_lines)) + 1  # Rough estimate
        max_iterations_keyline = expected_stages + 10  # Set a reasonable limit based on input features (+10 for safety)

        # Iterate until no new start points are found or max iterations reached
        while not current_start_points.empty and stage <= max_iterations_keyline:
            # Progress: 5% at start, 95% spread over expected_stages
            progress = 5 + int((stage - 1) * (95 / expected_stages))
            if feedback:
                feedback.pushInfo(f"**** Stage {stage}/~{expected_stages}: Processing {len(current_start_points)} start points...***")
                feedback.setProgress(min(progress, 99))
            else:
                print(f"**** Stage {stage}/~{expected_stages}: Processing {len(current_start_points)} start points...****")
                print(f"Progress: {progress}%")
            
            # Determine destination and barrier features based on stage
            if stage % 2 == 1:  # Odd stages: trace to ridges, valleys as barriers
                destination_features = [ridge_lines]
                barrier_features = [valley_lines]
                target_type = "ridges"
                use_slope = slope  # Use slope as is for downhill tracing
            else:  # Even stages: trace to valleys, ridges as barriers
                destination_features = [valley_lines] 
                barrier_features = [ridge_lines]
                target_type = "valleys"
                use_slope = -slope  # Invert slope for uphill tracing
            
            # Always add perimeter als destination feature if provided
            if perimeter is not None:
                destination_features.append(perimeter)
                
            if feedback:
                feedback.pushInfo(f"Stage {stage}: Tracing to {target_type}...")
            else:
                print(f"Stage {stage}: Tracing to {target_type}...")
        
            if feedback:
                feedback.pushInfo(f"* for more details on tracing of stage {stage}, see in python console")

            if feedback:
                feedback.pushInfo(f"Current start points: {current_start_points}")

            # Trace constant slope lines
            stage_lines = self.get_constant_slope_lines(
                dtm_path=dtm_path,
                start_points=current_start_points,
                destination_features=destination_features,
                slope=use_slope,
                barrier_features=barrier_features,
                allow_barriers_as_temp_destination=False,
                slope_deviation_threshold=slope_deviation_threshold,
                max_iterations_slope=max_iterations_slope,
                feedback=None # want to keep feedback for the main loop, not for each tracing call here
            )

            if feedback:
                feedback.pushInfo(f"Stage {stage}: Tracing complete, processing results...")

            if stage_lines.empty:
                if feedback:
                    feedback.pushInfo(f"Stage {stage}: No lines generated, stopping...")
                else:
                    print(f"Stage {stage}: No lines generated, stopping...")
                break
                
            if feedback:
                feedback.pushInfo(f"Stage {stage} complete: {len(stage_lines)} lines to {target_type}")
                feedback.pushInfo(f"Stage lines: {stage_lines}")
            else:
                print(f"Stage {stage} complete: {len(stage_lines)} lines to {target_type}")

            # Apply slope adjustment if parameters are provided
            if change_after is not None and slope_after is not None:
                if feedback:
                    feedback.pushInfo(f"Stage {stage}: Applying slope adjustment after {change_after*100:.1f}% with new slope {slope_after}")
                else:
                    print(f"Stage {stage}: Applying slope adjustment after {change_after*100:.1f}% with new slope {slope_after}")
                
                # Apply the slope adjustment using the adjust_constant_slope_after method
                stage_lines = self.adjust_constant_slope_after(
                    dtm_path=dtm_path,
                    input_lines=stage_lines,
                    change_after=change_after,
                    slope_after=slope_after,  # Use slope_after as-is because line direction is already handled in stage_lines
                    destination_features=destination_features,
                    barrier_features=barrier_features,
                    allow_barriers_as_temp_destination=False,
                    slope_deviation_threshold=slope_deviation_threshold,
                    max_iterations_slope=max_iterations_slope,
                    feedback=None  # Keep feedback at main level
                )
                
                if feedback:
                    feedback.pushInfo(f"Stage {stage}: Slope adjustment complete, {len(stage_lines)} adjusted lines")
                else:
                    print(f"Stage {stage}: Slope adjustment complete, {len(stage_lines)} adjusted lines")

            # Check endpoints and create new start points if they're on target features
            new_start_points = []
            # Define which raster mask to use for barrier checking
            new_barrier_raster_path = ridge_lines_raster_path if target_type == "ridges" else valley_lines_raster_path
            new_barrier_mask = ridge_mask if target_type == "ridges" else valley_mask
            new_barrier_feature = ridge_lines if target_type == "ridges" else valley_lines
            
            if feedback:    
                feedback.pushInfo(f"Stage {stage}: Checking endpoints on {target_type} using raster-based detection...")
            else:
                print(f"Stage {stage}: Checking endpoints on {target_type} using raster-based detection...")

            # Iterate through each line in the stage_lines GeoDataFrame
            with rasterio.open(dtm_path) as dtm_src:
                for _, line_row in stage_lines.iterrows():
                    line_geom = line_row.geometry
                    if hasattr(line_geom, 'coords') and len(line_geom.coords) >= 2:
                        end_point = Point(line_geom.coords[-1])
                        if feedback:
                            feedback.pushInfo(f"Stage {stage}: Checking endpoint {end_point.wkt} for new start point...")
                        
                        # Get raster coordinates for endpoint
                        end_r, end_c = dtm_src.index(end_point.x, end_point.y)
                        
                        # Check if endpoint is within raster bounds
                        if not (0 <= end_r < dtm_src.height and 0 <= end_c < dtm_src.width):
                            if feedback:
                                feedback.pushInfo(f"Stage {stage}: Endpoint outside raster bounds, skipping...")
                            continue
                        
                        # Check if endpoint reached the perimeter using raster lookup
                        reached_perimeter = False
                        if perimeter is not None and perimeter_mask is not None:
                            perimeter_value = int(perimeter_mask[end_r, end_c])
                            reached_perimeter = (perimeter_value == 1)
                            
                            if feedback:
                                feedback.pushInfo(f"Stage {stage}: Perimeter raster value at endpoint: {perimeter_value}")
                            
                            if reached_perimeter:
                                if feedback:
                                    feedback.pushInfo(f"Stage {stage}: Endpoint is on perimeter raster cell, adding line and skipping further tracing.")
                                
                                # For even stages, we traced from ridges to valleys (uphill)
                                # We want to reverse these lines so all keylines go valley → ridge
                                if stage % 2 == 0:  # Even stage: reverse direction to ensure valley → ridge
                                    reversed_coords = list(line_geom.coords)[::-1]  # Reverse coordinate order
                                    line_geom = LineString(reversed_coords)
                                    if feedback:
                                        feedback.pushInfo(f"Stage {stage}: Reversed line direction (ridge→valley to valley→ridge)")    
                                # Add final line to all_keylines
                                all_keylines.append(line_geom)
                                continue  # Skip further processing for this line
                                
                        # Check if endpoint is on barrier using raster lookup
                        barrier_value = int(new_barrier_mask[end_r, end_c])
                        is_on_barrier = (barrier_value == 1)
                        
                        if feedback:
                            feedback.pushInfo(f"Stage {stage}: Barrier raster value at endpoint: {barrier_value}")
                        
                        if is_on_barrier:
                            if feedback:
                                feedback.pushInfo(f"Stage {stage}: Endpoint is on barrier raster cell, trying to get a new start point...")
                            new_start_point = TopoDrainCore._get_linedirection_start_point(
                                new_barrier_raster_path, line_geom, max_offset=5
                            )
                            if new_start_point:
                                if feedback:
                                    feedback.pushInfo(f"Stage {stage}: New start point found at {new_start_point.wkt}")
                                new_start_points.append(new_start_point)
                                # Extend line_geom to include the new start point
                                line_geom = TopoDrainCore._snap_line_to_point(
                                    line_geom, new_start_point, "end"
                                )
                            else:
                                if feedback:
                                    feedback.pushInfo(f"Stage {stage}: No valid start point found.")
                                else:
                                    print(f"Stage {stage}: No valid start point found.")

                        # For even stages, we traced from ridges to valleys (uphill)
                        # We want to reverse these lines so all keylines go valley → ridge
                        if stage % 2 == 0:  # Even stage: reverse direction to ensure valley → ridge
                            reversed_coords = list(line_geom.coords)[::-1]  # Reverse coordinate order
                            line_geom = LineString(reversed_coords)
                            if feedback:
                                feedback.pushInfo(f"Stage {stage}: Reversed line direction (ridge→valley to valley→ridge)")    
                        all_keylines.append(line_geom)

            if not new_start_points:
                if feedback:
                    feedback.pushInfo(f"Stage {stage}: No endpoints on {target_type}, stopping iteration...")
                else:
                    print(f"Stage {stage}: No endpoints on {target_type}, stopping iteration...")
                break
            
            # Create GeoDataFrame from new start points
            current_start_points = gpd.GeoDataFrame(
                geometry=new_start_points,
                crs=start_points.crs
            )
        
            if feedback:
                feedback.pushInfo(f"Stage {stage}: Generated {len(new_start_points)} new start points beyond {target_type}")
            else:
                print(f"Stage {stage}: Generated {len(new_start_points)} new start points beyond {target_type}")
            
            # Increment stage
            stage += 1

        if stage > max_iterations_keyline:
            if feedback:
                feedback.reportError(f"Warning: Maximum iterations ({max_iterations_keyline}) reached, stopping iteration...")
            else:
                print(f"Warning: Maximum iterations ({max_iterations_keyline}) reached, stopping iteration...")

        # Create combined GeoDataFrame
        if all_keylines:
            combined_gdf = gpd.GeoDataFrame(
                geometry=all_keylines,
                crs=start_points.crs
            )
        else:
            combined_gdf = gpd.GeoDataFrame(crs=start_points.crs)
        
        if feedback:
            feedback.setProgress(100)
            feedback.pushInfo(f"Keyline creation complete: {len(combined_gdf)} total keylines from {stage-1} stages")
        else:
            print(f"Keyline creation complete: {len(combined_gdf)} total keylines from {stage-1} stages")
            print("Progress: 100%")
            
        return combined_gdf

    def adjust_constant_slope_after(
        self,
        dtm_path: str,
        input_lines: gpd.GeoDataFrame,
        change_after: float,
        slope_after: float,
        destination_features: list[gpd.GeoDataFrame],
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
            feedback.pushInfo(f"[AdjustConstantSlopeAfter] Starting adjustment of {len(input_lines)} lines...")
            feedback.setProgress(0)
        else:
            print(f"[AdjustConstantSlopeAfter] Starting adjustment of {len(input_lines)} lines...")
        
        # Validate change_after parameter
        if not (0.0 <= change_after <= 1.0):
            raise ValueError("change_after must be between 0.0 and 1.0")
        
        # Phase 1: Process all lines to create first parts and collect start points for second parts
        if feedback:
            feedback.pushInfo(f"[AdjustConstantSlopeAfter] Phase 1: Processing {len(input_lines)} lines to create first parts...")
            feedback.setProgress(10)
        else:
            print(f"[AdjustConstantSlopeAfter] Phase 1: Processing {len(input_lines)} lines to create first parts...")
        
        first_parts_data = []  # Store first part data with mapping info
        all_start_points = []  # Collect all start points for second parts
        line_mapping = {}      # Map start point index to original line index
        
        for idx, row in input_lines.iterrows():
            line_geom = row.geometry
            
            # Handle MultiLineString by stitching into a single LineString
            if isinstance(line_geom, MultiLineString):
                if feedback:
                    feedback.pushInfo(f"[AdjustConstantSlopeAfter] Converting MultiLineString to LineString using _merge_lines_by_distance")
                line_geom = TopoDrainCore._merge_lines_by_distance(line_geom)
            elif not isinstance(line_geom, LineString):
                if feedback:
                    feedback.pushInfo(f"[AdjustConstantSlopeAfter] Skipping unsupported geometry type: {type(line_geom)} at index {idx}")
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
                feedback.pushInfo(f"[AdjustConstantSlopeAfter] Phase 2: Tracing {len(all_start_points)} second parts in parallel...")
                feedback.setProgress(50)
            else:
                print(f"[AdjustConstantSlopeAfter] Phase 2: Tracing {len(all_start_points)} second parts in parallel...")
            
            # Set CRS after creation when we have geometry column
            second_part_lines = second_part_lines.set_crs(input_lines.crs)
            
            # Create GeoDataFrame with all start points and add mapping information
            start_points_gdf = gpd.GeoDataFrame(geometry=all_start_points, crs=input_lines.crs)
            start_points_gdf['original_line_idx'] = [line_mapping[i] for i in range(len(all_start_points))]
            
            try:
                # Trace all second parts with new slope in a single call
                second_part_lines = self.get_constant_slope_lines(
                    dtm_path=dtm_path,
                    start_points=start_points_gdf,
                    destination_features=destination_features,
                    slope=slope_after,
                    barrier_features=barrier_features,
                    allow_barriers_as_temp_destination=allow_barriers_as_temp_destination,
                    max_iterations_barrier=max_iterations_barrier,
                    slope_deviation_threshold=slope_deviation_threshold,
                    max_iterations_slope=max_iterations_slope,
                    feedback=feedback  # Pass feedback to the main tracing call
                )
                
                if feedback:
                    feedback.pushInfo(f"[AdjustConstantSlopeAfter] Successfully traced {len(second_part_lines)} second parts")
            except Exception as e:
                if feedback:
                    feedback.pushInfo(f"[AdjustConstantSlopeAfter] Error tracing second parts: {str(e)}")
                else:
                    print(f"[AdjustConstantSlopeAfter] Error tracing second parts: {str(e)}")
                # Continue with empty second_part_lines
        else:
            if feedback:
                feedback.pushInfo(f"[AdjustConstantSlopeAfter] No second parts to trace")
        
        # Phase 3: Combine first and second parts
        if feedback:
            feedback.pushInfo(f"[AdjustConstantSlopeAfter] Phase 3: Combining first and second parts...")
            feedback.setProgress(80)
        else:
            print(f"[AdjustConstantSlopeAfter] Phase 3: Combining first and second parts...")
        
        adjusted_lines = []
        
        for data_idx, part_data in enumerate(first_parts_data):
            original_row = part_data['original_row']
            first_part_line = part_data['first_part_line']
            needs_second_part = part_data['needs_second_part']
            start_point_index = part_data['start_point_index']
            
            if not needs_second_part or first_part_line is None:
                # Keep original line
                adjusted_lines.append(original_row)
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
                        
                        # Create new row with combined geometry and original attributes
                        new_row = original_row.copy()
                        new_row['geometry'] = combined_line
                        adjusted_lines.append(new_row)
                        
                        if feedback:
                            feedback.pushInfo(f"[AdjustConstantSlopeAfter] Successfully combined line parts for line {data_idx}")
                    else:
                        if feedback:
                            feedback.pushInfo(f"[AdjustConstantSlopeAfter] Second part is not LineString for line {data_idx}, keeping first part only")
                        new_row = original_row.copy()
                        new_row['geometry'] = first_part_line
                        adjusted_lines.append(new_row)
                else:
                    if feedback:
                        feedback.pushInfo(f"[AdjustConstantSlopeAfter] No matching second part found for line {data_idx}, keeping first part only")
                    new_row = original_row.copy()
                    new_row['geometry'] = first_part_line
                    adjusted_lines.append(new_row)
            else:
                if feedback:
                    feedback.pushInfo(f"[AdjustConstantSlopeAfter] No second parts available for line {data_idx}, keeping first part only")
                new_row = original_row.copy()
                new_row['geometry'] = first_part_line
                adjusted_lines.append(new_row)
        
        # Create result GeoDataFrame
        if adjusted_lines:
            result_gdf = gpd.GeoDataFrame(adjusted_lines, crs=input_lines.crs).reset_index(drop=True)
        else:
            result_gdf = gpd.GeoDataFrame(crs=input_lines.crs)
        
        if feedback:
            feedback.setProgress(100)
            feedback.pushInfo(f"[AdjustConstantSlopeAfter] Adjustment complete: {len(result_gdf)} adjusted lines")
        else:
            print(f"[AdjustConstantSlopeAfter] Adjustment complete: {len(result_gdf)} adjusted lines")
        
        return result_gdf


if __name__ == "__main__":
    print("No main part")
