# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: utils.py
#
# Purpose: Utility functions for the TopoDrain plugin
#
# -----------------------------------------------------------------------------

import os
import re
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsFeatureSource,
    QgsProcessingException
)


def ensure_whiteboxtools_configured(processing_instance, feedback=None):
    """
    Utility function to ensure WhiteboxTools is configured before running algorithms.
    Handles plugin connection and WhiteboxTools verification with proper feedback.
    
    Args:
        processing_instance: The processing algorithm instance that has a 'plugin' attribute
        feedback: QgsProcessingFeedback object for progress reporting
        
    Returns:
        bool: True if WhiteboxTools is configured or verification was successful, False otherwise
        
    Raises:
        QgsProcessingException: If WhiteboxTools is not configured and feedback is None
    """
    try:
        # First try to use existing plugin reference
        if hasattr(processing_instance, 'plugin') and processing_instance.plugin:
            if hasattr(processing_instance.plugin, 'ensure_whiteboxtools_configured'):
                if not processing_instance.plugin.ensure_whiteboxtools_configured():
                    error_msg = "WhiteboxTools is not configured. Please install and configure the WhiteboxTools for QGIS plugin."
                    if feedback:
                        feedback.pushWarning(error_msg)
                        return False
                    else:
                        raise QgsProcessingException(error_msg)
                else:
                    if feedback:
                        feedback.pushInfo("WhiteboxTools configuration verified")
                    return True
            else:
                if feedback:
                    feedback.pushWarning("Plugin found but WhiteboxTools configuration method not available")
                return False
        else:
            # Try to automatically find and connect to the TopoDrain plugin
            if feedback:
                feedback.pushInfo("Plugin reference not available - attempting to connect to TopoDrain plugin")
            
            try:
                from qgis.utils import plugins
                if 'topo_drain' in plugins:
                    topo_drain_plugin = plugins['topo_drain']
                    # Set the plugin reference for this instance
                    processing_instance.plugin = topo_drain_plugin
                    if feedback:
                        feedback.pushInfo("Successfully connected to TopoDrain plugin")
                    
                    # Now try to configure WhiteboxTools
                    if hasattr(topo_drain_plugin, 'ensure_whiteboxtools_configured'):
                        if not topo_drain_plugin.ensure_whiteboxtools_configured():
                            if feedback:
                                feedback.pushWarning("WhiteboxTools is not configured. Please install and configure the WhiteboxTools for QGIS plugin.")
                            return False
                        else:
                            if feedback:
                                feedback.pushInfo("WhiteboxTools configuration verified")
                            return True
                    else:
                        if feedback:
                            feedback.pushWarning("TopoDrain plugin found but configuration method not available")
                        return False
                else:
                    if feedback:
                        feedback.pushWarning("TopoDrain plugin not found in QGIS registry - cannot verify WhiteboxTools configuration")
                    return False
            except Exception as e:
                if feedback:
                    feedback.pushWarning(f"Could not connect to TopoDrain plugin: {e} - continuing without WhiteboxTools verification")
                return False
                
    except Exception as e:
        error_msg = f"Error during WhiteboxTools configuration check: {e}"
        if feedback:
            feedback.pushWarning(error_msg)
            return False
        else:
            raise QgsProcessingException(error_msg)


def update_core_crs_if_needed(core, input_crs, feedback=None):
    """
    Update core CRS if input CRS is different and valid.
    
    Args:
        core: TopoDrainCore instance
        input_crs: CRS string from input layer
        feedback: QGIS feedback object for logging
    
    Returns:
        bool: True if CRS was updated, False otherwise
    """
    if not core or not hasattr(core, 'crs'):
        return False
    
    if not input_crs or not input_crs.strip():
        return False
    
    current_crs = getattr(core, 'crs', None)
    
    if current_crs != input_crs:
        if feedback:
            feedback.pushInfo(f"Updating core CRS from '{current_crs}' to '{input_crs}' to match input layer")
        else:
            print(f"[TopoDrain Utils] Updating core CRS from '{current_crs}' to '{input_crs}' to match input layer")
        core.crs = input_crs
        return True
    
    return False


def parse_epsg_from_wkt_or_description(text):
    """
    Try to extract EPSG code from WKT string or CRS description
    
    Args:
        text: WKT string or CRS description
    
    Returns:
        str: EPSG code (e.g., 'EPSG:2056') or None if not found
    """
    if not text:
        return None
    
    # Direct EPSG pattern search
    epsg_match = re.search(r'EPSG["\s]*[:\s]*["\s]*(\d+)', text, re.IGNORECASE)
    if epsg_match:
        epsg_code = f"EPSG:{epsg_match.group(1)}"
        print(f"[TopoDrain Utils] Found EPSG in text: {epsg_code}")
        return epsg_code
    # Common coordinate system patterns, including Swiss systems
    coordinate_systems = {
        "CH1903+ / LV95": "EPSG:2056",
        "CH1903+": "EPSG:2056",
        "LV95": "EPSG:2056",
        "CH1903 / LV03": "EPSG:21781",
        "CH1903": "EPSG:21781",
        "LV03": "EPSG:21781",
        "WGS 84": "EPSG:4326",
        "WGS84": "EPSG:4326",
        "Web Mercator": "EPSG:3857",
        "Pseudo-Mercator": "EPSG:3857",
        "UTM zone 32N": "EPSG:32632",
        "UTM zone 33N": "EPSG:32633",
    }
    
    for pattern, epsg in coordinate_systems.items():
        if pattern.lower() in text.lower():
            print(f"[TopoDrain Utils] Detected {pattern} - using {epsg}")
            return epsg
    
    print(f"[TopoDrain Utils] Could not extract EPSG from: {text[:100]}...")
    return None

def get_crs_from_project():
    """
    Get the current QGIS project CRS as authid (e.g., 'EPSG:4326')
    """
    try:
        crs = QgsProject.instance().crs()
        
        # Get authid once before checking if it's empty
        crs_authid = crs.authid()
        print(f"[TopoDrain Utils] Project CRS authid: '{crs_authid}'")
        
        # If authid is empty, try alternative methods
        if not crs_authid or crs_authid.strip() == "":
            print("[TopoDrain Utils] Project authid() returned empty, trying alternatives...")
            description = crs.description()
            print(f"[TopoDrain Utils] Project CRS description: {description}")
            
            # Try to extract EPSG from description
            epsg_from_desc = parse_epsg_from_wkt_or_description(description)
            if epsg_from_desc:
                print(f"[TopoDrain Utils] Extracted EPSG from project description: {epsg_from_desc}")
                return epsg_from_desc
            
            # Try to get WKT and parse it
            wkt = crs.toWkt()
            print(f"[TopoDrain Utils] Project CRS WKT: {wkt}")
            epsg_from_wkt = parse_epsg_from_wkt_or_description(wkt)
            if epsg_from_wkt:
                print(f"[TopoDrain Utils] Extracted EPSG from project WKT: {epsg_from_wkt}")
                return epsg_from_wkt
            
            # If still no EPSG, return the WKT as fallback
            if wkt:
                print(f"[TopoDrain Utils] Using project WKT as fallback: {wkt[:100]}...")
                return wkt
            
            warnings.warn("[TopoDrain Utils] Project CRS is empty, returning None")
            return None
        else:
            print(f"[TopoDrain Utils] Returning project authid: {crs_authid}")
            return crs_authid
            
    except Exception as e:
        warnings.warn(f"[TopoDrain Utils] Could not get project CRS: {e}, returning None")
        return None

def get_crs_from_layer(layer_source):
    """
    Get CRS from a raster or vector layer file or object, or from layer source object
    
    Args:
        layer_source: Can be a QgsRasterLayer, QgsVectorLayer, QgsFeatureSource, or file path string
    
    Returns:
        str or None: CRS authid (e.g., 'EPSG:2056'), WKT string, or None if not found
    """    
    crs = None
    print(f"[TopoDrain Utils] Getting CRS from layer source {layer_source}...")
    try:
        # Check if layer_source is already a QGIS layer object
        if isinstance(layer_source, (QgsRasterLayer, QgsVectorLayer)):
            print(f"[TopoDrain Utils] Layer source is a QGIS layer object: {type(layer_source)}")
            if layer_source.isValid():
                crs = layer_source.crs()
                print(f"[TopoDrain Utils] Layer is valid.")
            else:
                print("[TopoDrain Utils] Layer object is not valid.")
                return None
        
        elif isinstance(layer_source, QgsFeatureSource):
            # If it's a layer source object (e.g., QgsProcessingParameterFeatureSource)
            print(f"[TopoDrain Utils] Layer source is a layer source object: {type(layer_source)}")
            crs = layer_source.sourceCrs()
            print(f"[TopoDrain Utils] Found crs from source.")

        # Otherwise it's probably a file path string
        elif isinstance(layer_source, str) and os.path.exists(layer_source):
            # Try as raster first
            print(f"[TopoDrain Utils] Trying as QgsRasterLayer: {layer_source}")
            layer = QgsRasterLayer(layer_source)
            if layer.isValid():
                crs = layer.crs()
                print(f"[TopoDrain Utils] Raster layer is valid.")
            else:
                print("[TopoDrain Utils] Raster layer is not valid, trying as vector...")
                # Try as vector
                print(f"[TopoDrain Utils] Trying as QgsVectorLayer: {layer_source}")
                layer = QgsVectorLayer(layer_source)
                if layer.isValid():
                    crs = layer.crs()
                    print(f"[TopoDrain Utils] Vector layer is valid.")
                else:
                    print("[TopoDrain Utils] Vector layer is not valid.")
                    return None
        else:
            if isinstance(layer_source, str):
                print(f"[TopoDrain Utils] File does not exist: {layer_source}")
            else:
                print(f"[TopoDrain Utils] layer_source is not a valid type: {type(layer_source)}")
            return None

        # If we have a crs object, try to extract valid CRS information
        if crs is None:
            print("[TopoDrain Utils] No CRS object available")
            return None
        
        # Get authid once before checking if it's empty
        crs_authid = crs.authid()
        print(f"[TopoDrain Utils] CRS authid: '{crs_authid}'")
        
        # If authid is empty, try alternative methods
        if not crs_authid or crs_authid.strip() == "":
            print("[TopoDrain Utils] authid() returned empty, trying alternatives...")
            description = crs.description()
            print(f"[TopoDrain Utils] CRS description: {description}")
            
            # Try to extract EPSG from description
            epsg_from_desc = parse_epsg_from_wkt_or_description(description)
            if epsg_from_desc:
                print(f"[TopoDrain Utils] Extracted EPSG from description: {epsg_from_desc}")
                return epsg_from_desc
            
            # Try to get WKT and parse it
            wkt = crs.toWkt()
            print(f"[TopoDrain Utils] CRS WKT: {wkt}")
            epsg_from_wkt = parse_epsg_from_wkt_or_description(wkt)
            if epsg_from_wkt:
                print(f"[TopoDrain Utils] Extracted EPSG from WKT: {epsg_from_wkt}")
                return epsg_from_wkt
            
            # If still no EPSG, return the WKT as fallback
            if wkt:
                print(f"[TopoDrain Utils] Using WKT as fallback: {wkt[:100]}...")
                return wkt
            
            print("[TopoDrain Utils] No valid CRS information found")
            return None
        else:
            print(f"[TopoDrain Utils] Returning authid: {crs_authid}")
            return crs_authid

    except Exception as e:
        print(f"[TopoDrain Utils] Could not get CRS from layer {layer_source}: {e}")
        return None



def clean_qvariant_data(gdf):
    """
    Clean QVariant objects from GeoDataFrame to avoid field type errors during saving.
    Converts QVariant values to appropriate Python native types while preserving integer types.
    
    Args:
        gdf (GeoDataFrame): Input GeoDataFrame that may contain QVariant objects
        
    Returns:
        GeoDataFrame: Cleaned GeoDataFrame with native Python types and proper dtypes
    """
    try:
        from qgis.PyQt.QtCore import QVariant
    except ImportError:
        # If QVariant is not available, return the GeoDataFrame as-is
        return gdf
        
    cleaned_gdf = gdf.copy()
    
    for column in cleaned_gdf.columns:
        if column == 'geometry':
            continue
            
        # Check if column contains QVariant objects
        has_qvariant = False
        for value in cleaned_gdf[column]:
            if isinstance(value, QVariant):
                has_qvariant = True
                break
        
        if has_qvariant:
            # Convert QVariant values to native Python types
            def convert_qvariant(value):
                if isinstance(value, QVariant):
                    if value.isNull():
                        return None
                    else:
                        # Get the actual value from QVariant
                        return value.value()
                return value
            
            cleaned_gdf[column] = cleaned_gdf[column].apply(convert_qvariant)
            
            # After conversion, try to infer and preserve integer types
            # Check if all non-null values are integer-like
            non_null_values = cleaned_gdf[column].dropna()
            if len(non_null_values) > 0:
                try:
                    # Try to convert to numeric if it's not already
                    if cleaned_gdf[column].dtype == object:
                        cleaned_gdf[column] = pd.to_numeric(cleaned_gdf[column], errors='ignore')
                    
                    # Check if all values are integers (no decimal parts)
                    if pd.api.types.is_numeric_dtype(cleaned_gdf[column]):
                        if all(pd.isna(v) or (isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())) 
                               for v in cleaned_gdf[column]):
                            # Convert to Int64 (nullable integer type)
                            cleaned_gdf[column] = cleaned_gdf[column].astype('Int64')
                except Exception as e:
                    # If conversion fails, keep the column as-is
                    print(f"[TopoDrain Utils] Warning: Could not convert column '{column}' to integer: {e}")
                    pass
    
    return cleaned_gdf


def load_gdf_from_qgis_source(qgis_source, feedback=None):
    """
    Load a GeoDataFrame from a QGIS QgsProcessingParameterFeatureSource.
    Automatically cleans QVariant data types for compatibility.
    
    Args:
        qgis_source: QgsProcessingParameterFeatureSource object with getFeatures() method
        feedback (QgsProcessingFeedback, optional): Processing feedback for logging
    
    Returns:
        gpd.GeoDataFrame: Loaded GeoDataFrame with cleaned data types and crs=None for safe handling
    
    Raises:
        Exception: If source loading fails
    """
    
    try:
        if feedback:
            feedback.pushInfo("Loading features from QGIS source...")
        
        # Load GeoDataFrame from QGIS source features
        gdf = gpd.GeoDataFrame.from_features(qgis_source.getFeatures())
        
        if gdf.empty:
            if feedback:
                feedback.pushInfo("No features found in QGIS source")
            return gdf
        
        # Automatically clean QVariant data types
        if feedback:
            feedback.pushInfo("Cleaning data types...")
        gdf = clean_qvariant_data(gdf)
        
        if feedback:
            feedback.pushInfo(f"Successfully loaded and cleaned {len(gdf)} features from QGIS source")
        
        return gdf
    
    except Exception as e:
        error_msg = f"Failed to load GeoDataFrame from QGIS source: {e}"
        if feedback:
            feedback.pushInfo(error_msg)
        raise Exception(error_msg)


def load_gdf_from_file(file_path, feedback=None):
    """
    Load a GeoDataFrame from a file path, handling GeoPackage layer syntax.
    Automatically cleans QVariant data types for compatibility.
    
    This function handles both regular file paths and QGIS GeoPackage layer paths
    in the format "/path/file.gpkg|layername=layer_name".
    
    Args:
        file_path (str): Path to the vector file, may include GeoPackage layer syntax
        feedback (QgsProcessingFeedback, optional): Processing feedback for logging
    
    Returns:
        gpd.GeoDataFrame: Loaded GeoDataFrame with cleaned data types
    
    Raises:
        Exception: If file loading fails
    """

    try:
        # Handle GeoPackage layer paths for GeoPandas
        if '|' in file_path and 'layername=' in file_path:
            # Parse GeoPackage path: "/path/file.gpkg|layername=layer_name"
            gpkg_file = file_path.split('|')[0]
            layer_part = file_path.split('|')[1]
            layer_name = layer_part.split('=')[1] if '=' in layer_part else layer_part
            
            if feedback:
                feedback.pushInfo(f"Loading GeoPackage layer: {gpkg_file}, layer: {layer_name}")
            
            gdf = gpd.read_file(gpkg_file, layer=layer_name)
        else:
            # Regular file path
            if feedback:
                feedback.pushInfo(f"Loading vector file: {file_path}")
            
            gdf = gpd.read_file(file_path)
        
        # Automatically clean QVariant data types
        if feedback:
            feedback.pushInfo("Cleaning data types...")
        gdf = clean_qvariant_data(gdf)
        
        if feedback:
            feedback.pushInfo(f"Successfully loaded and cleaned {len(gdf)} features")
        
        return gdf
    
    except Exception as e:
        error_msg = f"Failed to load vector file '{file_path}': {e}"
        if feedback:
            feedback.pushInfo(error_msg)
        raise Exception(error_msg)


def _remove_id_columns_and_retry(gdf, file_path, driver, feedback):
    """Helper to remove ID columns and retry save after ID conflict error."""
    columns_to_drop = [col for col in ['id', 'fid', 'ID', 'FID'] if col in gdf.columns]
    if columns_to_drop:
        gdf = gdf.drop(columns=columns_to_drop)
        if feedback:
            feedback.pushInfo(f"Dropped columns: {columns_to_drop}")
    
    if driver:
        gdf.to_file(file_path, driver=driver)
    else:
        gdf.to_file(file_path)
    
    return gdf


def save_gdf_to_file(gdf, file_path, core, feedback):
    """
    Save GeoDataFrame to file with proper format handling using core's OGR driver mapping.
    Automatically cleans QVariant data types before saving to prevent field type errors.
    
    Args:
        gdf: GeoDataFrame to save
        file_path (str): Output file path
        core: TopoDrainCore instance with ogr_driver_mapping
        feedback: QGIS processing feedback object
        
    Raises:
        QgsProcessingException: If saving fails
    """
    
    try:
        # Automatically clean QVariant data types before saving
        if feedback:
            feedback.pushInfo("Cleaning data types before saving...")
        cleaned_gdf = clean_qvariant_data(gdf)
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Check if format is in our supported mapping
        if hasattr(core, 'ogr_driver_mapping') and file_ext in core.ogr_driver_mapping:
            # Use the mapped driver
            ogr_driver = core._get_ogr_driver_from_path(file_path)
            feedback.pushInfo(f"Detected output format: {file_ext} (using driver: {ogr_driver})")
            
            try:
                cleaned_gdf.to_file(file_path, driver=ogr_driver)
                feedback.pushInfo(f"GeoDataFrame saved to: {file_path}")
            except Exception as driver_error:
                error_str_lower = str(driver_error).lower()
                # Check if it's an ID conflict error (common with GPKG)
                if ("id" in error_str_lower or "fid" in error_str_lower) and "failed to write" in error_str_lower:
                    if feedback:
                        feedback.pushWarning(f"ID conflict detected, removing 'id' and 'fid' columns and retrying: {driver_error}")
                    
                    try:
                        _remove_id_columns_and_retry(cleaned_gdf, file_path, ogr_driver, feedback)
                        feedback.pushInfo(f"GeoDataFrame saved successfully after removing ID columns: {file_path}")
                    except Exception as retry_error:
                        # If still fails, try auto-detection
                        feedback.pushWarning(f"Retry with driver '{ogr_driver}' failed, trying auto-detection: {retry_error}")
                        _remove_id_columns_and_retry(cleaned_gdf, file_path, None, feedback)
                        feedback.pushInfo(f"GeoDataFrame saved using auto-detection: {file_path}")
                else:
                    # If not an ID conflict, try auto-detection
                    feedback.pushWarning(f"Driver '{ogr_driver}' failed, trying auto-detection: {driver_error}")
                    cleaned_gdf.to_file(file_path)
                    feedback.pushInfo(f"GeoDataFrame saved using auto-detection: {file_path}")
        else:
            # Format not in our mapping - try auto-detection
            feedback.pushWarning(f"Format {file_ext} not in available driver mapping, using auto-detection")
            
            try:
                cleaned_gdf.to_file(file_path)
                feedback.pushInfo(f"GeoDataFrame saved using auto-detection: {file_path}")
            except Exception as auto_error:
                error_str_lower = str(auto_error).lower()
                # Check if it's an ID conflict error
                if ("id" in error_str_lower or "fid" in error_str_lower) and "failed to write" in error_str_lower:
                    if feedback:
                        feedback.pushWarning(f"ID conflict detected, removing 'id' and 'fid' columns and retrying: {auto_error}")
                    _remove_id_columns_and_retry(cleaned_gdf, file_path, None, feedback)
                    feedback.pushInfo(f"GeoDataFrame saved successfully after removing ID columns: {file_path}")
                else:
                    raise
            
    except Exception as e:
        # Provide helpful error message with format suggestions
        if hasattr(core, 'ogr_driver_mapping'):
            supported_formats = list(core.ogr_driver_mapping.keys())
            error_msg = f"Failed to save GeoDataFrame output: {e}\n"
            error_msg += f"Recommended formats: {', '.join(supported_formats[:5])} (and others)"
        else:
            error_msg = f"Failed to save GeoDataFrame output: {e}"
        raise QgsProcessingException(error_msg)


def get_raster_ext(raster_path, feedback=None, check_existence=True):
    """Get file extension from raster path, handling GDAL virtual file system paths.
    
    Args:
        raster_path: Path to raster file (may include GDAL virtual paths)
        feedback: Optional feedback object for logging
        check_existence: If True, checks if file exists (for input files). 
                        If False, skips existence check (for output files)
    
    Returns:
        str: File extension (lowercase, with dot)
    
    Raises:
        Exception: If check_existence=True and path doesn't exist, or can't be processed
    """

    try:
        if feedback:
            feedback.pushInfo(f"Processing raster path: {raster_path}")
        
        # Handle GDAL virtual file system paths (e.g., GPKG raster layers)
        base_path = raster_path
        if ':' in raster_path:
            # Check if it's a GDAL virtual file system path
            parts = raster_path.split(':')
            if len(parts) >= 2:
                # Format: GPKG:/path/to/file.gpkg:layer_name
                if parts[0].upper() == 'GPKG':
                    if len(parts) >= 3:
                        # Extract the actual file path (between first and last colon)
                        base_path = ':'.join(parts[1:-1])
                        if feedback:
                            feedback.pushInfo(f"Extracted GPKG raster base path: {base_path}")
                else:
                    # For other GDAL virtual paths, try to extract base path
                    # Format might be like "driver:/path/file.ext"
                    if len(parts) >= 2:
                        base_path = parts[1]
        
        # Check if the base path exists (only if requested)
        if check_existence and not os.path.exists(base_path):
            raise Exception(f"Raster file not found: {base_path}")
        
        # Get file extension
        ext = os.path.splitext(base_path)[1].lower()
        
        if feedback:
            feedback.pushInfo(f"Extracted extension: {ext}")
        
        return ext
    
    except Exception as e:
        error_msg = f"Failed to get raster extension from '{raster_path}': {e}"
        if feedback:
            feedback.pushInfo(error_msg)
        raise Exception(error_msg)


def get_vector_ext(vector_path, feedback=None, check_existence=True):
    """Get file extension from vector path, handling GDAL/OGR virtual file system paths.
    
    Args:
        vector_path: Path to vector file (may include OGR virtual paths)
        feedback: Optional feedback object for logging
        check_existence: If True, checks if file exists (for input files). 
                        If False, skips existence check (for output files)
    
    Returns:
        str: File extension (lowercase, with dot)
    
    Raises:
        Exception: If check_existence=True and path doesn't exist, or can't be processed
    """

    try:
        if feedback:
            feedback.pushInfo(f"Processing vector path: {vector_path}")
        
        # Handle OGR virtual file system paths (e.g., GPKG vector layers)
        base_path = vector_path
        if '|' in vector_path:
            # For paths like "/path/file.gpkg|layername=layer_name", extract "/path/file.gpkg"
            base_path = vector_path.split('|')[0]
            if feedback:
                feedback.pushInfo(f"Extracted vector base path: {base_path}")
        
        # Check if the base path exists (only if requested)
        if check_existence and not os.path.exists(base_path):
            raise Exception(f"Vector file not found: {base_path}")
        
        # Get file extension
        ext = os.path.splitext(base_path)[1].lower()
        
        if feedback:
            feedback.pushInfo(f"Extracted extension: {ext}")
        
        return ext
    
    except Exception as e:
        error_msg = f"Failed to get vector extension from '{vector_path}': {e}"
        if feedback:
            feedback.pushInfo(error_msg)
        raise Exception(error_msg)

