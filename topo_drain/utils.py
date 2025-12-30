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

WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")

def clear_pyproj_cache(feedback=None):
    """
    Clear PyProj's internal CRS cache to prevent Windows access violations on repeated runs.
    
    This function aggressively clears PyProj's cached CRS objects that contain C library
    pointers. These pointers become stale when a QThread terminates on Windows, causing
    "access violation" crashes when the next QThread tries to use them.
    
    The issue: PyProj uses thread-local storage (_local) that stores CRS objects with
    C library pointers. When a QThread terminates, Windows immediately releases memory,
    making these pointers invalid. On the next run in a new QThread, PyProj tries to
    access these stale pointers → crash.
    
    Args:
        feedback: Optional feedback object for logging
        
    Returns:
        bool: True if cache was cleared or attempted, False if PyProj not available
    """
    try:
        import pyproj
        import gc
        import sys
        
        if feedback:
            feedback.pushInfo("[PyProj Cache Clear] Aggressively clearing PyProj CRS cache and stale pointers...")
        
        cleared_something = False
        
        # Approach 1: Clear any global CRS cache if it exists
        cache_attrs = ['_CRS_CACHE', '__crs_cache__', '_cache', '__pyproj_crs__']
        for attr in cache_attrs:
            if hasattr(pyproj, attr):
                try:
                    cache = getattr(pyproj, attr)
                    if hasattr(cache, 'clear'):
                        cache.clear()
                        if feedback:
                            feedback.pushInfo(f"[PyProj Cache Clear] Cleared pyproj.{attr}")
                        cleared_something = True
                    elif isinstance(cache, dict):
                        cache.clear()
                        if feedback:
                            feedback.pushInfo(f"[PyProj Cache Clear] Cleared dict pyproj.{attr}")
                        cleared_something = True
                except Exception as e:
                    if feedback:
                        feedback.pushInfo(f"[PyProj Cache Clear] Could not clear {attr}: {e}")
        
        # Approach 2: Aggressively clear CRS class-level caches and thread-local storage
        if hasattr(pyproj, 'CRS'):
            CRS = pyproj.CRS
            
            # CRITICAL: Clear thread-local storage (the main culprit on Windows)
            if hasattr(CRS, '_local'):
                try:
                    # Get the current thread-local object
                    local_obj = CRS._local
                    
                    # Clear all attributes from thread-local storage
                    if hasattr(local_obj, '__dict__'):
                        local_keys = list(local_obj.__dict__.keys())
                        if local_keys:
                            if feedback:
                                feedback.pushInfo(f"[PyProj Cache Clear] Found thread-local keys: {local_keys}")
                            # Delete each attribute to release C pointers
                            for key in local_keys:
                                try:
                                    delattr(local_obj, key)
                                    if feedback:
                                        feedback.pushInfo(f"[PyProj Cache Clear] Deleted thread-local.{key}")
                                except Exception as del_e:
                                    if feedback:
                                        feedback.pushInfo(f"[PyProj Cache Clear] Could not delete {key}: {del_e}")
                            cleared_something = True
                    
                    # Recreate thread-local storage with fresh object
                    try:
                        import threading
                        old_local = CRS._local
                        CRS._local = threading.local()
                        # Delete reference to old thread-local object
                        del old_local
                        if feedback:
                            feedback.pushInfo("[PyProj Cache Clear] ✓ Recreated CRS._local thread-local storage")
                        cleared_something = True
                    except Exception as e:
                        if feedback:
                            feedback.pushInfo(f"[PyProj Cache Clear] Could not recreate _local: {e}")
                            
                except Exception as e:
                    if feedback:
                        feedback.pushInfo(f"[PyProj Cache Clear] Could not clear CRS._local: {e}")
            
            # Clear other CRS class caches that might hold references
            crs_cache_attrs = ['_cache', '__cache__', '__dict__']
            for attr in crs_cache_attrs:
                if hasattr(CRS, attr):
                    try:
                        cache = getattr(CRS, attr)
                        if hasattr(cache, 'clear') and callable(cache.clear):
                            cache.clear()
                            if feedback:
                                feedback.pushInfo(f"[PyProj Cache Clear] Cleared CRS.{attr}")
                            cleared_something = True
                        elif isinstance(cache, dict):
                            # For __dict__, only clear CRS-related keys to avoid breaking the class
                            if attr == '__dict__':
                                cache_keys = [k for k in cache.keys() if 'cache' in k.lower() or 'crs' in k.lower()]
                                for key in cache_keys:
                                    try:
                                        if isinstance(cache[key], dict):
                                            cache[key].clear()
                                            if feedback:
                                                feedback.pushInfo(f"[PyProj Cache Clear] Cleared CRS.__dict__[{key}]")
                                            cleared_something = True
                                    except Exception:
                                        pass
                            else:
                                cache.clear()
                                if feedback:
                                    feedback.pushInfo(f"[PyProj Cache Clear] Cleared dict CRS.{attr}")
                                cleared_something = True
                    except Exception as e:
                        if feedback:
                            feedback.pushInfo(f"[PyProj Cache Clear] Could not clear CRS.{attr}: {e}")
        
        # Approach 3: Clear any PyProj module-level CRS instances
        # Look for any CRS objects in pyproj module namespace and delete them
        try:
            pyproj_dict = pyproj.__dict__
            crs_instances = []
            for key, value in list(pyproj_dict.items()):
                # Find any CRS instances stored at module level
                if hasattr(pyproj, 'CRS') and isinstance(value, pyproj.CRS):
                    crs_instances.append(key)
            
            if crs_instances:
                if feedback:
                    feedback.pushInfo(f"[PyProj Cache Clear] Found CRS instances in module: {crs_instances}")
                for key in crs_instances:
                    try:
                        del pyproj_dict[key]
                        if feedback:
                            feedback.pushInfo(f"[PyProj Cache Clear] Deleted pyproj.{key}")
                        cleared_something = True
                    except Exception as e:
                        if feedback:
                            feedback.pushInfo(f"[PyProj Cache Clear] Could not delete {key}: {e}")
        except Exception as e:
            if feedback:
                feedback.pushInfo(f"[PyProj Cache Clear] Could not scan module for CRS instances: {e}")
        
        # Approach 4: Force aggressive garbage collection with multiple generations
        if feedback:
            feedback.pushInfo("[PyProj Cache Clear] Running aggressive garbage collection...")
        # Collect multiple times to ensure all generations are cleaned
        collected_total = 0
        for i in range(3):
            collected = gc.collect()
            collected_total += collected
            if feedback and collected > 0:
                feedback.pushInfo(f"[PyProj Cache Clear] GC pass {i+1}: collected {collected} objects")
        
        if feedback:
            feedback.pushInfo(f"[PyProj Cache Clear] Total garbage collected: {collected_total} objects")
        cleared_something = True
        
        if cleared_something:
            if feedback:
                feedback.pushInfo("[PyProj Cache Clear] ✓ PyProj cache and pointer cleanup completed")
        else:
            if feedback:
                feedback.pushInfo("[PyProj Cache Clear] No known caches found to clear")
        
        return True
        
    except ImportError:
        if feedback:
            feedback.pushInfo("[PyProj Cache Clear] PyProj not available - skipping")
        return False
    except Exception as e:
        if feedback:
            feedback.pushWarning(f"[PyProj Cache Clear] Error during cache clear: {e}")
        return False

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
    Get the current QGIS project CRS as a string authid (e.g., 'EPSG:4326').
    
    Returns:
        str or None: CRS string (EPSG code or WKT), never a CRS object.
                     Returns None if CRS cannot be determined.
    
    Note:
        Always returns a string to prevent PyProj thread-safety issues.
        Never returns QgsCoordinateReferenceSystem or pyproj.CRS objects.
    """
    try:
        crs = QgsProject.instance().crs()
        
        # Get authid and ensure it's a string (thread-safe)
        crs_authid = str(crs.authid()) if crs.authid() else ""
        print(f"[TopoDrain Utils] Project CRS authid: '{crs_authid}'")
        
        # If authid is empty, try alternative methods
        if not crs_authid or crs_authid.strip() == "":
            print("[TopoDrain Utils] Project authid() returned empty, trying alternatives...")
            description = str(crs.description()) if crs.description() else ""
            print(f"[TopoDrain Utils] Project CRS description: {description}")
            
            # Try to extract EPSG from description
            epsg_from_desc = parse_epsg_from_wkt_or_description(description)
            if epsg_from_desc:
                print(f"[TopoDrain Utils] Extracted EPSG from project description: {epsg_from_desc}")
                return str(epsg_from_desc)  # Ensure string
            
            # Try to get WKT and parse it
            wkt = str(crs.toWkt()) if crs.toWkt() else ""
            print(f"[TopoDrain Utils] Project CRS WKT: {wkt}")
            epsg_from_wkt = parse_epsg_from_wkt_or_description(wkt)
            if epsg_from_wkt:
                print(f"[TopoDrain Utils] Extracted EPSG from project WKT: {epsg_from_wkt}")
                return str(epsg_from_wkt)  # Ensure string
            
            # If still no EPSG, return the WKT as fallback
            if wkt:
                print(f"[TopoDrain Utils] Using project WKT as fallback: {wkt[:100]}...")
                return str(wkt)  # Ensure string
            
            warnings.warn("[TopoDrain Utils] Project CRS is empty, returning None")
            return None
        else:
            print(f"[TopoDrain Utils] Returning project authid: {crs_authid}")
            return str(crs_authid)  # Ensure string
            
    except Exception as e:
        warnings.warn(f"[TopoDrain Utils] Could not get project CRS: {e}, returning None")
        return None

def get_crs_from_layer(layer_source):
    """
    Get CRS from a raster or vector layer file or object, or from layer source object.
    
    Args:
        layer_source: Can be a QgsRasterLayer, QgsVectorLayer, QgsFeatureSource, or file path string
    
    Returns:
        str or None: CRS string (EPSG authid or WKT), never a CRS object.
                     Returns None if CRS cannot be determined.
    
    Note:
        Always returns a string to prevent PyProj thread-safety issues.
        Never returns QgsCoordinateReferenceSystem or pyproj.CRS objects.
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

        # If we have a crs object, try to extract valid CRS information as string
        if crs is None:
            print("[TopoDrain Utils] No CRS object available")
            return None
        
        # Get authid and ensure it's a string (thread-safe)
        crs_authid = str(crs.authid()) if crs.authid() else ""
        print(f"[TopoDrain Utils] CRS authid: '{crs_authid}'")
        
        # If authid is empty, try alternative methods
        if not crs_authid or crs_authid.strip() == "":
            print("[TopoDrain Utils] authid() returned empty, trying alternatives...")
            description = str(crs.description()) if crs.description() else ""
            print(f"[TopoDrain Utils] CRS description: {description}")
            
            # Try to extract EPSG from description
            epsg_from_desc = parse_epsg_from_wkt_or_description(description)
            if epsg_from_desc:
                print(f"[TopoDrain Utils] Extracted EPSG from description: {epsg_from_desc}")
                return str(epsg_from_desc)  # Ensure string
            
            # Try to get WKT and parse it
            wkt = str(crs.toWkt()) if crs.toWkt() else ""
            print(f"[TopoDrain Utils] CRS WKT: {wkt}")
            epsg_from_wkt = parse_epsg_from_wkt_or_description(wkt)
            if epsg_from_wkt:
                print(f"[TopoDrain Utils] Extracted EPSG from WKT: {epsg_from_wkt}")
                return str(epsg_from_wkt)  # Ensure string
            
            # If still no EPSG, return the WKT as fallback
            if wkt:
                print(f"[TopoDrain Utils] Using WKT as fallback: {wkt[:100]}...")
                return str(wkt)  # Ensure string
            
            print("[TopoDrain Utils] No valid CRS information found")
            return None
        else:
            print(f"[TopoDrain Utils] Returning authid: {crs_authid}")
            return str(crs_authid)  # Ensure string

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


def load_gdf_from_file_ogr(file_path, feedback=None):
    """
    Load a GeoDataFrame from a file path using OGR directly (bypasses GeoPandas read_file).
    This avoids PyProj completely by using GDAL/OGR for reading - CRS is NOT loaded.
    
    This function handles both regular file paths and QGIS GeoPackage layer paths
    in the format "/path/file.gpkg|layername=layer_name".
    
    Args:
        file_path (str): Path to the vector file, may include GeoPackage layer syntax
        feedback (QgsProcessingFeedback, optional): Processing feedback for logging
    
    Returns:
        gpd.GeoDataFrame: Loaded GeoDataFrame with cleaned data types and crs=None
    
    Raises:
        Exception: If file loading fails
        
    Note:
        The returned GeoDataFrame will have crs=None to avoid PyProj issues.
        CRS information is discarded during loading.
    """
    try:
        from osgeo import ogr
        from shapely.wkt import loads as wkt_loads
        
        # Handle GeoPackage layer paths
        layer_name = None
        actual_file_path = file_path
        
        if '|' in file_path and 'layername=' in file_path:
            # Parse GeoPackage path: "/path/file.gpkg|layername=layer_name"
            actual_file_path = file_path.split('|')[0]
            layer_part = file_path.split('|')[1]
            layer_name = layer_part.split('=')[1] if '=' in layer_part else layer_part
            
            if feedback:
                feedback.pushInfo(f"Loading GeoPackage layer with OGR: {actual_file_path}, layer: {layer_name}")
        else:
            if feedback:
                feedback.pushInfo(f"Loading vector file with OGR: {actual_file_path}")
        
        # Open data source
        ds = ogr.Open(actual_file_path)
        if ds is None:
            raise Exception(f"Could not open data source: {actual_file_path}")
        
        # Get layer
        if layer_name:
            layer = ds.GetLayerByName(layer_name)
        else:
            layer = ds.GetLayer(0)
        
        if layer is None:
            ds = None
            raise Exception(f"Could not get layer from: {actual_file_path}")
        
        # Get layer definition
        layer_defn = layer.GetLayerDefn()
        field_count = layer_defn.GetFieldCount()
        
        # Prepare lists for data
        geometries = []
        attributes = {layer_defn.GetFieldDefn(i).GetName(): [] for i in range(field_count)}
        
        # Read features
        feature_count = 0
        for feature in layer:
            feature_count += 1
            
            # Get geometry
            geom = feature.GetGeometryRef()
            if geom is not None:
                # Convert OGR geometry to Shapely using WKT (more reliable than WKB)
                try:
                    wkt_data = geom.ExportToWkt()
                    shapely_geom = wkt_loads(wkt_data)
                    geometries.append(shapely_geom)
                except Exception as geom_error:
                    if feedback:
                        feedback.pushWarning(f"Could not convert geometry for feature {feature_count}: {geom_error}")
                    geometries.append(None)
            else:
                geometries.append(None)
            
            # Get attributes
            for i in range(field_count):
                field_name = layer_defn.GetFieldDefn(i).GetName()
                value = feature.GetField(i)
                attributes[field_name].append(value)
        
        # Close data source
        ds = None
        
        if feature_count == 0:
            if feedback:
                feedback.pushInfo("No features found in file")
            return gpd.GeoDataFrame()
        
        # Create GeoDataFrame WITHOUT CRS to avoid PyProj
        gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=None)
        
        # Clean QVariant data types
        if feedback:
            feedback.pushInfo("Cleaning data types...")
        gdf = clean_qvariant_data(gdf)
        
        if feedback:
            feedback.pushInfo(f"✓ Successfully loaded {len(gdf)} features using OGR (no CRS)")
        
        return gdf
    
    except Exception as e:
        error_msg = f"Failed to load vector file with OGR '{file_path}': {e}"
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
    
    # Write with driver if specified
    if driver:
        gdf.to_file(file_path, driver=driver)
    else:
        gdf.to_file(file_path)
    
    return gdf



def save_gdf_to_file(gdf, file_path, core, feedback, all_upper=False):
    """
    Save GeoDataFrame to file with proper format handling using core's OGR driver mapping.
    Automatically cleans QVariant data types before saving to prevent field type errors.
    
    Args:
        gdf: GeoDataFrame to save
        file_path (str): Output file path
        core: TopoDrainCore instance with ogr_driver_mapping and crs
        feedback: QGIS processing feedback object
        all_upper (bool): If True, rename all columns to uppercase (except 'geometry' and columns containing 'id')
        
    Raises:
        QgsProcessingException: If saving fails
    """
    
    try:
        # Rename columns to uppercase if requested
        if all_upper:
            column_mapping = {}
            for col in gdf.columns:
                # Don't rename geometry column or columns equal to 'id'
                # Also skip if already uppercase
                if col != 'geometry' and 'id' != col.lower() and col != col.upper():
                    column_mapping[col] = col.upper()
            
            if column_mapping:
                gdf = gdf.rename(columns=column_mapping)
                if feedback:
                    feedback.pushInfo(f"Renamed columns to uppercase: {list(column_mapping.values())}")
        
        # Clean QVariant data types before saving
        if feedback:
            feedback.pushInfo("Cleaning data types before saving...")
        cleaned_gdf = clean_qvariant_data(gdf)
        
        # Check current CRS status
        if feedback:
            current_crs = str(cleaned_gdf.crs) if cleaned_gdf.crs else "None"
            feedback.pushInfo(f"GeoDataFrame CRS: {current_crs}")
        
        # Set CRS if needed using GeoPandas set_crs
        if cleaned_gdf.crs is None:
            if hasattr(core, 'crs') and core.crs:
                crs_for_save = core.crs  # Already guaranteed to be a string by set_crs()
                
                if feedback:
                    feedback.pushInfo(f"Setting CRS: {crs_for_save}")
                
                try:
                    # Set CRS - core.crs is already a proper string
                    cleaned_gdf = cleaned_gdf.set_crs(crs_for_save, allow_override=True)
                    if feedback:
                        feedback.pushInfo(f"✓ CRS successfully set")
                except Exception as crs_error:
                    if feedback:
                        feedback.pushWarning(f"Could not set CRS: {crs_error}")
                        feedback.pushInfo(f"Continuing without explicit CRS - file will still save correctly...")

        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Check if format is in our supported mapping
        if hasattr(core, 'ogr_driver_mapping') and file_ext in core.ogr_driver_mapping:
            # Use the mapped driver
            ogr_driver = core._get_ogr_driver_from_path(file_path)
            feedback.pushInfo(f"Detected output format: {file_ext} (using driver: {ogr_driver})")
            
            try:
                # Save GeoDataFrame
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
                # Save GeoDataFrame
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


def save_gdf_to_file_ogr(gdf, file_path, core, feedback, all_upper=False):
    """
    Save GeoDataFrame to file using OGR directly (bypasses GeoPandas to_file).
    This avoids PyProj completely by using GDAL/OGR for both geometry writing and CRS setting.
    
    Args:
        gdf: GeoDataFrame to save
        file_path (str): Output file path
        core: TopoDrainCore instance with ogr_driver_mapping and crs
        feedback: QGIS processing feedback object
        all_upper (bool): If True, rename all columns to uppercase (except 'geometry' and columns containing 'id')
        
    Raises:
        QgsProcessingException: If saving fails
    """
    try:
        from osgeo import ogr, osr
        
        # Rename columns to uppercase if requested
        if all_upper:
            column_mapping = {}
            for col in gdf.columns:
                if col != 'geometry' and 'id' != col.lower() and col != col.upper():
                    column_mapping[col] = col.upper()
            
            if column_mapping:
                gdf = gdf.rename(columns=column_mapping)
                if feedback:
                    feedback.pushInfo(f"Renamed columns to uppercase: {list(column_mapping.values())}")
        
        # Clean QVariant data types
        if feedback:
            feedback.pushInfo("Cleaning data types before saving...")
        cleaned_gdf = clean_qvariant_data(gdf)
        
        # Get file extension and driver
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Determine OGR driver using core's mapping
        driver_name = None
        if hasattr(core, 'ogr_driver_mapping') and file_ext in core.ogr_driver_mapping:
            driver_name = core.ogr_driver_mapping[file_ext]
            if feedback:
                feedback.pushInfo(f"Using driver from core mapping: {driver_name}")
        else:
            # Fallback to common drivers
            driver_map = {
                '.gpkg': 'GPKG',
                '.shp': 'ESRI Shapefile',
                '.geojson': 'GeoJSON',
                '.json': 'GeoJSON',
                '.gml': 'GML'
            }
            if file_ext in driver_map:
                driver_name = driver_map[file_ext]
                if feedback:
                    feedback.pushInfo(f"Using fallback driver: {driver_name}")
            else:
                raise QgsProcessingException(f"Unsupported format for OGR direct save: {file_ext}")
        
        driver = ogr.GetDriverByName(driver_name)
        if driver is None:
            raise QgsProcessingException(f"OGR driver not available: {driver_name}")
        
        if feedback:
            feedback.pushInfo(f"Using OGR driver: {driver_name}")
        
        # Create spatial reference if CRS is available
        srs = None
        if hasattr(core, 'crs') and core.crs:
            srs = osr.SpatialReference()
            result = srs.SetFromUserInput(core.crs)
            if result == 0:
                if feedback:
                    feedback.pushInfo(f"Created SRS from: {core.crs}")
            else:
                if feedback:
                    feedback.pushWarning(f"Could not parse CRS: {core.crs}")
                srs = None
        
        # Remove existing file if it exists
        if os.path.exists(file_path):
            driver.DeleteDataSource(file_path)
        
        # Create data source
        ds = driver.CreateDataSource(file_path)
        if ds is None:
            raise QgsProcessingException(f"Could not create data source: {file_path}")
        
        # Determine geometry type from first geometry
        geom_type = ogr.wkbUnknown
        if len(cleaned_gdf) > 0:
            first_geom = cleaned_gdf.iloc[0].geometry
            if first_geom is not None:
                from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
                if isinstance(first_geom, Point):
                    geom_type = ogr.wkbPoint
                elif isinstance(first_geom, LineString):
                    geom_type = ogr.wkbLineString
                elif isinstance(first_geom, Polygon):
                    geom_type = ogr.wkbPolygon
                elif isinstance(first_geom, MultiPoint):
                    geom_type = ogr.wkbMultiPoint
                elif isinstance(first_geom, MultiLineString):
                    geom_type = ogr.wkbMultiLineString
                elif isinstance(first_geom, MultiPolygon):
                    geom_type = ogr.wkbMultiPolygon
        
        # Create layer
        layer_name = os.path.splitext(os.path.basename(file_path))[0]
        layer = ds.CreateLayer(layer_name, srs=srs, geom_type=geom_type)
        if layer is None:
            ds = None
            raise QgsProcessingException("Could not create layer")
        
        # Create fields (excluding geometry column)
        for column in cleaned_gdf.columns:
            if column == 'geometry':
                continue
            
            # Determine OGR field type
            dtype = cleaned_gdf[column].dtype
            if pd.api.types.is_integer_dtype(dtype):
                field_type = ogr.OFTInteger64
            elif pd.api.types.is_float_dtype(dtype):
                field_type = ogr.OFTReal
            else:
                field_type = ogr.OFTString
            
            field_defn = ogr.FieldDefn(column, field_type)
            layer.CreateField(field_defn)
        
        # Add features
        for idx, row in cleaned_gdf.iterrows():
            # Create feature
            feature = ogr.Feature(layer.GetLayerDefn())
            
            # Set geometry
            if row.geometry is not None:
                geom = ogr.CreateGeometryFromWkb(row.geometry.wkb)
                feature.SetGeometry(geom)
            
            # Set attributes
            for column in cleaned_gdf.columns:
                if column == 'geometry':
                    continue
                value = row[column]
                if pd.isna(value):
                    continue
                feature.SetField(column, value)
            
            # Add feature to layer
            layer.CreateFeature(feature)
            feature = None
        
        # Close dataset
        ds = None
        
        if feedback:
            feedback.pushInfo(f"✓ GeoDataFrame saved using OGR: {file_path}")
            feedback.pushInfo(f"  Features written: {len(cleaned_gdf)}")
            if srs:
                feedback.pushInfo(f"  CRS: {core.crs}")
        
    except Exception as e:
        error_msg = f"Failed to save GeoDataFrame using OGR: {e}"
        if feedback:
            feedback.pushWarning(error_msg)
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
            # Check if it's a Windows drive letter path (e.g., C:/, D:\)
            # Use the regex pattern defined at module level
            is_windows_path = WINDOWS_DRIVE_RE.match(raster_path) is not None
            
            # Only treat as GDAL virtual path if it's NOT a Windows drive letter
            if not is_windows_path:
                parts = raster_path.split(':')
                # It's a GDAL virtual file system path
                # Format: GPKG:/path/to/file.gpkg:layer_name or HDF5:/path/file.h5:dataset
                if parts[0].upper() == 'GPKG':
                    if len(parts) >= 3:
                        # Extract the actual file path (between first and last colon)
                        # Handles both: GPKG:/unix/path:layer and GPKG:C:/windows/path:layer
                        base_path = ':'.join(parts[1:-1])
                        if feedback:
                            feedback.pushInfo(f"Extracted GPKG raster base path: {base_path}")
                else:
                    # For other GDAL virtual paths (HDF5, NETCDF, etc.)
                    # Format: driver:/path/file.ext or driver:/path/file.ext:dataset
                    base_path = ':'.join(parts[1:]) if len(parts) > 2 else parts[1]
                    if feedback:
                        feedback.pushInfo(f"Extracted GDAL virtual path base: {base_path}")
        
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

