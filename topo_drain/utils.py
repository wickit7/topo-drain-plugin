# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: utils.py
#
# Purpose: Utility functions for the TopoDrain plugin
#
# -----------------------------------------------------------------------------

import os
import re
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsLineSymbol,
    QgsMarkerLineSymbolLayer,
    QgsSimpleMarkerSymbolLayer,
    QgsSingleSymbolRenderer,
    QgsMarkerSymbol
)

def get_crs_from_project(fallback_crs="EPSG:2056"):
    """
    Get the current QGIS project CRS as authid (e.g., 'EPSG:4326')
    Returns 'EPSG:2056' as fallback if project CRS is invalid or unavailable
    """
    try:
        project_crs = QgsProject.instance().crs().authid()
        if project_crs and project_crs.strip():
            return project_crs
        else:
            print(f"[TopoDrain Utils] Project CRS is empty, using fallback: {fallback_crs}")
            return fallback_crs
    except Exception as e:
        print(f"[TopoDrain Utils] Could not get project CRS: {e}, using fallback: {fallback_crs}")
        return fallback_crs

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

def get_crs_from_layer(layer_source, fallback_crs="EPSG:2056"):
    """
    Get CRS from a raster or vector layer file or object
    
    Args:
        layer_source: Can be a QgsRasterLayer, QgsVectorLayer, or file path string
        fallback_crs: CRS to use if layer CRS cannot be determined
    
    Returns:
        str: Valid CRS authid (never None or empty)
    """    
    try:
        # Check if layer_source is already a QGIS layer object
        if isinstance(layer_source, (QgsRasterLayer, QgsVectorLayer)):
            print(f"[TopoDrain Utils] layer_source is a QGIS layer object: {type(layer_source)}")
            if layer_source.isValid():
                crs = layer_source.crs()
                crs_authid = crs.authid()
                print(f"[TopoDrain Utils] Layer is valid. CRS authid: '{crs_authid}'")
                
                # If authid is empty, try alternative methods
                if not crs_authid or crs_authid.strip() == "":
                    print("[TopoDrain Utils] authid() returned empty, trying alternatives...")
                    # Try to get from description or other methods
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
                    
                    # If still no EPSG, return the WKT as fallback for GeoPandas
                    if wkt:
                        print(f"[TopoDrain Utils] Using WKT as fallback: {wkt[:100]}...")
                        return wkt
                else:
                    print(f"[TopoDrain Utils] Returning authid: {crs_authid}")
                    return crs_authid
            else:
                print("[TopoDrain Utils] Layer object is not valid.")
        
        # Otherwise treat it as a file path
        elif isinstance(layer_source, str) and os.path.exists(layer_source):
            # Try as raster first
            print(f"[TopoDrain Utils] Trying as QgsRasterLayer: {layer_source}")
            layer = QgsRasterLayer(layer_source)
            if layer.isValid():
                crs = layer.crs()
                crs_authid = crs.authid()
                print(f"[TopoDrain Utils] Raster layer is valid. CRS authid: '{crs_authid}'")
                
                # If authid is empty, try alternative methods
                if not crs_authid or crs_authid.strip() == "":
                    print("[TopoDrain Utils] authid() returned empty for raster, trying alternatives...")
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
                        print(f"[TopoDrain Utils] Using WKT as fallback for raster: {wkt[:100]}...")
                        return wkt
                else:
                    print(f"[TopoDrain Utils] Returning raster authid: {crs_authid}")
                    return crs_authid
            else:
                print("[TopoDrain Utils] Raster layer is not valid.")
            
            # Try as vector
            print(f"[TopoDrain Utils] Trying as QgsVectorLayer: {layer_source}")
            layer = QgsVectorLayer(layer_source)
            if layer.isValid():
                crs = layer.crs()
                crs_authid = crs.authid()
                print(f"[TopoDrain Utils] Vector layer is valid. CRS authid: '{crs_authid}'")
                
                # If authid is empty, try alternative methods
                if not crs_authid or crs_authid.strip() == "":
                    print("[TopoDrain Utils] authid() returned empty for vector, trying alternatives...")
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
                    wkt = crs.toWkt()
                    if wkt:
                        print(f"[TopoDrain Utils] Using WKT as fallback for vector: {wkt[:100]}...")
                        return wkt
                else:
                    print(f"[TopoDrain Utils] Returning vector authid: {crs_authid}")
                    return crs_authid
            else:
                print("[TopoDrain Utils] Vector layer is not valid.")
        else:
            if isinstance(layer_source, str):
                print(f"[TopoDrain Utils] File does not exist: {layer_source}")
            else:
                print(f"[TopoDrain Utils] layer_source is not a valid type: {type(layer_source)}")

    except Exception as e:
        print(f"[TopoDrain Utils] Could not get CRS from layer {layer_source}: {e}")
    
    # Fallback if no valid CRS could be determined
    print(f"[TopoDrain Utils] No valid CRS found, using fallback: {fallback_crs}")
    return fallback_crs

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
        core.crs = input_crs
        return True
    
    return False


def clean_qvariant_data(gdf):
    """
    Clean QVariant objects from GeoDataFrame to avoid field type errors during saving.
    Converts QVariant values to appropriate Python native types.
    
    Args:
        gdf (GeoDataFrame): Input GeoDataFrame that may contain QVariant objects
        
    Returns:
        GeoDataFrame: Cleaned GeoDataFrame with native Python types
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
    
    return cleaned_gdf

