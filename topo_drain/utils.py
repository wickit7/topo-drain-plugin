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

def get_crs_from_project():
    """Get the current QGIS project CRS as authid (e.g., 'EPSG:4326')"""
    try:
        return QgsProject.instance().crs().authid()
    except Exception as e:
        print(f"[TopoDrain Utils] Could not get project CRS: {e}")
        return None

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
    
    # Swiss coordinate system patterns
    if "CH1903" in text and "LV95" in text:
        print("[TopoDrain Utils] Detected CH1903+ / LV95 - using EPSG:2056")
        return "EPSG:2056"
    elif "CH1903" in text:
        print("[TopoDrain Utils] Detected CH1903 - using EPSG:21781")
        return "EPSG:21781"
    
    # Common coordinate system patterns
    coordinate_systems = {
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

def get_crs_from_layer(layer_source):
    """
    Get CRS from a raster or vector layer file or object
    
    Args:
        layer_source: Can be a QgsRasterLayer, QgsVectorLayer, or file path string
    
    Returns:
        str: CRS authid (e.g., 'EPSG:4326') or None if not found
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
                return None
        
        # Otherwise treat it as a file path
        if not isinstance(layer_source, str):
            print(f"[TopoDrain Utils] layer_source is not a string: {type(layer_source)}")
            return None
        if not os.path.exists(layer_source):
            print(f"[TopoDrain Utils] File does not exist: {layer_source}")
            return None
        
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

    except Exception as e:
        print(f"[TopoDrain Utils] Could not get CRS from layer {layer_source}: {e}")
    
    print("[TopoDrain Utils] Returning None for CRS.")
    return None


def apply_line_arrow_symbology(vlayer, linecolor, markercolor, linewidth=0.4, markersize=4, feedback=None):
    """
    Apply line symbology with flow direction arrows to a vector layer.
    
    Args:
        vlayer: QgsVectorLayer to apply symbology to
        linecolor: Color string for the line (e.g., '#0066CC')
        markercolor: Color string for the arrow markers (e.g., '#003366')
        linewidth: Width of the line in map units (default: 0.4)
        markersize: Size of the arrow markers (default: 4)
        feedback: Optional feedback object for error reporting
    """
    try:
        # Create a line symbol with specified color
        line_symbol = QgsLineSymbol.createSimple({
            'color': linecolor,
            'width': str(linewidth),
            'capstyle': 'round',
            'joinstyle': 'round'
        })
        
        # Create marker line symbol layer for flow direction arrows
        marker_line = QgsMarkerLineSymbolLayer()
        marker_line.setPlacement(QgsMarkerLineSymbolLayer.Interval)
        marker_line.setInterval(20)  # Place markers every 20 map units
        marker_line.setRotateMarker(True)  # Rotate markers along line direction
        
        # Create marker symbol for arrows
        marker_symbol = QgsMarkerSymbol()
        marker_symbol.deleteSymbolLayer(0)  # Remove default layer
        
        # Create arrow marker layer
        arrow_marker = QgsSimpleMarkerSymbolLayer()
        arrow_marker.setShape(QgsSimpleMarkerSymbolLayer.ArrowHead)
        arrow_marker.setSize(markersize)  # Arrow size
        arrow_marker.setColor(QColor(markercolor))  # Marker color
        arrow_marker.setStrokeColor(QColor(linecolor))  # Stroke color matches line
        arrow_marker.setStrokeWidth(0.2)
        arrow_marker.setAngle(0)  # Don't add extra rotation, let the marker line handle it
        
        # Add arrow to marker symbol
        marker_symbol.appendSymbolLayer(arrow_marker)
        
        # Set the marker symbol to the marker line
        marker_line.setSubSymbol(marker_symbol)
        
        # Add marker line to the main line symbol
        line_symbol.appendSymbolLayer(marker_line)
        
        # Apply the symbol to the layer
        renderer = QgsSingleSymbolRenderer(line_symbol)
        vlayer.setRenderer(renderer)
        vlayer.triggerRepaint()
        
    except Exception as e:
        # If symbology fails, just continue without it
        if feedback:
            feedback.reportError(f"Failed to apply symbology: {str(e)}")
        pass

