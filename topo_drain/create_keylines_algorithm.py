# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: create_keylines_algorithm.py
#
# Purpose: QGIS Processing Algorithm to create keylines using iterative tracing
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorLayer, QgsProcessingParameterMultipleLayers,
                       QgsProcessingParameterVectorDestination, QgsProcessingParameterNumber,
                       QgsProcessingParameterBoolean, QgsProcessing, QgsProcessingException, 
                       QgsProcessingParameterFeatureSource)
import os
import geopandas as gpd
from .utils import get_crs_from_layer, update_core_crs_if_needed, ensure_whiteboxtools_configured, save_gdf_to_file, load_gdf_from_file, load_gdf_from_qgis_source, get_raster_ext, get_vector_ext

pluginPath = os.path.dirname(__file__)

class CreateKeylinesAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for creating keylines using iterative tracing between ridges and valleys.

    This algorithm creates comprehensive keyline networks by iteratively tracing constant slope lines:
    1. Traces from start points (keypoints) to ridges using valleys as barriers
    2. Creates new start points beyond ridge endpoints and traces to valleys using ridges as barriers
    3. Continues iteratively while endpoints reach target features (ridges or valleys)
    4. Returns all traced keylines as a combined vector layer

    The iterative approach ensures that keylines follow natural topographic flow patterns,
    creating comprehensive drainage and access line networks that respect the landscape's
    ridge-valley structure.

    This is particularly useful for:
    - Agricultural keyline design
    - Drainage planning
    - Access road planning following natural contours
    - Watershed management
    """

    INPUT_DTM = 'INPUT_DTM'
    INPUT_START_POINTS = 'INPUT_START_POINTS'
    INPUT_VALLEY_LINES = 'INPUT_VALLEY_LINES'
    INPUT_RIDGE_LINES = 'INPUT_RIDGE_LINES'
    INPUT_PERIMETER = 'INPUT_PERIMETER'
    OUTPUT_KEYLINES = 'OUTPUT_KEYLINES'
    SLOPE = 'SLOPE'
    CHANGE_AFTER = 'CHANGE_AFTER'
    SLOPE_AFTER = 'SLOPE_AFTER'
    SLOPE_DEVIATION_THRESHOLD = 'SLOPE_DEVIATION_THRESHOLD'
    MAX_ITERATIONS_SLOPE = 'MAX_ITERATIONS_SLOPE'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        instance = CreateKeylinesAlgorithm(core=self.core)
        if hasattr(self, 'plugin'):
            instance.plugin = self.plugin
        return instance

    def name(self):
        return 'create_keylines'

    def displayName(self):
        return self.tr('Create Keylines')

    def group(self):
        return self.tr('Slope Line Analysis')

    def groupId(self):
        return 'slope_line_analysis'

    def shortHelpString(self):
        return self.tr(
            """QGIS Processing Algorithm for creating keylines using iterative tracing between ridges and valleys.

This algorithm creates comprehensive keyline networks by iteratively tracing constant slope lines:
1. Traces from start points (keypoints) to ridges using valleys as barriers
2. Creates new start points beyond ridge endpoints and traces to valleys using ridges as barriers  
3. Continues iteratively while endpoints reach target features (ridges or valleys)
4. Returns all traced keylines as a combined vector layer

All output keylines will be oriented from valley to ridge (valley → ridge direction).

Parameters:
- Input DTM: Digital Terrain Model for slope calculations
- Start Points: Point features where keylines should begin (start points can be positioned on valley lines, on ridge lines, or mixed - the algorithm automatically detects and handles each type appropriately)
- Valley Lines: Valley line features to use as barriers/destinations during tracing
- Ridge Lines: Ridge line features to use as barriers/destinations during tracing
- Perimeter: Polygon features defining area of interest (always acts as final destination)
- Slope: Desired slope as a decimal (e.g., 0.01 for 1% downhill, -0.01 for 1% uphill) - always defined from valley to ridge perspective, regardless of where start points are located
- Change Slope At Distance: Creates two segments - Desired Slope from start to this point, then New Slope to end (e.g., 0.5 = change at middle) - always defined from valley to ridge perspective, regardless of where start points are located
- New Slope After Change Point: New Slope to apply for the second segment (required if Change Slope At Distance is set) - always defined from valley to ridge perspective, regardless of where start points are located
- Slope Deviation Threshold: Maximum allowed slope deviation before triggering slope refinement iterations (0.0-1.0, e.g., 0.2 = 20%)
- Max Iterations Slope: Maximum iterations for slope refinement (1-50, default: 20)

Example Use Cases:
• Agricultural Keyline Design: Use slope of about 1% downhill (slope = 0.01) with start points on valley lines to create water-harvesting keylines that move water across the landscape towards ridges
• Advanced Slope Control: Set "Change Slope At Distance" to e.g. 0.5 (middle of line) and New Slope to 0.0 to create keylines that start steep (1%) then level out (0%) for better water infiltration
• Starting on ridge line (start point)?: Think in perspective valley to ridge! -->  For 0.5% uphill from ridge for 40% of length then 1% uphill to valley --> use: slope=0.01, change_after=0.6 (because 1-0.4=0.6), slope_after=0.005

The algorithm alternates between tracing to ridges and valleys, creating new start points
beyond endpoints that intersect target features, and continues until no more valid
connections can be made. Ensure that a valley line is followed by a ridge line and then 
another valley line, alternating between the two."""
        )

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'icons', 'topo_drain.svg'))

    def initAlgorithm(self, config=None):        
        # Input parameters
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DTM,
                self.tr('Input DTM (Digital Terrain Model)')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_START_POINTS,
                self.tr('Start Points (lying on valley or ridge lines, e.g. keypoints on valley lines)'),
                types=[QgsProcessing.TypeVectorPoint]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VALLEY_LINES,
                self.tr('Main Valley Lines'),
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_RIDGE_LINES,
                self.tr('Main Ridge Lines'),
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_PERIMETER,
                self.tr('Perimeter (Area of Interest)'),
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True
            )
        )
        
        # Algorithm parameters
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE,
                self.tr('Desired Slope (decimal, e.g., 0.01 for 1% downhill, -0.01 for 1% uphill) - always valley to ridge perspective!'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.01,
                minValue=-1.0,
                maxValue=1.0
            )
        )
        
        # Optional slope adjustment parameters
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CHANGE_AFTER,
                self.tr('Change Slope At Distance (0.5 = Desired Slope from start to middle, then New Slope from middle to end) - valley to ridge perspective!'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=None,
                minValue=0.01,
                maxValue=0.99,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE_AFTER,
                self.tr('New Slope After Change Point (decimal, e.g., 0.005 for 0.5% downhill) - valley to ridge perspective!'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=None,
                minValue=-1.0,
                maxValue=1.0,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE_DEVIATION_THRESHOLD,
                self.tr('Advanced: Slope Deviation Threshold (max allowed deviation before slope refinement, 0.0-1.0, default: 0.2 = 20%)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.2,
                minValue=0.01,
                maxValue=1.0,
                optional=False
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ITERATIONS_SLOPE,
                self.tr('Advanced: Max Iterations Slope (maximum iterations for line refinement, 1-100, default: 20)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=30,
                minValue=1,
                maxValue=500
            )
        )
        
        # Output parameters
        keylines_param = QgsProcessingParameterVectorDestination(
            self.OUTPUT_KEYLINES,
            self.tr('Output Keylines'),
            type=QgsProcessing.TypeVectorLine,
            defaultValue=None
        )
        self.addParameter(keylines_param)

    def processAlgorithm(self, parameters, context, feedback):
        # Ensure WhiteboxTools is configured before running
        if not ensure_whiteboxtools_configured(self, feedback):
            return {}
        
        # Validate and read input parameters
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        start_points_source = self.parameterAsSource(parameters, self.INPUT_START_POINTS, context)
        valley_lines_layer = self.parameterAsVectorLayer(parameters, self.INPUT_VALLEY_LINES, context)
        ridge_lines_layer = self.parameterAsVectorLayer(parameters, self.INPUT_RIDGE_LINES, context)
        perimeter_layer = self.parameterAsVectorLayer(parameters, self.INPUT_PERIMETER, context)
        
        # Get DTM path and validate format
        dtm_path = dtm_layer.source()
        dtm_ext = get_raster_ext(dtm_path, feedback)
        
        # Validate raster format compatibility with GDAL driver mapping
        supported_raster_formats = list(self.core.gdal_driver_mapping.keys())
        if hasattr(self.core, 'gdal_driver_mapping') and dtm_ext not in self.core.gdal_driver_mapping:
            raise QgsProcessingException(f"DTM raster format '{dtm_ext}' is not supported. Supported formats: {supported_raster_formats}")

        # Validate vector formats (warning only)
        supported_vector_formats = list(self.core.ogr_driver_mapping.keys()) if hasattr(self.core, 'ogr_driver_mapping') else []
        valley_lines_path = valley_lines_layer.source() if valley_lines_layer else None
        valley_ext = get_vector_ext(valley_lines_path, feedback)
        if hasattr(self.core, 'ogr_driver_mapping') and valley_ext not in self.core.ogr_driver_mapping:
            feedback.pushWarning(f"Valley lines format '{valley_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to load it automatically.")
        ridge_lines_path = ridge_lines_layer.source() if ridge_lines_layer else None
        ridge_ext = get_vector_ext(ridge_lines_path, feedback)
        if hasattr(self.core, 'ogr_driver_mapping') and ridge_ext not in self.core.ogr_driver_mapping:
            feedback.pushWarning(f"Ridge lines format '{ridge_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to load it automatically.")
        if perimeter_layer and perimeter_layer.source():
            perimeter_layer_path = perimeter_layer.source()
            perimeter_ext = get_vector_ext(perimeter_layer_path, feedback)
            if hasattr(self.core, 'ogr_driver_mapping') and perimeter_ext not in self.core.ogr_driver_mapping:
                feedback.pushWarning(f"Perimeter format '{perimeter_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to load it automatically.")
       
        # Get output parameters
        keylines_output = self.parameterAsOutputLayer(parameters, self.OUTPUT_KEYLINES, context)
        slope = self.parameterAsDouble(parameters, self.SLOPE, context)

        # Extract file paths
        keylines_path = keylines_output
        
        # Validate output vector format compatibility with OGR driver mapping
        output_ext = get_vector_ext(keylines_path, feedback, check_existence=False)
        if hasattr(self.core, 'ogr_driver_mapping') and output_ext not in self.core.ogr_driver_mapping:
            feedback.pushWarning(f"Output file format '{output_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to save it automatically.")

        # Optional slope adjustment parameters
        change_after = self.parameterAsDouble(parameters, self.CHANGE_AFTER, context) if parameters.get(self.CHANGE_AFTER) is not None else None
        slope_after = self.parameterAsDouble(parameters, self.SLOPE_AFTER, context) if parameters.get(self.SLOPE_AFTER) is not None else None
        slope_deviation_threshold = self.parameterAsDouble(parameters, self.SLOPE_DEVIATION_THRESHOLD, context)
        max_iterations_slope = self.parameterAsInt(parameters, self.MAX_ITERATIONS_SLOPE, context)
        
        # Validate that both change_after and slope_after are provided together
        if (change_after is not None) != (slope_after is not None):
            raise QgsProcessingException("Both 'Change After' and 'Slope After' parameters must be provided together or both left empty.")

        
        # Read CRS from the DTM using QGIS layer with safe fallback
        feedback.pushInfo("Reading CRS from DTM...")
        dtm_crs = get_crs_from_layer(dtm_layer)
        feedback.pushInfo(f"DTM Layer crs: {dtm_crs}")

        # Update core CRS if needed (dtm_crs is guaranteed to be valid)
        update_core_crs_if_needed(self.core, dtm_crs, feedback)

        # Convert QGIS layers to GeoDataFrames
        feedback.pushInfo("Converting start points to GeoDataFrame...")
        start_points_gdf = load_gdf_from_qgis_source(start_points_source, feedback)
        if start_points_gdf.empty:
            raise QgsProcessingException("No start points found in input layer")
        
        feedback.pushInfo(f"Start points: {len(start_points_gdf)} features")

        # Convert valley lines to GeoDataFrame with Windows-safe CRS handling
        feedback.pushInfo("Converting valley lines to GeoDataFrame...")
        try:
            # Load GeoDataFrame using utility function
            valley_lines_gdf = load_gdf_from_file(valley_lines_path, feedback)
            feedback.pushInfo(f"Successfully loaded {len(valley_lines_gdf)} valley line features")
        except Exception as e:
            feedback.pushInfo(f"Failed to load valley lines: {e}")
            raise QgsProcessingException(f"Failed to load valley lines: {e}")
            
        if valley_lines_gdf.empty:
            raise QgsProcessingException("No valley lines found in input layer")
        
        valley_lines_gdf = valley_lines_gdf.to_crs(self.core.crs)
        feedback.pushInfo(f"Valley lines: {len(valley_lines_gdf)} features")

        # Convert ridge lines to GeoDataFrame with Windows-safe CRS handling
        feedback.pushInfo("Converting ridge lines to GeoDataFrame...")
        try:
            # Load GeoDataFrame using utility function
            ridge_lines_gdf = load_gdf_from_file(ridge_lines_path, feedback)
            feedback.pushInfo(f"Successfully loaded {len(ridge_lines_gdf)} ridge line features")
        except Exception as e:
            feedback.pushInfo(f"Failed to load ridge lines: {e}")
            raise QgsProcessingException(f"Failed to load ridge lines: {e}")
            
        if ridge_lines_gdf.empty:
            raise QgsProcessingException("No ridge lines found in input layer")
        
        ridge_lines_gdf = ridge_lines_gdf.to_crs(self.core.crs)
        feedback.pushInfo(f"Ridge lines: {len(ridge_lines_gdf)} features")

        # Convert perimeter to GeoDataFrame (optional) with Windows-safe CRS handling
        perimeter_gdf = None
        if perimeter_layer:
            feedback.pushInfo("Converting perimeter to GeoDataFrame...")
            try:
                # Load GeoDataFrame using utility function
                perimeter_gdf = load_gdf_from_file(perimeter_layer_path, feedback)
                feedback.pushInfo(f"Successfully loaded {len(perimeter_gdf)} perimeter")
            except Exception as e:
                feedback.pushInfo(f"Failed to load perimeter: {e}")
                raise QgsProcessingException(f"Failed to load perimeter: {e}")
                
            if not perimeter_gdf.empty:
                perimeter_gdf = perimeter_gdf.to_crs(self.core.crs)
                feedback.pushInfo(f"Perimeter: {len(perimeter_gdf)} features")
            else:
                feedback.pushInfo("Warning: Empty perimeter layer provided")
                perimeter_gdf = None
        else:
            feedback.pushInfo("No perimeter layer provided (optional)")

        # Report slope adjustment settings
        if change_after is not None and slope_after is not None:
            feedback.pushInfo(f"Slope adjustment enabled: change after {change_after*100:.1f}% to slope {slope_after}")
        else:
            feedback.pushInfo("No slope adjustment will be applied")

        feedback.pushInfo("Running keylines creation...")
        keylines_gdf = self.core.create_keylines(
            dtm_path=dtm_path,
            start_points=start_points_gdf,
            valley_lines=valley_lines_gdf,
            ridge_lines=ridge_lines_gdf,
            slope=slope,
            perimeter=perimeter_gdf,
            change_after=change_after,
            slope_after=slope_after,
            slope_deviation_threshold=slope_deviation_threshold,
            max_iterations_slope=max_iterations_slope,
            feedback=feedback
        )

        if keylines_gdf.empty:
            raise QgsProcessingException("No keylines were created")

        # Ensure the keylines GeoDataFrame has the correct CRS
        keylines_gdf = keylines_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Keylines CRS: {keylines_gdf.crs}")

        # Save result with proper format handling (all_upper=True to rename columns to uppercase)
        save_gdf_to_file(keylines_gdf, keylines_path, self.core, feedback, all_upper=True)

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters: 
                results[outputName] = parameters[outputName]
                
        return results
