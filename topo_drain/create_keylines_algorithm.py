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
from .utils import get_crs_from_layer

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
        return CreateKeylinesAlgorithm(core=self.core)

    def name(self):
        return 'create_keylines'

    def displayName(self):
        return self.tr('Create Keylines')

    def group(self):
        return self.tr('Slope Analysis')

    def groupId(self):
        return 'slope_analysis'

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
- Start Points: Point features where keylines should begin (typically keypoints from valleys)
- Valley Lines: Valley line features to use as barriers/destinations during tracing
- Ridge Lines: Ridge line features to use as barriers/destinations during tracing
- Perimeter: Polygon features defining area of interest (always acts as final destination)
- Slope: Desired slope as a decimal (e.g., 0.01 for 1% downhill, -0.01 for 1% uphill)
- Change Slope At Distance: Creates two segments - Desired Slope from start to this point, then New Slope to end (e.g., 0.5 = change at middle)
- New Slope After Change Point: New Slope to apply for the second segment (required if Change Slope At Distance is set)
- Slope Deviation Threshold: Maximum allowed slope deviation before triggering slope refinement iterations (0.0-1.0, e.g., 0.2 = 20%)
- Max Iterations Slope: Maximum iterations for slope refinement (1-50, default: 20)

The algorithm alternates between tracing to ridges and valleys, creating new start points
beyond endpoints that intersect target features, and continues until no more valid
connections can be made."""
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
                self.tr('Start Points (keypoints)'),
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
                self.tr('Desired Slope (decimal, e.g., 0.01 for 1% downhill, -0.01 for 1% uphill)'),
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
                self.tr('Change Slope At Distance (0.5 = Desired Slope from start to middle, then New Slope from middle to end)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=None,
                minValue=0.0,
                maxValue=1.0,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE_AFTER,
                self.tr('New Slope After Change Point (decimal, e.g., 0.005 for 0.5% downhill)'),
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
                minValue=0.0,
                maxValue=1.0,
                optional=False
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ITERATIONS_SLOPE,
                self.tr('Advanced: Max Iterations Slope (maximum iterations for line refinement, 1-50, default: 20)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=20,
                minValue=1,
                maxValue=50
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
        # Validate and read input parameters
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        start_points_source = self.parameterAsSource(parameters, self.INPUT_START_POINTS, context)
        valley_lines_layer = self.parameterAsVectorLayer(parameters, self.INPUT_VALLEY_LINES, context)
        ridge_lines_layer = self.parameterAsVectorLayer(parameters, self.INPUT_RIDGE_LINES, context)
        perimeter_layer = self.parameterAsVectorLayer(parameters, self.INPUT_PERIMETER, context)
        
        dtm_path = dtm_layer.source()
        if not dtm_path or not os.path.exists(dtm_path):
            raise QgsProcessingException(f"DTM file not found: {dtm_path}")
        
        keylines_output = self.parameterAsOutputLayer(parameters, self.OUTPUT_KEYLINES, context)
        slope = self.parameterAsDouble(parameters, self.SLOPE, context)
        
        # Optional slope adjustment parameters
        change_after = self.parameterAsDouble(parameters, self.CHANGE_AFTER, context) if parameters.get(self.CHANGE_AFTER) is not None else None
        slope_after = self.parameterAsDouble(parameters, self.SLOPE_AFTER, context) if parameters.get(self.SLOPE_AFTER) is not None else None
        slope_deviation_threshold = self.parameterAsDouble(parameters, self.SLOPE_DEVIATION_THRESHOLD, context)
        max_iterations_slope = self.parameterAsInt(parameters, self.MAX_ITERATIONS_SLOPE, context)
        
        # Validate that both change_after and slope_after are provided together
        if (change_after is not None) != (slope_after is not None):
            raise QgsProcessingException("Both 'Change After' and 'Slope After' parameters must be provided together or both left empty.")

        # Extract file paths
        keylines_path = keylines_output if isinstance(keylines_output, str) else keylines_output

        feedback.pushInfo("Reading CRS from DTM...")
        # Read CRS from the DTM using QGIS layer
        dtm_crs = get_crs_from_layer(dtm_layer)
        feedback.pushInfo(f"DTM Layer crs: {dtm_crs}")

        # Check if self.core.crs matches dtm_crs, warn and update if not
        if dtm_crs:
            if self.core and hasattr(self.core, "crs"):
                if self.core.crs != dtm_crs:
                    feedback.reportError(f"Warning: TopoDrainCore CRS ({self.core.crs}) does not match DTM CRS ({dtm_crs}). Updating TopoDrainCore CRS to match DTM.")
                    self.core.crs = dtm_crs

        feedback.pushInfo("Processing keylines via TopoDrainCore...")
        if not self.core:
            from topo_drain.core.topo_drain_core import TopoDrainCore
            feedback.reportError("TopoDrainCore not set, creating default instance.")
            self.core = TopoDrainCore()  # fallback: create default instance (not recommended for plugin use)

        # Convert QGIS layers to GeoDataFrames
        feedback.pushInfo("Converting start points to GeoDataFrame...")
        start_points_gdf = gpd.GeoDataFrame.from_features(start_points_source.getFeatures())
        if start_points_gdf.empty:
            raise QgsProcessingException("No start points found in input layer")
        
        feedback.pushInfo(f"Start points: {len(start_points_gdf)} features")

        # Convert valley lines to GeoDataFrame
        feedback.pushInfo("Converting valley lines to GeoDataFrame...")
        if not valley_lines_layer or not valley_lines_layer.source():
            raise QgsProcessingException("No valley lines layer provided")
        
        valley_lines_gdf = gpd.read_file(valley_lines_layer.source())
        if valley_lines_gdf.empty:
            raise QgsProcessingException("No valley lines found in input layer")
        
        valley_lines_gdf = valley_lines_gdf.to_crs(self.core.crs)
        feedback.pushInfo(f"Valley lines: {len(valley_lines_gdf)} features")

        # Convert ridge lines to GeoDataFrame
        feedback.pushInfo("Converting ridge lines to GeoDataFrame...")
        if not ridge_lines_layer or not ridge_lines_layer.source():
            raise QgsProcessingException("No ridge lines layer provided")
        
        ridge_lines_gdf = gpd.read_file(ridge_lines_layer.source())
        if ridge_lines_gdf.empty:
            raise QgsProcessingException("No ridge lines found in input layer")
        
        ridge_lines_gdf = ridge_lines_gdf.to_crs(self.core.crs)
        feedback.pushInfo(f"Ridge lines: {len(ridge_lines_gdf)} features")

        # Convert perimeter to GeoDataFrame (optional)
        perimeter_gdf = None
        if perimeter_layer and perimeter_layer.source():
            feedback.pushInfo("Converting perimeter to GeoDataFrame...")
            perimeter_gdf = gpd.read_file(perimeter_layer.source())
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

        # Save result
        try:
            keylines_gdf.to_file(keylines_path)
            feedback.pushInfo(f"Keylines saved to: {keylines_path}")
        except Exception as e:
            raise QgsProcessingException(f"Failed to save keylines output: {e}")

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters: 
                results[outputName] = parameters[outputName]
                
        return results
