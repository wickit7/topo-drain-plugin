# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: create_constant_slope_lines_algorithm.py
#
# Purpose: QGIS Processing Algorithm to create constant slope lines from keypoints
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorLayer, QgsProcessingParameterMultipleLayers,
                       QgsProcessingParameterVectorDestination, QgsProcessingParameterNumber,
                       QgsProcessing, QgsProcessingException, QgsProcessingParameterFeatureSource)
import os
import geopandas as gpd
from .utils import get_crs_from_layer

pluginPath = os.path.dirname(__file__)

class CreateConstantSlopeLinesAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for creating constant slope lines from starting points (e.g., keypoints).

    This algorithm traces lines with constant slope starting from given points using a cost-distance approach
    based on slope deviation. The algorithm can handle barrier features to avoid and destination features
    to target. Start points that overlap with barrier lines are automatically offset in both orthogonal directions to avoid conflicts.

    The algorithm performs the following steps:
    1. Creates a cost raster based on deviation from the desired slope
    2. Handles barrier features by rasterizing them and offsetting overlapping start points
    3. Uses WhiteboxTools cost-distance analysis to find optimal paths
    4. Traces least-cost pathways from start points to destination features
    5. Returns the traced constant slope lines as vector features

    This is useful for creating drainage lines, access paths, or other linear features that need
    to maintain a specific gradient across the terrain.
    """

    INPUT_DTM = 'INPUT_DTM'
    INPUT_START_POINTS = 'INPUT_START_POINTS'
    INPUT_DESTINATION_FEATURES = 'INPUT_DESTINATION_FEATURES'
    INPUT_BARRIER_FEATURES = 'INPUT_BARRIER_FEATURES'
    OUTPUT_SLOPE_LINES = 'OUTPUT_SLOPE_LINES'
    SLOPE = 'SLOPE'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return CreateConstantSlopeLinesAlgorithm(core=self.core)

    def name(self):
        return 'create_constant_slope_lines'

    def displayName(self):
        return self.tr('Create Constant Slope Lines')

    def group(self):
        return self.tr('Slope Analysis')

    def groupId(self):
        return 'slope_analysis'

    def shortHelpString(self):
        return self.tr(
            """QGIS Processing Algorithm for creating constant slope lines from starting points (e.g., keypoints).

This algorithm traces lines with constant slope starting from given points using a cost-distance approach
based on slope deviation. The algorithm can handle barrier features to avoid and destination features
to target. Start points that overlap with barrier lines are automatically offset to avoid conflicts.

The algorithm performs the following steps:
1. Creates a cost raster based on deviation from the desired slope
2. Handles barrier features by rasterizing them and offsetting overlapping start points
3. Uses WhiteboxTools cost-distance analysis to find optimal paths
4. Traces least-cost pathways from start points to destination features
5. Returns the traced constant slope lines as vector features

This is useful for creating drainage lines, access paths, or other linear features that need
to maintain a specific gradient across the terrain.

Parameters:
- Input DTM: Digital Terrain Model for slope calculations
- Start Points: Point features where slope lines should begin (e.g., keypoints)
- Destination Features: Line or polygon features that slope lines should reach (e.g. main ridge lines, area of interest)
- Barrier Features (optional): Line or polygon features to avoid during tracing (e.g. main valley lines)
- Slope: Desired slope as a decimal (e.g., 0.01 for 1% downhill, -0.01 for 1% uphill)"""
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
                self.tr('Start Points (e.g., keypoints)'),
                types=[QgsProcessing.TypeVectorPoint]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.INPUT_DESTINATION_FEATURES,
                self.tr('Destination Features (lines or polygons that slope lines should reach)'),
                layerType=QgsProcessing.TypeVectorAnyGeometry
            )
        )
        
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.INPUT_BARRIER_FEATURES,
                self.tr('Barrier Features (lines or polygons to avoid during tracing)'),
                layerType=QgsProcessing.TypeVectorAnyGeometry,
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
        
        # Output parameters
        slope_lines_param = QgsProcessingParameterVectorDestination(
            self.OUTPUT_SLOPE_LINES,
            self.tr('Output Constant Slope Lines'),
            type=QgsProcessing.TypeVectorLine,
            defaultValue=None
        )
        self.addParameter(slope_lines_param)

    def processAlgorithm(self, parameters, context, feedback):
        # Validate and read input parameters
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        start_points_source = self.parameterAsSource(parameters, self.INPUT_START_POINTS, context)
        destination_layers = self.parameterAsLayerList(parameters, self.INPUT_DESTINATION_FEATURES, context)
        barrier_layers = self.parameterAsLayerList(parameters, self.INPUT_BARRIER_FEATURES, context)
        
        dtm_path = dtm_layer.source()
        if not dtm_path or not os.path.exists(dtm_path):
            raise QgsProcessingException(f"DTM file not found: {dtm_path}")
        
        slope_lines_output = self.parameterAsOutputLayer(parameters, self.OUTPUT_SLOPE_LINES, context)
        slope = self.parameterAsDouble(parameters, self.SLOPE, context)

        # Extract file paths
        slope_lines_path = slope_lines_output if isinstance(slope_lines_output, str) else slope_lines_output

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

        feedback.pushInfo("Processing constant slope lines via TopoDrainCore...")
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

        # Convert destination layers to GeoDataFrames
        feedback.pushInfo("Converting destination features to GeoDataFrames...")
        destination_gdfs = []
        for layer in destination_layers:
            if layer:
                gdf = gpd.read_file(layer.source())
                if not gdf.empty:
                    gdf = gdf.to_crs(self.core.crs)
                    destination_gdfs.append(gdf)
                    feedback.pushInfo(f"Destination layer: {len(gdf)} features")
        
        if not destination_gdfs:
            raise QgsProcessingException("No valid destination features found")

        # Convert barrier layers to GeoDataFrames (optional)
        barrier_gdfs = []
        if barrier_layers:
            feedback.pushInfo("Converting barrier features to GeoDataFrames...")
            for layer in barrier_layers:
                if layer:
                    gdf = gpd.read_file(layer.source())
                    if not gdf.empty:
                        gdf = gdf.to_crs(self.core.crs)
                        barrier_gdfs.append(gdf)
                        feedback.pushInfo(f"Barrier layer: {len(gdf)} features")

        feedback.pushInfo("Running constant slope lines tracing...")
        slope_lines_gdf = self.core.get_constant_slope_lines(
            dtm_path=dtm_path,
            start_points=start_points_gdf,
            destination_features=destination_gdfs,
            slope=slope,
            barrier_features=barrier_gdfs if barrier_gdfs else None,
            feedback=feedback
        )

        if slope_lines_gdf.empty:
            raise QgsProcessingException("No slope lines were created")

        # Ensure the slope lines GeoDataFrame has the correct CRS
        slope_lines_gdf = slope_lines_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Slope lines CRS: {slope_lines_gdf.crs}")

        # Save result
        try:
            slope_lines_gdf.to_file(slope_lines_path)
            feedback.pushInfo(f"Constant slope lines saved to: {slope_lines_path}")
        except Exception as e:
            raise QgsProcessingException(f"Failed to save slope lines output: {e}")

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters: 
                results[outputName] = parameters[outputName]
                
        return results
