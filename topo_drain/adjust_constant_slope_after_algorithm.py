# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: adjust_constant_slope_after_algorithm.py
#
# Purpose: QGIS Processing Algorithm to adjust constant slope lines with secondary slopes
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

class AdjustConstantSlopeAfterAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for adjusting constant slope lines with secondary slopes after a specified distance.

    This algorithm modifies existing constant slope lines by changing to a secondary slope after 
    a specified fraction of the line length. The algorithm performs the following steps:
    1. Splits each input line at the specified distance fraction
    2. Keeps the first part of the line unchanged  
    3. Uses get_constant_slope_lines to trace a new second part with the secondary slope
    4. Combines both parts into a single modified line

    This is useful for creating more complex keyline profiles where different slopes are needed
    along different sections of the line, such as gentler slopes near the end to reduce erosion
    or steeper slopes at the beginning for better water collection.
    """

    INPUT_DTM = 'INPUT_DTM'
    INPUT_LINES = 'INPUT_LINES'
    INPUT_DESTINATION_FEATURES = 'INPUT_DESTINATION_FEATURES'
    INPUT_BARRIER_FEATURES = 'INPUT_BARRIER_FEATURES'
    OUTPUT_ADJUSTED_LINES = 'OUTPUT_ADJUSTED_LINES'
    CHANGE_AFTER = 'CHANGE_AFTER'
    SLOPE_AFTER = 'SLOPE_AFTER'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return AdjustConstantSlopeAfterAlgorithm(core=self.core)

    def name(self):
        return 'adjust_constant_slope_after'

    def displayName(self):
        return self.tr('Adjust Constant Slope After Distance')

    def group(self):
        return self.tr('Slope Analysis')

    def groupId(self):
        return 'slope_analysis'

    def shortHelpString(self):
        return self.tr(
            """QGIS Processing Algorithm for adjusting constant slope lines with secondary slopes after a specified distance.

This algorithm modifies existing constant slope lines by changing to a secondary slope after 
a specified fraction of the line length. This is useful for creating more complex keyline 
profiles where different slopes are needed along different sections.

The algorithm performs the following steps:
1. Splits each input line at the specified distance fraction
2. Keeps the first part of the line unchanged  
3. Uses cost-distance analysis to trace a new second part with the secondary slope
4. Combines both parts into a single modified line

Use cases:
- Creating gentler slopes near ridge lines to reduce erosion
- Steeper initial slopes for better water collection, then gentler continuation
- Adapting keylines to local terrain variations
- Optimizing agricultural drainage and water management systems

Parameters:
- Input DTM: Digital Terrain Model for slope calculations
- Input Lines: Existing constant slope lines to modify (e.g., from Create Keylines)
- Change After: Fraction of line length where slope changes (0.0-1.0, e.g., 0.5 = halfway)
- Slope After: New slope for the second part (e.g., 0.005 for 0.5% downhill)
- Destination Features: Features that the new slope sections should reach (e.g., ridge lines)
- Barrier Features (optional): Features to avoid during new slope tracing (e.g., valley lines)"""
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
                self.INPUT_LINES,
                self.tr('Input Constant Slope Lines (e.g., keylines to modify)'),
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        
        # Algorithm parameters
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CHANGE_AFTER,
                self.tr('Change After (fraction of line length, 0.0-1.0, e.g., 0.5 = halfway)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                minValue=0.0,
                maxValue=1.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE_AFTER,
                self.tr('Slope After (decimal, e.g., 0.005 for 0.5% downhill, -0.005 for 0.5% uphill)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.005,
                minValue=-1.0,
                maxValue=1.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.INPUT_DESTINATION_FEATURES,
                self.tr('Destination Features (lines or polygons that new slope sections should reach)'),
                layerType=QgsProcessing.TypeVectorAnyGeometry
            )
        )
        
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.INPUT_BARRIER_FEATURES,
                self.tr('Barrier Features (lines or polygons to avoid during new slope tracing)'),
                layerType=QgsProcessing.TypeVectorAnyGeometry,
                optional=True
            )
        )
        
        # Output parameters
        adjusted_lines_param = QgsProcessingParameterVectorDestination(
            self.OUTPUT_ADJUSTED_LINES,
            self.tr('Output Adjusted Constant Slope Lines'),
            type=QgsProcessing.TypeVectorLine,
            defaultValue=None
        )
        self.addParameter(adjusted_lines_param)

    def processAlgorithm(self, parameters, context, feedback):
        # Validate and read input parameters
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        input_lines_source = self.parameterAsSource(parameters, self.INPUT_LINES, context)
        destination_layers = self.parameterAsLayerList(parameters, self.INPUT_DESTINATION_FEATURES, context)
        barrier_layers = self.parameterAsLayerList(parameters, self.INPUT_BARRIER_FEATURES, context)
        
        dtm_path = dtm_layer.source()
        if not dtm_path or not os.path.exists(dtm_path):
            raise QgsProcessingException(f"DTM file not found: {dtm_path}")
        
        adjusted_lines_output = self.parameterAsOutputLayer(parameters, self.OUTPUT_ADJUSTED_LINES, context)
        change_after = self.parameterAsDouble(parameters, self.CHANGE_AFTER, context)
        slope_after = self.parameterAsDouble(parameters, self.SLOPE_AFTER, context)

        # Extract file paths
        adjusted_lines_path = adjusted_lines_output if isinstance(adjusted_lines_output, str) else adjusted_lines_output

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

        feedback.pushInfo("Processing constant slope line adjustment via TopoDrainCore...")
        if not self.core:
            from topo_drain.core.topo_drain_core import TopoDrainCore
            feedback.reportError("TopoDrainCore not set, creating default instance.")
            self.core = TopoDrainCore()  # fallback: create default instance (not recommended for plugin use)

        # Convert QGIS layers to GeoDataFrames
        feedback.pushInfo("Converting input lines to GeoDataFrame...")
        input_lines_gdf = gpd.GeoDataFrame.from_features(input_lines_source.getFeatures())
        if input_lines_gdf.empty:
            raise QgsProcessingException("No input lines found in input layer")
        
        # Set CRS from the source layer if GeoDataFrame doesn't have one
        if input_lines_gdf.crs is None:
            source_crs = input_lines_source.sourceCrs()
            if source_crs.isValid():
                input_lines_gdf = input_lines_gdf.set_crs(source_crs.authid())
                feedback.pushInfo(f"Set input lines CRS to: {source_crs.authid()}")
            else:
                feedback.pushInfo("Warning: Input lines layer has no valid CRS")
        
        # Ensure input lines have correct CRS
        if input_lines_gdf.crs != self.core.crs:
            input_lines_gdf = input_lines_gdf.to_crs(self.core.crs)
            feedback.pushInfo(f"Transformed input lines from {input_lines_gdf.crs} to {self.core.crs}")
        
        feedback.pushInfo(f"Input lines: {len(input_lines_gdf)} features")

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

        feedback.pushInfo("Running constant slope line adjustment...")
        adjusted_lines_gdf = self.core.adjust_constant_slope_after(
            dtm_path=dtm_path,
            input_lines=input_lines_gdf,
            change_after=change_after,
            slope_after=slope_after,
            destination_features=destination_gdfs,
            barrier_features=barrier_gdfs if barrier_gdfs else None,
            feedback=feedback
        )

        if adjusted_lines_gdf.empty:
            raise QgsProcessingException("No adjusted lines were created")

        # Ensure the adjusted lines GeoDataFrame has the correct CRS
        adjusted_lines_gdf = adjusted_lines_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Adjusted lines CRS: {adjusted_lines_gdf.crs}")

        # Save result
        try:
            adjusted_lines_gdf.to_file(adjusted_lines_path)
            feedback.pushInfo(f"Adjusted constant slope lines saved to: {adjusted_lines_path}")
        except Exception as e:
            raise QgsProcessingException(f"Failed to save adjusted lines output: {e}")

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters: 
                results[outputName] = parameters[outputName]
                
        return results
