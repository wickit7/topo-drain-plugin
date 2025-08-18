# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: adjust_keylines_after_algorithm.py
#
# Purpose: QGIS Processing Algorithm to adjust keylines with secondary slopes after distance
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

class AdjustKeylinesAfterAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for adjusting keylines with secondary slopes after a specified distance.

    This algorithm is specifically designed for keyline workflows and modifies existing keylines 
    by changing to a secondary slope after a specified fraction of the line length. It uses 
    familiar keyline terminology to make the interface more intuitive for agricultural and 
    landscape design applications.

    The algorithm performs the following steps:
    1. Splits each keyline at the specified distance fraction
    2. Keeps the first part of the keyline unchanged  
    3. Traces a new second part with the secondary slope toward ridges
    4. Uses valleys as barriers and ridges/perimeter as destinations
    5. Combines both parts into a single modified keyline

    This is particularly useful for:
    - Creating gentler slopes near ridge lines to reduce erosion
    - Steeper initial slopes for better water collection, then gentler continuation
    - Adapting keylines to local terrain variations
    - Optimizing agricultural keyline systems for better water management
    """

    INPUT_DTM = 'INPUT_DTM'
    INPUT_KEYLINES = 'INPUT_KEYLINES'
    INPUT_VALLEY_LINES = 'INPUT_VALLEY_LINES'
    INPUT_RIDGE_LINES = 'INPUT_RIDGE_LINES'
    INPUT_PERIMETER = 'INPUT_PERIMETER'
    OUTPUT_ADJUSTED_KEYLINES = 'OUTPUT_ADJUSTED_KEYLINES'
    CHANGE_AFTER = 'CHANGE_AFTER'
    SLOPE_AFTER = 'SLOPE_AFTER'
    ALLOW_BARRIERS_AS_TEMP_DESTINATION = 'ALLOW_BARRIERS_AS_TEMP_DESTINATION'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return AdjustKeylinesAfterAlgorithm(core=self.core)

    def name(self):
        return 'adjust_keylines_after'

    def displayName(self):
        return self.tr('Adjust Keylines After Distance')

    def group(self):
        return self.tr('Slope Analysis')

    def groupId(self):
        return 'slope_analysis'

    def shortHelpString(self):
        return self.tr(
            """QGIS Processing Algorithm for adjusting keylines with secondary slopes after a specified distance.

This algorithm is specifically designed for keyline workflows and modifies existing keylines 
by changing to a secondary slope after a specified fraction of the line length. It uses 
familiar keyline terminology to make the interface more intuitive for agricultural and 
landscape design applications.

The algorithm performs the following steps:
1. Splits each keyline at the specified distance fraction
2. Keeps the first part of the keyline unchanged  
3. Traces a new second part with the secondary slope toward ridges
4. Uses valleys as barriers and ridges/perimeter as destinations
5. Combines both parts into a single modified keyline

Use cases in keyline design:
- Creating gentler slopes near ridge lines to reduce erosion
- Steeper initial slopes for better water collection from valleys
- Adapting keylines to local terrain variations
- Optimizing agricultural keyline systems for better water management
- Implementing variable slope keylines for different land uses

Parameters:
- Input DTM: Digital Terrain Model for slope calculations
- Input Keylines: Existing keylines to modify (e.g., from Create Keylines algorithm)
- Valley Lines: Valley features to avoid during new slope tracing (barriers)
- Ridge Lines: Ridge features that new slope sections should reach (destinations)
- Perimeter: Optional area boundary that also acts as destination
- Change After: Fraction of keyline length where slope changes (0.0-1.0, e.g., 0.5 = halfway)
- Slope After: New slope for the second part (e.g., 0.005 for 0.5% downhill toward ridges)

The algorithm ensures that all adjusted keylines maintain the valley → ridge orientation
and follow keyline design principles for effective water management."""
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
                self.INPUT_KEYLINES,
                self.tr('Input Keylines (to modify)'),
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VALLEY_LINES,
                self.tr('Valley Lines (barriers to avoid)'),
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_RIDGE_LINES,
                self.tr('Ridge Lines (destinations to reach)'),
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_PERIMETER,
                self.tr('Perimeter (area boundary, also acts as destination)'),
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True
            )
        )
        
        # Algorithm parameters
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CHANGE_AFTER,
                self.tr('Change After (fraction of keyline length, 0.0-1.0, e.g., 0.5 = halfway)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                minValue=0.0,
                maxValue=1.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE_AFTER,
                self.tr('Slope After (decimal, e.g., 0.005 for 0.5% downhill toward ridges)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.005,
                minValue=-1.0,
                maxValue=1.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ALLOW_BARRIERS_AS_TEMP_DESTINATION,
                self.tr('Allow barriers as temporary destinations (iterative tracing)'),
                defaultValue=False
            )
        )
        
        # Output parameters
        adjusted_keylines_param = QgsProcessingParameterVectorDestination(
            self.OUTPUT_ADJUSTED_KEYLINES,
            self.tr('Output Adjusted Keylines'),
            type=QgsProcessing.TypeVectorLine,
            defaultValue=None
        )
        self.addParameter(adjusted_keylines_param)

    def processAlgorithm(self, parameters, context, feedback):
        # Validate and read input parameters
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        input_keylines_source = self.parameterAsSource(parameters, self.INPUT_KEYLINES, context)
        valley_lines_layer = self.parameterAsVectorLayer(parameters, self.INPUT_VALLEY_LINES, context)
        ridge_lines_layer = self.parameterAsVectorLayer(parameters, self.INPUT_RIDGE_LINES, context)
        perimeter_layer = self.parameterAsVectorLayer(parameters, self.INPUT_PERIMETER, context)
        
        dtm_path = dtm_layer.source()
        if not dtm_path or not os.path.exists(dtm_path):
            raise QgsProcessingException(f"DTM file not found: {dtm_path}")
        
        adjusted_keylines_output = self.parameterAsOutputLayer(parameters, self.OUTPUT_ADJUSTED_KEYLINES, context)
        change_after = self.parameterAsDouble(parameters, self.CHANGE_AFTER, context)
        slope_after = self.parameterAsDouble(parameters, self.SLOPE_AFTER, context)
        allow_barriers_as_temp_destination = self.parameterAsBool(parameters, self.ALLOW_BARRIERS_AS_TEMP_DESTINATION, context)

        # Extract file paths
        adjusted_keylines_path = adjusted_keylines_output if isinstance(adjusted_keylines_output, str) else adjusted_keylines_output

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

        feedback.pushInfo("Processing keyline adjustment via TopoDrainCore...")
        if not self.core:
            from topo_drain.core.topo_drain_core import TopoDrainCore
            feedback.reportError("TopoDrainCore not set, creating default instance.")
            self.core = TopoDrainCore()  # fallback: create default instance (not recommended for plugin use)

        # Convert QGIS layers to GeoDataFrames
        feedback.pushInfo("Converting input keylines to GeoDataFrame...")
        input_keylines_gdf = gpd.GeoDataFrame.from_features(input_keylines_source.getFeatures())
        if input_keylines_gdf.empty:
            raise QgsProcessingException("No input keylines found in input layer")
        
        # Set CRS from the source layer if GeoDataFrame doesn't have one
        if input_keylines_gdf.crs is None:
            source_crs = input_keylines_source.sourceCrs()
            if source_crs.isValid():
                input_keylines_gdf = input_keylines_gdf.set_crs(source_crs.authid())
                feedback.pushInfo(f"Set input keylines CRS to: {source_crs.authid()}")
            else:
                feedback.pushInfo("Warning: Input keylines layer has no valid CRS")
        
        # Ensure input keylines have correct CRS
        if input_keylines_gdf.crs != self.core.crs:
            input_keylines_gdf = input_keylines_gdf.to_crs(self.core.crs)
            feedback.pushInfo(f"Transformed input keylines from {input_keylines_gdf.crs} to {self.core.crs}")
        
        feedback.pushInfo(f"Input keylines: {len(input_keylines_gdf)} features")

        # Convert ridge lines to GeoDataFrame (destinations)
        feedback.pushInfo("Converting ridge lines to GeoDataFrame...")
        if not ridge_lines_layer or not ridge_lines_layer.source():
            raise QgsProcessingException("No ridge lines layer provided")
        
        ridge_lines_gdf = gpd.read_file(ridge_lines_layer.source())
        if ridge_lines_gdf.empty:
            raise QgsProcessingException("No ridge lines found in input layer")
        
        ridge_lines_gdf = ridge_lines_gdf.to_crs(self.core.crs)
        feedback.pushInfo(f"Ridge lines (destinations): {len(ridge_lines_gdf)} features")

        # Build destination features list
        destination_gdfs = [ridge_lines_gdf]

        # Add perimeter as destination if provided
        if perimeter_layer and perimeter_layer.source():
            feedback.pushInfo("Converting perimeter to GeoDataFrame...")
            perimeter_gdf = gpd.read_file(perimeter_layer.source())
            if not perimeter_gdf.empty:
                perimeter_gdf = perimeter_gdf.to_crs(self.core.crs)
                destination_gdfs.append(perimeter_gdf)
                feedback.pushInfo(f"Perimeter (additional destination): {len(perimeter_gdf)} features")
            else:
                feedback.pushInfo("Warning: Empty perimeter layer provided")
        else:
            feedback.pushInfo("No perimeter layer provided (optional)")

        # Convert valley lines to GeoDataFrame (barriers)
        barrier_gdfs = []
        if valley_lines_layer and valley_lines_layer.source():
            feedback.pushInfo("Converting valley lines to GeoDataFrame...")
            valley_lines_gdf = gpd.read_file(valley_lines_layer.source())
            if not valley_lines_gdf.empty:
                valley_lines_gdf = valley_lines_gdf.to_crs(self.core.crs)
                barrier_gdfs.append(valley_lines_gdf)
                feedback.pushInfo(f"Valley lines (barriers): {len(valley_lines_gdf)} features")
            else:
                feedback.pushInfo("Warning: Empty valley lines layer provided")
        else:
            raise QgsProcessingException("No valley lines layer provided")

        feedback.pushInfo("Running keyline adjustment...")
        adjusted_keylines_gdf = self.core.adjust_constant_slope_after(
            dtm_path=dtm_path,
            input_lines=input_keylines_gdf,
            change_after=change_after,
            slope_after=slope_after,
            destination_features=destination_gdfs,
            barrier_features=barrier_gdfs if barrier_gdfs else None,
            allow_barriers_as_temp_destination=allow_barriers_as_temp_destination,
            feedback=feedback
        )

        if adjusted_keylines_gdf.empty:
            raise QgsProcessingException("No adjusted keylines were created")

        # Ensure the adjusted keylines GeoDataFrame has the correct CRS
        adjusted_keylines_gdf = adjusted_keylines_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Adjusted keylines CRS: {adjusted_keylines_gdf.crs}")

        # Save result
        try:
            adjusted_keylines_gdf.to_file(adjusted_keylines_path)
            feedback.pushInfo(f"Adjusted keylines saved to: {adjusted_keylines_path}")
        except Exception as e:
            raise QgsProcessingException(f"Failed to save adjusted keylines output: {e}")

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters: 
                results[outputName] = parameters[outputName]
                
        return results
