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
                       QgsProcessingParameterBoolean, QgsProcessing, QgsProcessingException, 
                       QgsProcessingParameterFeatureSource)
import os
import geopandas as gpd
from .utils import get_crs_from_layer, update_core_crs_if_needed, clean_qvariant_data

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
    SLOPE_DEVIATION_THRESHOLD = 'SLOPE_DEVIATION_THRESHOLD'
    ALLOW_BARRIERS_AS_TEMP_DESTINATION = 'ALLOW_BARRIERS_AS_TEMP_DESTINATION'
    MAX_ITERATIONS_SLOPE = 'MAX_ITERATIONS_SLOPE'
    MAX_ITERATIONS_BARRIER = 'MAX_ITERATIONS_BARRIER'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        instance = AdjustConstantSlopeAfterAlgorithm(core=self.core)
        if hasattr(self, 'plugin'):
            instance.plugin = self.plugin
        return instance

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
- Change Slope At Distance: Creates two segments - Original Slope from start to this point, then New Slope to end (e.g., 0.5 = change at middle)
- New Slope After Change Point: New Slope for the second segment (e.g., 0.005 for 0.5% downhill)
- Destination Features: Features that the new slope sections should reach (e.g., ridge lines)
- Barrier Features (optional): Features to avoid during new slope tracing (e.g., valley lines)
- Slope Deviation Threshold: Maximum allowed slope deviation before triggering slope refinement iterations (0.0-1.0, e.g., 0.2 = 20%)
- Max Iterations Slope: Maximum iterations for slope refinement (1-50, default: 20)
- Max Iterations Barrier: Maximum iterations when using barriers as temporary destinations (1-50, default: 30)
"""
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
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ALLOW_BARRIERS_AS_TEMP_DESTINATION,
                self.tr('Allow Barriers as Temporary Destination (enables zig-zag tracing between barriers)'),
                defaultValue=False
            )
        )

        
        # Algorithm parameters
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CHANGE_AFTER,
                self.tr('Change Slope At Distance (0.5 = Original Slope from start to middle, then New Slope from middle to end)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                minValue=0.0,
                maxValue=1.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE_AFTER,
                self.tr('New Slope After Change Point (decimal, e.g., 0.005 for 0.5% downhill, -0.005 for 0.5% uphill)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.005,
                minValue=-1.0,
                maxValue=1.0
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
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ITERATIONS_BARRIER,
                self.tr('Advanced: Max Iterations Barrier (maximum iterations when using barriers as temporary destinations, 1-50, default: 30)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=30,
                minValue=1,
                maxValue=50
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
        slope_deviation_threshold = self.parameterAsDouble(parameters, self.SLOPE_DEVIATION_THRESHOLD, context)
        allow_barriers_as_temp_destination = self.parameterAsBoolean(parameters, self.ALLOW_BARRIERS_AS_TEMP_DESTINATION, context)
        max_iterations_slope = self.parameterAsInt(parameters, self.MAX_ITERATIONS_SLOPE, context)
        max_iterations_barrier = self.parameterAsInt(parameters, self.MAX_ITERATIONS_BARRIER, context)

        # Extract file paths
        adjusted_lines_path = adjusted_lines_output if isinstance(adjusted_lines_output, str) else adjusted_lines_output

        feedback.pushInfo("Reading CRS from DTM...")
        # Read CRS from the DTM using QGIS layer
        dtm_crs = get_crs_from_layer(dtm_layer, fallback_crs="EPSG:2056")
        feedback.pushInfo(f"DTM Layer crs: {dtm_crs}")

        # Ensure WhiteboxTools is configured before running
        if hasattr(self, 'plugin') and self.plugin:
            if not self.plugin.ensure_whiteboxtools_configured():
                raise QgsProcessingException("WhiteboxTools is not configured. Please install and configure the WhiteboxTools for QGIS plugin.")
        else:
            # Try to automatically find and connect to the TopoDrain plugin
            feedback.pushInfo("Plugin reference not available - attempting to connect to TopoDrain plugin")
            try:
                from qgis.utils import plugins
                if 'topo_drain' in plugins:
                    topo_drain_plugin = plugins['topo_drain']
                    # Set the plugin reference for this instance
                    self.plugin = topo_drain_plugin
                    feedback.pushInfo("Successfully connected to TopoDrain plugin")
                    
                    # Now try to configure WhiteboxTools
                    if hasattr(topo_drain_plugin, 'ensure_whiteboxtools_configured'):
                        if not topo_drain_plugin.ensure_whiteboxtools_configured():
                            feedback.pushWarning("WhiteboxTools is not configured. Please install and configure the WhiteboxTools for QGIS plugin.")
                        else:
                            feedback.pushInfo("WhiteboxTools configuration verified")
                    else:
                        feedback.pushWarning("TopoDrain plugin found but configuration method not available")
                else:
                    feedback.pushWarning("TopoDrain plugin not found in QGIS registry - cannot verify WhiteboxTools configuration")
            except Exception as e:
                feedback.pushWarning(f"Could not connect to TopoDrain plugin: {e} - continuing without WhiteboxTools verification")

        # Update core CRS if needed (dtm_crs is guaranteed to be valid)
        update_core_crs_if_needed(self.core, dtm_crs, feedback)

        feedback.pushInfo("Processing constant slope line adjustment via TopoDrainCore...")
        
        # Convert QGIS layers to GeoDataFrames
        feedback.pushInfo("Converting input lines to GeoDataFrame...")
        input_lines_gdf = gpd.GeoDataFrame.from_features(input_lines_source.getFeatures())
        if input_lines_gdf.empty:
            raise QgsProcessingException("No input lines found in input layer")

        # Clean QVariant objects from input lines data to avoid field type errors
        feedback.pushInfo("Cleaning input lines data types...")
        input_lines_gdf = clean_qvariant_data(input_lines_gdf)
        
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

        # Convert destination layers to GeoDataFrames with Windows-safe CRS handling
        feedback.pushInfo("Converting destination features to GeoDataFrames...")
        destination_gdfs = []
        for layer in destination_layers:
            if layer:
                try:
                    # Read without CRS first to avoid Windows PROJ crashes
                    gdf = gpd.read_file(layer.source(), crs=None)
                    # Manually set the safe CRS
                    gdf.crs = dtm_crs
                    feedback.pushInfo(f"Successfully loaded {len(gdf)} destination features with safe CRS: {dtm_crs}")
                except Exception as e:
                    feedback.pushInfo(f"Failed to load destination layer with safe CRS handling: {e}")
                    raise QgsProcessingException(f"Failed to load destination layer: {e}")
                    
                if not gdf.empty:
                    gdf = gdf.to_crs(self.core.crs)
                    destination_gdfs.append(gdf)
                    feedback.pushInfo(f"Destination layer: {len(gdf)} features")
        
        if not destination_gdfs:
            raise QgsProcessingException("No valid destination features found")

        # Convert barrier layers to GeoDataFrames (optional) with Windows-safe CRS handling
        barrier_gdfs = []
        if barrier_layers:
            feedback.pushInfo("Converting barrier features to GeoDataFrames...")
            for layer in barrier_layers:
                if layer:
                    try:
                        # Read without CRS first to avoid Windows PROJ crashes
                        gdf = gpd.read_file(layer.source(), crs=None)
                        # Manually set the safe CRS
                        gdf.crs = dtm_crs
                        feedback.pushInfo(f"Successfully loaded {len(gdf)} barrier features with safe CRS: {dtm_crs}")
                    except Exception as e:
                        feedback.pushInfo(f"Failed to load barrier layer with safe CRS handling: {e}")
                        raise QgsProcessingException(f"Failed to load barrier layer: {e}")
                        
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
            allow_barriers_as_temp_destination=allow_barriers_as_temp_destination,
            max_iterations_barrier=max_iterations_barrier,
            slope_deviation_threshold=slope_deviation_threshold,
            max_iterations_slope=max_iterations_slope,
            feedback=feedback
        )

        if adjusted_lines_gdf.empty:
            raise QgsProcessingException("No adjusted lines were created")

        # Ensure the adjusted lines GeoDataFrame has the correct CRS
        adjusted_lines_gdf = adjusted_lines_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Adjusted lines CRS: {adjusted_lines_gdf.crs}")

        # Clean any QVariant objects from the GeoDataFrame before saving
        feedback.pushInfo("Cleaning adjusted lines data types before saving...")
        adjusted_lines_gdf = clean_qvariant_data(adjusted_lines_gdf)

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
