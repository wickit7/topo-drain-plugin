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
                       QgsProcessing, QgsProcessingException, QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterBoolean)
import os
import geopandas as gpd
from .utils import get_crs_from_layer, update_core_crs_if_needed

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
    ALLOW_BARRIERS_AS_TEMP_DESTINATION = 'ALLOW_BARRIERS_AS_TEMP_DESTINATION'
    CHANGE_AFTER = 'CHANGE_AFTER'
    SLOPE_AFTER = 'SLOPE_AFTER'
    SLOPE_DEVIATION_THRESHOLD = 'SLOPE_DEVIATION_THRESHOLD'
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
        instance = CreateConstantSlopeLinesAlgorithm(core=self.core)
        if hasattr(self, 'plugin'):
            instance.plugin = self.plugin
        return instance

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
5. Optionally adjusts slope after a specified distance along each line
6. Returns the traced constant slope lines as vector features

This is useful for creating drainage lines, roads and paths, or other linear features that need
to maintain a specific gradient across the terrain.

Parameters:
- Input DTM: Digital Terrain Model for slope calculations
- Start Points: Point features where slope lines should begin (e.g., keypoints)
- Destination Features: Line or polygon features that slope lines should reach (e.g. main ridge lines, area of interest)
- Barrier Features (optional): Line or polygon features to avoid during tracing (e.g. main valley lines)
- Slope: Desired slope as a decimal (e.g., 0.01 for 1% downhill, -0.01 for 1% uphill)
- Change Slope At Distance (optional): Creates two segments - Desired Slope from start to this point, then New Slope to end (e.g., 0.5 = change at middle)
- New Slope After Change Point (optional): New Slope to apply for the second segment (required if Change Slope At Distance is set)
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
                self.SLOPE,
                self.tr('Desired Slope (decimal, e.g., 0.01 for 1% downhill, -0.01 for 1% uphill)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.01,
                minValue=-1.0,
                maxValue=1.0
            )
        )
        

        # Slope adjustment parameters
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
                self.tr('New Slope After Change Point (decimal, e.g., 0.005 for 0.5%)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=None,
                minValue=-1.0,
                maxValue=1.0,
                optional=True
            )
        )
        
        # Advanced parameters
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE_DEVIATION_THRESHOLD,
                self.tr('Advanced: Slope Deviation Threshold (max allowed deviation before slope refinement, 0.0-1.0, default: 0.2 = 20%)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.2,
                minValue=0.0,
                maxValue=1.0
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
        allow_barriers_as_temp_destination = self.parameterAsBoolean(parameters, self.ALLOW_BARRIERS_AS_TEMP_DESTINATION, context)
        
        # Read slope adjustment parameters
        change_after = self.parameterAsDouble(parameters, self.CHANGE_AFTER, context) if self.CHANGE_AFTER in parameters and parameters[self.CHANGE_AFTER] is not None else None
        slope_after = self.parameterAsDouble(parameters, self.SLOPE_AFTER, context) if self.SLOPE_AFTER in parameters and parameters[self.SLOPE_AFTER] is not None else None
        slope_deviation_threshold = self.parameterAsDouble(parameters, self.SLOPE_DEVIATION_THRESHOLD, context)
        max_iterations_slope = self.parameterAsInt(parameters, self.MAX_ITERATIONS_SLOPE, context)
        max_iterations_barrier = self.parameterAsInt(parameters, self.MAX_ITERATIONS_BARRIER, context)

        # Validate slope adjustment parameters
        if change_after is not None and slope_after is None:
            raise QgsProcessingException("Slope After Change Point is required when Change Slope After is specified")
        if slope_after is not None and change_after is None:
            raise QgsProcessingException("Change Slope After is required when Slope After Change Point is specified")
        
        # Provide feedback about slope adjustment
        if change_after is not None and slope_after is not None:
            feedback.pushInfo(f"Slope adjustment will be applied after {change_after*100:.1f}% of line length with new slope {slope_after}")
        else:
            feedback.pushInfo("No slope adjustment will be applied")

        # Extract file paths
        slope_lines_path = slope_lines_output if isinstance(slope_lines_output, str) else slope_lines_output

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
                    
        # Convert QGIS layers to GeoDataFrames
        feedback.pushInfo("Converting start points to GeoDataFrame...")
        start_points_gdf = gpd.GeoDataFrame.from_features(start_points_source.getFeatures())
        if start_points_gdf.empty:
            raise QgsProcessingException("No start points found in input layer")
        
        feedback.pushInfo(f"Start points: {len(start_points_gdf)} features")

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

        feedback.pushInfo("Running constant slope lines tracing...")
        slope_lines_gdf = self.core.get_constant_slope_lines(
            dtm_path=dtm_path,
            start_points=start_points_gdf,
            destination_features=destination_gdfs,
            slope=slope,
            barrier_features=barrier_gdfs if barrier_gdfs else None,
            allow_barriers_as_temp_destination=allow_barriers_as_temp_destination,
            max_iterations_barrier=max_iterations_barrier,
            slope_deviation_threshold=slope_deviation_threshold,
            max_iterations_slope=max_iterations_slope,
            feedback=feedback
        )

        if slope_lines_gdf.empty:
            raise QgsProcessingException("No slope lines were created")

        # Apply slope adjustment if parameters are provided
        if change_after is not None and slope_after is not None:
            feedback.pushInfo(f"Applying slope adjustment after {change_after*100:.1f}% with new slope {slope_after}")
            
            # Apply the slope adjustment using the adjust_constant_slope_after method
            slope_lines_gdf = self.core.adjust_constant_slope_after(
                dtm_path=dtm_path,
                input_lines=slope_lines_gdf,
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
            
            feedback.pushInfo(f"Slope adjustment complete, {len(slope_lines_gdf)} adjusted lines")
            
            if slope_lines_gdf.empty:
                raise QgsProcessingException("No lines remained after slope adjustment")

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
