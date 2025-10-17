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
from .utils import get_crs_from_layer, update_core_crs_if_needed, ensure_whiteboxtools_configured, save_gdf_to_file, load_gdf_from_file, load_gdf_from_qgis_source, get_raster_ext, get_vector_ext

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
    INPUT_PERIMETER = 'INPUT_PERIMETER'
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
- Perimeter (optional): Polygon features defining area of interest. Acts as both barrier (boundary cannot be crossed) and is used to check if points are inside the perimeter area.
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
            QgsProcessingParameterVectorLayer(
                self.INPUT_PERIMETER,
                self.tr('Perimeter (Area of Interest) - acts as barrier and limits processing to interior points'),
                types=[QgsProcessing.TypeVectorPolygon],
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
                minValue=0.01,
                maxValue=0.99,
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
                minValue=0.01,
                maxValue=1.0
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
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ITERATIONS_BARRIER,
                self.tr('Advanced: Max Iterations Barrier (maximum iterations when using barriers as temporary destinations, 1-100, default: 30)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=30,
                minValue=1,
                maxValue=500
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
        # Ensure WhiteboxTools is configured before running
        if not ensure_whiteboxtools_configured(self, feedback):
            return {}
        
        # Validate and read input parameters
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        start_points_source = self.parameterAsSource(parameters, self.INPUT_START_POINTS, context)
        destination_layers = self.parameterAsLayerList(parameters, self.INPUT_DESTINATION_FEATURES, context)
        barrier_layers = self.parameterAsLayerList(parameters, self.INPUT_BARRIER_FEATURES, context)
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
        
        # Validate vector formats for destination layers (warning only)
        for i, layer in enumerate(destination_layers):
            if layer and layer.source():
                dest_path = layer.source()
                dest_ext = get_vector_ext(dest_path, feedback)
                if hasattr(self.core, 'ogr_driver_mapping') and dest_ext not in self.core.ogr_driver_mapping:
                    feedback.pushWarning(f"Destination layer {i+1} format '{dest_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to load it automatically.")
        
        # Validate vector formats for barrier layers (warning only)
        for i, layer in enumerate(barrier_layers):
            if layer and layer.source():
                barrier_path = layer.source()
                barrier_ext = get_vector_ext(barrier_path, feedback)
                if hasattr(self.core, 'ogr_driver_mapping') and barrier_ext not in self.core.ogr_driver_mapping:
                    feedback.pushWarning(f"Barrier layer {i+1} format '{barrier_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to load it automatically.")
        
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
        slope_lines_path = slope_lines_output
        
        # Validate output vector format compatibility with OGR driver mapping
        output_ext = get_vector_ext(slope_lines_path, feedback, check_existence=False)
        if hasattr(self.core, 'ogr_driver_mapping') and output_ext not in self.core.ogr_driver_mapping:
            feedback.pushWarning(f"Output file format '{output_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to save it automatically.")

        feedback.pushInfo("Reading CRS from DTM...")
        # Read CRS from the DTM using QGIS layer
        dtm_crs = get_crs_from_layer(dtm_layer, fallback_crs="EPSG:2056")
        feedback.pushInfo(f"DTM Layer crs: {dtm_crs}")

        # Update core CRS if needed (dtm_crs is guaranteed to be valid)
        update_core_crs_if_needed(self.core, dtm_crs, feedback)
                    
        # Convert QGIS layers to GeoDataFrames
        feedback.pushInfo("Converting start points to GeoDataFrame...")
        start_points_gdf = load_gdf_from_qgis_source(start_points_source, feedback)
        if start_points_gdf.empty:
            raise QgsProcessingException("No start points found in input layer")
        
        feedback.pushInfo(f"Start points: {len(start_points_gdf)} features")

        # Convert destination layers to GeoDataFrames with Windows-safe CRS handling
        feedback.pushInfo("Converting destination features to GeoDataFrames...")
        destination_gdfs = []
        for layer in destination_layers:
            if layer:
                try:
                    # Load GeoDataFrame using utility function
                    gdf = load_gdf_from_file(layer.source(), feedback)
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
                        # Load GeoDataFrame using utility function
                        gdf = load_gdf_from_file(layer.source(), feedback)
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

        # Convert perimeter to GeoDataFrame (optional) with Windows-safe CRS handling
        perimeter_gdf = None
        if perimeter_layer and perimeter_layer.source():
            feedback.pushInfo("Converting perimeter to GeoDataFrame...")
            try:
                # Load GeoDataFrame using utility function
                perimeter_layer_path = perimeter_layer.source()
                perimeter_gdf = load_gdf_from_file(perimeter_layer_path, feedback)
                # Manually set the safe CRS
                perimeter_gdf.crs = dtm_crs
                feedback.pushInfo(f"Successfully loaded {len(perimeter_gdf)} perimeter features with safe CRS: {dtm_crs}")
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

        if change_after is not None and slope_after is not None:
            feedback.pushInfo(f"***** Phase 1/2 - Progress Reporting 1-100% - Creating first parts of line *****")

        feedback.pushInfo("Running constant slope lines tracing...")
        slope_lines_gdf = self.core.get_constant_slope_lines(
            dtm_path=dtm_path,
            start_points=start_points_gdf,
            destination_features=destination_gdfs,
            slope=slope,
            perimeter=perimeter_gdf,
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
            feedback.pushInfo("***** Phase 2/2 - Progress Reporting 1-100% - Creating second parts of line *****")
            feedback.pushInfo(f"Applying slope adjustment after {change_after} with new slope {slope_after}")
            
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

        # Save result with proper format handling
        save_gdf_to_file(slope_lines_gdf, slope_lines_path, self.core, feedback)

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters: 
                results[outputName] = parameters[outputName]
                
        return results
