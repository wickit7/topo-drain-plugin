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
from .utils import get_crs_from_layer, update_core_crs_if_needed, ensure_whiteboxtools_configured, save_gdf_to_file, load_gdf_from_file, load_gdf_from_qgis_source, get_raster_ext, get_vector_ext

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
    INPUT_PERIMETER = 'INPUT_PERIMETER'
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
        return self.tr('Slope Line Analysis')

    def groupId(self):
        return 'slope_line_analysis'

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
- Perimeter (optional): Polygon features defining area of interest. Acts as both barrier (boundary cannot be crossed) and is used to check if points are inside the perimeter area.
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
                self.CHANGE_AFTER,
                self.tr('Change Slope At Distance (0.5 = Original Slope from start to middle, then New Slope from middle to end)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                minValue=0.01,
                maxValue=0.99
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
        adjusted_lines_param = QgsProcessingParameterVectorDestination(
            self.OUTPUT_ADJUSTED_LINES,
            self.tr('Output Adjusted Constant Slope Lines'),
            type=QgsProcessing.TypeVectorLine,
            defaultValue=None
        )
        self.addParameter(adjusted_lines_param)

    def processAlgorithm(self, parameters, context, feedback):
        # Ensure WhiteboxTools is configured before running
        if not ensure_whiteboxtools_configured(self, feedback):
            return {}
        
        # Validate and read input parameters
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        input_lines_source = self.parameterAsSource(parameters, self.INPUT_LINES, context)
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
        
        adjusted_lines_output = self.parameterAsOutputLayer(parameters, self.OUTPUT_ADJUSTED_LINES, context)
        change_after = self.parameterAsDouble(parameters, self.CHANGE_AFTER, context)
        slope_after = self.parameterAsDouble(parameters, self.SLOPE_AFTER, context)
        slope_deviation_threshold = self.parameterAsDouble(parameters, self.SLOPE_DEVIATION_THRESHOLD, context)
        allow_barriers_as_temp_destination = self.parameterAsBoolean(parameters, self.ALLOW_BARRIERS_AS_TEMP_DESTINATION, context)
        max_iterations_slope = self.parameterAsInt(parameters, self.MAX_ITERATIONS_SLOPE, context)
        max_iterations_barrier = self.parameterAsInt(parameters, self.MAX_ITERATIONS_BARRIER, context)

        # Extract file paths
        adjusted_lines_path = adjusted_lines_output
        
        # Validate output vector format compatibility with OGR driver mapping
        output_ext = get_vector_ext(adjusted_lines_path, feedback, check_existence=False)
        if hasattr(self.core, 'ogr_driver_mapping') and output_ext not in self.core.ogr_driver_mapping:
            feedback.pushWarning(f"Output file format '{output_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to save it automatically.")

        feedback.pushInfo("Reading CRS from DTM...")
        # Read CRS from the DTM using QGIS layer
        dtm_crs = get_crs_from_layer(dtm_layer)
        feedback.pushInfo(f"DTM Layer crs: {dtm_crs}")

        # Update core CRS if needed (dtm_crs is guaranteed to be valid)
        update_core_crs_if_needed(self.core, dtm_crs, feedback)

        feedback.pushInfo("Processing constant slope line adjustment via TopoDrainCore...")
        
        # Convert QGIS layers to GeoDataFrames
        feedback.pushInfo("Converting input lines to GeoDataFrame...")
        input_lines_gdf = load_gdf_from_qgis_source(input_lines_source, feedback)
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

        # Convert destination layers to GeoDataFrames with Windows-safe CRS handling
        feedback.pushInfo("Converting destination features to GeoDataFrames...")
        destination_gdfs = []
        for layer in destination_layers:
            if layer:
                try:
                    # Load GeoDataFrame using utility function
                    gdf = load_gdf_from_file(layer.source(), feedback)
                    # Manually set the safe CRS
                    gdf.crs = self.crs
                    feedback.pushInfo(f"Successfully loaded {len(gdf)} destination features with safe CRS: {self.crs}")
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
                        gdf.crs = self.crs
                        feedback.pushInfo(f"Successfully loaded {len(gdf)} barrier features with safe CRS: {self.crs}")
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
                perimeter_gdf.crs = self.crs
                feedback.pushInfo(f"Successfully loaded {len(perimeter_gdf)} perimeter features with safe CRS: {self.crs}")
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

        feedback.pushInfo("Running constant slope line adjustment...")
        adjusted_lines_gdf = self.core.adjust_constant_slope_after(
            dtm_path=dtm_path,
            input_lines=input_lines_gdf,
            change_after=change_after,
            slope_after=slope_after,
            destination_features=destination_gdfs,
            perimeter=perimeter_gdf,
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

        # Add slope adjustment attributes to output
        adjusted_lines_gdf['change_after'] = change_after
        adjusted_lines_gdf['slope_after'] = slope_after
        feedback.pushInfo(f"Added attributes: change_after={change_after}, slope_after={slope_after}")

        # Save result with proper format handling
        save_gdf_to_file(adjusted_lines_gdf, adjusted_lines_path, self.core, feedback)

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters: 
                results[outputName] = parameters[outputName]
                
        return results
