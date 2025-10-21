# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: delineate_watersheds_algorithm.py
#
# Purpose: QGIS Processing Algorithm to delineate watersheds for given outlet points
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterNumber)
import geopandas as gpd
import os
from .utils import get_crs_from_layer, update_core_crs_if_needed, ensure_whiteboxtools_configured, save_gdf_to_file, load_gdf_from_qgis_source, load_gdf_from_file, get_raster_ext, get_vector_ext

pluginPath = os.path.dirname(__file__)

class DelineateWatershedsAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for delineating watersheds for given outlet points using WhiteboxTools.

    This algorithm delineates watersheds (drainage basins) for specified outlet points using
    flow direction data and optional stream snapping. The workflow includes:
    
    1. Preparation of outlet points as pour points
    2. Optional snapping of pour points to stream network (if streams provided and snap distance > 0)
    3. Watershed delineation using D8 flow direction algorithm
    4. Conversion of watershed raster to vector polygons with attributes
    
    The algorithm uses WhiteboxTools for hydrological processing and provides comprehensive
    watershed analysis capabilities including:
    - Automatic outlet point snapping to stream networks for improved accuracy
    - Cross-platform compatibility with reliable watershed identification
    - Area calculation and watershed numbering for easy analysis
    
    This is particularly useful for:
    - Hydrological modeling and analysis
    - Catchment delineation for water management
    - Environmental impact assessment
    - Agricultural watershed planning
    - Flood risk assessment and management
    """

    # Constants used to refer to parameters and outputs
    INPUT_OUTLET_POINTS = 'INPUT_OUTLET_POINTS'
    INPUT_FDIR = 'INPUT_FDIR'
    INPUT_STREAMS = 'INPUT_STREAMS'
    MAX_SNAP_DISTANCE = 'MAX_SNAP_DISTANCE'
    OUTPUT_WATERSHEDS = 'OUTPUT_WATERSHEDS'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        instance = DelineateWatershedsAlgorithm(core=self.core)
        if hasattr(self, 'plugin'):
            instance.plugin = self.plugin
        return instance

    def name(self):
        return 'delineate_watersheds'

    def displayName(self):
        return self.tr('Delineate Watersheds')

    def group(self):
        return self.tr('Basic Watershed Analysis')

    def groupId(self):
        return 'basic_watershed_analysis'

    def shortHelpString(self):
        return self.tr(
            """Delineate watersheds for given outlet points using WhiteboxTools.
            
This algorithm delineates watersheds (drainage basins) for specified outlet points using flow direction data and optional stream snapping.

The algorithm workflow:
1. Prepares outlet points as pour points for watershed delineation
2. Optionally snaps pour points to stream network (if streams provided and snap distance > 0)
3. Delineates watersheds using D8 flow direction algorithm from WhiteboxTools
4. Converts watershed raster to vector polygons with calculated area and watershed ID attributes

Key Features:
- Automatic outlet point snapping to stream networks for improved accuracy
- Cross-platform compatibility with reliable watershed identification
- Area calculation and watershed numbering for easy analysis
- Robust error handling and progress reporting

Input Requirements:
- Outlet Points: Point features where watersheds should be delineated (typically stream outlets or monitoring locations)
- Flow Direction: D8 flow direction raster (typically from Create Valleys algorithm)
- Streams (optional): Stream raster for outlet point snapping (from Create Valleys algorithm)
- Max Snap Distance: Maximum distance to snap outlet points to streams (0 = no snapping)

Output:
- Watershed polygons with WATERSHED_ID and AREA attributes

Use Cases:
- Hydrological modeling and catchment analysis
- Water management and planning
- Environmental impact assessment
- Agricultural watershed design
- Flood risk assessment"""
        )

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'icons', 'topo_drain.svg'))

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm.
        """
        # Input outlet points
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_OUTLET_POINTS,
                self.tr('Outlet Points (for watershed delineation)'),
                [QgsProcessing.TypeVectorPoint]
            )
        )

        # Input flow direction raster
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_FDIR,
                self.tr('Flow Direction Raster (D8 format)')
            )
        )

        # Input streams raster (optional)
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_STREAMS,
                self.tr('Streams Raster (for outlet point snapping)'),
                optional=True
            )
        )

        # Maximum snap distance
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_SNAP_DISTANCE,
                self.tr('Maximum Snap Distance (distance to snap outlet points to streams, 0 = no snapping)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0
            )
        )

        # Output watersheds
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_WATERSHEDS,
                self.tr('Output Watersheds'),
                type=QgsProcessing.TypeVectorPolygon
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Ensure WhiteboxTools is configured before running
        if not ensure_whiteboxtools_configured(self, feedback):
            return {}
        
        # Validate and read input parameters
        outlet_points_source = self.parameterAsSource(parameters, self.INPUT_OUTLET_POINTS, context)
        fdir_layer = self.parameterAsRasterLayer(parameters, self.INPUT_FDIR, context)
        streams_layer = self.parameterAsRasterLayer(parameters, self.INPUT_STREAMS, context)
        max_snap_distance = self.parameterAsDouble(parameters, self.MAX_SNAP_DISTANCE, context)

        # Get file paths and validate formats
        fdir_path = fdir_layer.source()
        fdir_ext = get_raster_ext(fdir_path, feedback)

        # Validate raster format compatibility with GDAL driver mapping
        supported_raster_formats = list(self.core.gdal_driver_mapping.keys())
        if hasattr(self.core, 'gdal_driver_mapping') and fdir_ext not in self.core.gdal_driver_mapping:
            raise QgsProcessingException(f"Flow direction raster format '{fdir_ext}' is not supported. Supported formats: {supported_raster_formats}")

        # Handle optional streams raster
        streams_path = None
        if streams_layer:
            streams_path = streams_layer.source()
            streams_ext = get_raster_ext(streams_path, feedback)
            if hasattr(self.core, 'gdal_driver_mapping') and streams_ext not in self.core.gdal_driver_mapping:
                raise QgsProcessingException(f"Streams raster format '{streams_ext}' is not supported. Supported formats: {supported_raster_formats}")

        # Validate snap distance logic
        if max_snap_distance > 0 and not streams_layer:
            raise QgsProcessingException("Streams raster is required when maximum snap distance > 0")

        # Get output file path using parameterAsOutputLayer
        watersheds_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_WATERSHEDS, context)
        watersheds_file_path = watersheds_output_layer
        
        # Validate output vector format compatibility with OGR driver mapping
        output_ext = get_vector_ext(watersheds_file_path, feedback, check_existence=False)
        supported_vector_formats = list(self.core.ogr_driver_mapping.keys()) if hasattr(self.core, 'ogr_driver_mapping') else []
        if hasattr(self.core, 'ogr_driver_mapping') and output_ext not in self.core.ogr_driver_mapping:
            feedback.pushWarning(f"Output file format '{output_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to save it automatically.")

        feedback.pushInfo("Reading CRS from flow direction raster...")
        # Read CRS from the flow direction raster with safe fallback
        fdir_crs = get_crs_from_layer(fdir_layer, fallback_crs="EPSG:2056")
        feedback.pushInfo(f"Flow Direction CRS: {fdir_crs}")

        # Update core CRS if needed (fdir_crs is guaranteed to be valid)
        update_core_crs_if_needed(self.core, fdir_crs, feedback)

        # Load input data as GeoDataFrame
        feedback.pushInfo("Loading outlet points...")
        outlet_points_gdf = load_gdf_from_qgis_source(outlet_points_source, feedback)
        
        if outlet_points_gdf.empty:
            raise QgsProcessingException("No features found in outlet points input")

        # Ensure outlet points are in the correct CRS
        # Check if CRS is set, if not set it to the same as flow direction raster
        if outlet_points_gdf.crs is None:
            feedback.pushInfo(f"Setting outlet points CRS to flow direction CRS: {self.core.crs}")
            outlet_points_gdf = outlet_points_gdf.set_crs(self.core.crs)
        elif outlet_points_gdf.crs != self.core.crs:
            feedback.pushInfo(f"Transforming outlet points from {outlet_points_gdf.crs} to {self.core.crs}")
            outlet_points_gdf = outlet_points_gdf.to_crs(self.core.crs)
        
        feedback.pushInfo(f"Outlet Points: {len(outlet_points_gdf)} features")

        # Report snapping configuration
        if max_snap_distance > 0 and streams_path:
            feedback.pushInfo(f"Outlet point snapping enabled: max distance = {max_snap_distance}")
        else:
            feedback.pushInfo("Outlet point snapping disabled (max_snap_distance = 0 or no streams provided)")

        # Run watershed delineation
        feedback.pushInfo("Running watershed delineation...")
        watersheds_gdf = self.core.delineate_watersheds(
            outlet_points=outlet_points_gdf,
            fdir_input_path=fdir_path,
            streams_input_path=streams_path,
            max_snap_distance=max_snap_distance,
            feedback=feedback
        )

        if watersheds_gdf.empty:
            raise QgsProcessingException("No watersheds were delineated")

        feedback.pushInfo(f"Delineated {len(watersheds_gdf)} watersheds")

        # Ensure the watersheds GeoDataFrame has the correct CRS
        watersheds_gdf = watersheds_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Watersheds CRS: {watersheds_gdf.crs}")

        # Save result with proper format handling
        save_gdf_to_file(watersheds_gdf, watersheds_file_path, self.core, feedback)

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters:
                results[outputName] = parameters[outputName]

        return results