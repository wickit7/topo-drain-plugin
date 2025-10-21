# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: get_points_along_lines_algorithm.py
#
# Purpose: QGIS Processing Algorithm to distribute points along input lines
#          at specified intervals with optional reference point optimization
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterNumber)
import geopandas as gpd
import os
from .utils import get_crs_from_layer, update_core_crs_if_needed, ensure_whiteboxtools_configured, save_gdf_to_file, load_gdf_from_qgis_source, get_vector_ext

pluginPath = os.path.dirname(__file__)

class GetPointsAlongLinesAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for distributing points along input lines at specified intervals.

    This algorithm distributes points along input lines at regular intervals. If reference points
    are provided, the points will be placed optimally to maintain minimum distance from reference 
    points while maximizing the number of points that can be created.

    The algorithm:
    - Distributes points along lines at specified distance intervals
    - Handles both LineString and MultiLineString geometries
    - Preserves original line attributes in output points
    - Optionally optimizes placement considering reference points
    - Uses efficient bidirectional expansion from reference points when provided

    Without reference points (simple mode):
    - Places points evenly along lines at the specified distance intervals
    - Fast and straightforward placement

    With reference points (optimized mode):
    - Projects reference points onto lines to find their positions
    - Starts from the first reference point along each line
    - Expands bidirectionally placing points at distance intervals
    - Validates all placements to maintain minimum distance from reference points

    This is useful for creating sampling points, establishing measurement locations,
    or generating evenly spaced features along linear infrastructure while avoiding
    existing important points like keypoints or other critical locations.
    """

    # Constants used to refer to parameters and outputs
    INPUT_LINES = 'INPUT_LINES'
    INPUT_REFERENCE_POINTS = 'INPUT_REFERENCE_POINTS'
    DISTANCE_BETWEEN_POINTS = 'DISTANCE_BETWEEN_POINTS'
    OUTPUT_POINTS = 'OUTPUT_POINTS'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        instance = GetPointsAlongLinesAlgorithm(core=self.core)
        if hasattr(self, 'plugin'):
            instance.plugin = self.plugin
        return instance

    def name(self):
        return 'get_points_along_lines'

    def displayName(self):
        return self.tr('Get Points Along Lines')

    def group(self):
        return self.tr('Point Analysis')

    def groupId(self):
        return 'point_analysis'

    def shortHelpString(self):
        return self.tr(
            """Distribute points along input lines at specified intervals with optional reference point optimization.
            
This algorithm distributes points along input lines at regular intervals. If reference points are provided, the points will be placed optimally to maintain minimum distance from reference points while maximizing the number of points that can be created.

The algorithm:
- Distributes points along lines at specified distance intervals
- Handles both LineString and MultiLineString geometries  
- Preserves original line attributes in output points
- Optionally optimizes placement considering reference points
- Uses efficient bidirectional expansion from reference points when provided

**Simple Mode (no reference points):**
- Places points evenly along lines at the specified distance intervals
- Fast and straightforward placement

**Optimized Mode (with reference points):**
- Projects reference points onto lines to find their positions
- Starts from the first reference point along each line
- Expands bidirectionally placing points at distance intervals
- Validates all placements to maintain minimum distance from reference points

This is useful for creating sampling points, establishing measurement locations, or generating evenly spaced features along linear infrastructure while avoiding existing important points like keypoints or other critical locations.

**Input Requirements:**
- Input Lines: Any line layer (e.g., valley lines, ridge lines, contours)
- Reference Points (optional): Points to maintain distance from (e.g., keypoints)

**OUTPUT_POINTS:**
Point layer containing distributed points with attributes:
- All original line attributes preserved
- line_id: Identifier of the source line
- distance_along_line: Distance from line start to the point
- point_index: Sequential index of the point along the line
- optimized_placement: Boolean flag (only when reference points used)

**Parameters:**
- Distance between points: Target spacing between generated points (meters)
- Reference points: Optional points to maintain minimum distance from"""
        )

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'icons', 'topo_drain.svg'))

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm.
        """
        # Input lines
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_LINES,
                self.tr('Input Lines (e.g., valley lines, ridge lines, contours)'),
                [QgsProcessing.TypeVectorLine]
            )
        )

        # Input reference points (optional)
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_REFERENCE_POINTS,
                self.tr('Reference Points (optional, e.g., keypoints)'),
                [QgsProcessing.TypeVectorPoint],
                optional=True
            )
        )

        # Distance between points
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_BETWEEN_POINTS,
                self.tr('Distance between points (m)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=0.1
            )
        )

        # Output points
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_POINTS,
                self.tr('Output Points')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Ensure WhiteboxTools is configured before running (inherited pattern from other algorithms)
        if not ensure_whiteboxtools_configured(self, feedback):
            return {}
        
        # Validate and read input parameters
        lines_source = self.parameterAsSource(parameters, self.INPUT_LINES, context)
        reference_points_source = self.parameterAsSource(parameters, self.INPUT_REFERENCE_POINTS, context)
        distance_between_points = self.parameterAsDouble(parameters, self.DISTANCE_BETWEEN_POINTS, context)

        # Validate distance parameter
        if distance_between_points <= 0:
            raise QgsProcessingException("Distance between points must be positive")

        # Get output file path
        points_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_POINTS, context)
        points_file_path = points_output_layer
        
        # Validate output vector format compatibility with OGR driver mapping
        output_ext = get_vector_ext(points_file_path, feedback, check_existence=False)
        supported_vector_formats = list(self.core.ogr_driver_mapping.keys()) if hasattr(self.core, 'ogr_driver_mapping') else []
        if hasattr(self.core, 'ogr_driver_mapping') and output_ext not in self.core.ogr_driver_mapping:
            feedback.pushWarning(f"Output file format '{output_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to save it automatically.")

        feedback.pushInfo("Reading CRS from input lines...")
        # Read CRS from the lines layer
        lines_crs = get_crs_from_layer(lines_source, fallback_crs="EPSG:2056")
        feedback.pushInfo(f"Lines CRS: {lines_crs}")

        # Update core CRS if needed
        update_core_crs_if_needed(self.core, lines_crs, feedback)

        # Load input data as GeoDataFrames
        feedback.pushInfo("Loading input lines...")
        lines_gdf = load_gdf_from_qgis_source(lines_source, feedback)
        
        if lines_gdf.empty:
            raise QgsProcessingException("No features found in input lines")

        # Load reference points if provided
        reference_points_gdf = None
        if reference_points_source is not None:
            feedback.pushInfo("Loading reference points...")
            reference_points_gdf = load_gdf_from_qgis_source(reference_points_source, feedback)
            
            if reference_points_gdf.empty:
                feedback.pushInfo("No reference points found, proceeding with simple point distribution")
                reference_points_gdf = None
            else:
                feedback.pushInfo(f"Loaded {len(reference_points_gdf)} reference points for optimized placement")

        # Run point distribution
        feedback.pushInfo("Running point distribution along lines...")
        points_gdf = self.core.get_points_along_lines(
            input_lines=lines_gdf,
            reference_points=reference_points_gdf,
            distance_between_points=distance_between_points,
            feedback=feedback
        )

        if points_gdf.empty:
            raise QgsProcessingException("No points could be generated along the input lines")

        feedback.pushInfo(f"Generated {len(points_gdf)} points along {len(lines_gdf)} lines")

        # Ensure the points GeoDataFrame has the correct CRS
        points_gdf = points_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Points CRS: {points_gdf.crs}")

        # Save result with proper format handling
        save_gdf_to_file(points_gdf, points_file_path, self.core, feedback)

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters:
                results[outputName] = parameters[outputName]

        return results