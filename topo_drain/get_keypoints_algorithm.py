# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: get_keypoints_algorithm.py
#
# Purpose: QGIS Processing Algorithm to detect keypoints along valley lines
#          based on curvature analysis of elevation profiles
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterFileDestination,
                       QgsProcessingParameterNumber)
import geopandas as gpd
import os
from .utils import get_crs_from_layer, ensure_whiteboxtools_configured, save_gdf_to_file, load_gdf_from_qgis_source, get_raster_ext, get_vector_ext, get_crs_from_project, clear_pyproj_cache

pluginPath = os.path.dirname(__file__)

class GetKeypointsAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for detecting keypoints along valley lines based on curvature analysis of elevation profiles.

    This algorithm identifies keypoints (points of high convexity) along valley lines by analyzing
    the curvature of elevation profiles extracted from a DTM. The elevation profile is extracted
    along each valley line and fitted with a polynomial. The analytical second derivative
    (curvature) is then computed, and locations with the strongest convex curvature are selected
    as keypoints.

    The algorithm:
    - Extracts elevation profiles along each valley line using the DTM at pixel resolution
    - Fits a polynomial to the elevation data for smoothing
    - Computes the analytical second derivative (curvature) of the polynomial
    - Identifies inflection points using mathematical analysis (zero crossings, local extrema)
    - Selects all valid keypoints based on curvature strength and distance constraints
    - Ensures minimum distance between selected keypoints

    This is useful for identifying significant morphological features along drainage channels,
    such as knickpoints, channel transitions, or locations suitable for water retention
    structures in keyline design applications.
    """

    # Constants used to refer to parameters and outputs
    INPUT_VALLEY_LINES = 'INPUT_VALLEY_LINES'
    INPUT_DTM = 'INPUT_DTM'
    POLYNOMIAL_DEGREE = 'POLYNOMIAL_DEGREE'
    MIN_DISTANCE = 'MIN_DISTANCE'
    MIN_KEYPOINTS = 'MIN_KEYPOINTS'
    OUTPUT_KEYPOINTS = 'OUTPUT_KEYPOINTS'
    CSV_OUTPUT_PATH = 'CSV_OUTPUT_PATH'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        instance = GetKeypointsAlgorithm(core=self.core)
        if hasattr(self, 'plugin'):
            instance.plugin = self.plugin
        return instance

    def name(self):
        return 'get_keypoints'

    def displayName(self):
        return self.tr('Get Keypoints')

    def group(self):
        return self.tr('Point Analysis')

    def groupId(self):
        return 'point_analysis'

    def shortHelpString(self):
        return self.tr(
            """Detect keypoints along valley lines based on curvature analysis of elevation profiles.
            
This algorithm identifies keypoints (points of high convexity) along valley lines by analyzing the curvature of elevation profiles extracted from a DTM. 

The algorithm:
- Extracts elevation profiles along each valley line using the DTM at pixel resolution
- Fits a polynomial to the elevation data for smoothing and noise reduction
- Computes the analytical second derivative (curvature) of the polynomial
- Identifies inflection points using mathematical analysis (zero crossings, local extrema)
- Selects all valid keypoints based on curvature strength and distance constraints
- Ensures minimum distance between selected keypoints

This is useful for identifying significant morphological features along drainage channels, such as knickpoints, channel transitions, or locations suitable for water retention structures in keyline design applications.

Input Requirements:
- Valley Lines: Should have 'LINK_ID' attribute (from Create Valleys algorithm)
- DTM: Digital Terrain Model for elevation profile extraction

OUTPUTS:
- Keypoints: Point layer with attributes: VALLEY_ID, RANK, CURVATURE
- CSV Output (optional): Elevation profiles and curvature data

Parameters:
- Minimum keypoint candidates: Controls inflection detection sensitivity 
- Minimum distance: Ensures spatial separation between keypoints  
- Polynomial degree: Controls smoothing level (higher degree = more flexible fitting)

The algorithm uses sophisticated mathematical techniques:
- Exact zero crossings in curvature (most precise)
- Local extrema in curvature (morphological features)
- Fallback methods ensure keypoints are always found"""
        )

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'icons', 'topo_drain.svg'))

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm.
        """
        # Input valley lines
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_VALLEY_LINES,
                self.tr('Input Main Valley Lines'),
                [QgsProcessing.TypeVectorLine]
            )
        )

        # Input DTM
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DTM,
                self.tr('Input DTM (Digital Terrain Model)')
            )
        )

        # Minimum keypoint candidates
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_KEYPOINTS,
                self.tr('Minimum keypoint candidates to find'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1
           )
        )

        # Minimum distance between keypoints
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_DISTANCE,
                self.tr('Minimum distance between keypoints (m)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=20.0,
                minValue=1.0
            )
        )

        # Polynomial degree
        self.addParameter(
            QgsProcessingParameterNumber(
                self.POLYNOMIAL_DEGREE,
                self.tr('Polynomial degree for elevation profile fitting'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=1,
                maxValue=8
            )
        )

        # Output keypoints
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_KEYPOINTS,
                self.tr('Output Keypoints')
            )
        )

        # CSV output file for elevation profiles and curvature data
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.CSV_OUTPUT_PATH,
                self.tr('CSV Output File (Elevation Profiles and Curvature Data)'),
                fileFilter='CSV files (*.csv)',
                optional=True
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # CRITICAL: Clear PyProj cache at start to prevent Windows crashes on repeated runs
        clear_pyproj_cache(feedback)
        
        # Ensure WhiteboxTools is configured before running
        if not ensure_whiteboxtools_configured(self, feedback):
            return {}
        
        # Reset core CRS to None to prevent PyProj crashes on Windows
        self.core.reset_crs()
        
        # Validate and read input parameters
        valley_lines_source = self.parameterAsSource(parameters, self.INPUT_VALLEY_LINES, context)
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        polynomial_degree = self.parameterAsInt(parameters, self.POLYNOMIAL_DEGREE, context)
        min_distance = self.parameterAsDouble(parameters, self.MIN_DISTANCE, context)
        min_keypoints = self.parameterAsInt(parameters, self.MIN_KEYPOINTS, context)
        
        # Only get CSV output path if user actually provided one
        csv_output_path = None
        if self.CSV_OUTPUT_PATH in parameters and parameters[self.CSV_OUTPUT_PATH]:
            param_value = parameters[self.CSV_OUTPUT_PATH]
            # Check if it's not a temporary/automatic path generated by QGIS
            if not (str(param_value).endswith('CSV_OUTPUT_PATH.csv') or str(param_value) == 'TEMPORARY_OUTPUT'):
                csv_output_path = self.parameterAsFileOutput(parameters, self.CSV_OUTPUT_PATH, context)
                # Ensure it's not empty or just whitespace
                if not csv_output_path or not csv_output_path.strip():
                    csv_output_path = None

        # Get DTM path and validate format
        dtm_path = dtm_layer.source()
        dtm_ext = get_raster_ext(dtm_path, feedback)

        # Validate raster format compatibility with GDAL driver mapping
        supported_raster_formats = list(self.core.gdal_driver_mapping.keys())
        if hasattr(self.core, 'gdal_driver_mapping') and dtm_ext not in self.core.gdal_driver_mapping:
            raise QgsProcessingException(f"DTM raster format '{dtm_ext}' is not supported. Supported formats: {supported_raster_formats}")

        # Get output file path using parameterAsOutputLayer
        keypoints_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_KEYPOINTS, context)
        keypoints_file_path = keypoints_output_layer
        
        # Validate output vector format compatibility with OGR driver mapping
        output_ext = get_vector_ext(keypoints_file_path, feedback, check_existence=False)
        supported_vector_formats = list(self.core.ogr_driver_mapping.keys()) if hasattr(self.core, 'ogr_driver_mapping') else []
        if hasattr(self.core, 'ogr_driver_mapping') and output_ext not in self.core.ogr_driver_mapping:
            feedback.pushWarning(f"Output file format '{output_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to save it automatically.")

        # Adjust core crs with project crs if needed
        feedback.pushInfo(f"Core CRS: {self.core.crs}")
        project_crs = get_crs_from_project()
        feedback.pushInfo(f"Project CRS: {project_crs}")
        if self.core.crs is None and project_crs is None:
            feedback.pushWarning("Both core CRS and project CRS are None - CRS may not be properly set")
        elif project_crs != self.core.crs:
            if project_crs is None:
                feedback.pushWarning("Project CRS is None - keeping core CRS") 
            else:
                feedback.pushInfo(f"Setting core CRS from project CRS: {project_crs}")
                self.core.set_crs(project_crs)

        # Check input crs against core crs
        feedback.pushInfo("Reading CRS from DTM layer...")
        dtm_crs = get_crs_from_layer(dtm_layer)
        feedback.pushInfo(f"DTM CRS: {dtm_crs}")
        # Adjust core crs with input crs but only if it is None
        if self.core.crs is None:
            feedback.pushInfo(f"Setting core CRS from DTM CRS: {dtm_crs}")
            self.core.set_crs(dtm_crs)
        elif dtm_crs != self.core.crs:
            # Add warning if input crs not equal to core crs
            feedback.pushWarning(f"DTM CRS {dtm_crs} differs from core (project) CRS {self.core.crs}!")

        # Load input data as GeoDataFrame
        feedback.pushInfo("Loading valley lines...")
        valley_lines_gdf = load_gdf_from_qgis_source(valley_lines_source, feedback)
        
        if valley_lines_gdf.empty:
            raise QgsProcessingException("No features found in valley lines input")

        # Check for LINK_ID attribute (create if missing)
        if 'LINK_ID' not in valley_lines_gdf.columns:
            feedback.pushWarning("Valley lines do not have a 'LINK_ID' attribute. Creating LINK_ID field with unique values.")
            valley_lines_gdf['LINK_ID'] = range(1, len(valley_lines_gdf) + 1)
            feedback.pushInfo(f"Created LINK_ID field with values from 1 to {len(valley_lines_gdf)}")

        # Run keypoint detection
        feedback.pushInfo("Running keypoint detection...")
        keypoints_gdf = self.core.get_keypoints(
            valley_lines=valley_lines_gdf,
            dtm_path=dtm_path,
            polynomial_degree=polynomial_degree,
            min_distance=min_distance,
            min_keypoints=min_keypoints,
            csv_output_path=csv_output_path,
            feedback=feedback
        )

        if keypoints_gdf.empty:
            raise QgsProcessingException("No keypoints were detected")

        feedback.pushInfo(f"Detected {len(keypoints_gdf)} keypoints")

        # Note: CRS is already set by core.get_keypoints() - no need to call .set_crs() here
        # Calling .set_crs() triggers PyProj CRS object creation which causes crashes on Windows
        feedback.pushInfo(f"Keypoints CRS: {keypoints_gdf.crs}")

        # Save result with proper format handling
        save_gdf_to_file(keypoints_gdf, keypoints_file_path, self.core, feedback)

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters:
                # Only include CSV output in results if it was actually created (user provided a real path)
                if outputName == self.CSV_OUTPUT_PATH and csv_output_path is None:
                    continue  # Skip CSV output if no real path was provided
                results[outputName] = parameters[outputName]

        return results
