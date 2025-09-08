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
                       QgsProcessingParameterNumber)
import geopandas as gpd
import os
from .utils import get_crs_from_source

pluginPath = os.path.dirname(__file__)

class GetKeypointsAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for detecting keypoints along valley lines based on curvature analysis of elevation profiles.

    This algorithm identifies keypoints (points of high convexity) along valley lines by analyzing
    the curvature of elevation profiles extracted from a DTM. The elevation profile is extracted
    along each valley line and smoothed using a Savitzky-Golay filter. The second derivative
    (curvature) is then computed, and locations with the strongest convex curvature are selected
    as keypoints.

    The algorithm:
    - Extracts elevation profiles along each valley line using the DTM
    - Applies Savitzky-Golay smoothing to reduce noise
    - Computes the second derivative (curvature) of the elevation profile
    - Identifies inflection points where curvature changes from concave to convex
    - Selects the top N keypoints per valley line based on curvature strength
    - Ensures minimum distance between selected keypoints

    This is useful for identifying significant morphological features along drainage channels,
    such as knickpoints, channel transitions, or locations suitable for water retention
    structures in keyline design applications.
    """

    # Constants used to refer to parameters and outputs
    INPUT_VALLEY_LINES = 'INPUT_VALLEY_LINES'
    INPUT_DTM = 'INPUT_DTM'
    SMOOTHING_WINDOW = 'SMOOTHING_WINDOW'
    POLYORDER = 'POLYORDER'
    MIN_DISTANCE = 'MIN_DISTANCE'
    MAX_KEYPOINTS = 'MAX_KEYPOINTS'
    OUTPUT_KEYPOINTS = 'OUTPUT_KEYPOINTS'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return GetKeypointsAlgorithm(core=self.core)

    def name(self):
        return 'get_keypoints'

    def displayName(self):
        return self.tr('Get Keypoints (along Main Valley Lines)')

    def group(self):
        return self.tr('Basic Watershed Analysis')

    def groupId(self):
        return 'basic_watershed_analysis'

    def shortHelpString(self):
        return self.tr(
            """Detect keypoints along valley lines based on curvature analysis of elevation profiles.
            
This algorithm identifies keypoints (points of high convexity) along valley lines by analyzing the curvature of elevation profiles extracted from a DTM. 

The algorithm:
- Extracts elevation profiles along each valley line using the DTM at pixel resolution
- Applies smoothing to reduce noise depending on the parameters "smoothing window" and "polyorder"
- Computes the second derivative (curvature) of the elevation profile
- Identifies inflection points where curvature changes from concave to convex
- Selects the top N keypoints ("Number of keypoints per valley line") per valley line based on curvature strength
- Ensures minimum distance between selected keypoints ("Minimum distance between keypoints (m)")

This is useful for identifying significant morphological features along drainage channels, such as knickpoints, channel transitions, or locations suitable for water retention structures in keyline design applications.

Input Requirements:
- Valley Lines: Should have 'FID' attribute (e.g., from Create Valleys algorithm)
- DTM: Digital Terrain Model for elevation profile extraction

OUTPUT_KEYPOINTS:
Point layer containing detected keypoints with attributes: valley_id, elev_index, rank, curvature

Simplified Parameters:
- Maximum keypoints: Limits output per valley line
- Minimum distance: Ensures spatial separation between keypoints
- Smoothing parameters: Control noise reduction in elevation profiles"""
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
                self.tr('Input Main Valley Lines (must have FID attribute)'),
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

        # Maximum keypoints per valley line
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_KEYPOINTS,
                self.tr('(Max.) Number of keypoints per valley line'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=2,
                minValue=1
           )
        )

        # Minimum distance between keypoints
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_DISTANCE,
                self.tr('Minimum distance between keypoints (m)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=1.0
            )
        )

        # Polynomial order
        self.addParameter(
            QgsProcessingParameterNumber(
                self.POLYORDER,
                self.tr('Advanced: Polynomial order for smoothing'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=2,
                minValue=1,
                maxValue=5
            )
        )
        
        # Smoothing window
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SMOOTHING_WINDOW,
                self.tr('Advanced: Smoothing window size (must be odd and larger than Polynomial order)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=9,
                minValue=3,
                maxValue=51
            )
        )

        # Output keypoints
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_KEYPOINTS,
                self.tr('Output Keypoints')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Validate and read input parameters
        valley_lines_source = self.parameterAsSource(parameters, self.INPUT_VALLEY_LINES, context)
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        smoothing_window = self.parameterAsInt(parameters, self.SMOOTHING_WINDOW, context)
        polyorder = self.parameterAsInt(parameters, self.POLYORDER, context)
        min_distance = self.parameterAsDouble(parameters, self.MIN_DISTANCE, context)
        max_keypoints = self.parameterAsInt(parameters, self.MAX_KEYPOINTS, context)

        # Validate smoothing parameters
        if smoothing_window <= polyorder:
            raise QgsProcessingException(f"Smoothing window ({smoothing_window}) must be larger than polynomial order ({polyorder})")
        
        if smoothing_window % 2 == 0:
            new_window = smoothing_window - 1
            feedback.reportError(f"Smoothing window ({smoothing_window}) must be odd. Adjusting to {new_window}.")
            smoothing_window = new_window

        # Get file paths
        dtm_path = dtm_layer.source()
        
        # Validate file existence
        if not dtm_path or not os.path.exists(dtm_path):
            raise QgsProcessingException(f"DTM file not found: {dtm_path}")

        # Get output file path using parameterAsOutputLayer
        keypoints_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_KEYPOINTS, context)
        keypoints_file_path = keypoints_output_layer if isinstance(keypoints_output_layer, str) else keypoints_output_layer

        feedback.pushInfo("Reading CRS from valley source...")
        # Read CRS from the valley lines source using unified function
        valley_crs = get_crs_from_source(valley_lines_source)
        feedback.pushInfo(f"Valley lines CRS: {valley_crs}")

        # Check if self.core.crs matches valley_crs, warn and update if not
        if valley_crs:
            if self.core and hasattr(self.core, "crs"):
                if self.core.crs != valley_crs and valley_crs != "":
                    feedback.reportError(f"Warning: TopoDrainCore CRS ({self.core.crs}) does not match valley lines CRS ({valley_crs}). Updating TopoDrainCore CRS to match valley lines.")
                    self.core.crs = valley_crs

        feedback.pushInfo("Processing get_keypoints via TopoDrainCore...")
        if not self.core:
            from .core.topo_drain_core import TopoDrainCore
            feedback.reportError("TopoDrainCore not set, creating default instance.")
            self.core = TopoDrainCore()  # fallback: create default instance (not recommended for plugin use)

        # Load input data as GeoDataFrame
        feedback.pushInfo("Loading valley lines...")
        valley_lines_gdf = gpd.GeoDataFrame.from_features(valley_lines_source.getFeatures())
        
        if valley_lines_gdf.empty:
            raise QgsProcessingException("No features found in valley lines input")

        # Check for FID attribute
        if 'FID' not in valley_lines_gdf.columns:
            raise QgsProcessingException("Valley lines must have an 'FID' attribute")

        # Run keypoint detection
        feedback.pushInfo("Running keypoint detection...")
        keypoints_gdf = self.core.get_keypoints(
            valley_lines=valley_lines_gdf,
            dtm_path=dtm_path,
            smoothing_window=smoothing_window,
            polyorder=polyorder,
            min_distance=min_distance,
            max_keypoints=max_keypoints,
            feedback=feedback
        )

        if keypoints_gdf.empty:
            raise QgsProcessingException("No keypoints were detected")

        feedback.pushInfo(f"Detected {len(keypoints_gdf)} keypoints")

        # Ensure the keypoints GeoDataFrame has the correct CRS
        keypoints_gdf = keypoints_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Keypoints CRS: {keypoints_gdf.crs}")

        # Save result
        try:
            keypoints_gdf.to_file(keypoints_file_path)
            feedback.pushInfo(f"Keypoints saved to: {keypoints_file_path}")
        except Exception as e:
            raise QgsProcessingException(f"Failed to save keypoints output: {e}")

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters:
                results[outputName] = parameters[outputName]

        return results
