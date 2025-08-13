# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: get_keypoints_algorithm.py
#
# Purpose: QGIS Processing Algorithm to detect keypoints along valley lines
#          based on curvature analysis of elevation profiles
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterBoolean)
from qgis import processing
import geopandas as gpd
import tempfile
import os
from .utils import get_crs_from_layer

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
    SAMPLING_DISTANCE = 'SAMPLING_DISTANCE'
    SMOOTHING_WINDOW = 'SMOOTHING_WINDOW'
    POLYORDER = 'POLYORDER'
    TOP_N = 'TOP_N'
    MIN_DISTANCE = 'MIN_DISTANCE'
    FIND_WINDOW_DISTANCE = 'FIND_WINDOW_DISTANCE'
    PLOT_DEBUG = 'PLOT_DEBUG'
    OUTPUT = 'OUTPUT'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return GetKeypointsAlgorithm(core=self.core)

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'get_keypoints'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Get Keypoints along Valley Lines')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr('Basic Hydrological Analysis')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'basic_hydrological_analysis'

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it.
        """
        return self.tr(
            """QGIS Processing Algorithm for detecting keypoints along valley lines based on curvature analysis of elevation profiles.
            
This algorithm identifies keypoints (points of high convexity) along valley lines by analyzing the curvature of elevation profiles extracted from a DTM. The elevation profile is extracted along each valley line and smoothed using a Savitzky-Golay filter. The second derivative (curvature) is then computed, and locations with the strongest convex curvature are selected as keypoints.

The algorithm:
- Extracts elevation profiles along each valley line using the DTM
- Applies Savitzky-Golay smoothing to reduce noise
- Computes the second derivative (curvature) of the elevation profile
- Identifies inflection points where curvature changes from concave to convex
- Selects the top N keypoints per valley line based on curvature strength
- Ensures minimum distance between selected keypoints

This is useful for identifying significant morphological features along drainage channels, such as knickpoints, channel transitions, or locations suitable for water retention structures in keyline design applications.

Input Requirements:
- Valley Lines: Should have 'FID' attribute (e.g., from Extract Valleys algorithm)
- DTM: Digital Terrain Model for elevation profile extraction

Output:
Point layer containing detected keypoints with attributes: valley_id, elev_index, rank, curvature"""
        )

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

        # Top N keypoints
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TOP_N,
                self.tr('(Maximum) Number of keypoints per valley line'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=2,
                minValue=1,
                maxValue=9999
            )
        )

        # Minimum distance
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_DISTANCE,
                self.tr('Minimum distance between keypoints (m) (if Number of keypoints > 1)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=1.0,
                maxValue=1000.0
            )
        )

        # Sampling distance
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SAMPLING_DISTANCE,
                self.tr('Sampling distance along lines (m) (samples dtm values every x meters)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=2.0,
                minValue=0.1,
                maxValue=100.0
            )
        )

        # Smoothing window
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SMOOTHING_WINDOW,
                self.tr('Smoothing window size (m) (smoothes elevation profile using Savitzky-Golay filter)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=9,
                minValue=3,
                maxValue=51
            )
        )

        # Polynomial order
        self.addParameter(
            QgsProcessingParameterNumber(
                self.POLYORDER,
                self.tr('Polynomial order for smoothing (for Savitzky-Golay filter)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=2,
                minValue=1,
                maxValue=5
            )
        )

        # Find window distance
        self.addParameter(
            QgsProcessingParameterNumber(
                self.FIND_WINDOW_DISTANCE,
                self.tr('Curvature analysis window distance (m)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=1.0,
                maxValue=100.0
            )
        )

        # Plot debug
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PLOT_DEBUG,
                self.tr('Show elevation profile plots for debugging'),
                defaultValue=False
            )
        )

        # Output keypoints
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT,
                self.tr('Output Keypoints')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        try:
            # Validate and read input parameters
            valley_lines_source = self.parameterAsSource(parameters, self.INPUT_VALLEY_LINES, context)
            dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
            sampling_distance = self.parameterAsDouble(parameters, self.SAMPLING_DISTANCE, context)
            smoothing_window = self.parameterAsInt(parameters, self.SMOOTHING_WINDOW, context)
            polyorder = self.parameterAsInt(parameters, self.POLYORDER, context)
            top_n = self.parameterAsInt(parameters, self.TOP_N, context)
            min_distance = self.parameterAsDouble(parameters, self.MIN_DISTANCE, context)
            find_window_distance = self.parameterAsDouble(parameters, self.FIND_WINDOW_DISTANCE, context)
            plot_debug = self.parameterAsBool(parameters, self.PLOT_DEBUG, context)

            if feedback is None:
                raise QgsProcessingException("Feedback object is None")

            # Get file paths
            dtm_path = dtm_layer.source()
            
            # Validate file existence
            if not dtm_path or not os.path.exists(dtm_path):
                raise FileNotFoundError(f"[Input Error] DTM file not found: {dtm_path}")

            # Validate smoothing window (must be odd)
            if smoothing_window % 2 == 0:
                smoothing_window += 1
                feedback.pushInfo(f"Smoothing window adjusted to {smoothing_window} (must be odd)")

            # Get output file path using parameterAsOutputLayer
            keypoints_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            keypoints_file_path = keypoints_output_layer if isinstance(keypoints_output_layer, str) else keypoints_output_layer

            feedback.pushInfo("Reading CRS from valley lines...")
            # Read CRS from the valley lines layer
            valley_crs = valley_lines_source.sourceCrs().authid()
            feedback.pushInfo(f"Valley lines CRS: {valley_crs}")

            # Check if self.core.crs matches valley_crs, warn and update if not
            if self.core and hasattr(self.core, "crs"):
                if self.core.crs != valley_crs:
                    feedback.reportError(f"Warning: TopoDrainCore CRS ({self.core.crs}) does not match valley lines CRS ({valley_crs}). Updating TopoDrainCore CRS to match valley lines.")
                    self.core.crs = valley_crs

            feedback.pushInfo("Processing get_keypoints via TopoDrainCore...")
            if not self.core:
                from .core.topo_drain_core import TopoDrainCore
                feedback.reportError("TopoDrainCore not set, creating default instance.")
                self.core = TopoDrainCore()  # fallback: create default instance (not recommended for plugin use)

            # Convert QGIS features to GeoDataFrame
            feedback.pushInfo("Converting main valley lines to GeoDataFrame...")
            valley_lines_gdf = gpd.GeoDataFrame.from_features(valley_lines_source.getFeatures())
            feedback.pushInfo(f"Loaded {len(valley_lines_gdf)} valley lines")
    
            # Check for required attributes
            required_attrs = ['FID']
            missing_attrs = [attr for attr in required_attrs if attr not in valley_lines_gdf.columns]
            if missing_attrs:
                raise ValueError(f"[Input Error] Main valley lines missing required attributes: {missing_attrs}. Please use output from Extract Main Valleys algorithm.")


            # Run keypoint detection
            feedback.pushInfo("Running keypoint detection...")
            keypoints_gdf = self.core.get_keypoints(
                valley_lines=valley_lines_gdf,
                dtm_path=dtm_path,
                sampling_distance=sampling_distance,
                smoothing_window=smoothing_window,
                polyorder=polyorder,
                top_n=top_n,
                min_distance=min_distance,
                find_window_distance=find_window_distance,
                plot_debug=plot_debug,
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
                raise RuntimeError(f"[GetKeypointsAlgorithm] failed to save keypoints output: {e}")

            return {self.OUTPUT: keypoints_output_layer}

        except Exception as e:
            feedback.reportError(f"Error in keypoint detection: {str(e)}", fatalError=True)
            raise QgsProcessingException(f"Keypoint detection failed: {str(e)}")
