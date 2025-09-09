# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: create_ridges_algorithm.py
#
# Purpose: QGIS Processing Algorithm to create ridge lines (inverted river network) based on WhiteboxTools
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterVectorDestination, QgsProcessingParameterNumber,
                       QgsProcessing, QgsProject, QgsProcessingException)
import os
import geopandas as gpd
from .utils import get_crs_from_layer

pluginPath = os.path.dirname(__file__)

class CreateRidgesAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for creating ridge lines (inverted stream network) from a DEM resp. DTM.

    After inverting the DEM, this algorithm leverages the same WhiteboxTools (WBT) processes as Extract Valleys:
    - BreachDepressionsLeastCost: Optimally breaches depressions in the DEM to prepare it for hydrological analysis, providing a lower-impact alternative to depression filling.
    - D8Pointer: Generates a flow direction raster using the D8 algorithm, assigning flow from each cell to its steepest downslope neighbor.
    - D8FlowAccumulation: Calculates flow accumulation (contributing area) using the FD8 algorithm, distributing flow among downslope neighbors.
    - ExtractStreams: Extracts stream networks from the flow accumulation raster based on a user-defined threshold, identifying significant flow paths.
    - RasterStreamsToVector: Converts rasterized stream networks into vector line features for further analysis or export.
    - StreamLinkIdentifier: Assigns unique identifiers to each stream segment (link) in the raster stream network.
    - VectorStreamNetworkAnalysis: Analyzes the vectorized stream network, calculating stream order (e.g., HORTON), tributary IDs (TRIB_ID), and additional attributes such as the downstream link ID (DS_LINK_ID) for each stream segment (FID).

    For more customization, you can use individual WhiteboxTools algorithms directly in the QGIS Processing Toolbox (WhiteBox Plugin) step by step.
    """

    INPUT_DTM = 'INPUT_DTM'
    OUTPUT_RIDGES = 'OUTPUT_RIDGES'
    OUTPUT_FILLED_INVERTED = 'OUTPUT_FILLED_INVERTED'
    OUTPUT_FDIR_INVERTED = 'OUTPUT_FDIR_INVERTED'
    OUTPUT_FACC_INVERTED = 'OUTPUT_FACC_INVERTED'
    OUTPUT_FACC_LOG_INVERTED = 'OUTPUT_FACC_LOG_INVERTED'
    OUTPUT_STREAMS_INVERTED = 'OUTPUT_STREAMS_INVERTED'
    ACCUM_THRESHOLD = 'ACCUM_THRESHOLD'
    DIST_FACC = 'DIST_FACC'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core
    
    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return CreateRidgesAlgorithm(core=self.core)

    def name(self):
        return 'create_ridges'

    def displayName(self):
        return self.tr('Create Ridges (inverted stream network)')

    def group(self):
        return self.tr('Basic Watershed Analysis')

    def groupId(self):
        return 'basic_watershed_analysis'

    def shortHelpString(self):
        return self.tr(
            """QGIS Processing Algorithm for creating ridge lines (inverted stream network) from a DEM resp. DTM.
                After inverting the DEM, this algorithm leverages the same WhiteboxTools (WBT) processes as Extract Valleys:
                - BreachDepressionsLeastCost: Optimally breaches depressions in the DEM to prepare it for hydrological analysis, providing a lower-impact alternative to depression filling.
                - D8Pointer: Generates a flow direction raster using the D8 algorithm, assigning flow from each cell to its steepest downslope neighbor.
                - D8FlowAccumulation: Calculates flow accumulation (contributing area) using the FD8 algorithm, distributing flow among downslope neighbors.
                - ExtractStreams: Extracts stream networks from the flow accumulation raster based on a user-defined threshold, identifying significant flow paths.
                - RasterStreamsToVector: Converts rasterized stream networks into vector line features for further analysis or export.
                - StreamLinkIdentifier: Assigns unique identifiers to each stream segment (link) in the raster stream network.
                - VectorStreamNetworkAnalysis: Analyzes the vectorized stream network, calculating stream order (e.g., HORTON), tributary IDs (TRIB_ID), and additional attributes such as the downstream link ID (DS_LINK_ID) for each stream segment (FID).
                For more customization, you can use individual WhiteboxTools algorithms directly in the QGIS Processing Toolbox (WhiteBox Plugin) step by step."""
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
        # Algorithm parameters
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DIST_FACC,
                self.tr('Advanced: Maximum search distance for breach paths in cells (see WBT `BreachDepressionsLeastCost`)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.ACCUM_THRESHOLD,
                self.tr('Accumulation Threshold (see WBT `ExtractStreams`)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1000
            )
        )
        # Output parameters
        ridges_param = QgsProcessingParameterVectorDestination(
            self.OUTPUT_RIDGES,
            self.tr('Output Ridge Lines'),
            type=QgsProcessing.TypeVectorLine,
            defaultValue=None
        )
        self.addParameter(ridges_param)
        
        filled_inverted_param = QgsProcessingParameterRasterDestination(
            self.OUTPUT_FILLED_INVERTED,
            self.tr('Output Inverted Filled DTM'),
            defaultValue=None,
            optional=True
        )
        self.addParameter(filled_inverted_param)
        
        fdir_inverted_param = QgsProcessingParameterRasterDestination(
            self.OUTPUT_FDIR_INVERTED,
            self.tr('Output Inverted Flow Direction Raster'),
            defaultValue=None,
            optional=True
        )
        self.addParameter(fdir_inverted_param)
        
        facc_inverted_param = QgsProcessingParameterRasterDestination(
            self.OUTPUT_FACC_INVERTED,
            self.tr('Output Inverted Flow Accumulation Raster'),
            defaultValue=None,
            optional=True
        )
        self.addParameter(facc_inverted_param)
        
        facc_log_inverted_param = QgsProcessingParameterRasterDestination(
            self.OUTPUT_FACC_LOG_INVERTED,
            self.tr('Output Inverted Log Accumulation Raster'),
            defaultValue=None,
            optional=True
        )
        self.addParameter(facc_log_inverted_param)
        
        streams_inverted_param = QgsProcessingParameterRasterDestination(
            self.OUTPUT_STREAMS_INVERTED,
            self.tr('Output Inverted Stream Raster'),
            defaultValue=None,
            optional=True
        )
        self.addParameter(streams_inverted_param)

    def processAlgorithm(self, parameters, context, feedback):
        # Validate and read input parameters
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
       
        dtm_path = dtm_layer.source()
        if not dtm_path or not os.path.exists(dtm_path):
            raise QgsProcessingException(f"DTM file not found: {dtm_path}")
        
        # Use parameterAsOutputLayer to preserve checkbox state information
        ridge_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_RIDGES, context)
        filled_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_FILLED_INVERTED, context)
        fdir_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_FDIR_INVERTED, context)
        facc_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_FACC_INVERTED, context)
        facc_log_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_FACC_LOG_INVERTED, context)
        streams_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_STREAMS_INVERTED, context)
        
        accumulation_threshold = self.parameterAsInt(parameters, self.ACCUM_THRESHOLD, context)
        dist_facc = self.parameterAsDouble(parameters, self.DIST_FACC, context)

        # Extract actual file paths from layer objects for processing
        ridge_file_path = ridge_output_layer if isinstance(ridge_output_layer, str) else ridge_output_layer
        filled_file_path = filled_output_layer if isinstance(filled_output_layer, str) else filled_output_layer if filled_output_layer else None
        fdir_file_path = fdir_output_layer if isinstance(fdir_output_layer, str) else fdir_output_layer if fdir_output_layer else None
        facc_file_path = facc_output_layer if isinstance(facc_output_layer, str) else facc_output_layer if facc_output_layer else None
        facc_log_file_path = facc_log_output_layer if isinstance(facc_log_output_layer, str) else facc_log_output_layer if facc_log_output_layer else None
        streams_file_path = streams_output_layer if isinstance(streams_output_layer, str) else streams_output_layer if streams_output_layer else None

        feedback.pushInfo("Reading CRS from DTM...")
        # Read CRS from the DTM using QGIS layer
        dtm_crs = get_crs_from_layer(dtm_layer)
        feedback.pushInfo(f"DTM Layer crs: {dtm_crs}")

        feedback.pushInfo("Processing extract_ridges via TopoDrainCore...")
        if not self.core:
            from topo_drain.core.topo_drain_core import TopoDrainCore
            feedback.reportError("TopoDrainCore not set, creating default instance.")
            self.core = TopoDrainCore()  # fallback: create default instance (not recommended for plugin use)

        # Check if self.core.crs matches dtm_crs, warn and update if not
        if dtm_crs:
            if self.core and hasattr(self.core, "crs"):
                if self.core.crs != dtm_crs:
                    feedback.reportError(f"Warning: TopoDrainCore CRS ({self.core.crs}) does not match DTM CRS ({dtm_crs}). Updating TopoDrainCore CRS to match DTM.")
                    self.core.crs = dtm_crs

        # Ensure WhiteboxTools is configured before running
        if hasattr(self, 'plugin') and self.plugin:
            if not self.plugin.ensure_whiteboxtools_configured():
                raise QgsProcessingException("WhiteboxTools is not configured. Please install and configure the WhiteboxTools for QGIS plugin.")
        else:
            feedback.pushInfo("Warning: Plugin reference not available - WhiteboxTools configuration cannot be checked")

        feedback.pushInfo("Running extract ridges...")
        ridge_gdf = self.core.extract_ridges(
            dtm_path=dtm_path,
            inverted_filled_output_path=filled_file_path,
            inverted_fdir_output_path=fdir_file_path,
            inverted_facc_output_path=facc_file_path,
            inverted_facc_log_output_path=facc_log_file_path,
            inverted_streams_output_path=streams_file_path,
            accumulation_threshold=accumulation_threshold,
            dist_facc=dist_facc,
            feedback=feedback
        )

        if ridge_gdf.empty:
            raise QgsProcessingException("No ridges were created")
        
        # Ensure the ridges GeoDataFrame has the correct CRS
        ridge_gdf = ridge_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Ridge lines CRS: {ridge_gdf.crs}")

        # Save result
        try:
            ridge_gdf.to_file(ridge_file_path)
            feedback.pushInfo(f"Ridge lines saved to: {ridge_file_path}")
        except Exception as e:
            raise QgsProcessingException(f"Failed to save ridge output: {e}")

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters:
                results[outputName] = parameters[outputName]

        return results
