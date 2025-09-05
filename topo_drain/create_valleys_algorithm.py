# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: create_valleys_algorithm.py
#
# Purpose: QGIS Processing Algorithm to create valley lines (river network) based on WhiteboxTools
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterVectorDestination, QgsProcessingParameterNumber,
                       QgsProcessing, QgsProcessingException)
import os
import geopandas as gpd
from .utils import get_crs_from_layer

pluginPath = os.path.dirname(__file__)

class CreateValleysAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for creating valley lines (stream network) from a DEM resp. DTM.

    This algorithm leverages several WhiteboxTools (WBT) processes:
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
    OUTPUT_VALLEYS = 'OUTPUT_VALLEYS'
    OUTPUT_FILLED = 'OUTPUT_FILLED'
    OUTPUT_FDIR = 'OUTPUT_FDIR'
    OUTPUT_FACC = 'OUTPUT_FACC'
    OUTPUT_FACC_LOG = 'OUTPUT_FACC_LOG'
    OUTPUT_STREAMS = 'OUTPUT_STREAMS'
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
        return CreateValleysAlgorithm(core=self.core)

    def name(self):
        return 'create_valleys'

    def displayName(self):
        return self.tr('Create Valleys (stream network)')

    def group(self):
        return self.tr('Basic Watershed Analysis')

    def groupId(self):
        return 'basic_watershed_analysis'

    def shortHelpString(self):
        return self.tr(
            """QGIS Processing Algorithm for creating valley lines (stream network) from a DEM resp. DTM
                This algorithm leverages several WhiteboxTools (WBT) processes:
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
                self.tr('Maximum search distance for breach paths in cells (see WBT `BreachDepressionsLeastCost`)'),
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
        valleys_param = QgsProcessingParameterVectorDestination(
            self.OUTPUT_VALLEYS,
            self.tr('Output Valley Lines'),
            type=QgsProcessing.TypeVectorLine,
            defaultValue=None
        )
        self.addParameter(valleys_param)
        
        filled_param = QgsProcessingParameterRasterDestination(
            self.OUTPUT_FILLED,
            self.tr('Output Filled DTM'),
            defaultValue=None,
            optional=True
        )
        self.addParameter(filled_param)
        
        fdir_param = QgsProcessingParameterRasterDestination(
            self.OUTPUT_FDIR,
            self.tr('Output Flow Direction Raster'),
            defaultValue=None,
            optional=True
        )
        self.addParameter(fdir_param)
        
        facc_param = QgsProcessingParameterRasterDestination(
            self.OUTPUT_FACC,
            self.tr('Output Flow Accumulation Raster'),
            defaultValue=None,
            optional=True
        )
        self.addParameter(facc_param)
        
        facc_log_param = QgsProcessingParameterRasterDestination(
            self.OUTPUT_FACC_LOG,
            self.tr('Output Log Accumulation Raster'),
            defaultValue=None,
            optional=True
        )
        self.addParameter(facc_log_param)
        
        streams_param = QgsProcessingParameterRasterDestination(
            self.OUTPUT_STREAMS,
            self.tr('Output Stream Raster'),
            defaultValue=None,
            optional=True
        )
        self.addParameter(streams_param)

    def processAlgorithm(self, parameters, context, feedback):
        # Validate and read input parameters
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)

        dtm_path = dtm_layer.source()
        if not dtm_path or not os.path.exists(dtm_path):
            raise QgsProcessingException(f"DTM file not found: {dtm_path}")
        
        # Use parameterAsOutputLayer to preserve checkbox state information
        valley_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_VALLEYS, context)
        filled_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_FILLED, context)
        fdir_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_FDIR, context)
        facc_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_FACC, context)
        facc_log_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_FACC_LOG, context)
        streams_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_STREAMS, context)
        
        accumulation_threshold = self.parameterAsInt(parameters, self.ACCUM_THRESHOLD, context)
        dist_facc = self.parameterAsDouble(parameters, self.DIST_FACC, context)

        # Extract actual file paths from layer objects for processing
        valley_file_path = valley_output_layer if isinstance(valley_output_layer, str) else valley_output_layer
        filled_file_path = filled_output_layer if isinstance(filled_output_layer, str) else filled_output_layer if filled_output_layer else None
        fdir_file_path = fdir_output_layer if isinstance(fdir_output_layer, str) else fdir_output_layer if fdir_output_layer else None
        facc_file_path = facc_output_layer if isinstance(facc_output_layer, str) else facc_output_layer if facc_output_layer else None
        facc_log_file_path = facc_log_output_layer if isinstance(facc_log_output_layer, str) else facc_log_output_layer if facc_log_output_layer else None
        streams_file_path = streams_output_layer if isinstance(streams_output_layer, str) else streams_output_layer if streams_output_layer else None

        feedback.pushInfo("Reading CRS from DTM...")
        # Read CRS from the DTM using QGIS layer
        dtm_crs = get_crs_from_layer(dtm_layer)
        feedback.pushInfo(f"DTM Layer crs: {dtm_crs}")

        # Check if self.core.crs matches dtm_crs, warn and update if not
        if dtm_crs:
            if self.core and hasattr(self.core, "crs"):
                if self.core.crs != dtm_crs:
                    feedback.reportError(f"Warning: TopoDrainCore CRS ({self.core.crs}) does not match DTM CRS ({dtm_crs}). Updating TopoDrainCore CRS to match DTM.")
                    self.core.crs = dtm_crs

        feedback.pushInfo("Processing extract_valleys via TopoDrainCore...")
        if not self.core:
            from topo_drain.core.topo_drain_core import TopoDrainCore
            feedback.reportError("TopoDrainCore not set, creating default instance.")
            self.core = TopoDrainCore()  # fallback: create default instance (not recommended for plugin use)

        feedback.pushInfo("Running extract valleys...")
        valleys_gdf = self.core.extract_valleys(
            dtm_path=dtm_path,
            filled_output_path=filled_file_path,
            fdir_output_path=fdir_file_path,
            facc_output_path=facc_file_path,
            facc_log_output_path=facc_log_file_path,
            streams_output_path=streams_file_path,
            accumulation_threshold=accumulation_threshold,
            dist_facc=dist_facc,
            feedback=feedback
        )

        if valleys_gdf.empty:
            raise QgsProcessingException("No valleys were created")

        # Ensure the valleys GeoDataFrame has the correct CRS
        valleys_gdf = valleys_gdf.set_crs(self.core.crs, allow_override=True)
        feedback.pushInfo(f"Valley lines CRS: {valleys_gdf.crs}")

        # Save result
        try:
            valleys_gdf.to_file(valley_file_path)
            feedback.pushInfo(f"Valley lines saved to: {valley_file_path}")
        except Exception as e:
            raise QgsProcessingException(f"Failed to save valley output: {e}")

        results = {}
        # Add ouput parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters: 
                results[outputName] = parameters[outputName]
                
        return results
