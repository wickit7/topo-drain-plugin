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
                       QgsProcessing, QgsProcessingException)
import os
from .utils import get_crs_from_layer, ensure_whiteboxtools_configured, save_gdf_to_file, save_gdf_to_file_ogr, get_raster_ext, get_vector_ext, get_crs_from_project, clear_pyproj_cache

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
    - VectorStreamNetworkAnalysis: Analyzes the vectorized stream network, calculating stream order (e.g., HORTON), tributary IDs (TRIB_ID), and additional attributes such as the downstream link ID (DS_LINK_ID) for each stream segment (LINK_ID).

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
        instance = CreateRidgesAlgorithm(core=self.core)
        if hasattr(self, 'plugin'):
            instance.plugin = self.plugin
        return instance

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
                - VectorStreamNetworkAnalysis: Analyzes the vectorized stream network, calculating stream order (e.g., HORTON), tributary IDs (TRIB_ID), and additional attributes such as the downstream link ID (DS_LINK_ID) for each stream segment (LINK_ID).
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
                defaultValue=1000,
                minValue=1
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
        # CRITICAL: Clear PyProj cache at start to prevent Windows crashes on repeated runs
        #clear_pyproj_cache(feedback) # seems not to resolve the issue
        
        # Ensure WhiteboxTools is configured before running
        if not ensure_whiteboxtools_configured(self, feedback):
            return {}

        # Validate and read input parameters
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
       
        # Get DTM path and validate format
        dtm_path = dtm_layer.source()
        dtm_ext = get_raster_ext(dtm_path, feedback)
        
        # Validate raster format compatibility with GDAL driver mapping
        supported_raster_formats = list(self.core.gdal_driver_mapping.keys())
        if hasattr(self.core, 'gdal_driver_mapping') and dtm_ext not in self.core.gdal_driver_mapping:
            raise QgsProcessingException(f"DTM raster format '{dtm_ext}' is not supported. Supported formats: {supported_raster_formats}")
        
        # Use parameterAsOutputLayer to preserve checkbox state information
        ridge_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_RIDGES, context)
        filled_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_FILLED_INVERTED, context)
        fdir_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_FDIR_INVERTED, context)
        facc_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_FACC_INVERTED, context)
        facc_log_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_FACC_LOG_INVERTED, context)
        streams_output_layer = self.parameterAsOutputLayer(parameters, self.OUTPUT_STREAMS_INVERTED, context)
        
        # Read numeric parameters
        accumulation_threshold = self.parameterAsInt(parameters, self.ACCUM_THRESHOLD, context)
        dist_facc = self.parameterAsDouble(parameters, self.DIST_FACC, context)

        # Extract actual file paths from layer objects for processing
        ridge_file_path = ridge_output_layer
        
        # Validate output vector format compatibility with OGR driver mapping
        output_ext = get_vector_ext(ridge_file_path, feedback, check_existence=False)
        supported_vector_formats = list(self.core.ogr_driver_mapping.keys()) if hasattr(self.core, 'ogr_driver_mapping') else []
        if hasattr(self.core, 'ogr_driver_mapping') and output_ext not in self.core.ogr_driver_mapping:
            feedback.pushWarning(f"Output file format '{output_ext}' is not in OGR driver mapping. Supported formats: {supported_vector_formats}. GeoPandas will attempt to save it automatically.")
        
        filled_file_path = filled_output_layer if filled_output_layer else None
        fdir_file_path = fdir_output_layer if fdir_output_layer else None
        facc_file_path = facc_output_layer if facc_output_layer else None
        facc_log_file_path = facc_log_output_layer if facc_log_output_layer else None
        streams_file_path = streams_output_layer if streams_output_layer else None

        # Validate output raster format compatibility with GDAL driver mapping for optional outputs
        supported_raster_formats = list(self.core.gdal_driver_mapping.keys()) if hasattr(self.core, 'gdal_driver_mapping') else []
        for output_name, output_path in [
            ("filled inverted DTM", filled_file_path),
            ("inverted flow direction", fdir_file_path), 
            ("inverted flow accumulation", facc_file_path),
            ("log inverted flow accumulation", facc_log_file_path),
            ("inverted streams", streams_file_path)
        ]:
            if output_path:  # Only validate if output is specified
                output_raster_ext = get_raster_ext(output_path, feedback, check_existence=False)
                if hasattr(self.core, 'gdal_driver_mapping') and output_raster_ext not in self.core.gdal_driver_mapping:
                    feedback.pushWarning(f"Output {output_name} format '{output_raster_ext}' is not in GDAL driver mapping. Supported formats: {supported_raster_formats}. GDAL will attempt to save it automatically.")

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
        feedback.pushInfo("Reading CRS from DTM...")
        dtm_crs = get_crs_from_layer(dtm_layer)
        feedback.pushInfo(f"DTM Layer crs: {dtm_crs}")
        # Adjust core crs with input crs but only if it is None
        if self.core.crs is None:
            feedback.pushInfo(f"Setting core CRS from DTM CRS: {dtm_crs}")
            self.core.set_crs(dtm_crs)
        elif dtm_crs != self.core.crs:
            # Add warning if input crs not equal to core crs
            feedback.pushWarning(f"DTM CRS {dtm_crs} differs from core (project) CRS {self.core.crs}!")

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
        
        feedback.pushInfo(f"Ridge lines CRS: {ridge_gdf.crs}")

        # Save result - use OGR on Windows to avoid PyProj crashes
        if self.core.disable_crs_operations:
            feedback.pushInfo("Saving ridges WITHOUT setting CRS to avoid WINDOWS PyProj issues...")   
            save_gdf_to_file_ogr(ridge_gdf, ridge_file_path, self.core, feedback)
        else:
            feedback.pushInfo("Saving ridges WITH setting CRS pyproj (geopandas)...")
            save_gdf_to_file(ridge_gdf, ridge_file_path, self.core, feedback)

        results = {}
        # Add output parameters to results
        for output in self.outputDefinitions():
            outputName = output.name()
            if outputName in parameters:
                results[outputName] = parameters[outputName]

        return results
