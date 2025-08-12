# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name: extract_ridges_algorithm.py
#
# Purpose: QGIS Processing Algorithm to create ridge lines based on WhiteboxTools
#
# -----------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QColor
from qgis.core import (QgsRasterLayer, QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFileDestination, QgsProcessingParameterNumber,
                       QgsVectorLayer, QgsProject, QgsProcessingParameterBoolean)
import os
import geopandas as gpd
from .utils import get_crs_from_layer, apply_line_arrow_symbology

class ExtractRidgesAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm for extracting ridge lines (inverted stream network) from a DEM resp. DTM.

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

    ADD_RIDGES_TO_MAP = 'ADD_RIDGES_TO_MAP'
    ADD_FILLED_INVERTED_TO_MAP = 'ADD_FILLED_INVERTED_TO_MAP'
    ADD_FDIR_INVERTED_TO_MAP = 'ADD_FDIR_INVERTED_TO_MAP'
    ADD_FACC_INVERTED_TO_MAP = 'ADD_FACC_INVERTED_TO_MAP'
    ADD_FACC_LOG_INVERTED_TO_MAP = 'ADD_FACC_LOG_INVERTED_TO_MAP'
    ADD_STREAMS_INVERTED_TO_MAP = 'ADD_STREAMS_INVERTED_TO_MAP'

    def __init__(self, core=None):
        super().__init__()
        self.core = core  # Should be set to a TopoDrainCore instance by the plugin

    def set_core(self, core):
        self.core = core
    
    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ExtractRidgesAlgorithm(core=self.core)

    def name(self):
        return 'extract_ridges'

    def displayName(self):
        return self.tr('Extract Ridges (inverted stream network)')

    def group(self):
        return self.tr('Basic Hydrological Analysis')

    def groupId(self):
        return 'basic_hydrological_analysis'

    def shortHelpString(self):
        return self.tr(
            """QGIS Processing Algorithm for extracting ridge lines (inverted stream network) from a DEM resp. DTM.
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

    def initAlgorithm(self, config=None):        
        # Input parameters
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DTM,
                self.tr('Input DTM (GeoTIFF)')
            )
        )
        # Algorithm parameters
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DIST_FACC,
                self.tr('Maximum search distance for breach paths in cells (for WBT algorithm `BreachDepressionsLeastCost`)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.ACCUM_THRESHOLD,
                self.tr('Accumulation Threshold (for ridge processing)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1000
            )
        )
        # Output parameters with "Add to Map" options
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_RIDGES,
                self.tr('Output Ridge Lines'),
                fileFilter='Shapefile (*.shp);;GeoPackage (*.gpkg)'
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ADD_RIDGES_TO_MAP,
                self.tr('Add Ridge Lines to QGIS map'),
                defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_FILLED_INVERTED,
                self.tr('Output Inverted Filled DTM (output of WBT `BreachDepressionsLeastCost` on inverted DTM)'),
                fileFilter='GeoTIFF (*.tif)',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ADD_FILLED_INVERTED_TO_MAP,
                self.tr('Add Inverted Filled DTM to QGIS map'),
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_FDIR_INVERTED,
                self.tr('Output Inverted Flow Direction Raster (output of WBT `D8Pointer` on inverted DTM)'),
                fileFilter='GeoTIFF (*.tif)',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ADD_FDIR_INVERTED_TO_MAP,
                self.tr('Add Inverted Flow Direction to QGIS map'),
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_FACC_INVERTED,
                self.tr('Output Inverted Flow Accumulation Raster (output of WBT `D8FlowAccumulation` on inverted DTM)'),
                fileFilter='GeoTIFF (*.tif)',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ADD_FACC_INVERTED_TO_MAP,
                self.tr('Add Inverted Flow Accumulation to QGIS map'),
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_FACC_LOG_INVERTED,
                self.tr('Output Inverted Log-Scaled Accumulation Raster'),
                fileFilter='GeoTIFF (*.tif)',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ADD_FACC_LOG_INVERTED_TO_MAP,
                self.tr('Add Inverted Log-Scaled Accumulation to QGIS map'),
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_STREAMS_INVERTED,
                self.tr('Output Inverted Stream Raster'),
                fileFilter='GeoTIFF (*.tif)',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ADD_STREAMS_INVERTED_TO_MAP,
                self.tr('Add Inverted Stream Raster to QGIS map'),
                defaultValue=False
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Validate and read input parameters
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        dtm_path = dtm_layer.source()
        if not dtm_path or not os.path.exists(dtm_path):
            raise FileNotFoundError(f"[Input Error] DTM file not found: {dtm_path}")
        ridge_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_RIDGES, context)
        filled_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FILLED_INVERTED, context)
        fdir_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FDIR_INVERTED, context)
        facc_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FACC_INVERTED, context)
        facc_log_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FACC_LOG_INVERTED, context)
        streams_output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_STREAMS_INVERTED, context)
        accumulation_threshold = self.parameterAsInt(parameters, self.ACCUM_THRESHOLD, context)
        dist_facc = self.parameterAsDouble(parameters, self.DIST_FACC, context)
        # Read boolean parameters for adding layers to the map
        add_ridges = self.parameterAsBool(parameters, self.ADD_RIDGES_TO_MAP, context)
        add_filled = self.parameterAsBool(parameters, self.ADD_FILLED_INVERTED_TO_MAP, context)
        add_fdir = self.parameterAsBool(parameters, self.ADD_FDIR_INVERTED_TO_MAP, context)
        add_facc = self.parameterAsBool(parameters, self.ADD_FACC_INVERTED_TO_MAP, context)
        add_facc_log = self.parameterAsBool(parameters, self.ADD_FACC_LOG_INVERTED_TO_MAP, context)
        add_streams = self.parameterAsBool(parameters, self.ADD_STREAMS_INVERTED_TO_MAP, context)

        feedback.pushInfo("Input:")
        feedback.pushInfo(f"DTM Input: {dtm_path}")
        feedback.pushInfo(f"Ridge Output: {ridge_output_path}")
        if filled_output_path:
            feedback.pushInfo(f"Inverted Filled DTM Output: {filled_output_path}")
        if fdir_output_path:
            feedback.pushInfo(f"Inverted Flow Direction Output: {fdir_output_path}")
        if facc_output_path:
            feedback.pushInfo(f"Inverted Flow Accumulation Output: {facc_output_path}")
        if facc_log_output_path:
            feedback.pushInfo(f"Inverted Log-Scaled Accumulation Output: {facc_log_output_path}") 
        if streams_output_path:
            feedback.pushInfo(f"Inverted Stream Raster Output: {streams_output_path}")
        feedback.pushInfo(f"Accumulation Threshold: {accumulation_threshold}")
        feedback.pushInfo(f"Max Search Distance for Breach Paths: {dist_facc}")

        feedback.pushInfo("Validating input DTM...")
        if not os.path.exists(dtm_path):
            raise FileNotFoundError(f"[Input Error] DTM file not found: {dtm_path}")
        if not dtm_path.lower().endswith(('.tif', '.tiff')):
            raise ValueError(f"[Input Error] DTM must be a GeoTIFF file: {dtm_path}")
        feedback.pushInfo("Validating output paths...")
        if not os.path.isdir(os.path.dirname(ridge_output_path)):
            raise FileNotFoundError(f"[Input Error] directory not found: {os.path.dirname(ridge_output_path)}")
        if filled_output_path and not os.path.isdir(os.path.dirname(filled_output_path)):
            raise FileNotFoundError(f"[Input Error] directory not found: {os.path.dirname(filled_output_path)}")
        if fdir_output_path and not os.path.isdir(os.path.dirname(fdir_output_path)):
            raise FileNotFoundError(f"[Input Error] directory not found: {os.path.dirname(fdir_output_path)}")
        if facc_output_path and not os.path.isdir(os.path.dirname(facc_output_path)):
            raise FileNotFoundError(f"[Input Error] directory not found: {os.path.dirname(facc_output_path)}")
        if facc_log_output_path and not os.path.isdir(os.path.dirname(facc_log_output_path)):
            raise FileNotFoundError(f"[Input Error] directory not found: {os.path.dirname(facc_log_output_path)}")
        if streams_output_path and not os.path.isdir(os.path.dirname(streams_output_path)):
            raise FileNotFoundError(f"[Input Error] directory not found: {os.path.dirname(streams_output_path)}")
        
        feedback.pushInfo("Reading CRS from DTM...")
        # Read CRS from the DTM using QGIS layer
        dtm_crs = get_crs_from_layer(dtm_layer)
        feedback.pushInfo(f"DTM Layer crs: {dtm_crs}")

        # Check if self.core.crs matches dtm_crs, warn and update if not
        if self.core and hasattr(self.core, "crs"):
            if self.core.crs != dtm_crs:
                feedback.reportError(f"Warning: TopoDrainCore CRS ({self.core.crs}) does not match DTM CRS ({dtm_crs}). Updating TopoDrainCore CRS to match DTM.")
                self.core.crs = dtm_crs
                
        feedback.pushInfo("Processing extract_ridges via TopoDrainCore...")
        if not self.core:
            from topo_drain.core.topo_drain_core import TopoDrainCore
            print("TopoDrainCore not set, creating default instance.")
            self.core = TopoDrainCore()  # fallback: create default instance (not recommended for plugin use)

        gdf_ridges = self.core.extract_ridges(
            dtm_path=dtm_path,
            inverted_filled_output_path=filled_output_path,
            inverted_fdir_output_path=fdir_output_path,
            inverted_facc_output_path=facc_output_path,
            inverted_facc_log_output_path=facc_log_output_path,
            inverted_streams_output_path=streams_output_path,
            accumulation_threshold=float(accumulation_threshold),
            dist_facc=dist_facc,
            feedback=feedback
        )

        feedback.pushInfo(f"Ridge lines CRS: {gdf_ridges.crs}")

        # Save result
        try:
            gdf_ridges.to_file(ridge_output_path)
            feedback.pushInfo(f"Ridge lines saved to: {ridge_output_path}")
        except Exception as e:
            raise RuntimeError(f"[ExtractRidgesAlgorithm] failed to save ridge output: {e}")

        # Optionally add each output to QGIS project
        # Add ridge lines
        if add_ridges:
            try:
                feedback.pushInfo("Add Ridge lines layer to QGIS Map...")  
                vlayer = QgsVectorLayer(ridge_output_path, "Ridge Lines", "ogr")
                if not vlayer.isValid():
                    feedback.reportError(f"Failed to load ridge lines layer: {ridge_output_path}")
                else:
                    # Apply red line symbology with flow direction arrows
                    apply_line_arrow_symbology(vlayer, '#CC3300', '#660000', linewidth=0.5, markersize=5, feedback=feedback)
                    QgsProject.instance().addMapLayer(vlayer)
                    feedback.pushInfo("Ridge lines layer added to QGIS project with red symbology and flow direction arrows.")
            except Exception as e:
                feedback.reportError(f"Could not add ridge lines to QGIS project: {e}")
        # Add inverted filled DTM
        if add_filled and filled_output_path and os.path.exists(filled_output_path):
            try:
                rlayer = QgsRasterLayer(filled_output_path, "Inverted Filled DTM")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    feedback.pushInfo("Inverted Filled DTM layer added to QGIS project.")
                else:
                    feedback.reportError(f"Failed to load inverted filled DTM: {filled_output_path}")
            except Exception as e:
                feedback.reportError(f"Could not add inverted filled DTM to QGIS project: {e}")
        # Add inverted flow direction
        if add_fdir and fdir_output_path and os.path.exists(fdir_output_path):
            try:
                rlayer = QgsRasterLayer(fdir_output_path, "Inverted Flow Direction")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    feedback.pushInfo("Inverted Flow Direction layer added to QGIS project.")
                else:
                    feedback.reportError(f"Failed to load inverted flow direction: {fdir_output_path}")
            except Exception as e:
                feedback.reportError(f"Could not add inverted flow direction to QGIS project: {e}")
        # Add inverted flow accumulation
        if add_facc and facc_output_path and os.path.exists(facc_output_path):
            try:
                rlayer = QgsRasterLayer(facc_output_path, "Inverted Flow Accumulation")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    feedback.pushInfo("Inverted Flow Accumulation layer added to QGIS project.")
                else:
                    feedback.reportError(f"Failed to load inverted flow accumulation: {facc_output_path}")
            except Exception as e:
                feedback.reportError(f"Could not add inverted flow accumulation to QGIS project: {e}")
        # Add inverted log-scaled accumulation
        if add_facc_log and facc_log_output_path and os.path.exists(facc_log_output_path):
            try:
                rlayer = QgsRasterLayer(facc_log_output_path, "Inverted Log-Scaled Accumulation")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    feedback.pushInfo("Inverted Log-Scaled Accumulation layer added to QGIS project.")
                else:
                    feedback.reportError(f"Failed to load inverted log-scaled accumulation: {facc_log_output_path}")
            except Exception as e:
                feedback.reportError(f"Could not add inverted log-scaled accumulation to QGIS project: {e}")
        # Add inverted streams
        if add_streams and streams_output_path and os.path.exists(streams_output_path):
            try:
                rlayer = QgsRasterLayer(streams_output_path, "Inverted Streams")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    feedback.pushInfo("Inverted Streams layer added to QGIS project.")
                else:
                    feedback.reportError(f"Failed to load inverted streams: {streams_output_path}")
            except Exception as e:
                feedback.reportError(f"Could not add inverted streams to QGIS project: {e}")

        results = {self.OUTPUT_RIDGES: ridge_output_path}
        if filled_output_path:
            results[self.OUTPUT_FILLED_INVERTED] = filled_output_path
        if fdir_output_path:
            results[self.OUTPUT_FDIR_INVERTED] = fdir_output_path
        if facc_output_path:
            results[self.OUTPUT_FACC_INVERTED] = facc_output_path
        if facc_log_output_path:
            results[self.OUTPUT_FACC_LOG_INVERTED] = facc_log_output_path
        if streams_output_path:
            results[self.OUTPUT_STREAMS_INVERTED] = streams_output_path

        return results
