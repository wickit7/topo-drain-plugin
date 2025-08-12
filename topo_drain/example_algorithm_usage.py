# Example of how to use the utils functions in other algorithms

from .utils import (
    get_crs_from_layer,
    get_crs_object_from_layer,
    get_crs_from_project,
    convert_qgis_crs_to_geopandas_format,
    get_nodata_value_from_layer
)

class YourAlgorithm(QgsProcessingAlgorithm):
    def processAlgorithm(self, parameters, context, feedback):
        # Get input layer
        input_layer = self.parameterAsRasterLayer(parameters, 'INPUT', context)
        
        # Method 1: Get CRS as string (EPSG code)
        crs_string = get_crs_from_layer(input_layer)
        feedback.pushInfo(f"Layer CRS: {crs_string}")  # e.g., "EPSG:4326"
        
        # Method 2: Get CRS as QGIS object for more operations
        crs_object = get_crs_object_from_layer(input_layer)
        if crs_object:
            feedback.pushInfo(f"CRS Description: {crs_object.description()}")
            feedback.pushInfo(f"Is Geographic: {crs_object.isGeographic()}")
        
        # Method 3: Convert QGIS CRS to GeoPandas-compatible format
        geopandas_crs = convert_qgis_crs_to_geopandas_format(crs_object)
        feedback.pushInfo(f"GeoPandas CRS: {geopandas_crs}")
        
        # Method 4: Get project CRS
        project_crs = get_crs_from_project()
        feedback.pushInfo(f"Project CRS: {project_crs}")
        
        # Method 5: Get NoData value
        nodata_value = get_nodata_value_from_layer(input_layer)
        feedback.pushInfo(f"NoData value: {nodata_value}")
        
        # Use with GeoPandas
        if geopandas_crs:
            import geopandas as gpd
            # Create or load your GeoDataFrame
            gdf = gpd.read_file("your_file.shp")
            # Set CRS using the converted format
            gdf = gdf.set_crs(geopandas_crs, allow_override=True)
        
        return {}
