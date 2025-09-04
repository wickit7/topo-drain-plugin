#!/usr/bin/env python3
"""
Quick test to verify the _vector_to_raster method returns correct geometry mapping
when unique_values=True
"""

import sys
import os
sys.path.append('/Users/aquaplus_tiw/Documents/gitprojects/topo-drain-plugin')

import geopandas as gpd
from shapely.geometry import LineString, Point
import tempfile
import rasterio
import numpy as np
from topo_drain.core.topo_drain_core import TopoDrainCore

def test_vector_to_raster_mapping():
    """Test that _vector_to_raster returns correct geometry mapping"""
    
    # Create a test DTM
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_dtm:
        dtm_path = tmp_dtm.name
    
    # Create test DTM
    height, width = 100, 100
    transform = rasterio.transform.from_bounds(0, 0, 100, 100, width, height)
    dtm_data = np.random.rand(height, width).astype(np.float32) * 100
    
    with rasterio.open(dtm_path, 'w', 
                       driver='GTiff',
                       height=height, width=width,
                       count=1, dtype=np.float32,
                       crs='EPSG:4326',
                       transform=transform) as dst:
        dst.write(dtm_data, 1)
    
    # Create test geometries
    line1 = LineString([(10, 10), (20, 20)])
    line2 = LineString([(30, 30), (40, 40)]) 
    line3 = LineString([(50, 50), (60, 60)])
    
    test_gdf = gpd.GeoDataFrame({
        'id': ['A', 'B', 'C'],
        'geometry': [line1, line2, line3]
    }, crs='EPSG:4326')
    
    print("Original geometries:")
    for i, geom in enumerate(test_gdf.geometry):
        print(f"  Index {i}: {geom}")
    
    # Initialize TopoDrainCore
    core = TopoDrainCore()
    core.temp_directory = tempfile.mkdtemp()
    
    # Test with unique_values=False (should return only path)
    print("\nTesting unique_values=False:")
    result_path = core._vector_to_raster([test_gdf], dtm_path, unique_values=False)
    print(f"  Result type: {type(result_path)}")
    print(f"  Result: {result_path}")
    
    # Test with unique_values=True (should return path and mapping)
    print("\nTesting unique_values=True:")
    result_path, geometry_mapping = core._vector_to_raster([test_gdf], dtm_path, unique_values=True)
    print(f"  Result types: path={type(result_path)}, mapping={type(geometry_mapping)}")
    print(f"  Path: {result_path}")
    print("  Geometry mapping:")
    for raster_value, geom in geometry_mapping.items():
        print(f"    Raster value {raster_value}: {geom}")
    
    # Verify mapping correctness
    print("\nVerifying mapping correctness:")
    with rasterio.open(result_path) as src:
        raster_data = src.read(1)
        unique_values = np.unique(raster_data[raster_data > 0])
        print(f"  Unique raster values: {unique_values}")
        print(f"  Mapping keys: {list(geometry_mapping.keys())}")
        
        # Check that all raster values have corresponding geometries
        mapping_correct = True
        for val in unique_values:
            if val not in geometry_mapping:
                print(f"  ERROR: Raster value {val} not in geometry mapping!")
                mapping_correct = False
            else:
                print(f"  ✓ Raster value {val} maps to geometry: {geometry_mapping[val]}")
    
    if mapping_correct:
        print("\n✓ SUCCESS: Geometry mapping is correct!")
    else:
        print("\n✗ FAILURE: Geometry mapping is incorrect!")
    
    # Cleanup
    try:
        os.unlink(dtm_path)
        os.unlink(result_path)
    except:
        pass
    
    return mapping_correct

if __name__ == "__main__":
    success = test_vector_to_raster_mapping()
    sys.exit(0 if success else 1)
