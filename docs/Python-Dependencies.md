# Python Dependencies

The TopoDrain plugin requires several Python packages: `numpy`, `pandas`, `geopandas`, `shapely`, `scipy`, `whitebox_workflows`. While these packages are widely used in geospatial data processing, not all of them are included in the default QGIS installation (particularly `pandas`, `geopandas`, `scipy`, and `whitebox_workflows` could be missing). 

**If a package is missing in your QGIS installation:**

## Windows (OSGeo4W installation)

**Method 1: Using OSGeo4W Shell (Recommended)**
1. Open the **OSGeo4W Shell** as Administrator (search for "OSGeo4W Shell" in Start menu, right-click → Run as administrator)
2. (Optional) Check currently installed packages:
   ```bash
   python -m pip list
   ```
3. Install the missing packages using pip:
   ```bash
   python -m pip install pandas
   python -m pip install geopandas
   python -m pip install scipy
   ```
4. Restart QGIS after installation

For more informations: https://landscapearchaeology.org/2018/installing-python-packages-in-qgis-3-for-windows/

**Note:** If you encounter issues with `python`, try using `python3` instead:
```bash
python3 -m pip install pandas geopandas scipy
```

**Method 2: Using OSGeo4W Setup Installer**
- Run the OSGeo4W Setup installer (`osgeo4w-setup.exe`)
- Search for and select the missing Python packages
- Complete the installation wizard


## macOS Installation

The installation method depends on your QGIS version:

**Method 1: Terminal-based installation**

Older versions provide a standalone Python executable. Common locations:
- **QGIS LTR:** `/Applications/QGIS-LTR.app/Contents/MacOS/bin/python3`
- **QGIS regular:** `/Applications/QGIS.app/Contents/MacOS/bin/python3`
- **Alternative location:** `/Applications/QGIS.app/Contents/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python` or at user location if python in not bundled within QGIS.app `/usr/local/bin/python3.12`(Instead of 3.12, it could be a different version of Python)
If you don't succeed, see at https://gis.stackexchange.com/questions/351280/installing-python-modules-for-qgis-3-on-mac for more informations on how to find python executable path or use Method 2 to install missing python packages)

**Installation steps:**

1. Open Terminal
2. Locate your QGIS Python executable:
   ```bash
   # For QGIS LTR
   ls /Applications/QGIS-LTR.app/Contents/MacOS/bin/python3
   
   # For regular QGIS
   ls /Applications/QGIS.app/Contents/MacOS/bin/python3
   ```
3. If found, use the full path to install packages:
   ```bash
   /Applications/QGIS-LTR.app/Contents/MacOS/bin/python3 -m pip install pandas geopandas scipy
   ```
4. Restart QGIS after installation!

**If you cannot find a Python executable, use Method 2 (QGIS Python Console) instead.**

**Method 2: Use QGIS Python Console**

Newer QGIS versions seem to embed Python directly inside the application without a standalone Python executable. Packages must be installed from within QGIS:

1. Open QGIS and go to **Plugins → Python Console**
2. Install the missing packages (run one command line after one another):
   ```python
   from pip._internal.cli.main import main
   main(["install", "pandas"])
   main(["install", "geopandas"])
   main(["install", "scipy"])
   ```
   
3. Restart QGIS completely!
4. Verify installation in the Python Console:
   ```python
   import pandas as pd
   import geopandas as gpd
   import scipy
   ```
