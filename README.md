# AlphaPolyp

## 🏷️ Labeling System (Using Blender)

### 💡 Overview

This labeling system is designed to automatically measure the **size** and **volume** of polyps inside the colon by analyzing 3D intersections between objects in `.obj` files. Given synthetic 3D models that contain both a colon and a polyp, the script:

1. Imports the 3D models into Blender
2. Calculates the intersection between the polyp and colon using a boolean operation
3. Computes the volume of the intersection
4. Extracts the bounding box dimensions (X, Y, Z)
5. Saves the results for each file into a structured `.csv` file

---

### 🛠️ How to Run the Labeling Script

To use this script, ensure you have [Blender](https://www.blender.org/download/) installed (tested with **Blender 3.x**) and that your `.obj` files are located inside the `data/sample_data/mesh` directory.


You can run the labeling in **one of two ways**:

---

### 🟠 Option 1: Run Using the `.blend` File

1. Open **Blender**.
2. Click **Open**, then select the file:  
   `label_polyp_data.blend` (located in the `AlphaPolyp/` main folder).
3. Go to the **Scripting** workspace.
4. Click **Run Script** to process the `.obj` files.

---

### 🔵 Option 2: Run Using the Python Script Directly

1. Open **Blender**.
2. Go to the **Scripting** workspace.
3. Click **Open**, and select the Python script:  
   `labeling/label_polyp_data.py`
4. Click **Run Script** to start processing.

---

### 📁 Output

The script will:
- Process all `.obj` files in `data/sample_data/meshes/`
- Save the results to:  `data/annotations/sample_polyp_intersection_results.csv`


⚠️ **Important**: Make sure you open either the `.blend` file or the `.py` file from the **project root folder** (`AlphaPolyp/`) to ensure the paths work correctly.
