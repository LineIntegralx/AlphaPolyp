# AlphaPolyp

## Labeling System (Using Blender)

### The Idea
The goal of this labeling process is to automatically measure the size and volume of polyps inside the colon using 3D model intersections. Given synthetic `.obj` files that contain both a colon and a polyp, this script:

1. Imports the 3D models into Blender
2. Computes the intersection between the polyp and colon using a boolean modifier
3. Calculates the volume of that intersection
4. Extracts the bounding box dimensions (X, Y, Z)
5. Saves the results (per file) to a `.csv` file

### How to Run the Labeling Script
To run the labeling script, make sure you have [Blender](https://www.blender.org/download/) installed (tested with **Blender 3.x**), and that your `.obj` files are correctly placed in the `data/sample_data/` directory.

1. Open **Blender**.
2. Go to the **Scripting** tab.
3. Click **Open** and select the file `labeling/label_polyp_data.py` from your project folder.
4. **Important**: Make sure you open Blender from the **project's main folder** (the one that contains the `data/` and `labeling/` folders).  
   This ensures that all relative paths like `data/sample_data/` work correctly.
   - If you're on Windows, right-click the folder → "Open in Terminal", then run `blender` from there.
   - If you're on macOS/Linux, navigate to the project folder in Terminal and run `blender`.
5. Once the script is open in Blender, click **Run Script**.
