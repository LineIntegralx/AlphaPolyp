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

To use this script, ensure you have [Blender](https://www.blender.org/download/) installed (tested with **Blender 3.x**) and that your `.obj` files are located inside the `data/sample_data/` directory.

#### ✅ Steps:

1. **Open Blender**.
2. Navigate to the **Scripting** workspace.
3. Click **Open** and select the script: `labeling/label_polyp_data.py`.
4. **Important**: You must run Blender from the **project’s root folder** (the one containing `data/` and `labeling/`) so that relative paths like `data/sample_data/` work properly.

   **To do this:**
   - On **Windows**: Right-click the project folder → choose **"Open in Terminal"** → type `blender` and press Enter.
   - On **macOS/Linux**: Open Terminal, `cd` into the project folder, and run `blender`.

5. Once inside Blender, with the script open, click **Run Script** to begin processing your `.obj` files.
