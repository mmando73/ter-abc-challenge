import numpy as np
import numpy.lib.recfunctions as recfun
from plyfile import PlyData, PlyElement
from pathlib import Path


def process_dataset(dataset='Train', save_ply=False):
    # 1) Path Definition
    # base directory for the chosen dataset
    base_path = Path('./Challenge-ABC') / dataset
    if not base_path.exists(): # verification
        print(f"Error: Dataset folder {base_path} not found, check 'dataset' argument.")
        return None

    # dynamic definition for the .ply and .lb files
    ply_dir = base_path / 'ply'
    lb_dir = base_path / 'lb'
    if save_ply:
        output_dir = base_path / 'ply_labeled'
        output_dir.mkdir(parents=True, exist_ok=True) # create the output folder if it doesn't exist

    # 2) Store each file data/stats
    ids = []
    points_counts = []
    edges_counts = []

    # 3) Process loop
    for ply_file in ply_dir.glob('*.ply'):
        # extract the prefix/file name
        file_id = ply_file.stem

        # define the .lb file path
        lb_file = lb_dir /f"{file_id}.lb"

        # check that the labels file exists
        if not lb_file.exists():
            print(f"Error: Didn't find labels file for {file_id}.ply -> ignored.")
            continue

        # reading data
        #print(f"Prcessing: {file_id}.ply")
        ply_data = PlyData.read(ply_file)
        labels = np.loadtxt(lb_file, dtype='int')
        vertex_data = ply_data.elements[0].data

        # Verification : check that nb_vertex == nb_labels, if not ignore it
        if len(vertex_data) != len(labels):         
            print(f"Dimension error with {file_id} : {len(vertex_data)} points (.ply) vs {len(labels)} labels (.lb) -> ignored.")
            continue
        
        # Data gathering for stats
        ids.append(file_id)
        points_counts.append(len(labels))
        edges_counts.append(np.sum(labels))

        # Optional combining+writing new files logic
        if save_ply:
            # create the new vertex_data which appends labels field to already existing fields  
            new_vertex_data = recfun.append_fields(
                vertex_data, 
                names='label', 
                data=labels, 
                dtypes='i4', 
                usemask=False
            )
            new_vertex_element = PlyElement.describe(new_vertex_data, 'vertex')
            # write the new file
            output_path = output_dir / f"{file_id}_labeled.ply"
            PlyData([new_vertex_element]).write(str(output_path))
        
    # 4) Compute Stats
    points_arr = np.array(points_counts)
    edges_arr = np.array(edges_counts)

    # avoid division by zero if empty
    if len(points_arr) == 0: return None

    percentages = (edges_arr / points_arr) * 100
    total_points = np.sum(points_arr)
    total_edges = np.sum(edges_arr)

    return {
        "total_points": total_points,
        "total_edges": total_edges,
        "mean_points" : int(np.mean(points_arr)),
        "global_edge_pct": (total_edges / total_points) * 100,
        "max_points": (ids[np.argmax(points_arr)], np.max(points_arr)),
        "min_points": (ids[np.argmin(points_arr)], np.min(points_arr)),
        "highest_pct": (ids[np.argmax(percentages)], np.max(percentages)),
        "lowest_pct": (ids[np.argmin(percentages)], np.min(percentages))
    }


train_stats_summary = process_dataset(dataset='Train', save_ply=True)
print("---Train Dataset---")
print(f"Total points = {train_stats_summary['total_points']} points.")
print(f"Mean points = {train_stats_summary['mean_points']} points.")
print(f"Total edges = {train_stats_summary['total_edges']} points.")
print(f"Global edge percentage = {train_stats_summary['global_edge_pct']:.2f}%.")
print(f"Model with max points = {train_stats_summary['max_points'][0]}.ply. Has {train_stats_summary['max_points'][1]} points.")
print(f"Model with min points = {train_stats_summary['min_points'][0]}.ply. Has {train_stats_summary['min_points'][1]} points.")
print(f"Model with the highest edge points percentage = {train_stats_summary['highest_pct'][0]}.ply. Has {train_stats_summary['highest_pct'][1]:.2f}%.")
print(f"Model with the lowest edge points percentage = {train_stats_summary['lowest_pct'][0]}.ply. Has {train_stats_summary['lowest_pct'][1]:.2f}%.")

validation_stats_summary = process_dataset(dataset='Validation', save_ply=True)
print("\n---Validation Dataset---")
print(f"Total points = {validation_stats_summary['total_points']} points.")
print(f"Mean points = {validation_stats_summary['mean_points']} points.")
print(f"Total edges = {validation_stats_summary['total_edges']} points.")
print(f"Global edge percentage = {validation_stats_summary['global_edge_pct']:.2f}%.")
print(f"Model with max points = {validation_stats_summary['max_points'][0]}.ply. Has {validation_stats_summary['max_points'][1]} points.")
print(f"Model with min points = {validation_stats_summary['min_points'][0]}.ply. Has {validation_stats_summary['min_points'][1]} points.")
print(f"Model with the highest edge points percentage = {validation_stats_summary['highest_pct'][0]}.ply. Has {validation_stats_summary['highest_pct'][1]:.2f}%.")
print(f"Model with the lowest edge points percentage = {validation_stats_summary['lowest_pct'][0]}.ply. Has {validation_stats_summary['lowest_pct'][1]:.2f}%.")

#process_dataset('Test') => Error: Dataset folder Challenge-ABC/Test not found, check 'dataset' argument.

# Output:
# ---Train Dataset---
# Total points = 3174768 points.
# Mean points = 16034 points.
# Total edges = 149469 points.
# Global edge percentage = 4.71%.
# Model with max points = 1452.ply. Has 118332 points.
# Model with min points = 2340.ply. Has 1246 points.
# Model with the highest edge points percentage = 2800.ply. Has 38.52%.
# Model with the lowest edge points percentage = 0310.ply. Has 0.61%.

# ---Validation Dataset---
# Total points = 690211 points.
# Mean points = 13804 points.
# Total edges = 40089 points.
# Global edge percentage = 5.81%.
# Model with max points = 0353.ply. Has 34340 points.
# Model with min points = 0713.ply. Has 1004 points.
# Model with the highest edge points percentage = 0713.ply. Has 87.65%.
# Model with the lowest edge points percentage = 0939.ply. Has 1.23%.

