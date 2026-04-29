import numpy as np
import numpy.lib.recfunctions as recfun
from plyfile import PlyData, PlyElement
from pathlib import Path


def combine_labels(dataset='Train'):
    # path definition
    # base directory for the chosen dataset
    base_path = Path('./Challenge-ABC') / dataset
    if not base_path.exists(): # verification
        print(f"Error: Dataset folder {base_path} not found, check 'dataset' argument.")
        return

    # dynamic definition
    ply_dir = base_path / 'ply'
    lb_dir = base_path / 'lb'
    output_dir = base_path / 'ply_labeled'

    output_dir.mkdir(parents=True, exist_ok=True) # create the output folder if it doesn't exist

    # we loop over the ply files
    for ply_file in ply_dir.glob('*.ply'):

        # extract the prefix/file name
        file_id = ply_file.stem

        # define the .lb file path
        lb_file = lb_dir / f"{file_id}.lb"

        # check that the label file exists
        if not lb_file.exists():
            print(f"Error : Didn't find label file for {file_id}.ply -> ignored.")
            continue
        
        print(f"Prcessing: {file_id}.ply")
        ply_data = PlyData.read(ply_file)           # read .ply content
        labels = np.loadtxt(lb_file, dtype='int')   # read .lb content
        vertex_data = ply_data.elements[0].data     # extract vertex data

        # check that nb_vertex == nb_labels, if not ignore it
        if len(vertex_data) != len(labels):         
            print(f"Dimension error with {file_id} : {len(vertex_data)} points (.ply) vs {len(labels)} labels (.lb) -> ignored.")
            continue

        # create the new vertex_data which appends labels field to already existing fields  
        new_vertex_data = recfun.append_fields(
            vertex_data, 
            names='label', 
            data=labels, 
            dtypes='i4', 
            usemask=False
        )

        # create the new Ply element to transform it after 
        new_vertex_element = PlyElement.describe(new_vertex_data, 'vertex')

        # write the new file
        output_path = output_dir / f"{file_id}_labeled.ply"
        PlyData([new_vertex_element]).write(str(output_path))   

    print(f"OK.")

def extract_stats(dataset='Train'):
    # path definition
    # base directory for the chosen dataset
    base_path = Path('./Challenge-ABC') / dataset
    if not base_path.exists(): # verification
        print(f"Error: Dataset folder {base_path} not found, check 'dataset' argument.")
        return

    # dynamic definition
    ply_dir = base_path / 'ply'
    lb_dir = base_path / 'lb'
  
    # to store each file data
    ids = []
    points_counts = []
    edges_counts = []

    # we loop over the ply files
    for ply_file in ply_dir.glob('*.ply'):

        # extract the prefix/file name
        file_id = ply_file.stem

        # define the .lb file path
        lb_file = lb_dir / f"{file_id}.lb"

        # check that the label file exists
        if not lb_file.exists():
            print(f"Error : Didn't find label file for {file_id}.ply -> ignored.")
            continue
        
        #print(f"Prcessing: {file_id}.ply")
        ply_data = PlyData.read(ply_file)           # read .ply content
        labels = np.loadtxt(lb_file, dtype='int')   # read .lb content
        vertex_data = ply_data.elements[0].data     # extract vertex data

        # check that nb_vertex == nb_labels, if not ignore it
        if len(vertex_data) != len(labels):         
            print(f"Dimension error with {file_id} : {len(vertex_data)} points (.ply) vs {len(labels)} labels (.lb) -> ignored.")
            continue
        
        ids.append(file_id)
        points_counts.append(len(labels))
        edges_counts.append(np.sum(labels))

    # convert to numpy arrays 
    points_arr = np.array(points_counts)
    edges_arr = np.array(edges_counts)

    # calculate metrics
    percentages = (edges_arr / points_arr) * 100

    idx_max_p = np.argmax(points_arr)
    idx_min_p = np.argmin(points_arr)
    idx_max_pct = np.argmax(percentages)
    idx_min_pct = np.argmin(percentages)

    total_points = np.sum(points_arr)
    mean_points = np.mean(points_arr, dtype='int')

    total_edges = np.sum(edges_arr)
    global_edge_pct = (total_edges / total_points) * 100

    stats_summary = {
        "total_points": total_points,
        "total_edges": total_edges,
        "mean_points" : mean_points,
        "global_edge_pct": global_edge_pct,
        "max_points": (ids[idx_max_p], points_arr[idx_max_p]),
        "min_points": (ids[idx_min_p], points_arr[idx_min_p]),
        "highest_pct": (ids[idx_max_pct], percentages[idx_max_pct]),
        "lowest_pct": (ids[idx_min_pct], percentages[idx_min_pct])
    }
    
    return stats_summary

combine_labels()
combine_labels('Validation')
#combine_labels('Test') => Error: Dataset folder Challenge-ABC/Test not found, check 'dataset' argument.

train_stats_summary = extract_stats()
print("---Train Dataset---")
print(f"Total points = {train_stats_summary['total_points']} points.")
print(f"Mean points = {train_stats_summary['mean_points']} points.")
print(f"Total edges = {train_stats_summary['total_edges']} points.")
print(f"Global edge percentage = {train_stats_summary['global_edge_pct']:.2f}%.")
print(f"Model with max points = {train_stats_summary['max_points'][0]}.ply. Has {train_stats_summary['max_points'][1]} points.")
print(f"Model with min points = {train_stats_summary['min_points'][0]}.ply. Has {train_stats_summary['min_points'][1]} points.")
print(f"Model with the highest edge points percentage = {train_stats_summary['highest_pct'][0]}.ply. Has {train_stats_summary['highest_pct'][1]:.2f}%.")
print(f"Model with the lowest edge points percentage = {train_stats_summary['lowest_pct'][0]}.ply. Has {train_stats_summary['lowest_pct'][1]:.2f}%.")

validation_stats_summary = extract_stats('Validation')
print("\n---Validation Dataset---")
print(f"Total points = {validation_stats_summary['total_points']} points.")
print(f"Mean points = {validation_stats_summary['mean_points']} points.")
print(f"Total edges = {validation_stats_summary['total_edges']} points.")
print(f"Global edge percentage = {validation_stats_summary['global_edge_pct']:.2f}%.")
print(f"Model with max points = {validation_stats_summary['max_points'][0]}.ply. Has {validation_stats_summary['max_points'][1]} points.")
print(f"Model with min points = {validation_stats_summary['min_points'][0]}.ply. Has {validation_stats_summary['min_points'][1]} points.")
print(f"Model with the highest edge points percentage = {validation_stats_summary['highest_pct'][0]}.ply. Has {validation_stats_summary['highest_pct'][1]:.2f}%.")
print(f"Model with the lowest edge points percentage = {validation_stats_summary['lowest_pct'][0]}.ply. Has {validation_stats_summary['lowest_pct'][1]:.2f}%.")



