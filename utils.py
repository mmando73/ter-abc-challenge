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

combine_labels()
combine_labels('Validation')
#combine_labels('Test') => Error: Dataset folder Challenge-ABC/Test not found, check 'dataset' argument.



