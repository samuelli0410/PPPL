import sys
from pathlib import Path
import csv
import numpy as np
import ast
csv.field_size_limit(sys.maxsize)
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
sys.path.append(str(project_root))
sav_files_dir = data_dir / "all"
from data.file_utils import GetEmission
em = GetEmission(file_path=sav_files_dir)
files = em.list_files(display=True)

def loadFramesAndInverted(i):
    inverted, radii, elevation, frames, times, vid_frames, vid_times, vid = em.load_all(files[i])

    framesInverted= []
    count = 0
    for j in frames:
        j = int(j)
        framesInverted.append([vid[j].tolist(), inverted[count].tolist()])
        count +=1
    data_dir = Path(__file__).parent.parent / "data/FrameInvertedData"
    output_file = data_dir / f"data{i}.csv"
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(framesInverted)


# for j in range(26):
#     loadFramesAndInverted(j)


def load_csv_data(file_index):
    data_dir = Path(__file__).parent.parent / "data/FrameInvertedData"
    file_path = data_dir / f"data{file_index}.csv"

    reconstructed = []

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue  # skip malformed rows

            try:
                arr1 = np.array(ast.literal_eval(row[0]))
                arr2 = np.array(ast.literal_eval(row[1]))
                reconstructed.append((arr1, arr2))
            except Exception as e:
                print(f"Failed to parse row: {row}\nError: {e}")

    return reconstructed

# Example usage
# print("1")
# framesInverted = load_csv_data(0)
# print(len(framesInverted))                  # Number of frame pairs
# print(framesInverted[0][0].shape)           # Shape of first `vid` frame
# print(framesInverted[0][1].shape) 