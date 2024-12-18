import subprocess
import pickle
import sys
import json
from pde.mesh_generation.generate_mesh import main

def run_subprocess():

    proc = subprocess.Popen(
        [sys.executable, '/home/maccyz/Documents/Neural_PDE/pde/mesh_generation/generate_mesh.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = proc.communicate()

    if proc.returncode == 0:
        # Deserialize the NumPy array from stdout
        array = pickle.loads(stdout)

        return array
    else:
        # Handle error message from stderr
        print("Error")
        print(f'{proc.returncode = }')
        error = stderr.decode()
        print(error)
        raise Exception(error)


if __name__ == "__main__":
    array = run_subprocess()
    print(array)
