# Rappi Anomaly Detection 

## Project structure: 

The project repo holds the following structure
```
 |-execution_scripts
 | |-Method.sh
 |-reqs
 | |-Method_PythonVersion.txt
 |-methods
 | |-method
 | | |-main.py
 | | |-__init__.py
 | |-Build_Data
 | | |-__init__.py
 | | |-dataBuilder.py

```

### execution_scripts
Holds sh scripts for sbatch execution. 
Should be setup in a way such that we only run 

```
sbatch /execution_scripts/method.sh
```

### reqs

Requirment files to create appropriate venvs for experiment. File name of requirments file should be in the following format: 

```
Method_PythonVersion.txt
```

To create and activate a venv for the specific method: 

```
module load python/PYTHON_VERSION
python3 -m venv $HOME/Method
source $HOME/Method
pip install -r Method_PythonVersion.txt
```

### Methods 

All method code should go inside the 

```
/methods/Method/main.py
```


### Build_Data

Data pipelines to set up data in appropriate structure for method to process. 

