
# Independent Research Project
## Package for Automated Slug Detection using Machine Learning 

slugdetection is a Python package created to automatically detect slug flow by classifying pressure and temperature data intervals. The package Slug_Detection module classifies with over 95% accuracy interval and presents a robust approach to labelling slug flow. Other functionalities include forecasting slug flow using times series modelling in the Slug_Forecasting module and clustering flow types together through the Slug_Labelling module.

## Author 

Deirdree A. Polak
Github: dapolak
Email: deirdree.polak@gmail.com
CID: 00973185

## Project Information

This package was developped as part of the Applied Computational Sciences and Engineering MSc 2018-19 at Imperial College London and as part of an internship at Wintershall Dea GmbH in Germany. The project was part of the Independant Research Project ACSE-9 module.

It was performed under the supervision of Prof Olivier Dubrule and Lukas Mosser from the Royal School of Mines, Imperial College London anf of Dr Meindert Dillen and Peter Kronberger from the Digital Transformation departement at Wintershall Dea GmbH. 

## Repository Structure




## Installation

### Local Installation



### DataBricks Installation

In the context of using the package in DataBricks to access the data, the wheel file `.whl` in the `\dist` folder is to be downloaded. 

## Usage

Once the package is installed locally or on Databricks, it can simply be imported and used as shown below.

```python
from slug_detection import *

de = Data_Engineering(spark_data)
sl = Slug_labelling(spark_data)
fr = Flow_Recognition(spark_data)
sd = Slug_Detection(spark_data)
sf = Slug_Forecasting(pands_whp_data)
```

More information about the various functionalities of the package can be found in the `IRP_slugdetection_2019` Python notebook. 

## Data

For confidentiality reasons, the data used to develop and test the package cannot be provided on this repository. The data used is raw pressure and temperature sensor data from an offshore oil well exhibiting slugging behaviour. Unfortunately, the code cannot be run without the data. A live demonstration can be arranged if required for assessment. 

## Requirements

The requirements are listed in `requirements.txt` and can be installed locally using the command:

```bash
pip install -r requirements.txt
```

On DataBricks, all the librairies are already installed on the ML clusters. If version upgrades are required run code
```python
dbutils.library.installPyPI("pypipackage", version="version")
```

## License

This package is licensed using a 
[MIT](https://choosealicense.com/licenses/mit/)
