# Data Quality anomalies detection leveraging Machine Learning 

## Description
This project implements Autoencoder (AE) models using PySpark and Keras for anomaly detection in a dataset stored in a Data Lakehouse. The goal is to preprocess the data stored as Iceberg tables on AWS S3 bucket (cataloged using Glue Catalog), train an AE or LSTM Autoencoder, and evaluate its performance in identifying anomalies based on reconstruction error (mean squared error).
The service could then be abstracted and used for several tables and orchestrated to detect anomalies based on user needs.



## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/francesco-camaione/dataquality_ML.git
   cd dataquality_ML
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```


## Notes
- The Autoencoder model training and evaluation sections are currently commented out in the code. Uncomment these sections to train the model.
- Adjust the parameters in the model as needed for your specific dataset and requirements.
- Currently, dataset partitions are stored as csv files in the dataset folder to minimize unnecessary AWS costs.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.