# Data Quality anomalies detection leveraging Machine Learning 

## Description
This project implements Autoencoder (AE) models using PySpark and Keras for anomaly detection in a dataset. The goal is to preprocess the data stored as Iceberg tables on AWS S3 bucket (cataloged using Glue Catalog), train an AE or LSTM Autoencoder, and evaluate its performance in identifying anomalies based on reconstruction error (mean squared error).



## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
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
This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE) file for details.