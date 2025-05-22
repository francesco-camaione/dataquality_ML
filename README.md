# Data Quality anomalies detection leveraging Machine Learning 

## Description
This project implements Autoencoder (AE) models using PySpark and Keras for anomaly detection in a dataset stored in a Data Lakehouse. The goal is to preprocess the data stored as Iceberg tables on AWS S3 bucket (cataloged using Glue Catalog), train an AE or LSTM Autoencoder, and evaluate its performance in identifying anomalies based on reconstruction error (mean absolute error).
The service could then be abstracted and used for several tables on a daily or weekly basis using orchestration tools.
Data is downloaded from the https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data, and the models are trained using the data (filtering failure == 0) related to the first 3 days of December 2024. The threshold of the reconstruction error after which records are considered anomalies is computed using the n percentile of the reconstruction error of the training data.
Models have been tested using data of 2024-12-15.

## Model Performance

### Latest Test Results using data of 2024-12-15

#### LSTM Autoencoder Performance
- **Detection Rate**: 100% (7/7 failure sequences detected)
- **False Positive Rate**: 0% (0 false positives out of 301,267 normal sequences)
- **Model Architecture**:
  - Input shape: (20, 127)
  - LSTM layers with decreasing dimensions (32 → 16 → 16 → 32)
  - Total parameters: 36,191
  - Training completed in 10 epochs with validation loss: 0.0843

#### Autoencoder Performance
- **Detection Rate**: 100% (7/7 failure records detected)
- **False Positive Rate**: 1% (3,013 false positives out of 301,286 normal records)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.