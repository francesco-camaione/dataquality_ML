# Data Quality anomalies detection leveraging Machine Learning 

## Description
This project implements Autoencoder (AE) models using PySpark and Keras for anomaly detection in a dataset stored in a Data Lakehouse. The goal is to preprocess the data stored as Iceberg tables on AWS S3 bucket (cataloged using Glue Catalog), train an Isolation Forest, AE and LSTM Autoencoder, and evaluate its performance in identifying anomalies based on reconstruction error (mean absolute error).
The service could then be abstracted and used for several tables on a daily or weekly basis using orchestration tools.
Data is downloaded from the https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data, and the models are trained using the data related to the first 3 days of December 2024. The threshold of the reconstruction error after which records are considered anomalies is computed using the n percentile of the reconstruction error of the training data.
Models have been tested using data of 2024-12-25.

## Model Performance

### Latest Test Results using data of 2024-12-25

#### Isolation Forest Performance
- **Total normal records processed**: 219744
- **Total failure records processed**: 4
- **True Positives**: 3
- **False Positives**: 21972
- **Detection Rate**: 75.00%
- **False Positive Rate**: 10%
- Model parameters:
  - n_estimators: 6000
  - max_samples: 16
  - contamination: 0.00025
  - max_features: 0.2
  - bootstrap: True
- Threshold set at 90th percentile of anomaly scores

#### LSTM Autoencoder Performance
- Training completed in 100 epochs 
- **Total normal records processed**: 219784
- **Total failure records processed**: 4
- **False positives (normal records flagged as anomaly)**: 116134
- **True positives (failure records correctly flagged)**: 1
- **Detection Rate**: 25% (1/4 failure sequences detected)
- **False Positive Rate**: 53% (1 false positives)
- LSTM layers with decreasing dimensions (128 → 64 → 32 → 64 → 128)

#### Autoencoder Performance
- Training completed in ~50 epochs
- **Total normal records processed**: 219784
- **Total failure records processed**: 4
- **False positives (normal records flagged as anomaly)**: 75834
- **True positives (failure records correctly flagged)**: 1
- **False positive rate**: 34.51%
- **Detection rate**: 25.00%
- AE layers with decreasing dimensions (512 → 256 → 128 → 48 -> 128 -> 256 -> 512)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.