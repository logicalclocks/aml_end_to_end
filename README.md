# AML End to End Example

## Project description
This project demonstrates end to end pipeline how to train binary anti money laundering (AML) classifier based on 
Generative Adversarial Networks (GANs) and Graph embeddings. Proposed solution includes following sub sections:  

* Data ingestion - We will use sample of transactions data generated by [AMLSim](https://github.com/IBM/AMLSim) 
* Feature store – We use Hopsworks Feature Store to compute features, organize them as feature groups and store for 
downstream analysis, such as creating training datasets for model training, as well as retrieving them      
* Graph Embeddings - We will use [StellarGraph](https://github.com/stellargraph/stellargraph) library to compute graph 
embeddings.
* Anomaly detection model - We will use keras implementation of [adversarial anomaly detection](https://arxiv.org/pdf/1905.11034.pdf) that was adapted to tabular data.
* Hyper parameter tuning - We will use [Maggy](https://github.com/logicalclocks/maggy) to conduct experiments for 
hyperparameter tuning.  
* Model serving - We will use Hopsworks model server to predict anomalous transactions. 

## Demo dataset
Sample of transactions data is provided in the folder ./demodata. Upload alert_transactions.csv, party.csv and 
transactions.csv in to the hdfs:///Projects/{}/Resources in your Hopsworks cluster. 

## Anomaly detection model
Keras implementation of [adversarial anomaly detection](https://arxiv.org/pdf/1905.11034.pdf) is provided in the folder
./adversarialaml. To use this library zip adversarialaml folder and attach dversarialaml.zip to Jupyter server or 
Hopsworks job.  
 
## End to End pipeline
To successfully complete this tutorial use one of 2 options bellow
 
### Jupyter notebooks step by step   
Run jupyter notebooks in the following order:
1) data_ingestion/1_transaction_feature_engineering_ingestion.ipynb 
2) data_ingestion/2_prep_training_dataset_for_embeddings.ipynb
3) training/3_maggy_node_embeddings.ipynb
4) training/4_compute_node_embeddings.ipynb 
5) mlserver/5_predict_node_embeddings_and_ingest_to_fs.ipynb
6) training/6_maggy_adversarial_aml.ipynb
7) training/7_train_adversarial_aml.ipynb
8) mlserver/8_aml_model_server.ipynb

### Airflow
In Hopsworks you can also create airflow pipeline. For this:  
1) Create notebook jobs. You can follow instructions [here](https://hopsworks.readthedocs.io/en/stable/user_guide/hopsworks/jobs.html?highlight=project.connect#python) 
how to create jobs in Hopsworks.
2) Create Airflow DAG using provided airflow_aml_end2end.py. 
