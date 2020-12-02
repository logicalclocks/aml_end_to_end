import airflow

from datetime import datetime, timedelta
from airflow import DAG

from hopsworks_plugin.operators.hopsworks_operator import HopsworksLaunchOperator
from hopsworks_plugin.operators.hopsworks_operator import HopsworksFeatureValidationResult
from hopsworks_plugin.sensors.hopsworks_sensor import HopsworksJobSuccessSensor

# Username in Hopsworks
# Click on Account from the top right drop-down menu
DAG_OWNER = 'meb10179'

# Project name this DAG belongs to
PROJECT_NAME = 'amlsim'


####################
## DAG definition ##
####################
delta = timedelta(minutes=-10)
now = datetime.now()

args = {
    'owner': DAG_OWNER,
    'depends_on_past': False,

    # DAG should have run 10 minutes before now
    # It will be automatically scheduled to run
    # when we upload the file in Hopsworks
    'start_date': now + delta,

    # Uncomment the following line if you want Airflow
    # to authenticate to Hopsworks using API key
    # instead of JWT
    #
    # NOTE: Edit only YOUR_API_KEY
    #
    
}

# Our DAG
dag = DAG(
    # Arbitrary identifier/name
    dag_id = "airflow_aml_end2end",
    default_args = args,

    # Run the DAG only one time
    # It can take Cron like expressions
    # E.x. run every 30 minutes: */30 * * * * 
    schedule_interval = "@once"
)



launch_transaction_feature_engineering_ingestion = HopsworksLaunchOperator(dag=dag,
					 project_name="amlsim",
					 task_id="launch_transaction_feature_engineering_ingestion",
					 job_name="transaction_feature_engineering_ingestion",
					 job_arguments="",
					 wait_for_completion=True)
					 

launch_prep_training_dataset_for_embeddings = HopsworksLaunchOperator(dag=dag,
					 project_name="amlsim",
					 task_id="launch_prep_training_dataset_for_embeddings",
					 job_name="prep_training_dataset_for_embeddings",
					 job_arguments="",
					 wait_for_completion=True)
					 

launch_maggy_node_embeddings = HopsworksLaunchOperator(dag=dag,
					 project_name="amlsim",
					 task_id="launch_maggy_node_embeddings",
					 job_name="maggy_node_embeddings",
					 job_arguments="",
					 wait_for_completion=True)
					 

launch_compute_node_embeddings = HopsworksLaunchOperator(dag=dag,
					 project_name="amlsim",
					 task_id="launch_compute_node_embeddings",
					 job_name="compute_node_embeddings",
					 job_arguments="",
					 wait_for_completion=True)
					 

launch_predict_node_embeddings_and_ingest_to_fs = HopsworksLaunchOperator(dag=dag,
					 project_name="amlsim",
					 task_id="launch_predict_node_embeddings_and_ingest_to_fs",
					 job_name="predict_node_embeddings_and_ingest_to_fs",
					 job_arguments="",
					 wait_for_completion=True)
					 

launch_maggy_adversarial_aml = HopsworksLaunchOperator(dag=dag,
					 project_name="amlsim",
					 task_id="launch_maggy_adversarial_aml",
					 job_name="maggy_adversarial_aml",
					 job_arguments="",
					 wait_for_completion=True)
					 

launch_train_adversarial_aml = HopsworksLaunchOperator(dag=dag,
					 project_name="amlsim",
					 task_id="launch_train_adversarial_aml",
					 job_name="train_adversarial_aml",
					 job_arguments="",
					 wait_for_completion=True)
					 

launch_aml_model_server = HopsworksLaunchOperator(dag=dag,
					 project_name="amlsim",
					 task_id="launch_aml_model_server",
					 job_name="aml_model_server",
					 job_arguments="",
					 wait_for_completion=True)
					 


launch_prep_training_dataset_for_embeddings.set_upstream(launch_transaction_feature_engineering_ingestion)
launch_maggy_node_embeddings.set_upstream(launch_prep_training_dataset_for_embeddings)
launch_compute_node_embeddings.set_upstream(launch_maggy_node_embeddings)
launch_predict_node_embeddings_and_ingest_to_fs.set_upstream(launch_compute_node_embeddings)
launch_maggy_adversarial_aml.set_upstream(launch_predict_node_embeddings_and_ingest_to_fs)
launch_train_adversarial_aml.set_upstream(launch_maggy_adversarial_aml)
launch_aml_model_server.set_upstream(launch_train_adversarial_aml)

