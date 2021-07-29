## NOTES:
## Example Airflow DAG for running a unit test job
## against a feature group in the Feature Store and
## fetch the validation result. If the validation was
## unsuccessful then it will fail the rest of the pipeline
##
## It assumes that you have already populated your Feature Store
## and composed some validation rules. You can run the
## Feature Store tour and follow the instructions on the
## documentation
## https://hopsworks.readthedocs.io/en/latest/user_guide/hopsworks/featurestore.html

# https://forum.astronomer.io/t/airflow-wait-for-previous-dag-run-to-complete/676



from datetime import datetime, timedelta

from airflow import DAG
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException

from hopsworks_plugin.sensors.hopsworks_sensor import HopsworksBaseSensor
from hopsworks_plugin.operators.hopsworks_operator import HopsworksLaunchOperator


class FeatureGroupCommitSensor(HopsworksBaseSensor):
    @apply_defaults
    def __init__(
            self,
            xcom_task_id_key,
            job_name=None,
            project_name=None,
            project_id=None,
            hopsworks_conn_id='hopsworks_default',
            poke_interval=10,
            timeout=3600,
            *args,
            **kwargs):
        super(FeatureGroupCommitSensor, self).__init__(hopsworks_conn_id,
                                                       job_name,
                                                       project_id,
                                                       project_name,
                                                       *args,
                                                       **kwargs)

        self.BASE_API = "hopsworks-api/api"

        self.xcom_task_id_key = xcom_task_id_key
        self.hook = self._get_hook()
        if project_id is None:
            self.project_id, _ = self.hook._get_project_info(project_id, project_name)
        else:
            self.project_id = project_id

        self.feature_store_id = self.hook.get_feature_store_id_by_name(project_name + "_featurestore")

        self.transactions_fg_id = self.get_feature_group_id_by_name_and_version(self.feature_store_id,
                                                                                "transactions_fg", 1)
        self.alert_transactions_fg_id = self.get_feature_group_id_by_name_and_version(self.feature_store_id,
                                                                                      "alert_transactions_fg", 1)
        self.party_fg_id = self.get_feature_group_id_by_name_and_version(self.feature_store_id, "party_fg", 1)

    def get_feature_group_latest_commit(self, feature_group_id):
        feature_group_commits = \
            ("GET", self.BASE_API + "/project/{project_id}/featurestores/{feature_store_id}/featuregroups/"
                                    "{feature_group_id}/commits?sort_by=committed_on:desc&offset=0")
        method, endpoint = feature_group_commits
        endpoint = endpoint.format(project_id=self.project_id, feature_store_id=self.feature_store_id,
                                   feature_group_id=feature_group_id)
        commit_metadata = self.hook._do_api_call(method, endpoint)

        items = commit_metadata["items"]
        # get latest available commit for this feature group
        current_max_commit_time = max(items, key=lambda x: x['commitTime'])['commitTime']

    def get_feature_group_id_by_name_and_version(self, feature_store_id, feature_group_name, feature_group_version):
        feature_group_id_by_name_and_version = \
            ("GET", self.BASE_API +
             "/project/{project_id}/featurestores/{feature_store_id}/featuregroups?version={feature_group_version}")
        method, endpoint = feature_group_id_by_name_and_version
        endpoint = endpoint.format(project_id=self.project_id, feature_store_id=feature_store_id,
                                   feature_group_version=feature_group_version)
        response = self.hook._do_api_call(method, endpoint)
        for fg in response:
            if 'name' in fg:
                if fg['name'] == feature_group_name:
                    feature_group_id = fg['id']
                    break
        if not feature_group_id:
            raise AirflowException("Could not find ID for feature group {}".format(feature_group_name))
        return feature_group_id

    def poke(self, context):
        task_instance = context['task_instance']

        # get latest available commit for this feature group
        current_max_commit_time = {}
        current_max_commit_time["transactions_fg"] = self.get_feature_group_latest_commit(self.transactions_fg_id)
        current_max_commit_time["alert_transactions_fg"] = self.get_feature_group_latest_commit(
            self.alert_transactions_fg_id)
        current_max_commit_time["party_fg"] = self.get_feature_group_latest_commit(self.party_fg_id)

        previous_max_commit_time = {}
        previous_max_commit_time["transactions_fg"] = 0
        previous_max_commit_time["alert_transactions_fg"] = 0
        previous_max_commit_time["party_fg"] = 0

        one_of_the_fg_updated = False
        if task_instance.previous_ti is not None and task_instance.previous_ti.state == "success":
            previous_max_commit_time = task_instance.previous_ti.xcom_pull(self.task_id, key=self.xcom_task_id_key)
            for fg in current_max_commit_time:
                if current_max_commit_time[fg] > previous_max_commit_time[fg]:
                    one_of_the_fg_updated = True
                    break

        if one_of_the_fg_updated:
            task_instance.xcom_push(self.xcom_task_id_key, current_max_commit_time)
            return True
        else:
            return False


# Username in Hopsworks
# Click on Account from the top right drop-down menu
DAG_OWNER = 'meb10180'

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
    dag_id="retrain_on_new_fg_commit_4",
    default_args=args,

    catchup=False,
    max_active_runs=1,
    # Run the DAG only one time
    # It can take Cron like expressions
    # E.x. run every 30 minutes: */30 * * * *
    schedule_interval="*/2 * * * *"
)

sensor_task = FeatureGroupCommitSensor(
    xcom_task_id_key='sensors_minute',
    job_name="sensorn_job_name",
    project_name=PROJECT_NAME,
    dag=dag,
    task_id='my_sensor_task'
)

launch_1_transaction_feature_engineering_ingestion = HopsworksLaunchOperator(dag=dag,
                                                                             project_name="amlsim",
                                                                             task_id="launch_1_transaction_feature_engineering_ingestion",
                                                                             job_name="1_transaction_feature_engineering_ingestion",
                                                                             job_arguments="",
                                                                             wait_for_completion=True)

launch_2_prep_training_dataset_for_embeddings = HopsworksLaunchOperator(dag=dag,
                                                                        project_name="amlsim",
                                                                        task_id="launch_2_prep_training_dataset_for_embeddings",
                                                                        job_name="2_prep_training_dataset_for_embeddings",
                                                                        job_arguments="",
                                                                        wait_for_completion=True)

launch_3_maggy_node_embeddings = HopsworksLaunchOperator(dag=dag,
                                                         project_name="amlsim",
                                                         task_id="launch_3_maggy_node_embeddings",
                                                         job_name="3_maggy_node_embeddings",
                                                         job_arguments="",
                                                         wait_for_completion=True)

launch_4_compute_node_embeddings = HopsworksLaunchOperator(dag=dag,
                                                           project_name="amlsim",
                                                           task_id="launch_4_compute_node_embeddings",
                                                           job_name="4_compute_node_embeddings",
                                                           job_arguments="",
                                                           wait_for_completion=True)

launch_5_predict_and_create_node_embeddings_fg = HopsworksLaunchOperator(dag=dag,
                                                                         project_name="amlsim",
                                                                         task_id="launch_5_predict_and_create_node_embeddings_fg",
                                                                         job_name="5_predict_and_create_node_embeddings_fg",
                                                                         job_arguments="",
                                                                         wait_for_completion=True)

launch_6_create_node_embeddings_td = HopsworksLaunchOperator(dag=dag,
                                                             project_name="amlsim",
                                                             task_id="launch_6_create_node_embeddings_td",
                                                             job_name="6_create_node_embeddings_td",
                                                             job_arguments="",
                                                             wait_for_completion=True)

launch_7_maggy_adversarial_aml = HopsworksLaunchOperator(dag=dag,
                                                         project_name="amlsim",
                                                         task_id="launch_7_maggy_adversarial_aml",
                                                         job_name="7_maggy_adversarial_aml",
                                                         job_arguments="",
                                                         wait_for_completion=True)

launch_8_train_adversarial_aml = HopsworksLaunchOperator(dag=dag,
                                                         project_name="amlsim",
                                                         task_id="launch_8_train_adversarial_aml",
                                                         job_name="8_train_adversarial_aml",
                                                         job_arguments="",
                                                         wait_for_completion=True)

launch_9_aml_model_server = HopsworksLaunchOperator(dag=dag,
                                                    project_name="amlsim",
                                                    task_id="launch_9_aml_model_server",
                                                    job_name="9_aml_model_server",
                                                    job_arguments="",
                                                    wait_for_completion=True)

launch_1_transaction_feature_engineering_ingestion.set_upstream(sensor_task)
launch_2_prep_training_dataset_for_embeddings.set_upstream(launch_1_transaction_feature_engineering_ingestion)
launch_3_maggy_node_embeddings.set_upstream(launch_2_prep_training_dataset_for_embeddings)
launch_4_compute_node_embeddings.set_upstream(launch_3_maggy_node_embeddings)
launch_5_predict_and_create_node_embeddings_fg.set_upstream(launch_4_compute_node_embeddings)
launch_6_create_node_embeddings_td.set_upstream(launch_5_predict_and_create_node_embeddings_fg)
launch_7_maggy_adversarial_aml.set_upstream(launch_6_create_node_embeddings_td)
launch_8_train_adversarial_aml.set_upstream(launch_7_maggy_adversarial_aml)
launch_9_aml_model_server.set_upstream(launch_8_train_adversarial_aml)
