from datetime import datetime,timedelta , date 

from airflow import models,DAG 

from airflow.contrib.operators.dataproc_operator import DataprocClusterCreateOperator,DataProcPySparkOperator,DataprocClusterDeleteOperator

from airflow.contrib.operators.gcs_to_bq import GoogleCloudStorageToBigQueryOperator

from airflow.operators import BashOperator 

from airflow.models import *

from airflow.utils.trigger_rule import TriggerRule


current_date = str(date.today())

BUCKET = "gs://naaru-etl"

BUCKET1 = "naaru-etl"

PROJECT_ID = "peppy-nation-295917"

PYSPARK_JOB = BUCKET + "/spark-job/spark_etl_job.py"

DEFAULT_DAG_ARGS = {
    'owner':"airflow",
    'depends_on_past' : False,
    "start_date":datetime.utcnow(),
    "email_on_failure":False,
    "email_on_retry":False,
    "retries": 1,
    "retry_delay":timedelta(minutes=5),
    "project_id":PROJECT_ID,
    "scheduled_interval":"30 2 * * *"
}

with DAG("spark_etl",default_args=DEFAULT_DAG_ARGS) as dag : 

    create_cluster = DataprocClusterCreateOperator(

        task_id ="create_dataproc_cluster",
        cluster_name="ephemeral-spark-cluster-{{ds_nodash}}",
        master_machine_type="n1-standard-1",
        worker_machine_type="n1-standard-2",
        num_workers=2,
        region="asia-east1",
        zone ="asia-east1-a"
    )

    submit_pyspark = DataProcPySparkOperator(
        task_id = "run_pyspark_etl",
        main = PYSPARK_JOB,
        cluster_name="ephemeral-spark-cluster-{{ds_nodash}}",
        region="asia-east1"
    )

    bq_load_results = GoogleCloudStorageToBigQueryOperator(

        task_id = "bq_load_results",
        bucket=BUCKET1,
        source_objects=["transformed-FIFA-data/"+current_date+"_results/part-*"],
        destination_project_dataset_table=PROJECT_ID+".data_analysis.results",
        schema_fields = [{
            "mode": "REQUIRED",
            "name": "date_id",
            "type": "INTEGER"
            },
            {
            "mode": "REQUIRED",
            "name": "date",
            "type": "DATE"
            },
            {
            "mode": "REQUIRED",
            "name": "game_id",
            "type": "INTEGER"
            },
            {
            "mode": "REQUIRED",
            "name": "home_team",
            "type": "STRING"
            },
            {
            "mode": "REQUIRED",
            "name": "away_team",
            "type": "STRING"
            },
            {
            "mode": "REQUIRED",
            "name": "home_score",
            "type": "INTEGER"
            },
            {
            "mode": "REQUIRED",
            "name": "away_score",
            "type": "INTEGER"
            },
            {
            "mode": "REQUIRED",
            "name": "tournament",
            "type": "STRING"
            },
            {
            "mode": "REQUIRED",
            "name": "city",
            "type": "STRING"
            },
            {
            "mode": "REQUIRED",
            "name": "country",
            "type": "STRING"
            },
            {
            "mode": "REQUIRED",
            "name": "neutral",
            "type": "BOOLEAN"
            }
        ],
        autodetect = False,
        source_format="CSV",
        create_disposition="CREATE_IF_NEEDED",
        skip_leading_rows=1,
        write_disposition="WRITE_APPEND",
        field_delimiter=',',
        max_bad_records=0
    )

    bq_load_facts = GoogleCloudStorageToBigQueryOperator(

        task_id = "bq_load_facts",
        bucket=BUCKET1,
        source_objects=["transformed-FIFA-data/star-schema/fact/"+str(current_date)+"_results/part-*"],
        destination_project_dataset_table=PROJECT_ID+".data_analysis.facts",
        schema_fields = [{
            "mode": "REQUIRED",
            "name": "date_id",
            "type": "INTEGER"
            },
            {
            "mode": "REQUIRED",
            "name": "game_id",
            "type": "INTEGER"
            },
            {
            "mode": "REQUIRED",
            "name": "home_score",
            "type": "INTEGER"
            },
            {
            "mode": "REQUIRED",
            "name": "away_score",
            "type": "INTEGER"
            }
        ],
        autodetect = False,
        source_format="CSV",
        create_disposition="CREATE_IF_NEEDED",
        skip_leading_rows=1,
        write_disposition="WRITE_APPEND",
        field_delimiter=',',
        max_bad_records=0
    )

    bq_load_dim1 = GoogleCloudStorageToBigQueryOperator(

        task_id = "bq_load_date_dim",
        bucket=BUCKET1,
        source_objects=["transformed-FIFA-data/star-schema/dim1/"+str(current_date)+"_results/part-*"],
        destination_project_dataset_table=PROJECT_ID+".data_analysis.date",
        schema_fields = [{
            "mode": "REQUIRED",
            "name": "date_id",
            "type": "INTEGER"
            },
            {
            "mode": "REQUIRED",
            "name": "date",
            "type": "DATE"
            }
        ],
        autodetect = False,
        source_format="CSV",
        create_disposition="CREATE_IF_NEEDED",
        skip_leading_rows=1,
        write_disposition="WRITE_APPEND",
        field_delimiter=',',
        max_bad_records=0
    )

    bq_load_dim2 = GoogleCloudStorageToBigQueryOperator(

        task_id = "bq_load_game_dim",
        bucket=BUCKET1,
        source_objects=["transformed-FIFA-data/star-schema/dim2/"+str(current_date)+"_results/part-*"],
        destination_project_dataset_table=PROJECT_ID+".data_analysis.game",
        schema_fields = [{
            "mode": "REQUIRED",
            "name": "game_id",
            "type": "INTEGER"
            },
            {
            "mode": "REQUIRED",
            "name": "home_team",
            "type": "STRING"
            },
            {
            "mode": "REQUIRED",
            "name": "away_team",
            "type": "STRING"
            },
            {
            "mode": "REQUIRED",
            "name": "tournament",
            "type": "STRING"
            },
            {
            "mode": "REQUIRED",
            "name": "city",
            "type": "STRING"
            },
            {
            "mode": "REQUIRED",
            "name": "country",
            "type": "STRING"
            },
            {
            "mode": "REQUIRED",
            "name": "neutral",
            "type": "BOOLEAN"
            }
        ],
        autodetect = False,
        source_format="CSV",
        create_disposition="CREATE_IF_NEEDED",
        skip_leading_rows=1,
        write_disposition="WRITE_APPEND",
        field_delimiter=',',
        max_bad_records=0
    )

    delete_cluster = DataprocClusterDeleteOperator(
        task_id = 'delete_dataproc_cluster',
        cluster_name = 'ephemeral-spark-cluster-{{ds_nodash}}',
        region = 'asia-east1',
        trigger_rule = TriggerRule.ALL_DONE
    )
    

    delete_transformed_files = BashOperator(
        task_id = 'delete_transformed_files',
        bash_command = 'gsutil -m rm -r ' +BUCKET+'/transformed-FIFA-data/*'
    )

    create_cluster.dag = dag

    create_cluster.set_downstream(submit_pyspark)

    submit_pyspark.set_downstream([bq_load_results, bq_load_facts, bq_load_dim1, bq_load_dim2, delete_cluster])

    delete_cluster.set_downstream(delete_transformed_files)
