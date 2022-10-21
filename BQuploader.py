from google.cloud import bigquery
import logging

from constants import GOOGLE_APPLICATION_CREDENTIALS


class BQuploader:
    
    def __init__(self, BQ_project_id, dataset_id):
        '''In the future use the bigquery_client module, which is not yet ready to use'''
        self.BQ_project_id = BQ_project_id
        self.dataset_id = dataset_id
        self.BQclient = bigquery.Client()
        self.BQdataset = self.BQclient.dataset(dataset_id)
        self.BQjob_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.CSV,
                skip_leading_rows=1, 
                autodetect=True,
                write_disposition = bigquery.WriteDisposition.WRITE_APPEND,
                max_bad_records = 5,
            )
    
    def get_config(self):
        return self.BQclient, self.BQdataset, self.BQjob_config

    def load_local_file_to_table(self, file_name, table_name):
        ''' Upload local file to BiQuery '''
        try:
            file_object = open(file_name,'rb')
            table_ref = self.BQdataset.table(table_name)
            load_job  = self.BQclient.load_table_from_file(file_object, table_ref, job_config=self.BQjob_config)
            load_job.result()
        except Exception as e:
            for e in load_job.errors:
                logging.error('%s', f'While loading to the table, errors were found : {e["message"]}')
            raise Exception(f'Upload failed for table {table_name}') 


    def create_table(self, table_name ):
        '''Creates a table in BigQuery named table_name, with schema set 
        accordingly if the table is for storing campaigns'''
        schema = [
            bigquery.SchemaField("field1", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("field2",  "STRING", mode="NULLABLE")]
        table = bigquery.Table(self.BQ_project_id + '.' + self.dataset_id + '.' + table_name, schema=schema)
        # can also create tables without schema, then schema will be set at first upload
        table = self.BQclient.create_table(table)  # Make an API request.



# use as follows:
# from BQuploader import BQuploader
# BQ_project_id = 'vertical-federated-learning'
# dataset_id = 'experiment_data'
# bq_uploader = BQuploader(BQ_project_id, dataset_id)
# bq_uploader.load_local_file_to_table(file_name, table_name)


