import json
from pathlib import Path
from time import sleep
from prefect_aws import S3Bucket, AwsCredentials



def create_aws_creds_block(key_data):
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id=key_data['aws']["access_key"], aws_secret_access_key=key_data['aws']["secret_access_key"]
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name="my-first-bucket-abc", credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="s3-bucket-example", overwrite=True)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve()
    # print(current_dir)
    key_file = current_dir.parents[2] / "key/key.json"
    with open(key_file) as in_file:
        key_data = json.load(in_file)
    print(key_file, key_data)

    create_aws_creds_block(key_data)
    sleep(5)
    create_s3_bucket_block()

    print()