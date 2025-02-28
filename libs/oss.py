import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

ENDPOINT = 'oss-cn-wuhan-lr.aliyuncs.com'
REGION = 'cn-wuhan-lr'
BUCKET_NAME = 'llmplayground'

class OSSService:
    def __init__(self):
        auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
        self.bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME, region=REGION)

    def put(self, object_name: str, content: str):
        self.bucket.put_object(object_name, content)
        return self.bucket.sign_url('GET', object_name, 600, slash_safe=True)