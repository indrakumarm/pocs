from fastapi import FastAPI
import boto3

app = FastAPI()
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("BUCKET_ENDPOINT"),
    aws_access_key_id=os.getenv("BUCKET_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("BUCKET_SECRET_ACCESS_KEY"),
    region_name=os.getenv("BUCKET_REGION")
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Railway deployment demo!"}

@app.get("/hello")
def hello(name: str = "User"):
    return {"message": f"Hello {name}, Railway deployed your app successfully!"}

@app.get("/check")
def hello(name: str = "User"):
    return {"message": f"Hello {name}, Railway updated your app successfully!"}

@app.route("/upload")
def upload_sample():
    now = datetime.utcnow().isoformat()
    data = f"Hello from Railway! Time is {now}"

    s3.put_object(
        Bucket=os.getenv("BUCKET_NAME"),
        Key="test/sample.txt",
        Body=data.encode("utf-8")
    )
    return f"Uploaded sample.txt successfully at {now}"

@app.route("/read")
def read_sample():
    obj = s3.get_object(Bucket=os.getenv("BUCKET_NAME"), Key="test/sample.txt")
    return obj["Body"].read().decode()
