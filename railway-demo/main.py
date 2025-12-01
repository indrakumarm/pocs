from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Railway deployment demo!"}

@app.get("/hello")
def hello(name: str = "User"):
    return {"message": f"Hello {name}, Railway deployed your app successfully!"}

@app.get("/check")
def hello(name: str = "User"):
    return {"message": f"Hello {name}, Railway updated your app successfully!"}