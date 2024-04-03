from fastapi import FastAPI
from fastapi import Request, Response
app = FastAPI()

@app.post("/api/camera/detected")
async def add_camera(request: Request):
    body = await request.json()
    dict_face = {"age": body["age"], "emotion": body["emotion"], 'gender': body["gender"]}
    print(dict_face)
    return Response(status_code=200, content="OK")

@app.post("/api/camera/logs")
async def add_camera(request: Request):
    body = await request.json()
    print(body["tracking_time"])
    return Response(status_code=200, content="OK")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=5001)