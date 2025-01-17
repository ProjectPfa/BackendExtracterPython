import uvicorn  # type: ignore
from fastapi import FastAPI, File, UploadFile, HTTPException  # type: ignore
from starlette.responses import RedirectResponse  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from serve.serve_model import *

app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>"""

app = FastAPI(title="Tensorflow FastAPI Start Pack", description=app_desc)

# Ajout du middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines. À personnaliser en fonction de tes besoins de sécurité
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes HTTP
    allow_headers=["*"],  # Permet tous les en-têtes
)

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    # Vérification de l'extension du fichier
    extension = file.filename.split(".")[-1].lower()
    if extension not in ("jpg", "jpeg", "png"):
        raise HTTPException(status_code=400, detail="Image must be in jpg, jpeg, or png format!")

    try:
        image = read_image_file(await file.read())
        prediction = predict(image)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
