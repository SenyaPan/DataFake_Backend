from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from inference.src.router import router

app = FastAPI(
    title="Inference",
    description="This API is used for analysis of face frames by neuron-network.",
    summary="API of auxiliary microservice for the DatAFake work.",
    version="1.0",
    root_path="/api/v1"
)

origins = [
    "http://localhost:5000"
]

app.add_middleware(  # i think this block should be rewritten
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
