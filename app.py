from fastapi import FastAPI
from api.create_clusters import create_clusters
from api.classify_record import classify_record

# Initialize FastAPI app
app = FastAPI()

# Include the routers
app.include_router(create_clusters)
app.include_router(classify_record)
