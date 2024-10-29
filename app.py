from fastapi import FastAPI
from api.create_clusters import create_clusters
from api.classify_record import classify_record
from api.multi_user_analysis import multi_user_analysis
from api.get_user_connections import get_user_connections

# Initialize FastAPI app
app = FastAPI()

# Include the routers
app.include_router(create_clusters)
app.include_router(classify_record)
app.include_router(multi_user_analysis)
app.include_router(get_user_connections)
