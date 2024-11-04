import time
import logging
from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from api.create_clusters import create_clusters
from api.classify_record import classify_record
from api.multi_user_analysis import multi_user_analysis
from api.user_connections import user_connections

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Include the routers
app.include_router(create_clusters)
app.include_router(classify_record)
app.include_router(multi_user_analysis)
app.include_router(user_connections)

# Middleware to log the execution time of each request
@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.time()  # Record the start time
    response = await call_next(request)
    process_time = time.time() - start_time  # Calculate the time taken
    logger.info(f"Endpoint '{request.url.path}' took {process_time:.2f} seconds.")
    response.headers["X-Process-Time"] = f"{process_time:.2f} seconds"  # Optional: add timing info to headers
    return response

# Define OpenAPI specification
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="User Closeness Analysis API",
        version="1.0.0",
        description="API for analyzing user connections based on KNN vectors and calculating closeness metrics.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Assign the custom OpenAPI schema to the FastAPI app
app.openapi = custom_openapi

# Serve the Swagger JSON on /docs/openapi.json
@app.get("/docs/openapi.json", include_in_schema=False)
async def get_openapi_json():
    return app.openapi()
