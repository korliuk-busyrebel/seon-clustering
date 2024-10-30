from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from api.create_clusters import create_clusters
from api.classify_record import classify_record
from api.multi_user_analysis import multi_user_analysis
from api.user_connections import user_connections

# Initialize FastAPI app
app = FastAPI()

# Include the routers
app.include_router(create_clusters)
app.include_router(classify_record)
app.include_router(multi_user_analysis)
app.include_router(user_connections)

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
