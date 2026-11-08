"""
================================================================================
  Router  —  Carbon Footprint & ML Risk Classification
================================================================================
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
import io
import pandas as pd

from backend.schemas.carbon import (
    CarbonCalculateRequest,
    CarbonCalculateResponse,
    CarbonActivityInput,
)
from backend.services.carbon_service import carbon_service

router = APIRouter(prefix="/api/carbon", tags=["Carbon Footprint ML Audit"])


@router.post(
    "/calculate",
    response_model=CarbonCalculateResponse,
    summary="Compute carbon footprint & perform KMeans risk clustering from activity metrics",
)
def calculate_carbon_footprint(req: CarbonCalculateRequest) -> CarbonCalculateResponse:
    try:
        return carbon_service.calculate(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Carbon audit error: {str(e)}")


@router.post(
    "/upload",
    response_model=CarbonCalculateResponse,
    summary="Upload enterprise emission CSV/Excel/PDF and execute ML risk classification",
)
async def upload_emission_file(file: UploadFile = File(...)) -> CarbonCalculateResponse:
    try:
        contents = await file.read()
        filename = file.filename.lower()
        activities = []

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        elif filename.endswith(".pdf"):
            try:
                import pymupdf
            except ImportError:
                import fitz as pymupdf
            doc = pymupdf.open(stream=contents, filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
            lines = [line.strip().split(",") for line in text.split("\n") if line.strip()]
            df = pd.DataFrame(lines)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Upload CSV, Excel, or PDF.")

        # Convert dataframe rows to CarbonActivityInput
        # Match columns or text rows
        from carbonfootprintmodel.carbonfootprint import extract_features_and_compute
        extracted_df = extract_features_and_compute(df)

        for _, row in extracted_df.iterrows():
            act_cat = row["Activity_Source"].lower().replace(" ", "_")
            activities.append(
                CarbonActivityInput(
                    category=act_cat,
                    consumption=float(row["Raw_Consumption"]),
                    label=row["Activity_Source"],
                )
            )

        req = CarbonCalculateRequest(activities=activities)
        return carbon_service.calculate(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File audit parsing error: {str(e)}")

