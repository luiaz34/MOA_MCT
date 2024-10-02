from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel 
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dto import DataDTO
from app import executeProgram

app = FastAPI()

class InputPayload(BaseModel):
    userPrompt: str
    proposerCount: int
    aggregatorCount: int
    ratingAgent: bool
    selectorAgent: bool
    useMCT: bool = False
    maxChildren: int = 3
    iteration: int = 2

@app.post("/execute")
async def mma_endpoint_method(payload: InputPayload):
    try:
        data_dto = DataDTO(
            user_prompt=payload.userPrompt,
            proposer_count=payload.proposerCount,
            aggregator_count=payload.aggregatorCount,
            rating_agent=payload.ratingAgent,
            selector_agent=payload.selectorAgent,
            use_mct=payload.useMCT,
            max_children=payload.maxChildren,
            iteration=payload.iteration
        )
        result = await executeProgram(data_dto)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)