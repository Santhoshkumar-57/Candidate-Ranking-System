from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import CandidateInput
from app.stage_1_skills import evaluate_semantic_skills
from typing import List

app = FastAPI(title="Talent Context Ranker API")

# Allow your React local dev server (Vite) to access your FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/rank/stage1")
async def rank_stage_1(
    job_description: str = Body(..., embed=True),
    candidates: List[CandidateInput] = Body(...)
):
    results = []
    
    for candidate in candidates:
        # Calculate Phase 1 score using local math & embeddings
        semantic_score = evaluate_semantic_skills(job_description, candidate.skills)
        
        results.append({
            "candidate_id": candidate.candidate_id,
            "stage_1_semantic_score": round(semantic_score, 2)
        })
        
    # Sort candidates dynamically from highest match to lowest match
    results.sort(key=lambda x: x["stage_1_semantic_score"], reverse=True)
    
    return {
        "status": "success",
        "total_processed": len(candidates),
        "rankings": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)