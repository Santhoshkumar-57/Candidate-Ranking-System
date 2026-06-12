from pydantic import BaseModel
from typing import List, Optional

class SkillModel(BaseModel):
    name: str
    proficiency: str
    endorsements: int
    duration_months: int

class CandidateInput(BaseModel):
    candidate_id: str
    skills: List[SkillModel]
    # (You can add profile, career_history, redrob_signals here later for Stage 2 & 3)