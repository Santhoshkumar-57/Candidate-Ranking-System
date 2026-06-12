import numpy as np
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
from app.schemas import SkillModel
from typing import List

# Load the 100MB model locally. It will download to your MSI Cyborg once 
# during the first boot and run completely offline after that.
print("Loading local semantic embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculates the mathematical angle similarity between two vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def evaluate_semantic_skills(job_description_text: str, candidate_skills: List[SkillModel]) -> float:
    if not candidate_skills:
        return 0.0

    # 1. Turn the Job Description text into a baseline requirement vector
    jd_vector = embedding_model.encode(job_description_text)
    
    # 2. Define our dataset multipliers for candidate experience
    proficiency_map = {
        "expert": 1.2,
        "advanced": 1.1,
        "intermediate": 1.0,
        "beginner": 0.7
    }
    
    all_skill_scores = []
    
    # 3. Process every structured skill entry from the JSON dataset
    for skill in candidate_skills:
        # Generate a vector for the skill name (e.g., "Next.js")
        skill_vector = embedding_model.encode(skill.name)
        
        # Check semantic similarity against the JD requirements
        base_similarity = calculate_cosine_similarity(jd_vector, skill_vector)
        
        # Grab proficiency adjustment factor
        p_multiplier = proficiency_map.get(skill.proficiency.lower(), 1.0)
        
        # Simple math to factor in experience duration (capped at 1.5x max)
        # 24 months acts as a solid benchmark baseline (1.0x)
        d_multiplier = min(1.5, max(0.5, skill.duration_months / 24.0))
        
        # Combine the semantic score with data signals
        final_skill_score = base_similarity * p_multiplier * d_multiplier
        all_skill_scores.append(final_skill_score)
        
    # 4. Sort and average out the top 5 highest matching skills to get the final Phase 1 Score
    all_skill_scores.sort(reverse=True)
    top_matches = all_skill_scores[:5]
    
    avg_score = sum(top_matches) / len(top_matches) if top_matches else 0.0
    
    # Normalize score between 0 and 100 for your UI charts
    return min(100.0, max(0.0, avg_score * 100))