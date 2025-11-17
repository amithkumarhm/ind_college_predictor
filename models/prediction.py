import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.database import get_db
except ImportError:
    # Fallback if utils is not available
    get_db = None


def load_model():
    """Load trained model and encoders"""
    try:
        model = pickle.load(open('models/saved/model.pkl', 'rb'))
        le_state = pickle.load(open('models/saved/le_state.pkl', 'rb'))
        le_exam = pickle.load(open('models/saved/le_exam.pkl', 'rb'))
        le_category = pickle.load(open('models/saved/le_category.pkl', 'rb'))
        le_type = pickle.load(open('models/saved/le_type.pkl', 'rb'))
        return model, le_state, le_exam, le_category, le_type
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None, None, None


def predict_colleges(field, exam_type, rank, marks_12th, category, state, db=None):
    """Enhanced college prediction with better ranking logic"""
    print(
        f"DEBUG: Predicting colleges - Field: {field}, Exam: {exam_type}, Rank: {rank}, Marks: {marks_12th}, Category: {category}, State: {state}")

    # Get database connection if not provided
    if db is None and get_db:
        db = get_db()
    elif db is None:
        print("❌ Database connection not available")
        return []

    # Build query based on inputs
    query = {
        'type': field.capitalize()
    }

    # Add category to query
    if category:
        query['category'] = category

    # Add exam_type to query if not "All India"
    if exam_type != 'All India':
        query['exam_type'] = exam_type

    print(f"DEBUG: Database query: {query}")

    colleges = list(db.colleges.find(query))

    print(f"DEBUG: Found {len(colleges)} colleges matching query")

    suitable_colleges = []
    for college in colleges:
        # Check state filter - if "All India", include all states
        if state == 'All India' or college['state'] == state:
            print(
                f"DEBUG: Considering college: {college['name']} - Cutoff: {college['cutoff_rank']}, Category: {college['category']}")

            # Enhanced probability calculation
            cutoff_rank = college['cutoff_rank']

            # Calculate rank difference percentage
            if rank <= cutoff_rank:
                # Student rank is better than cutoff - high probability
                rank_ratio = (cutoff_rank - rank) / cutoff_rank
                base_probability = 70 + (rank_ratio * 30)
            else:
                # Student rank is worse than cutoff - decreasing probability
                rank_diff = rank - cutoff_rank
                if rank_diff <= 1000:
                    base_probability = 60 - (rank_diff / 1000 * 20)
                elif rank_diff <= 5000:
                    base_probability = 40 - ((rank_diff - 1000) / 4000 * 20)
                else:
                    base_probability = 20 - min(15, (rank_diff - 5000) / 10000 * 15)

            # Adjust based on marks
            marks_cutoff = college['marks_cutoff']
            if marks_12th >= marks_cutoff:
                marks_bonus = min(20, ((marks_12th - marks_cutoff) / 10) * 5)
            else:
                marks_bonus = -10

            # Final probability calculation
            probability = base_probability + marks_bonus

            # Ensure probability is within bounds
            probability = max(5, min(95, probability))

            suitable_colleges.append({
                'name': college['name'],
                'state': college['state'],
                'cutoff_rank': college['cutoff_rank'],
                'marks_cutoff': college['marks_cutoff'],
                'website': college['website'],
                'image_url': college.get('image_url', '/static/images/default-college.jpg'),
                'probability': round(probability, 2)
            })

    # Sort by probability (descending) and return top colleges
    suitable_colleges.sort(key=lambda x: x['probability'], reverse=True)
    print(f"DEBUG: Returning {len(suitable_colleges[:15])} suitable colleges")

    # Debug: Print all found colleges
    for college in suitable_colleges[:5]:
        print(f"DEBUG College: {college['name']} - Probability: {college['probability']}%")

    return suitable_colleges[:15]


# Test function if run directly
if __name__ == '__main__':
    print("Testing prediction model...")

    # Load model to test
    model, le_state, le_exam, le_category, le_type = load_model()
    if model:
        print("✅ Model loaded successfully")
        print(f"Model features: {model.n_features_in_}")
    else:
        print("❌ Failed to load model")

    print("Prediction module loaded successfully")