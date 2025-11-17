import re
from datetime import datetime


def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_marks(marks):
    """Validate marks percentage"""
    try:
        marks_float = float(marks)
        return 0 <= marks_float <= 100
    except ValueError:
        return False


def validate_rank(rank):
    """Validate rank"""
    try:
        rank_int = int(rank)
        return rank_int > 0
    except ValueError:
        return False


def get_current_year():
    """Get current academic year"""
    return datetime.now().year


def calculate_probability(student_rank, cutoff_rank, student_marks, cutoff_marks):
    """Calculate admission probability"""
    rank_prob = max(0, min(100, (cutoff_rank - student_rank) / cutoff_rank * 100))
    marks_prob = max(0, min(100, (student_marks - cutoff_marks) / (100 - cutoff_marks) * 100))

    # Weighted average (rank has 70% weight, marks 30%)
    final_prob = (rank_prob * 0.7) + (marks_prob * 0.3)
    return round(final_prob, 2)