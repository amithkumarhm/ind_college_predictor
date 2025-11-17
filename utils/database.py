from pymongo import MongoClient
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

load_dotenv()

_client = None
_db = None


def get_db():
    global _db
    if _db is None:
        # Use MONGO_URI (consistent with app.py) instead of MONGODB_URI
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        client = MongoClient(mongo_uri)
        _db = client.college_predictor
        print(f"✅ Connected to MongoDB: {mongo_uri}")
    return _db


def init_db():
    """Initialize database with sample data"""
    db = get_db()

    # Create collections if they don't exist
    if 'users' not in db.list_collection_names():
        db.create_collection('users')
        print("✅ Users collection created")

    if 'colleges' not in db.list_collection_names():
        db.create_collection('colleges')
        print("✅ Colleges collection created")

    if 'email_verifications' not in db.list_collection_names():
        db.create_collection('email_verifications')
        print("✅ Email verifications collection created")
        # Create TTL index for email verifications (auto-delete after 10 minutes)
        db.email_verifications.create_index("expires_at", expireAfterSeconds=600)

    # Load CSV data
    try:
        # Check if colleges collection is empty
        if db.colleges.count_documents({}) == 0:
            # Load engineering colleges
            eng_df = pd.read_csv('data/engineering_colleges.csv')
            engineering_colleges = eng_df.to_dict('records')
            db.colleges.insert_many(engineering_colleges)
            print(f"✅ Inserted {len(engineering_colleges)} engineering colleges")

            # Load medical colleges
            med_df = pd.read_csv('data/medical_colleges.csv')
            medical_colleges = med_df.to_dict('records')
            db.colleges.insert_many(medical_colleges)
            print(f"✅ Inserted {len(medical_colleges)} medical colleges")

        print("✅ Database initialized successfully!")

    except Exception as e:
        print(f"❌ Error loading CSV data: {e}")
        # Create minimal sample data as fallback
        if db.colleges.count_documents({}) == 0:
            sample_data = [
                {
                    'name': 'IIT Bombay',
                    'state': 'Maharashtra',
                    'exam_type': 'JEE Main',
                    'category': 'General',
                    'cutoff_rank': 100,
                    'marks_cutoff': 85,
                    'website': 'https://www.iitb.ac.in',
                    'type': 'Engineering'
                },
                {
                    'name': 'AIIMS Delhi',
                    'state': 'Delhi',
                    'exam_type': 'NEET',
                    'category': 'General',
                    'cutoff_rank': 100,
                    'marks_cutoff': 85,
                    'website': 'https://www.aiims.edu',
                    'type': 'Medical'
                }
            ]
            db.colleges.insert_many(sample_data)
            print("✅ Inserted sample data as fallback")