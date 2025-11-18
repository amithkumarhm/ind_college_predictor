from pymongo import MongoClient
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import certifi

load_dotenv()

_client = None
_db = None


def get_mongo_client():
    """Get MongoDB client with proper SSL configuration for Python 3.8.18"""
    try:
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')

        # Check if we're in production (Render) or local development
        is_production = os.getenv('RENDER', False) or 'mongodb+srv' in mongo_uri

        if is_production:
            # Production configuration with SSL for Python 3.8.18 compatibility
            client = MongoClient(
                mongo_uri,
                tls=True,
                tlsAllowInvalidCertificates=True,  # More permissive for compatibility
                tlsCAFile=certifi.where(),
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                serverSelectionTimeoutMS=30000,
                retryWrites=True,
                w='majority',
                maxPoolSize=50,
                minPoolSize=10
            )
            print("✅ MongoDB connected with SSL in production mode")
        else:
            # Local development without SSL
            client = MongoClient(mongo_uri)
            print("✅ MongoDB connected in development mode")

        # Test connection
        client.admin.command('ismaster')
        return client

    except Exception as e:
        print(f"❌ MongoDB connection error: {e}")
        return None


def get_db():
    global _db
    if _db is None:
        client = get_mongo_client()
        if client:
            _db = client.college_predictor
            print(f"✅ Connected to MongoDB database")
        else:
            # Create a dummy database object to prevent crashes
            class DummyDB:
                def __getattr__(self, name):
                    return DummyCollection()

            class DummyCollection:
                def find_one(self, *args, **kwargs): return None

                def find(self, *args, **kwargs): return []

                def insert_one(self, *args, **kwargs): return DummyResult()

                def insert_many(self, *args, **kwargs): return DummyResult()

                def update_one(self, *args, **kwargs): return None

                def delete_one(self, *args, **kwargs): return None

                def count_documents(self, *args, **kwargs): return 0

                def list_collection_names(self): return []

                def create_collection(self, *args, **kwargs): return None

            class DummyResult:
                inserted_id = None

            _db = DummyDB()
            print("⚠️  Using dummy database - no persistence")
    return _db


def init_db():
    """Initialize database with sample data"""
    db = get_db()

    # Only proceed if we have a real database connection
    if hasattr(db, 'list_collection_names'):
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
    else:
        print("⚠️  Database not available - running in fallback mode")