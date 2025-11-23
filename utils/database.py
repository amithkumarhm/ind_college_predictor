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
    """Get MongoDB client with robust configuration and better error handling"""
    try:
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/college-predictor')

        print(f"🔧 Attempting to connect to MongoDB...")
        print(f"🔧 Connection URI: {mongo_uri.split('@')[0]}@***")  # Hide password in logs

        # Check connection type
        if 'mongodb+srv' in mongo_uri:
            # MongoDB Atlas connection
            print("🌐 Connecting to MongoDB Atlas...")
            client_options = {
                'tls': True,
                'tlsAllowInvalidCertificates': False,
                'tlsCAFile': certifi.where(),
                'connectTimeoutMS': 10000,
                'socketTimeoutMS': 10000,
                'serverSelectionTimeoutMS': 10000,
                'retryWrites': True,
                'w': 'majority',
                'maxPoolSize': 50,
                'minPoolSize': 10
            }

            client = MongoClient(mongo_uri, **client_options)

        elif 'localhost' in mongo_uri or '127.0.0.1' in mongo_uri:
            # Local MongoDB connection
            print("💻 Connecting to local MongoDB...")
            client_options = {
                'connectTimeoutMS': 5000,
                'socketTimeoutMS': 5000,
                'serverSelectionTimeoutMS': 5000
            }
            client = MongoClient(mongo_uri, **client_options)

        else:
            # Other MongoDB connection
            print("🔗 Connecting to MongoDB...")
            client = MongoClient(mongo_uri)

        # Test the connection
        print("🔄 Testing MongoDB connection...")
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")

        return client

    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("⚠️  Falling back to in-memory database")
        return None


def get_db():
    """Get database instance with fallback to in-memory database"""
    global _db
    if _db is None:
        client = get_mongo_client()
        if client:
            try:
                _db = client.college_predictor
                print(f"✅ Connected to database: college-predictor")

                # Test database operations
                collections = _db.list_collection_names()
                print(f"📁 Available collections: {collections}")

            except Exception as e:
                print(f"❌ Database connection test failed: {e}")
                _db = create_dummy_db()
        else:
            _db = create_dummy_db()

    return _db


def create_dummy_db():
    """Create a dummy database for fallback operation"""
    print("🔄 Creating in-memory database...")

    class DummyCollection:
        def __init__(self, name):
            self.name = name
            self.data = []
            self._id_counter = 1

        def find_one(self, query=None, **kwargs):
            if not query:
                return None if not self.data else self.data[0]

            for item in self.data:
                match = True
                for key, value in query.items():
                    if item.get(key) != value:
                        match = False
                        break
                if match:
                    return item
            return None

        def find(self, query=None, **kwargs):
            results = []
            if not query:
                return self.data.copy()

            for item in self.data:
                match = True
                for key, value in query.items():
                    if item.get(key) != value:
                        match = False
                        break
                if match:
                    results.append(item)
            return results

        def insert_one(self, document, **kwargs):
            from bson import ObjectId
            doc_id = self._id_counter
            document['_id'] = ObjectId(str(doc_id).zfill(24))
            self.data.append(document)
            self._id_counter += 1
            return DummyResult(document['_id'])

        def insert_many(self, documents, **kwargs):
            results = []
            for doc in documents:
                result = self.insert_one(doc)
                results.append(result)
            return DummyResult([r.inserted_id for r in results])

        def update_one(self, query, update, **kwargs):
            item = self.find_one(query)
            if item and '$set' in update:
                item.update(update['$set'])
            return DummyUpdateResult()

        def delete_one(self, query, **kwargs):
            for i, item in enumerate(self.data):
                match = True
                for key, value in query.items():
                    if item.get(key) != value:
                        match = False
                        break
                if match:
                    del self.data[i]
                    break
            return DummyDeleteResult()

        def delete_many(self, query=None, **kwargs):
            if not query:
                count = len(self.data)
                self.data.clear()
                return DummyDeleteResult(count)

            initial_count = len(self.data)
            self.data = [item for item in self.data if not all(
                item.get(key) == value for key, value in query.items()
            )]
            deleted_count = initial_count - len(self.data)
            return DummyDeleteResult(deleted_count)

        def count_documents(self, query=None, **kwargs):
            return len(self.find(query))

        def create_index(self, field, **kwargs):
            return None

    class DummyResult:
        def __init__(self, inserted_id=None):
            self.inserted_id = inserted_id
            if isinstance(inserted_id, list):
                self.inserted_ids = inserted_id

    class DummyUpdateResult:
        def __init__(self):
            self.matched_count = 1
            self.modified_count = 1

    class DummyDeleteResult:
        def __init__(self, deleted_count=1):
            self.deleted_count = deleted_count

    class DummyDB:
        def __init__(self):
            self.collections = {}

        def __getattr__(self, name):
            if name not in self.collections:
                self.collections[name] = DummyCollection(name)
            return self.collections[name]

        def list_collection_names(self):
            return list(self.collections.keys())

        def create_collection(self, name, **kwargs):
            if name not in self.collections:
                self.collections[name] = DummyCollection(name)
            return self.collections[name]

        def command(self, command_name):
            if command_name == 'ping':
                return {'ok': 1.0}
            return {'ok': 0.0}

    return DummyDB()


def init_db():
    """Initialize database with comprehensive data"""
    db = get_db()

    print("🔄 Initializing database...")

    # Only proceed if we have a real database connection
    if hasattr(db, 'command'):
        try:
            # Test if we have a real MongoDB connection
            db.command('ping')
            print("✅ Real MongoDB database detected")

            # Create collections if they don't exist
            collections_to_create = ['users', 'colleges', 'email_verifications']
            existing_collections = db.list_collection_names()

            for collection_name in collections_to_create:
                if collection_name not in existing_collections:
                    db.create_collection(collection_name)
                    print(f"✅ Created collection: {collection_name}")
                else:
                    print(f"📁 Collection exists: {collection_name}")

            # Load CSV data if colleges collection is empty
            if db.colleges.count_documents({}) == 0:
                print("📥 Loading college data from CSV files...")
                load_college_data_from_csv(db)
            else:
                print("📊 College data already exists in database")

                # Debug counts
                eng_count = db.colleges.count_documents({'type': 'Engineering'})
                med_count = db.colleges.count_documents({'type': 'Medical'})
                bca_count = db.colleges.count_documents({'type': 'BCA'})
                mca_count = db.colleges.count_documents({'type': 'MCA'})

                print(
                    f"📈 College counts - Engineering: {eng_count}, Medical: {med_count}, BCA: {bca_count}, MCA: {mca_count}")

            print("✅ Database initialization completed!")

        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            print("⚠️  Using in-memory database mode")
    else:
        print("🔄 Initializing in-memory database with sample data...")
        load_sample_data(db)


def load_college_data_from_csv(db):
    """Load college data from CSV files"""
    try:
        all_colleges = []

        # CSV file paths
        csv_files = {
            'Engineering': 'data/engineering_colleges.csv',
            'Medical': 'data/medical_colleges.csv',
            'BCA/MCA': 'data/bca_mca_colleges.csv'
        }

        for college_type, file_path in csv_files.items():
            if os.path.exists(file_path):
                print(f"📖 Reading {college_type} data from: {file_path}")
                df = pd.read_csv(file_path)

                for _, row in df.iterrows():
                    college_data = {
                        'college_id': int(row['college_id']),
                        'name': row['name'],
                        'state': row['state'],
                        'exam_type': row['exam_type'],
                        'category': row['category'],
                        'cutoff_rank': int(row['cutoff_rank']),
                        'marks_cutoff': int(row['marks_cutoff']),
                        'website': row['website'],
                        'type': row.get('type', college_type)  # Use type from CSV or fallback
                    }
                    all_colleges.append(college_data)

                print(f"✅ Loaded {len(df)} {college_type} colleges")
            else:
                print(f"⚠️  CSV file not found: {file_path}")

        if all_colleges:
            db.colleges.insert_many(all_colleges)
            print(f"🎉 Successfully inserted {len(all_colleges)} colleges into database")
        else:
            print("⚠️  No college data found in CSV files")
            load_sample_data(db)

    except Exception as e:
        print(f"❌ Error loading CSV data: {e}")
        print("🔄 Loading sample data as fallback...")
        load_sample_data(db)


def load_sample_data(db):
    """Load sample data when CSV files are not available"""
    try:
        sample_colleges = [
            # Engineering samples
            {
                'college_id': 1,
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
                'college_id': 2,
                'name': 'IIT Delhi',
                'state': 'Delhi',
                'exam_type': 'JEE Main',
                'category': 'General',
                'cutoff_rank': 500,
                'marks_cutoff': 80,
                'website': 'https://www.iitd.ac.in',
                'type': 'Engineering'
            },
            # Medical samples
            {
                'college_id': 101,
                'name': 'AIIMS Delhi',
                'state': 'Delhi',
                'exam_type': 'NEET',
                'category': 'General',
                'cutoff_rank': 100,
                'marks_cutoff': 85,
                'website': 'https://www.aiims.edu',
                'type': 'Medical'
            },
            # BCA samples
            {
                'college_id': 201,
                'name': 'Christ University',
                'state': 'Karnataka',
                'exam_type': 'UGCET',
                'category': 'General',
                'cutoff_rank': 14000,
                'marks_cutoff': 67,
                'website': 'https://www.christuniversity.in',
                'type': 'BCA'
            },
            # MCA samples
            {
                'college_id': 202,
                'name': 'Christ University',
                'state': 'Karnataka',
                'exam_type': 'PGCET',
                'category': 'General',
                'cutoff_rank': 7000,
                'marks_cutoff': 74,
                'website': 'https://www.christuniversity.in',
                'type': 'MCA'
            }
        ]

        db.colleges.insert_many(sample_colleges)
        print(f"✅ Loaded {len(sample_colleges)} sample colleges")

    except Exception as e:
        print(f"❌ Error loading sample data: {e}")


# Initialize database when module is imported
if __name__ != "__main__":
    init_db()