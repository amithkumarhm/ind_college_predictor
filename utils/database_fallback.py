"""
Fallback database implementation for when MongoDB is not available
"""
from bson import ObjectId
from datetime import datetime
import random


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

    def list_collection_names(self):
        return []


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
        """Mock database command for health checks"""
        if command_name == 'ping':
            return {'ok': 1.0}
        return {'ok': 0.0}