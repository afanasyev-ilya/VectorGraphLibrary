import pymongo
from bson import ObjectId

def connect_to_mongo():
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.demoCollection
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
        return False
    return True


def count_documents(graphs_criteria):
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.performance_data

        return collection.count_documents(graphs_criteria)
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
    return 0


def insert_many(documents):
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.performance_data
        collection.insert_many(documents)
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)


def cursor_to_array(cursor):
    result_list = []
    for res in cursor:
        result_list.append(res)
    return result_list


def distinct(search_criteria, field_name):
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.performance_data
        return cursor_to_array(collection.find(search_criteria).distinct(field_name))
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
        return []


def find(graphs_criteria, projection=None):
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.performance_data
        if projection is None:
            return cursor_to_array(collection.find(graphs_criteria))
        else:
            return cursor_to_array(collection.find(graphs_criteria, projection))
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
        return []


def find_one(graphs_criteria):
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.performance_data
        return collection.find_one(graphs_criteria)
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
        return None

def remove_one(name):
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.performance_data
        collection.remove({'arch_name': name});
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)

def update_many(search_criteria, new_data):
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.performance_data

        collection.update_many(search_criteria, {'$set': new_data}, upsert=False)
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
        return []


def update_perf(search_criteria, new_perf):
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.performance_data

        collection.update_one(search_criteria, {'$set': {'perf_val': new_perf}}, upsert=False)
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
        return []


def drop_collection():
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        db.performance_data.drop()
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)


def print_collection():
    documents = find({})
    for doc in documents:
        print(doc)

