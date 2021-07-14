import pymongo


def connect_to_mongo():
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.demoCollection
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
        return False
    return True


def check_if_results_for_arch_exist(arch_name):
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.performance_data
        if collection.count_documents({"arch": arch_name}):
            return True
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
    return False


def add_performance_stats(received_data, arch):
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.performance_data

        for received_document in received_data:
            print(received_document)

            extended_document = add_meta_data(received_document, arch)
            collection.insert_one(extended_document)
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
        return False


def dump_db_data():
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        collection = db.performance_data
        search_results = collection.find()
        for res in search_results:
            print(res)
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
    return False


def remove_collection():
    try:
        connect = pymongo.MongoClient('mongodb://localhost:27017/')
        db = connect.vgl_rankings_db
        db.performance_data.drop()
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)
    return False


def add_meta_data(received_document, arch):
    received_document["arch"] = arch
    received_document["graph_nature"] = "synthetic"
    received_document["scale"] = "small"
    return received_document
