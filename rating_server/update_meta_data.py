from web_rating.lib import mongo_api
from web_rating.lib.meta_data import get_meta_data


def print_db_stats():
    total_documents = mongo_api.count_documents({})
    print("TOTAL DOCUMENTS " + str(total_documents))
    unique_graphs = len(mongo_api.distinct({}, "graph_name"))
    unique_apps = len(mongo_api.distinct({}, "app_name"))
    unique_arch = len(mongo_api.distinct({}, "arch_name"))
    print("UNIQUE GRAPHS: " + str(unique_graphs))
    print("UNIQUE APPS: " + str(unique_apps))
    print("UNIQUE ARCH: " + str(unique_arch))
    print("check: " + str(total_documents) + " vs " + str(unique_graphs*unique_apps*unique_arch))


def update_all_meta_data():
    mongo_api.print_collection()
    print_db_stats()
    graph_names = mongo_api.distinct({}, "graph_name")
    for graph_name in graph_names:
        meta_data = get_meta_data(graph_name)
        mongo_api.update_many({"graph_name": graph_name}, meta_data)
    print_db_stats()


update_all_meta_data()
