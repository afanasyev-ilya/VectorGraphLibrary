import mongo_api


def print_db_contents():
    results = mongo_api.find({})
    for res in results:
        print(res)


def build_rating_for_graph():
    print("hehe")


def get_max_perf(perf_data):
    max_perf = 0
    for val in perf_data:
        if val["perf_val"] > max_perf:
            max_perf = val["perf_val"]
    return max_perf


def normalize(perf_data, max_perf):
    for val in perf_data:
        val["perf_val"] = val["perf_val"] / max_perf
    return perf_data


def get_coefficient(graph, app):
    return 1.0


def compute_weighted_normalized_rating(graph_filter_criteria, apps_filter_criteria):
    unique_graphs = mongo_api.distinct(graph_filter_criteria, "graph_name") # TODO select required graphs in query
    unique_apps = mongo_api.distinct(apps_filter_criteria, "app_name") # TODO select required apps in query
    unique_architectures = mongo_api.distinct({}, "arch_name")
    print(unique_graphs)
    print(unique_apps)

    rating_values = {}
    for arch in unique_architectures:
        rating_values[arch] = 0.0

    for app in unique_apps:
        for graph in unique_graphs:
            perf_data = mongo_api.find({"graph_name": graph, "app_name": app}, {"arch_name": 1, "perf_val": 1})
            max_perf = get_max_perf(perf_data)
            normalized_data = normalize(perf_data, max_perf)

            k = get_coefficient(graph, app)
            for data in normalized_data:
                rating_values[data["arch_name"]] += k * data["perf_val"]

    return rating_values


def print_rating_to_cmd():
    rating = compute_weighted_normalized_rating({}, {})
    data_sorted = {k: v for k, v in sorted(rating.items(), key=lambda x: x[1])}

    pos = 1
    for k in sorted(rating, key=rating.get, reverse=True):
        print(str(pos) + ") arch: " + str(k) + " rating: " + str(rating[k]))
        pos += 1


print_rating_to_cmd()