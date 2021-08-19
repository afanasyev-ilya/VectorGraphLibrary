from . import mongo_api


def print_db_contents():
    results = mongo_api.find({})
    for res in results:
        print(res)


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


def get_graph_coef(graph_name, slider_values):
    if slider_values == {}:
        return 1.0
    else:
        graph_category = mongo_api.find_one({"graph_name": graph_name})["graph_category"]
        k1 = float(slider_values[graph_category])/100.0

        graph_vertex_scale = mongo_api.find_one({"graph_name": graph_name})["vertex_scale"]
        k2 = float(slider_values[graph_vertex_scale])/100.0
        return k1 * k2


def get_app_coef(app_name, slider_values):
    '''
    if("bfs" not in slider_values):
        slider_values["bfs"] = 0.5

    if("synthetic" not in slider_values):
        slider_values["synthetic"] = 0.5

    if("pr" not in slider_values):
        slider_values["pr"] = 0.5

    if("tiny_vertex_scale" not in slider_values):
        slider_values["tiny_vertex_scale"] = 0.5

    if("Online social network" not in slider_values):
        slider_values["Online social network"] = 0.5

    if("small_vertex_scale" not in slider_values):
        slider_values["small_vertex_scale"] = 0.5

    if("Hyperlink network" not in slider_values):
        slider_values["Hyperlink network"] = 0.5

    if("sssp" not in slider_values):
        slider_values["sssp"] = 0.5

    if("medium_vertex_scale" not in slider_values):
        slider_values["medium_vertex_scale"] = 0.5

    if("Infrastructure network" not in slider_values):
        slider_values["Infrastructure network"] = 0.5

    if("large_vertex_scale" not in slider_values):
        slider_values["large_vertex_scale"] = 0.5

    if("hits" not in slider_values):
        slider_values["hits"] = 0.5
    '''

    if slider_values == {}:
        return 0.5
    else:
        k = float(slider_values[app_name])/100.0
        return k

def get_coefficient(graph, app, slider_values):
    return get_graph_coef(graph, slider_values) * get_app_coef(app, slider_values)


def compute_weighted_normalized_rating(graph_filter_criteria, apps_filter_criteria, slider_values):
    unique_graphs = mongo_api.distinct(graph_filter_criteria, "graph_name") # TODO select required graphs in query
    unique_apps = mongo_api.distinct(apps_filter_criteria, "app_name") # TODO select required apps in query
    unique_architectures = mongo_api.distinct({}, "arch_name")

    rating_values = {}
    for arch in unique_architectures:
        rating_values[arch] = 0.0

    for app in unique_apps:
        for graph in unique_graphs:
            perf_data = mongo_api.find({"graph_name": graph, "app_name": app}, {"arch_name": 1, "perf_val": 1}) # 1 means present
            max_perf = get_max_perf(perf_data)
            normalized_data = normalize(perf_data, max_perf)

            k = get_coefficient(graph, app, slider_values)
            for data in normalized_data:
                rating_values[data["arch_name"]] += k * data["perf_val"]

    return rating_values


def get_perf_table():
    unique_graphs = mongo_api.distinct({}, "graph_name")
    unique_apps = mongo_api.distinct({}, "app_name")

    perf_table = {}

    for app in unique_apps:
        perf_table[app] = {}
        for graph in unique_graphs:
            perf_data = mongo_api.find({"graph_name": graph, "app_name": app}, {"arch_name": 1, "perf_val": 1})  # 1 means present
            perf_table[app][graph] = perf_data
    return perf_table


def get_list_rating(slider_values):
    rating_list = []
    rating = compute_weighted_normalized_rating({}, {}, slider_values)
    data_sorted = {k: v for k, v in sorted(rating.items(), key=lambda x: x[1])}
    pos = 1
    for k in sorted(rating, key=rating.get, reverse=True):
        rating_val = float(rating[k])
        rating_val = round(rating_val, 2)
        rating_list.append({"pos": pos, "arch": str(k), "rating": str(rating_val)})
        pos += 1
    return rating_list


def get_text_rating(slider_values):
    rating = compute_weighted_normalized_rating({}, {}, slider_values)
    data_sorted = {k: v for k, v in sorted(rating.items(), key=lambda x: x[1])}

    text = ""
    pos = 1
    for k in sorted(rating, key=rating.get, reverse=True):
        text += (str(pos) + ") arch: " + str(k) + "         |         rating: " + str(rating[k]))
        pos += 1
    return text
