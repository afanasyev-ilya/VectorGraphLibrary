#include <random>
#include <algorithm>
#include <iterator>
#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void LP::seq_lp(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_labels)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    //Sequence vector used to iterate randomly over all vertices in a graph
    std::vector<int> seq_vector(vertices_count);
    for (int i = 0; i < vertices_count; i++) {
        seq_vector[i] = i;
    }
    bool updated;
    int iters = 0;
    int stop_value = 10;

    //To keep it simple, initial label is a vertice id
    for (int l = 0; l < vertices_count; ++l) {
        _labels[l] = l;
    }
    do {
        updated = false;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(seq_vector.begin(), seq_vector.end(), g);
        for (int bypass_i = 0; bypass_i < vertices_count; bypass_i++) {
            //Getting real vertice id
            int vertice_id = seq_vector[bypass_i];
            //Getting neighbour borders of a vertice
            long long int begin = outgoing_ptrs[vertice_id];
            long long int end = outgoing_ptrs[vertice_id + 1];
            //Map contains label and its frequency
            std::map<int, int> mp;
            //Filling up a map
            for (long long int neighbour = begin; neighbour< end; neighbour++) {
                if (mp.count(_labels[outgoing_ids[neighbour]])) {
                    mp[_labels[outgoing_ids[neighbour]]]++;
                } else {
                    mp[_labels[outgoing_ids[neighbour]]] = 1;
                }
            }

            int label_frequence = 0;
            int decision_label = -1;

            //Searching for a prevalent value - decision label
            for (auto it = mp.begin(); it != mp.end(); it++) {
                if (it->second > label_frequence)
                {
                    label_frequence = it->second;
                    decision_label = it->first;
                } else if (it->second == label_frequence)
                {
                    double change = (double)(random())/RAND_MAX;
                    if(change > 0.5) {
                        label_frequence = it->second;
                        decision_label = it->first;
                    }
                }
            }
            //Updating label
            if (decision_label != _labels[vertice_id]) {
                _labels[vertice_id] = decision_label;
                updated = true;
            }
        }
        iters++;
    } while ((updated) && (iters<stop_value));

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    cout << "sequential check labels: " << endl;
    PerformanceStats::component_stats(_labels, vertices_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
