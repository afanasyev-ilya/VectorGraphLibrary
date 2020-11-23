#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <random>
#include <algorithm>
#include <iterator>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void LP::seq_lp(UndirectedCSRGraph &_graph, int *_labels, int _max_iterations)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    // Sequence vector used to iterate randomly over all vertices in a graph
    std::vector<int> seq_vector(vertices_count);
    for (int src_id = 0; src_id < vertices_count; src_id++)
    {
        seq_vector[src_id] = src_id;
    }
    bool updated;
    int iters = 0;

    //To keep it simple, initial label is a vertice id
    for (int src_id = 0; src_id < vertices_count; src_id++)
    {
        _labels[src_id] = src_id;
    }

    do
    {
        updated = false;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(seq_vector.begin(), seq_vector.end(), g);
        for (int bypass_i = 0; bypass_i < vertices_count; bypass_i++)
        {
            //Getting real vertice id
            int vertice_id = seq_vector[bypass_i];
            //Getting neighbour borders of a vertice
            long long int begin = vertex_pointers[vertice_id];
            long long int end = vertex_pointers[vertice_id + 1];
            //Map contains label and its frequency
            std::map<int, int> mp;
            //Filling up a map
            for (long long int neighbour = begin; neighbour< end; neighbour++)
            {
                if (mp.count(_labels[adjacent_ids[neighbour]]))
                {
                    mp[_labels[adjacent_ids[neighbour]]]++;
                }
                else {
                    mp[_labels[adjacent_ids[neighbour]]] = 1;
                }
            }

            int label_frequence = 0;
            int decision_label = -1;

            //Searching for a prevalent value - decision label
            for (auto it = mp.begin(); it != mp.end(); it++)
            {
                if (it->second > label_frequence)
                {
                    label_frequence = it->second;
                    decision_label = it->first;
                }
                else if (it->second == label_frequence)
                {
                    double change = (double)(random())/RAND_MAX;
                    if(change > 0.5)
                    {
                        label_frequence = it->second;
                        decision_label = it->first;
                    }
                }
            }
            //Updating label
            if ((decision_label != _labels[vertice_id])&&(decision_label!=-1))
            {
                _labels[vertice_id] = decision_label;
                updated = true;
            }
        }
        iters++;
    } while ((updated) && (iters < _max_iterations));

    cout << "sequential check labels: " << endl;
    PerformanceStats::component_stats(_labels, vertices_count);

    /*cout << "seq labels: ";
    for(int i = 0; i < vertices_count; i++)
        cout << _labels[i] << " ";
    cout << endl;*/

    if(vertices_count < VISUALISATION_SMALL_GRAPH_VERTEX_THRESHOLD)
    {
        _graph.move_to_host();
        _graph.set_vertex_data_from_array(_labels);
        _graph.save_to_graphviz_file("lp_seq_graph.gv", VISUALISE_AS_DIRECTED);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
