#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define N 16
#define OTHER_PARAMS 3

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double* convert_graph_to_nn_input(EdgesContainer &_el_container)
{
    int vertices_count = _el_container.get_vertices_count();
    long long edges_count = _el_container.get_edges_count();
    int *src_ids = _el_container.get_src_ids();
    int *dst_ids = _el_container.get_dst_ids();

    long long *sparsity_data;
    double *normalized_sparsity_data;
    MemoryAPI::allocate_array(&sparsity_data, N*N);
    MemoryAPI::allocate_array(&normalized_sparsity_data, N*N);

    int seg_size = vertices_count / N;

    for(long long idx = 0; idx < edges_count; idx++)
    {
        int src_id = src_ids[idx];
        int dst_id = dst_ids[idx];
        int src_seg = src_id / seg_size;
        int dst_seg = dst_id / seg_size;
        sparsity_data[src_seg * N + dst_seg]++;
    }
    for(int x = 0; x < N; x++)
    {
        for(int y = 0; y < N; y++)
        {
            normalized_sparsity_data[x * N + y] = ((double)sparsity_data[x * N + y]) / ((double)edges_count);
        }
    }

    for(int x = 0; x < N; x++)
    {
        for(int y = 0; y < N; y++)
        {
            cout << normalized_sparsity_data[x * N + y] << " ";
        }
        cout << endl;
    }

    double *nn_input;
    MemoryAPI::allocate_array(&nn_input, N*N + OTHER_PARAMS);

    double vertices_scale = (double)vertices_count / pow(2.0, 26);
    double avg_degree = edges_count / vertices_count;
    double max_degree = 0;

    nn_input[0] = vertices_scale;
    nn_input[1] = avg_degree;
    nn_input[2] = 0;
    MemoryAPI::copy(nn_input + OTHER_PARAMS, normalized_sparsity_data, N*N);

    MemoryAPI::free_array(sparsity_data);
    MemoryAPI::free_array(normalized_sparsity_data);
    return nn_input;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void save_nn_input_to_file(double *_nn_input, int _label, string _graph_name)
{
    ofstream nn_file;
    nn_file.open(_graph_name + ".txt");

    int wall_nn_input_size = N * N + OTHER_PARAMS;
    for(int i = 0; i < wall_nn_input_size; i++)
    {
        nn_file << _nn_input[i] << " ";
    }

    nn_file.close();

    MemoryAPI::free_array(_nn_input);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
