#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphGenerationAPI::random_uniform(EdgesContainer &_edges_container,
                                        int _vertices_count, long long _edges_count,
                                        DirectionType _direction_type)
{
    int vertices_count = _vertices_count;
    long long edges_count = _edges_count;
    
    int directed_edges_count = edges_count;
    if(!_direction_type)
        edges_count *= 2;

    _edges_container.resize(vertices_count, edges_count);

    // get pointers
    int *src_ids = _edges_container.get_src_ids();
    int *dst_ids = _edges_container.get_dst_ids();

    #pragma omp parallel
    {};

    double t1 = omp_get_wtime();
    RandomGenerator rng_api;
    int max_id_val = vertices_count;
    rng_api.generate_array_of_random_values<int>(src_ids, directed_edges_count, max_id_val);
    rng_api.generate_array_of_random_values<int>(dst_ids, directed_edges_count, max_id_val);

    double t2 = omp_get_wtime();

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double work_per_edge = sizeof(int)*2.0;
    cout << "random_uniform gen time: " << t2 - t1 << " sec" << endl;
    cout << "random_uniform gen bandwidth: " << work_per_edge*directed_edges_count / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif
    
    if(!_direction_type)
    {
        #pragma omp parallel for
        for(long long i = 0; i < directed_edges_count; i++)
        {
            src_ids[i + directed_edges_count] = dst_ids[i];
            dst_ids[i + directed_edges_count] = src_ids[i];
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct BatchedRand
{
    int rand_buffer_size;
    int *rand_buffer;
    int rand_pos;

    BatchedRand()
    {
        rand_buffer_size = 100000;
        MemoryAPI::allocate_array(&rand_buffer, rand_buffer_size);
        rand_pos = 0;
        generate_new_portion();
    }

    ~BatchedRand()
    {
        MemoryAPI::free_array(rand_buffer);
    }

    void generate_new_portion()
    {
        cout << "gen " << omp_get_thread_num() << endl;
        RandomGenerator rng_api;
        rng_api.generate_array_of_random_values<int>(rand_buffer, rand_buffer_size, 100);
    }

    inline int rand()
    {
        if(rand_pos >= rand_buffer_size)
        {
            generate_new_portion();
            rand_pos = 0;
        }

        int rand_val = rand_buffer[rand_pos];
        rand_pos++;
        return rand_val;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphGenerationAPI::R_MAT(EdgesContainer &_edges_container,
                               int _vertices_count, long long _edges_count,
                               int _a_prob, int _b_prob, int _c_prob,
                               int _d_prob, DirectionType _direction_type)
{
    double t1 = omp_get_wtime();
    int n = (int)log2(_vertices_count);
    int vertices_count = _vertices_count;
    long long edges_count = _edges_count;

    int directed_edges_count = edges_count;
    if(!_direction_type)
        edges_count *= 2;
    
    int step = 1;
    if(_direction_type)
    {
        _edges_container.resize(vertices_count, edges_count);
    }
    else
    {
        step = 2;
        _edges_container.resize(vertices_count, 2*edges_count);
    }
    
    int *src_ids = _edges_container.get_src_ids();
    int *dst_ids = _edges_container.get_dst_ids();

    int threads_count = omp_get_max_threads();
    
    // generate and add edges to graph
    unsigned int seed = 0;
    #pragma omp parallel private(seed) num_threads(threads_count)
    {
        seed = int(time(NULL)) * omp_get_thread_num();
        
        #pragma omp for schedule(guided, 1024)
        for (long long cur_edge = 0; cur_edge < edges_count; cur_edge += step)
        {
            int x_middle = _vertices_count / 2, y_middle = _vertices_count / 2;
            for (long long i = 1; i < n; i++)
            {
                int a_beg = 0, a_end = _a_prob;
                int b_beg = _a_prob, b_end = b_beg + _b_prob;
                int c_beg = _a_prob + _b_prob, c_end = c_beg + _c_prob;
                int d_beg = _a_prob + _b_prob + _c_prob, d_end = d_beg + _d_prob;
                
                int step = (int)pow(2, n - (i + 1));
                
                int probability = rand_r(&seed) % 100;
                if (a_beg <= probability && probability < a_end)
                {
                    x_middle -= step, y_middle -= step;
                }
                else if (b_beg <= probability && probability < b_end)
                {
                    x_middle -= step, y_middle += step;
                }
                else if (c_beg <= probability && probability < c_end)
                {
                    x_middle += step, y_middle -= step;
                }
                else if (d_beg <= probability && probability < d_end)
                {
                    x_middle += step, y_middle += step;
                }
            }
            if (rand_r(&seed) % 2 == 0)
                x_middle--;
            if (rand_r(&seed) % 2 == 0)
                y_middle--;
            
            int from = x_middle;
            int to = y_middle;

            src_ids[cur_edge] = from;
            dst_ids[cur_edge] = to;
            
            if(!_direction_type)
            {
                src_ids[cur_edge + 1] = to;
                dst_ids[cur_edge + 1] = from;
            }
        }
    }

    double t2 = omp_get_wtime();

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double work_per_edge = sizeof(int)*2.0;
    cout << "rmat gen time: " << t2 - t1 << " sec" << endl;
    cout << "rmat gen bandwidth: " << work_per_edge*directed_edges_count / ((t2 - t1)*1e9) << " GB/s" << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct GenSCCdata
{
    int start_pos;
    int end_pos;
    int edges_count;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
void GraphGenerationAPI::SCC_uniform(EdgesListGraph &_graph,
                                     int _vertices_count, int _min_scc_size, int _max_scc_size)
{
    int vertices_count = _vertices_count;
    vector<GenSCCdata> SCC_data;
    int current_pos = 0;
    long long edges_count = 0;
    while(current_pos < vertices_count)
    {
        int current_size = rand() % (_max_scc_size - _min_scc_size) + _min_scc_size;
        
        GenSCCdata current_data;
        current_data.start_pos = current_pos;
        current_data.end_pos = current_pos + current_size;
        if(current_data.end_pos >= vertices_count)
            current_data.end_pos = vertices_count - 1;
        current_data.edges_count = current_size * 3;
        edges_count += current_data.edges_count;
        
        SCC_data.push_back(current_data);
        
        current_pos += current_size + 1;
    }
    
    int SCC_count = SCC_data.size();
    
    edges_count += SCC_count * 2;
    
    _graph.resize(vertices_count, edges_count);
    
    int *src_ids = _graph.get_src_ids();
    int *dst_ids = _graph.get_dst_ids();
    
    RandomGenerator rng_api;

    int current_edges_pos = 0;
    for(int i = 0; i < SCC_count; i++)
    {
        GenSCCdata current_data = SCC_data[i];
        
        int start_vertex = current_data.start_pos;
        int end_vertex = current_data.end_pos;
        int edges_to_generate = current_data.edges_count;
        
        for(int j = 0; j < edges_to_generate; j++)
        {
            int src_id = rand() % (end_vertex - start_vertex) + start_vertex;
            int dst_id = rand() % (end_vertex - start_vertex) + start_vertex;
            
            src_ids[current_edges_pos] = src_id;
            dst_ids[current_edges_pos] = dst_id;
            current_edges_pos++;
        }
    }
}*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphGenerationAPI::init_from_txt_file(EdgesContainer &_edges_container, string _txt_file_name,
                                            DirectionType _direction_type)
{
    ifstream infile(_txt_file_name.c_str());
    if (!infile.is_open())
        throw "can't open file during convert";
    
    int vertices_count = 0;
    long long edges_count = 0;
    string line;
    getline(infile, line); // read first line

    for(int i = 0; i < 5;i++)
        getline(infile, line);

    vector<int>tmp_src_ids;
    vector<int>tmp_dst_ids;
    
    long long i = 0;
    while (getline(infile, line))
    {
        istringstream iss(line);
        int src_id = 0, dst_id = 0;
        if (!(iss >> src_id >> dst_id))
        {
            continue;
        }
        
        if(src_id >= vertices_count)
            vertices_count = src_id + 1;
        
        if(dst_id >= vertices_count)
            vertices_count = dst_id + 1;
        
        tmp_src_ids.push_back(src_id);
        tmp_dst_ids.push_back(dst_id);
        i++;
        
        if(_direction_type == UNDIRECTED_GRAPH)
        {
            tmp_src_ids.push_back(dst_id);
            tmp_dst_ids.push_back(src_id);
            i++;
        }
    }

    cout << "direction type: " << _direction_type << endl;
    cout << "loaded " << vertices_count << " vertices_count" << endl;
    if(_direction_type == DIRECTED_GRAPH)
        cout << "loaded " << i << " edges" << endl;
    else
        cout << "loaded " << i << " directed edges, " << i/2 << " undirected" << endl;
    
    edges_count = i;

    _edges_container.resize(vertices_count, edges_count);
    int seed = int(time(NULL));
    for(i = 0; i < edges_count; i++)
    {
        _edges_container.get_src_ids()[i] = tmp_src_ids[i];
        _edges_container.get_dst_ids()[i] = tmp_dst_ids[i];
    }
    
    // validate
    for(i = 0; i < edges_count; i++)
    {
        int src_id = _edges_container.get_src_ids()[i];
        int dst_id = _edges_container.get_dst_ids()[i];
        if((src_id >= vertices_count) || (src_id < 0))
        {
            cout << "error src: " << src_id << endl;
            throw "Error: incorrect src id on conversion";
        }
        if((dst_id >= vertices_count) || (dst_id < 0))
        {
            cout << "error dst: " << dst_id << endl;
            throw "Error: incorrect dst id on conversion";
        }
    }
    
    infile.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
