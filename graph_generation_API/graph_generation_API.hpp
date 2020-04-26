//
//  graph_generation_API.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef graph_generation_API_hpp
#define graph_generation_API_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphGenerationAPI<_TVertexValue, _TEdgeWeight>::random_uniform(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                     int _vertices_count, long long _edges_count,
                                                                     DirectionType _direction_type)
{
    int vertices_count = _vertices_count;
    long long edges_count = _edges_count;
    
    int directed_edges_count = edges_count;
    if(!_direction_type)
        edges_count *= 2;
    
    _graph.resize(vertices_count, edges_count);
    
    RandomGenerationAPI rng_api;
    int max_id_val = vertices_count;
    rng_api.generate_array_of_random_values<_TVertexValue>(_graph.get_vertex_values(), vertices_count, 1000);
    rng_api.generate_array_of_random_values<int>(_graph.get_src_ids(), directed_edges_count, max_id_val);
    rng_api.generate_array_of_random_values<int>(_graph.get_dst_ids(), directed_edges_count, max_id_val);

    #ifdef __USE_WEIGHTED_GRAPHS__
    rng_api.generate_array_of_random_values<_TEdgeWeight>(_graph.get_weights(), directed_edges_count, 1.0);
    #endif
    
    if(!_direction_type)
    {
        for(long long i = 0; i < directed_edges_count; i++)
        {
            int src_id = (_graph.get_src_ids())[i];
            int dst_id = (_graph.get_dst_ids())[i];
            #ifdef __USE_WEIGHTED_GRAPHS__
            _TEdgeWeight weight = (_graph.get_weights())[i];
            #endif
            
            (_graph.get_src_ids())[i + directed_edges_count] = dst_id;
            (_graph.get_dst_ids())[i + directed_edges_count] = src_id;

            #ifdef __USE_WEIGHTED_GRAPHS__
            (_graph.get_weights())[i + directed_edges_count] = weight;
            #endif
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphGenerationAPI<_TVertexValue, _TEdgeWeight>::R_MAT(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                            int _vertices_count, long long _edges_count,
                                                            int _a_prob, int _b_prob, int _c_prob,
                                                            int _d_prob, DirectionType _direction_type)
{
    int n = (int)log2(_vertices_count);
    int vertices_count = _vertices_count;
    long long edges_count = _edges_count;
    
    int step = 1;
    if(_direction_type)
    {
        _graph.resize(vertices_count, edges_count);
    }
    else
    {
        step = 2;
        _graph.resize(vertices_count, 2*edges_count);
    }
    
    int *src_ids = _graph.get_src_ids();
    int *dst_ids = _graph.get_dst_ids();
    _TEdgeWeight *weights = _graph.get_weights();
    
    RandomGenerationAPI rng_api;
    rng_api.generate_array_of_random_values<_TVertexValue>(_graph.get_vertex_values(), vertices_count, 1000);
    
    int threads_count = omp_get_max_threads();
    
    // generate and add edges to graph
    unsigned int seed = 0;
    #pragma omp parallel private(seed) num_threads(threads_count)
    {
        seed = /*int(time(NULL)) * */omp_get_thread_num();
        
        #pragma omp for schedule(static)
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
            _TEdgeWeight edge_weight = (rand_r(&seed) % 10) + static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX);
            
            src_ids[cur_edge] = from;
            dst_ids[cur_edge] = to;

            #ifdef __USE_WEIGHTED_GRAPHS__
            weights[cur_edge] = edge_weight;
            #endif
            
            if(!_direction_type)
            {
                src_ids[cur_edge + 1] = to;
                dst_ids[cur_edge + 1] = from;

                #ifdef __USE_WEIGHTED_GRAPHS__
                weights[cur_edge + 1] = edge_weight;
                #endif
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef uint32_t
#define uint32_t int
#endif

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphGenerationAPI<_TVertexValue, _TEdgeWeight>::SSCA2(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                            int _vertices_count, int _max_clique_size)
{
    uint32_t TotVertices;
    uint32_t* clusterSizes;
    uint32_t* firstVsInCluster;
    uint32_t estTotClusters, totClusters;
    
    uint32_t *startVertex, *endVertex;
    long long numEdges;
    long long numIntraClusterEdges, numInterClusterEdges;
    _TEdgeWeight* weights;
    _TEdgeWeight MinWeight, MaxWeight;
    uint32_t MaxCliqueSize;
    uint32_t MaxParallelEdges = 1;
    double ProbUnidirectional = 1.0;
    double ProbIntercliqueEdges = 0.6;
    uint32_t i_cluster, currCluster;
    uint32_t *startV, *endV, *d;
    long long estNumEdges, edgeNum;
    
    long long i, j, k, t, t1, t2, dsize;
    double p;
    uint32_t* permV;
    
    // initialize RNG
    
    MinWeight = 0.0;
    MaxWeight = 1.0;
    TotVertices = _vertices_count;
    
    // generate clusters
    MaxCliqueSize = _max_clique_size;
    estTotClusters = 1.25 * TotVertices / (MaxCliqueSize/2);
    clusterSizes = (uint32_t *) malloc(estTotClusters*sizeof(uint32_t));
    
    for(i = 0; i < estTotClusters; i++)
    {
        clusterSizes[i] = 1 + (((double)(rand() % 10000)) / 10000.0 *MaxCliqueSize);
    }
    
    totClusters = 0;
    
    firstVsInCluster = (uint32_t *) malloc(estTotClusters*sizeof(uint32_t));
    
    firstVsInCluster[0] = 0;
    for (i=1; i<estTotClusters; i++)
    {
        firstVsInCluster[i] = firstVsInCluster[i-1] + clusterSizes[i-1];
        if (firstVsInCluster[i] > TotVertices-1)
            break;
    }
    
    totClusters = i;
    
    clusterSizes[totClusters-1] = TotVertices - firstVsInCluster[totClusters-1];
    
    // generate intra-cluster edges
    estNumEdges = (uint32_t) ((TotVertices * (double) MaxCliqueSize * (2-ProbUnidirectional)/2) +
                              (TotVertices * (double) ProbIntercliqueEdges/(1-ProbIntercliqueEdges))) * (1+MaxParallelEdges/2);
    
    if ((estNumEdges > ((1<<30) - 1)) && (sizeof(uint32_t*) < 8))
    {
        fprintf(stderr, "ERROR: long* should be 8 bytes for this problem size\n");
        fprintf(stderr, "\tPlease recompile the code in 64-bit mode\n");
        exit(-1);
    }
    
    edgeNum = 0;
    p = ProbUnidirectional;
    
    fprintf (stderr, "[allocating %3.3f GB memory ... ", (double) 2*estNumEdges*8/(1<<30));
    
    cout << "alloc of " << sizeof(uint32_t)*estNumEdges / (1024*1024) << " MB memory" << endl;
    startV = (uint32_t *) malloc(estNumEdges*sizeof(uint32_t));
    endV = (uint32_t *) malloc(estNumEdges*sizeof(uint32_t));
    
    fprintf(stderr, "done] ");
    
    for (i_cluster=0; i_cluster < totClusters; i_cluster++)
    {
        for (i = 0; i < clusterSizes[i_cluster]; i++)
        {
            for (j = 0; j < i; j++)
            {
                for (k = 0; k<1 + ((uint32_t)(MaxParallelEdges - 1) * ((double)(rand() % 10000)) / 10000.0); k++)
                {
                    startV[edgeNum] = j + \
                    firstVsInCluster[i_cluster];
                    endV[edgeNum] = i + \
                    firstVsInCluster[i_cluster];
                    edgeNum++;
                }
            }
            
        }
    }
    numIntraClusterEdges = edgeNum;
    
    //connect the clusters
    dsize = (uint32_t) (log((double)TotVertices)/log(2));
    d = (uint32_t *) malloc(dsize * sizeof(uint32_t));
    for (i = 0; i < dsize; i++) {
        d[i] = (uint32_t) pow(2, (double) i);
    }
    
    currCluster = 0;
    
    for (i = 0; i < TotVertices; i++)
    {
        p = ProbIntercliqueEdges;
        for (j = currCluster; j<totClusters; j++)
        {
            if ((i >= firstVsInCluster[j]) && (i < firstVsInCluster[j] + clusterSizes[j]))
            {
                currCluster = j;
                break;
            }
        }
        for (t = 1; t < dsize; t++)
        {
            j = (i + d[t] + (uint32_t)(((double)(rand() % 10000)) / 10000.0 * (d[t] - d[t - 1]))) % TotVertices;
            if ((j<firstVsInCluster[currCluster]) || (j>=firstVsInCluster[currCluster] + clusterSizes[currCluster]))
            {
                for (k = 0; k<1 + ((uint32_t)(MaxParallelEdges - 1)* ((double)(rand() % 10000)) / 10000.0); k++)
                {
                    if (p >  ((double)(rand() % 10000)) / 10000.0)
                    {
                        startV[edgeNum] = i;
                        endV[edgeNum] = j;
                        edgeNum++;
                    }
                }
            }
            p = p/2;
        }
    }
    
    numEdges = edgeNum;
    numInterClusterEdges = numEdges - numIntraClusterEdges;
    
    free(clusterSizes);
    free(firstVsInCluster);
    free(d);
    
    fprintf(stderr, "done\n");
    fprintf(stderr, "\tNo. of inter-cluster edges - %d\n", numInterClusterEdges);
    fprintf(stderr, "\tTotal no. of edges - %d\n", numEdges);
    
    // shuffle vertices to remove locality
    fprintf(stderr, "Shuffling vertices to remove locality ... ");
    fprintf(stderr, "[allocating %3.3f GB memory ... ", (double)(TotVertices + 2 * numEdges) * 8 / (1 << 30));
    
    permV = (uint32_t *)malloc(TotVertices*sizeof(uint32_t));
    startVertex = (uint32_t *)malloc(numEdges*sizeof(uint32_t));
    endVertex = (uint32_t *)malloc(numEdges*sizeof(uint32_t));
    
    for (i = 0; i<TotVertices; i++)
    {
        permV[i] = i;
    }
    
    for (i = 0; i<TotVertices; i++)
    {
        t1 = i + ((double)(rand() % 10000)) / 10000.0 * (TotVertices - i);
        if (t1 != i)
        {
            t2 = permV[t1];
            permV[t1] = permV[i];
            permV[i] = t2;
        }
    }
    
    for (i = 0; i<numEdges; i++)
    {
        startVertex[i] = permV[startV[i]];
        endVertex[i] = permV[endV[i]];
    }
    
    free(startV);
    free(endV);
    free(permV);
    
    // generate edge weights
    
    fprintf(stderr, "Generating edge weights ... ");
    weights = (_TEdgeWeight *)malloc(numEdges*sizeof(_TEdgeWeight));
    for (i = 0; i<numEdges; i++)
    {
        weights[i] = MinWeight + (_TEdgeWeight)(MaxWeight - MinWeight) * ((double)(rand() % 10000)) / 10000.0;
    }
    
    vector<vector<uint32_t> > dests(TotVertices);
    vector<vector<_TEdgeWeight> > weight_vect(TotVertices);
    
    // add data to vertices to graph
    _graph.resize(TotVertices, numEdges);
    
    RandomGenerationAPI rng_api;
    rng_api.generate_array_of_random_values<_TVertexValue>(_graph.get_vertex_values(), _graph.get_vertices_count(), 1000);
    
    // add edges to graph
    for (uint32_t i = 0; i < numEdges; i++)
    {
        _graph.get_src_ids()[i] = startVertex[i];
        _graph.get_dst_ids()[i] = endVertex[i];

        #ifdef __USE_WEIGHTED_GRAPHS__
        _graph.get_weights()[i] = weights[i];
        #endif
    }
    fprintf(stderr, "done\n");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct GenSCCdata
{
    int start_pos;
    int end_pos;
    int edges_count;
};

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphGenerationAPI<_TVertexValue, _TEdgeWeight>::SCC_uniform(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_graph,
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
    _TEdgeWeight *weights = _graph.get_weights();
    
    RandomGenerationAPI rng_api;
    rng_api.generate_array_of_random_values<_TVertexValue>(_graph.get_vertex_values(), vertices_count, 1000);
    
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
            _TEdgeWeight weight = weights[i] = ((double)(rand() % 10000)) / 10000.0;
            
            src_ids[current_edges_pos] = src_id;
            dst_ids[current_edges_pos] = dst_id;

            #ifdef __USE_WEIGHTED_GRAPHS__
            weights[current_edges_pos] = weight;
            #endif
            current_edges_pos++;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphGenerationAPI<_TVertexValue, _TEdgeWeight>::init_from_txt_file(EdgesListGraph<_TVertexValue, _TEdgeWeight>
                                                                         &_graph, string _txt_file_name,
                                                                         bool _append_with_reverse_edges)
{
    ifstream infile(_txt_file_name.c_str());
    if (!infile.is_open())
        throw "can't open file during convert";
    
    int vertices_count = 0;
    long long edges_count = 0;
    bool directed = true;
    string line;
    getline(infile, line); // read first line
    if(line == string("asym positive")) // try to understand which header it is
    {
        getline(infile, line); // get edges and vertices count line
        istringstream vert_iss(line);
        vert_iss >> edges_count >> vertices_count;
    }
    else if(line == string("% asym unweighted"))
    {
        cout << "asym positive detected" << endl;
        getline(infile, line);
    }
    else if(line == string("% bip unweighted"))
    {
        cout << "bip unweighted" << endl;
        directed = false;
    }
    else if(line == string("% sym unweighted"))
    {
        cout << "sym unweighted" << endl;
        directed = false;
        getline(infile, line); // get edges and vertices count line
        istringstream vert_iss(line);
        vert_iss >> edges_count >> vertices_count;
        cout << "edges: " << edges_count << " and vertices: " << vertices_count << endl;
    }
    else
    {
        getline(infile, line); // skip second line
        getline(infile, line); // get vertices and edges count line
        istringstream vert_iss(line);
        vert_iss >> vertices_count >> edges_count;
        getline(infile, line); // skip forth line
    }
    cout << "vc: " << vertices_count << " ec: " << edges_count << endl;
    
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
        
        if(!directed)
        {
            tmp_src_ids.push_back(dst_id);
            tmp_dst_ids.push_back(src_id);
        }
        i++;
        
        /*if((edges_count != 0) && (i > edges_count))
        {
            throw "ERROR: graph file is larger than expected";
        }*/
    }
    
    cout << "loaded " << vertices_count << " vertices_count" << endl;
    cout << "loaded " << i << " edges, expected amount " << edges_count << endl;
    
    edges_count = i;
    
    _graph.resize(vertices_count, edges_count);
    for(i = 0; i < edges_count; i++)
    {
        _graph.get_src_ids()[i] = tmp_src_ids[i];
        _graph.get_dst_ids()[i] = tmp_dst_ids[i];

        #ifdef __USE_WEIGHTED_GRAPHS__
        _graph.get_weights()[i] = 0.0;
        #endif
    }
    
    // validate
    for(i = 0; i < edges_count; i++)
    {
        int src_id = _graph.get_src_ids()[i];
        int dst_id = _graph.get_dst_ids()[i];
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

#endif /* graph_generation_API_hpp */
