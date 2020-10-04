/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../graph_library.h"
#include "nvgraph.h"
#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void check(nvgraphStatus_t status)
{
    if (status != NVGRAPH_STATUS_SUCCESS)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

void check_status(nvgraphStatus_t status)
{
    if (status != NVGRAPH_STATUS_SUCCESS)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void nvgraph_sssp(int vertices_count, int edges_count, int *source_indices, int *destination_offsets, float *weights,
                  float *sssp, ExtendedCSRGraph<int, float> &_graph)
{
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t vertex_dimT = CUDA_R_32F;

    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));

    // Инициализируем граф в библиотеке nvGRAPH
    check(nvgraphCreate(&handle));
    check(nvgraphCreateGraphDescr(handle, &graph));

    CSC_input->nvertices = vertices_count;
    CSC_input->nedges = edges_count;
    CSC_input->destination_offsets = destination_offsets;
    CSC_input->source_indices = source_indices;

    check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check(nvgraphAllocateVertexData(handle, graph, 1, &vertex_dimT));
    check(nvgraphAllocateEdgeData(handle, graph, 1, &edge_dimT));
    check(nvgraphSetEdgeData(handle, graph, (void*)weights, 0));
    
    // Запускаем алгоритм поиска кратчайших путей 50 раз
    // (для каждой из первых 50 вершин)
    int source_vert = 0;
    
    double total_time = 0;
    double t1 = omp_get_wtime();
    check(nvgraphSssp(handle, graph, 0,  &source_vert, 0));
    double t2 = omp_get_wtime();
    total_time += t2 - t1;
    
    // Усредняем время по 50 запускам
    // Вычисляем MTEPS и выводим в консоль
    cout << "sp nvGRAPH computation time: " << total_time << " sec" << endl;
    cout << "sp nvGRAPH performance (MTEPS): " << edges_count / (total_time * 1e6) << endl;
    
    // Получаем результаты вычислений
    check(nvgraphGetVertexData(handle, graph, (void*)sssp, 0));

    bool check_using_vgl = false;
    if(check_using_vgl)
    {
        float *vgl_result;

        ShortestPaths<int, float> sssp_operation(_graph);

        sssp_operation.allocate_result_memory(_graph.get_vertices_count(), &vgl_result);

        sssp_operation.seq_dijkstra(_graph, vgl_result, source_vert);

        int error_count = 0;
        for(int i = 0; i < _graph.get_vertices_count(); i++)
        {
            if(fabs(sssp[i] - vgl_result[i]) > 0.001)
            {
                if(error_count < 20)
                    cout << "Cached Error: " << sssp[i] << " vs " << vgl_result[i] << " in pos " << i << endl;
                error_count++;
            }
        }
        cout << "cached error count: " << error_count << endl;

        sssp_operation.free_result_memory(vgl_result);
    }
    
    // Освобождаем выделенные структуры
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
    
    free(CSC_input);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void nvgraph_page_rank(int vertices_count, int edges_count, int *source_indices, int *destination_offsets, float *weights,
                       float *sssp, ExtendedCSRGraph<int, float> &_graph)
{
    size_t vert_sets = 2, edge_sets = 1;
    float alpha1 = 0.85f; void *alpha1_p = (void *) &alpha1;
    
    // nvgraph variables
    nvgraphHandle_t handle; nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;
    
    // Allocate host data
    float *pr_1 = (float*)malloc(vertices_count*sizeof(float));
    void **vertex_dim = (void**)malloc(vert_sets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vert_sets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    
    // Initialize host data
    float *bookmark_h = new float[vertices_count];
    for(int i = 0; i < vertices_count; i++)
        bookmark_h[i] = 0;
    bookmark_h[0] = 1.0f;
    
    vertex_dim[0] = (void*)bookmark_h;
    vertex_dim[1]= (void*)pr_1;
    vertex_dimT[0] = CUDA_R_32F; vertex_dimT[1]= CUDA_R_32F, vertex_dimT[2]= CUDA_R_32F;
    
    // Starting nvgraph
    check(nvgraphCreate (&handle));
    check(nvgraphCreateGraphDescr (handle, &graph));
    
    CSC_input->nvertices = vertices_count;
    CSC_input->nedges = edges_count;
    CSC_input->destination_offsets = destination_offsets;
    CSC_input->source_indices = source_indices;
    
    // Set graph connectivity and properties (tranfers)
    check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check(nvgraphAllocateVertexData(handle, graph, vert_sets, vertex_dimT));
    check(nvgraphAllocateEdgeData  (handle, graph, edge_sets, &edge_dimT));
    
    for (int i = 0; i < 2; ++i)
        check(nvgraphSetVertexData(handle, graph, vertex_dim[i], i));
    check(nvgraphSetEdgeData(handle, graph, (void*)weights, 0));

    int desired_iterations_count = 1;
    if(desired_iterations_count > 0)
    {
        double total_time = 0;
        double t1 = omp_get_wtime();
        nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, 0.00001, desired_iterations_count);
        double t2 = omp_get_wtime();
        total_time += t2 - t1;

        cout << "pr nvGRAPH computation time: " << total_time << " sec" << endl;
        cout << "pr nvGRAPH performance (MTEPS): " << edges_count / (total_time * 1e6) << endl;
        cout << "pr nvGRAPH performance per iter (MTEPS): " << double(desired_iterations_count)*edges_count / (total_time * 1e6) << endl;
    }
    
    // Get result
    check(nvgraphGetVertexData(handle, graph, vertex_dim[1], 1));
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
    
    free(pr_1);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void nvgraph_bfs(int vertices_count, int edges_count, int *source_indices, int *destination_offsets,
                 ExtendedCSRGraph<int, float> &_graph)
{
    const size_t  n = vertices_count, nnz = edges_count, vertex_numsets = 2, edge_numset = 0;
    int *source_offsets_h = destination_offsets;
    int *destination_indices_h = source_indices;

    //where to store results (distances from source) and where to store results (predecessors in search tree)
    int *bfs_distances_h = new int[n];
    int *bfs_predecessors_h = new int[n];

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSRTopology32I_t CSR_input;
    cudaDataType_t* vertex_dimT;
    size_t distances_index = 0;
    size_t predecessors_index = 1;
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    vertex_dimT[distances_index] = CUDA_R_32I;
    vertex_dimT[predecessors_index] = CUDA_R_32I;

    //Creating nvgraph objects
    check_status(nvgraphCreate (&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));

    // Set graph connectivity and properties (tranfers)
    CSR_input = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    CSR_input->nvertices = n;
    CSR_input->nedges = nnz;
    CSR_input->source_offsets = source_offsets_h;
    CSR_input->destination_indices = destination_indices_h;
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSR_input, NVGRAPH_CSR_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));

    int source_vert = 1;

    //Setting the traversal parameters
    nvgraphTraversalParameter_t traversal_param;
    nvgraphTraversalParameterInit(&traversal_param);
    nvgraphTraversalSetDistancesIndex(&traversal_param, distances_index);
    nvgraphTraversalSetPredecessorsIndex(&traversal_param, predecessors_index);
    nvgraphTraversalSetUndirectedFlag(&traversal_param, false);

    //Computing traversal using BFS algorithm
    double t1 = omp_get_wtime();
    check_status(nvgraphTraversal(handle, graph, NVGRAPH_TRAVERSAL_BFS, &source_vert, traversal_param));
    double t2 = omp_get_wtime();
    double total_time = t2 - t1;

    cout << "bfs nvGRAPH computation time: " << total_time << " sec" << endl;
    cout << "bfs nvGRAPH performance (MTEPS): " << edges_count / (total_time * 1e6) << endl;

    // Get result
    check_status(nvgraphGetVertexData(handle, graph, (void*)bfs_distances_h, distances_index));
    check_status(nvgraphGetVertexData(handle, graph, (void*)bfs_predecessors_h, predecessors_index));

    // expect bfs distances_h = (1 0 1 3 3 2 2147483647)
    for (int i = 0; i<10; i++)
        printf("Distance to vertex %d: %i\n",i, bfs_distances_h[i]); printf("\n");

    // expect bfs predecessors = (1 -1 1 5 5 0 -1)
    for (int i = 0; i<10; i++)
        printf("Predecessor of vertex %d: %i\n",i, bfs_predecessors_h[i]); printf("\n");

    free(vertex_dimT);
    free(CSR_input);
    check_status(nvgraphDestroyGraphDescr (handle, graph));
    check_status(nvgraphDestroy (handle));

    delete []bfs_distances_h;
    delete []bfs_predecessors_h;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void convert_and_test(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, string alg)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph)

    vector<vector<TempEdgeData<float> > > csc_tmp_graph;

    for(int i = 0; i < vertices_count; i++)
    {
        vector<TempEdgeData<_TEdgeWeight> > empty_vector;
        csc_tmp_graph.push_back(empty_vector);
    }

    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        const long long int start = vertex_pointers[src_id];
        const long long int end = vertex_pointers[src_id + 1];
        int connections_count = end - start;

        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = adjacent_ids[start + edge_pos];
            _TEdgeWeight weight = adjacent_weights[start + edge_pos];
            csc_tmp_graph[src_id].push_back(TempEdgeData<_TEdgeWeight>(dst_id, weight));
        }
    }

    // Подготавлием структуры данных для передачи графа
    // в библиотеку nvGRAPH
    int *source_indices = (int*) malloc(edges_count * sizeof(int));
    int *destination_offsets = (int*) malloc((vertices_count + 1) * sizeof(int));
    float *weights = (float*) malloc(edges_count * sizeof(float));
    float *result = (float*) malloc(vertices_count * sizeof(float));

    int current_pos = 0;
    for (int i = 0; i < vertices_count; ++i)
    {
        destination_offsets[i] = current_pos;
        for (unsigned j = 0; j < csc_tmp_graph[i].size(); ++j)
        {
            source_indices[current_pos] = csc_tmp_graph[i][j].dst_id;
            weights[current_pos] = csc_tmp_graph[i][j].weight;
            current_pos += 1;
        }
    }
    destination_offsets[vertices_count] = edges_count;

    if(alg == string("sp"))
        nvgraph_sssp(vertices_count, edges_count, source_indices, destination_offsets, weights, result, _graph);
    else if(alg == string("pr"))
        nvgraph_page_rank(vertices_count, edges_count, source_indices, destination_offsets, weights, result, _graph);
    else if(alg == string("bfs"))
        nvgraph_bfs(vertices_count, edges_count, source_indices, destination_offsets, _graph);

    free(source_indices);
    free(destination_offsets);
    free(weights);
    free(result);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        string input_graph_name = string(argv[1]);
        cout << input_graph_name << endl;
        
        ExtendedCSRGraph<int, float> graph;
        if(!graph.load_from_binary_file(input_graph_name))
            throw "ERROR: no such file " + input_graph_name + " in nvgraph test";
        
        cout << "loaded" << endl;
        
        convert_and_test<int, float>(graph, string(argv[2]));
    }
    catch (string error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
