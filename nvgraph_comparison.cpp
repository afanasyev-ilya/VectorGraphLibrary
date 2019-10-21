//
//  nvgraph_comparison.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 16/05/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#include "graph_library.h"
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void convert_and_test(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_vect_graph, string alg)
{
    LOAD_VECTORISED_CSR_GRAPH_DATA(_vect_graph)
    
    vector<vector<TempEdgeData<float> > > csc_tmp_graph;
    
    for(int i = 0; i < vertices_count; i++)
    {
        vector<TempEdgeData<_TEdgeWeight> > empty_vector;
        csc_tmp_graph.push_back(empty_vector);
    }
    
    for(int src_id = 0; src_id < number_of_vertices_in_first_part; src_id++)
    {
        long long edge_start = first_part_ptrs[src_id];
        int connections_count = first_part_sizes[src_id];
            
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = incoming_ids[edge_start + edge_pos];
            _TEdgeWeight weight = incoming_weights[edge_start + edge_pos];
            csc_tmp_graph[src_id].push_back(TempEdgeData<_TEdgeWeight>(dst_id, weight));
        }
    }
    
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int segment_first_vertex = cur_vector_segment * VECTOR_LENGTH + number_of_vertices_in_first_part;
        long long segement_edges_start = vector_group_ptrs[cur_vector_segment];
        int segment_connections_count  = vector_group_sizes[cur_vector_segment];
        
        for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
        {
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = segment_first_vertex + i;
                int dst_id = incoming_ids[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                _TEdgeWeight weight = incoming_weights[segement_edges_start + edge_pos * VECTOR_LENGTH + i];
                
                csc_tmp_graph[src_id].push_back(TempEdgeData<_TEdgeWeight>(dst_id, weight));
            }
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
        nvgraph_sssp(vertices_count, edges_count, source_indices, destination_offsets, weights, result, _vect_graph);
    else if(alg == string("pr"))
        nvgraph_page_rank(vertices_count, edges_count, source_indices, destination_offsets, weights, result, _vect_graph);
    
    free(source_indices);
    free(destination_offsets);
    free(weights);
    free(result);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void nvgraph_sssp(int vertices_count, int edges_count, int *source_indices, int *destination_offsets, float *weights,
                  float *sssp, VectorisedCSRGraph<int, float> &_vect_graph)
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
    
    float *pgl_result;
    ShortestPaths<int, float>::allocate_result_memory(_vect_graph.get_vertices_count(), &pgl_result);
    
    ShortestPaths<int, float>::bellman_ford_cached(_vect_graph, 0, pgl_result, 12);
    
    int error_count = 0;
    for(int i = 0; i < _vect_graph.get_vertices_count(); i++)
    {
        if(fabs(sssp[i] - pgl_result[i]) > 0.001)
        {
            if(error_count < 20)
                cout << "Cached Error: " << sssp[i] << " vs " << pgl_result[i] << " in pos " << i << endl;
            error_count++;
        }
    }
    cout << "cached error count: " << error_count << endl;
    
    ShortestPaths<int, float>::free_result_memory(pgl_result);
    
    // Освобождаем выделенные структуры
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
    
    free(CSC_input);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void nvgraph_page_rank(int vertices_count, int edges_count, int *source_indices, int *destination_offsets, float *weights,
                       float *sssp, VectorisedCSRGraph<int, float> &_vect_graph)
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
    
    double total_time = 0;
    double t1 = omp_get_wtime();
    nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, 0.0001, 100);
    double t2 = omp_get_wtime();
    total_time += t2 - t1;
    
    // Усредняем время по 50 запускам
    // Вычисляем MTEPS и выводим в консоль
    cout << "pr nvGRAPH computation time: " << total_time << " sec" << endl;
    cout << "pr nvGRAPH performance (MTEPS): " << edges_count / (total_time * 1e6) << endl;
    
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

int main(int argc, char **argv)
{
    try
    {
        string input_graph_name = string(argv[1]);
        cout << input_graph_name << endl;
        
        VectorisedCSRGraph<int, float> vect_graph;
        if(!vect_graph.load_from_binary_file(input_graph_name))
            throw "ERROR: no such file " + input_graph_name + " in nvgraph test";
        
        cout << "loaded" << endl;
        
        convert_and_test<int, float>(vect_graph, string(argv[2]));
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
