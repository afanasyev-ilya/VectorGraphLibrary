#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void ConnectedComponents<_TVertexValue,_TEdgeWeight>::nec_random_mate(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                      int *_components)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    RandomGenerationAPI generator;

    int *rand_vals;
    MemoryAPI::allocate_array(&rand_vals, edges_count);

    double t1 = omp_get_wtime();
    generator.generate_array_of_random_values(rand_vals, edges_count, 2);
    double t2 = omp_get_wtime();
    cout << "rand time: " << t2 - t1 << endl;



    MemoryAPI::free_array(rand_vals);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
