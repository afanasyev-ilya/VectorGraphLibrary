/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_INTEL__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        int size = 64*1024*1024;

        float *a, *b, *c;
        MemoryAPI::allocate_array(&a, size);
        MemoryAPI::allocate_array(&c, size);
        MemoryAPI::allocate_array(&b, size);

        int large_size = 64*1024*1024;
        int cached_size = 2*4096;
        int *dst_ids;
        MemoryAPI::allocate_array(&dst_ids, large_size);
        float *result;
        MemoryAPI::allocate_array(&result, large_size);

        for(int i = 0; i < large_size; i++)
        {
            dst_ids[i] = rand()%cached_size;
        }

        for(int i = 0; i < size; i++)
        {
            a[i] = rand()%100;
            b[i] = rand()%100;
        }

        for(int tr = 0; tr < 5; tr++)
        {
            Timer tm;
            tm.start();
            #pragma omp parallel for simd
            for(int i = 0; i < size; i++)
            {
                c[i] = a[i] + b[i];
            }
            tm.end();
            tm.print_bandwidth_stats("KNL", size, 3.0*sizeof(float));

            tm.start();
            #pragma omp parallel for simd
            for(int i = 0; i < large_size; i++)
            {
                result[i] = a[dst_ids[i]];
            }
            tm.end();
            tm.print_bandwidth_stats("KNL gather", large_size, 3.0*sizeof(float));

        }
        cout << " --------------------------------------------- " << endl;

        MemoryAPI::free_array(a);
        MemoryAPI::free_array(c);
        MemoryAPI::free_array(b);
        MemoryAPI::free_array(result);
        MemoryAPI::free_array(dst_ids);

        /*
        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        VectCSRGraph graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            EdgesListGraph el_graph;
            int v = pow(2.0, 23);
            if(parser.get_graph_type() == RMAT)
                GraphGenerationAPI::R_MAT(el_graph, v, v * 5, 57, 19, 19, 5, DIRECTED_GRAPH);
            else if(parser.get_graph_type() == RANDOM_UNIFORM)
                GraphGenerationAPI::random_uniform(el_graph, v, v * 5, DIRECTED_GRAPH);
            graph.import(el_graph);
        }

        FrontierMulticore frontier(graph, SCATTER);
        frontier.set_all_active();

        GraphAbstractionsMulticore graph_API;

        VerticesArray<int> levels(graph, SCATTER);
        int _source_vertex = 1;
        int *levels_ptr = levels.get_ptr();
        auto init_levels = [levels_ptr, _source_vertex] (int src_id, int connections_count, int vector_index)
        {
            if(src_id == _source_vertex)
                levels_ptr[_source_vertex] = 2;
            else
                levels_ptr[src_id] = 1;
        };
        frontier.set_all_active();

        Timer tm;
        tm.start();
        graph_API.compute(graph, frontier, init_levels);
        tm.end();
        tm.print_bandwidth_stats("compute", graph.get_vertices_count(), sizeof(int));

        tm.start();
        graph_API.compute(graph, frontier, init_levels);
        tm.end();
        tm.print_bandwidth_stats("compute2", graph.get_vertices_count(), sizeof(int));

        tm.start();
        graph_API.compute(graph, frontier, init_levels);
        tm.end();
        tm.print_bandwidth_stats("compute3", graph.get_vertices_count(), sizeof(int));*/
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
