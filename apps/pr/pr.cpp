/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_NEC_SX_AURORA__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 4096
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void vgl_unpack(long long val, int &a, int &b)
{
    a = (int)((val & 0xFFFFFFFF00000000LL) >> 32);
    b = (int)(val & 0xFFFFFFFFLL);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void vgl_pack(long long &val, int a, int b)
{
    val = ((long long)a) << 32 | b;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    int a[256];
    int b[256];
    long long c[256];
    int idx[256];

    for(int i = 0; i < 256; i++)
    {
        a[i] = i;
        b[i] = i + 1;
        c[i] = 0;
        idx[i] = rand()%256;
    }

    // pack
    #pragma _NEC ivdep
    #pragma _NEC vector
    for(int i = 0; i < 256; i++)
    {
        vgl_pack(c[i], a[i], b[i]);
    }

    for(int i = 0; i < 10; i++)
    {
        cout << c[i] << endl;
    }

    // unpack
    #pragma _NEC ivdep
    #pragma _NEC vector
    for(int i = 0; i < 256; i++)
    {
        long long val = c[idx[i]];
        vgl_unpack(val, a[i], b[i]);
    }

    for(int i = 0; i < 10; i++)
    {
        cout << a[i] << " " << b[i] << endl;
    }
    cout << endl;

    try
    {
        cout << "PR (Page Rank) test..." << endl;

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        VectCSRGraph graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            EdgesListGraph el_graph;
            int v = pow(2.0, parser.get_scale());
            if(parser.get_graph_type() == RMAT)
                GraphGenerationAPI::R_MAT(el_graph, v, v * parser.get_avg_degree(), 57, 19, 19, 5, DIRECTED_GRAPH);
            else if(parser.get_graph_type() == RANDOM_UNIFORM)
                GraphGenerationAPI::random_uniform(el_graph, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);
            graph.import(el_graph);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            double t1 = omp_get_wtime();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "Error: graph file not found";
            double t2 = omp_get_wtime();
            cout << "file " << parser.get_graph_file_name() << " loaded in " << t2 - t1 << " sec" << endl;
        }

        VerticesArray<float> page_ranks(graph);
        PageRank::nec_page_rank(graph, page_ranks);

        if(parser.get_check_flag())
        {
            VerticesArray<float> seq_page_ranks(graph);
            PageRank::seq_page_rank(graph, seq_page_ranks);
            verify_results(page_ranks, seq_page_ranks, 10);
        }
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
