/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_NEC_SX_AURORA__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 4096
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    int size = 4*1024*1024;
    int *a_data = new int[size];
    int *b_data = new int[size];
    float *f_a_data = new float[size];
    float *f_b_data = new float[size];
    long long *packed_data = new long long[size];

    #pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        a_data[i] = i / 100;
        b_data[i] = (size - i) / 100;
        f_a_data[i] = i / 100;
        f_b_data[i] = (size - i) / 100;
    }

    cout << "before : " << a_data[0] << " " << b_data[0] << endl;

    Timer tm;
    tm.start();
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        int a = a_data[i];
        int b = b_data[i];
        packed_data[i] = ((long long)a) << 32 | b;
    }
    tm.end();
    tm.print_time_and_bandwidth_stats("int pack", size, 2.0*sizeof(int)+sizeof(long long));

    tm.start();
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        long long packed_val = packed_data[i];
        a_data[i] = (int)((packed_val & 0xFFFFFFFF00000000LL) >> 32);
        b_data[i] = (int)(packed_val & 0xFFFFFFFFLL);
    }
    tm.end();
    tm.print_time_and_bandwidth_stats("int unpack", size, 2.0*sizeof(int)+sizeof(long long));

    cout << "after : " << a_data[0] << " " << b_data[0] << endl;

    cout << "before : " << f_a_data[0] << " " << f_b_data[0] << endl;

    tm.start();
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        float fa = f_a_data[i];
        float fb = f_b_data[i];
        int a = *(int*)& fa;
        int b = *(int*)& fb;
        packed_data[i] = ((long long)a) << 32 | b;
    }
    tm.end();
    tm.print_time_and_bandwidth_stats("float pack", size, 2.0*sizeof(float)+sizeof(long long));

    tm.start();
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        long long packed_val = packed_data[i];
        int a = (int)((packed_val & 0xFFFFFFFF00000000LL) >> 32);
        int b = (int)(packed_val & 0xFFFFFFFFLL);
        f_a_data[i] = *(float*)& a;
        f_b_data[i] = *(float*)& b;
    }
    tm.end();
    tm.print_time_and_bandwidth_stats("float unpack", size, 2.0*sizeof(int)+sizeof(long long));

    cout << "after : " << f_a_data[0] << " " << f_b_data[0] << endl;

    delete[] a_data;
    delete[] b_data;
    delete[] f_a_data;
    delete[] f_b_data;
    delete[] packed_data;

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

            verify_results(page_ranks, seq_page_ranks);
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
