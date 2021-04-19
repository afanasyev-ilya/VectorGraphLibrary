/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "Printing graph stats..." << endl;

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // load graph
        VectCSRGraph graph;
        if(!graph.load_from_binary_file(parser.get_graph_file_name()))
            throw "Error: graph file not found";

        // print graphs stats
        graph.print_size();
        graph.print_stats();

        // check workload balance
        VGL_GRAPH_ABSTRACTIONS graph_API(graph, SCATTER);
        VGL_FRONTIER frontier(graph);
        frontier.set_all_active();

        long long edges_count = graph.get_edges_count();
        cout << "total edges: "<< edges_count << endl;

        int threads_count = omp_get_max_threads();
        int *work_array = new int[threads_count];
        #pragma omp parallel num_threads(threads_count)
        {
            int id = omp_get_thread_num();
            NEC_REGISTER_INT(work_size, 0);

            auto edge_op= [&reg_work_size](int src_id, int dst_id, int local_edge_pos,
                                           long long int global_edge_pos, int vector_index)
            {
                #pragma omp atomic
                reg_work_size[vector_index]++;
            };

            graph_API.enable_safe_stores();
            graph_API.scatter(graph, frontier, edge_op);
            graph_API.disable_safe_stores();
            int thread_work = register_sum_reduce(reg_work_size);

            #pragma omp critical
            {
                cout << "thread work: " << thread_work << endl;
            }
            work_array[id] = thread_work;
        }
        int max = 0, min = numeric_limits<int>::max(), avg = 0;
        for(int id = 0; id < threads_count; id++)
        {
            avg += work_array[id];
            if(max < work_array[id])
                max = work_array[id];
            if(min > work_array[id])
                min = work_array[id];
        }
        avg /= threads_count;
        cout << "AVG: " << avg << ", " << 100.0*((double)avg/edges_count) << "%" << endl;
        cout << "MAX: " << max << ", " << 100.0*((double)max/edges_count) << "%" << endl;
        cout << "MIN: " << min << ", " << 100.0*((double)min/edges_count) << "%" << endl;

        for(int id = 0; id < threads_count; id++)
        {
            cout << 100.0*((double)work_array[id]/edges_count) << "% " << endl;
        }
        cout << endl;
        delete[]work_array;
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
