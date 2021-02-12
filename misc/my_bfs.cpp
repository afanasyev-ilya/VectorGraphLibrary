// Specify that we want to use NEC SX-Aurora TSUBASA API part.
// This is the only mandatory constants to specify.
#define __USE_NEC_SX_AURORA__

// Specify graph splitting threshold
// It is not mandatory, but 256 value is desired for high-performance of BFS algorithm
#define VECTOR_CORE_THRESHOLD_VALUE 256

#include "graph_library.h"

int main(int argc, const char * argv[])
{
    // Declare graph in optimized Vect CSR representation
    VectCSRGraph graph;

    // you can initialize VectCSR in different ways. Here we generate
    // a random edges list graph, and copy it to Vect CSR graph. Format conversion may take some time.
    EdgesListGraph el_graph;
    int vertices_count = 1024*1024;
    int edges_count = 32*vertices_count;
    GraphGenerationAPI::random_uniform(el_graph, vertices_count, edges_count, DIRECTED_GRAPH);

    // Warning! Graph vertices is reordered and renumbered here. You can use special VGL API functions to renumber vertices.
    graph.import(el_graph);

    // define BFS-levels for each graph vertex
    VerticesArray<int> levels(graph, SCATTER);

    // since latest version of NCC compiler is bugged, you have to use raw pointers to vertices array data in lambda functions,
    // if VerticesArray is not templated (as in BFS algorithms)
    int *levels_ptr = levels.get_ptr();

    // Define class, which includes all VGL computaional abstractions
    GraphAbstractionsNEC graph_API(graph, SCATTER);

    // Define frontier - a subset, which contains vertices visited on each BFS level
    FrontierNEC frontier(graph, SCATTER);

    // We will launch BFS form vertex number 10 of original graph.
    // Renumber API has to be used here, since vertices of Vect CSR graph are renumbered.
    int source_vertex = graph.reorder(10, ORIGINAL, SCATTER);

    // Use Compute abstraction to initialize initial levels for each vertex
    auto init_levels = [levels_ptr, source_vertex] __VGL_COMPUTE_ARGS__
    {
        if(src_id == source_vertex)
            levels_ptr[source_vertex] = FIRST_LEVEL_VERTEX;
        else
            levels_ptr[src_id] = UNVISITED_VERTEX;
    };
    // Use all-active frontier to initialize each graph vertex
    frontier.set_all_active();
    graph_API.compute(graph, frontier, init_levels);

    // Start timing
    Timer tm;
    tm.start();

    // Clear the frontier.
    frontier.clear();
    // Add source vertex to the frontier (now it contains only this vertex).
    frontier.add_vertex(source_vertex);

    int current_level = FIRST_LEVEL_VERTEX;
    // Loop over BFS levels. If level (frontier) contains no vertices, stop the algorithm.
    while(frontier.size() > 0)
    {
        // For each vertex, visit all its outgoing edges (scatter direction).
        auto edge_op = [levels_ptr, current_level](int src_id, int dst_id, int local_edge_pos,
                                                 long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
        {
            int src_level = levels_ptr[src_id];
            int dst_level = levels_ptr[dst_id];
            if((src_level == current_level) && (dst_level == UNVISITED_VERTEX))
            {
                levels_ptr[dst_id] = current_level + 1;
            }
        };
        graph_API.scatter(graph, frontier, edge_op);

        // Generate a new level of graph vertices, which have been visited in scatter abstraction.
        auto on_next_level = [levels_ptr, current_level] (int src_id, int connections_count)->int
        {
            int result = NOT_IN_FRONTIER_FLAG;
            if(levels_ptr[src_id] == (current_level + 1))
                result = IN_FRONTIER_FLAG;
            return result;
        };
        graph_API.generate_new_frontier(graph, frontier, on_next_level);

        current_level++;
    }
    tm.end();

    // print performance stats
    PerformanceStats::print_algorithm_performance_stats("BFS (Top-down, NEC)", tm.get_time(), graph.get_edges_count(), current_level);

    // reorder levels, since the graph has been preprocessed during import
    levels.reorder(ORIGINAL);

    // print BFS levels for the first 10 graph vertices
    for(int i = 0; i < 10; i++)
        cout << "vertex " << i << " is on level " << levels[i] << endl;

    return 0;
}
