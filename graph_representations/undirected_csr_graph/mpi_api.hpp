#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MPI__
void UndirectedCSRGraph::estimate_mpi_thresholds()
{
    Timer tm;
    tm.start();

    int vector_engine_part_size = get_vector_engine_threshold_vertex();
    int vector_core_part_size = get_vector_core_threshold_vertex() - vector_engine_part_size;
    int collective_part_size = get_vertices_count() - vector_engine_part_size - vector_core_part_size;

    vector_engine_mpi_thresholds = get_mpi_thresholds(vgl_library_data.get_mpi_rank(), 0, vector_engine_part_size);
    vector_core_mpi_thresholds = get_mpi_thresholds(vgl_library_data.get_mpi_rank(), vector_engine_part_size,
                                                       vector_engine_part_size + vector_core_part_size);
    collective_mpi_thresholds = get_mpi_thresholds(vgl_library_data.get_mpi_rank(),
                                                   vector_engine_part_size + vector_core_part_size,
                                                   vector_engine_part_size + vector_core_part_size + collective_part_size);

    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("Estimate MPI thresholds");
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
