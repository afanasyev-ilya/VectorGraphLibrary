#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MPI_partitioner
{
private:
    int partitions_count;
    int current_partition;
    PartitioningAlgorithm algorithm;

    void do_edges_round_robin_partitioning(EdgesContainer &_edges_container,
                                           vector<int> &_partitioning_data,
                                           int &_partition_vertices_count,
                                           long long &_partition_edges_count)
    {
        _partition_vertices_count = _edges_container.get_vertices_count();
        _partition_edges_count = 0;

        for(long long i = 0; i < _edges_container.get_edges_count(); i++)
        {
            _partitioning_data[i] = i % partitions_count;
            if(_partitioning_data[i] == current_partition)
                _partition_edges_count++;
        }
    }

    void do_edges_consequent_partitioning(EdgesContainer &_edges_container,
                                          vector<int> &_partitioning_data,
                                          int &_partition_vertices_count,
                                          long long &_partition_edges_count)
    {
        _partition_vertices_count = _edges_container.get_vertices_count();
        _partition_edges_count = 0;

        for(long long i = 0; i < _edges_container.get_edges_count(); i++)
        {
            _partitioning_data[i] = i / _partition_edges_count;
            if(_partitioning_data[i] == current_partition)
                _partition_edges_count++;
        }
    }

    void do_vertices_1D_partitioning(EdgesContainer &_edges_container,
                                     vector<int> &_partitioning_data,
                                     int &_partition_vertices_count,
                                     long long &_partition_edges_count)
    {
        _partition_vertices_count = _edges_container.get_vertices_count();
        _partition_edges_count = 0;

        int part_size = (_partition_vertices_count - 1)/partitions_count + 1;

        for(long long i = 0; i < _edges_container.get_edges_count(); i++)
        {
            int src_id = _edges_container.get_src_ids()[i];
            int dst_id = _edges_container.get_dst_ids()[i];
            _partitioning_data[i] = src_id / part_size;
            if(_partitioning_data[i] == current_partition)
                _partition_edges_count++;
        }
    }

    void do_partitioning(EdgesContainer &_edges_container,
                         vector<int> &_partitioning_data,
                         int &_partition_vertices_count,
                         long long &_partition_edges_count)
    {
        if(algorithm == EDGES_CONSEQUENT_PARTITIONING)
        {
            cout << "EDGES_CONSEQUENT_PARTITIONING mode is used" << endl;
            do_edges_consequent_partitioning(_edges_container, _partitioning_data, _partition_vertices_count,
                                             _partition_edges_count);
        }
        else if(algorithm == EDGES_ROUND_ROBIN_PARTITIONING)
        {
            cout << "EDGES_ROUND_ROBIN_PARTITIONING mode is used" << endl;
            do_edges_round_robin_partitioning(_edges_container, _partitioning_data, _partition_vertices_count,
                                              _partition_edges_count);
        }
        else if(algorithm == VERTICES_1D_PARTITIONING)
        {
            cout << "VERTICES_1D_PARTITIONING mode is used" << endl;
            do_vertices_1D_partitioning(_edges_container, _partitioning_data, _partition_vertices_count,
                                        _partition_edges_count);
        }
        else
            throw "Error in MPI_partitioner::do_partitioning : unsupported algorithm";
    }
public:
    MPI_partitioner(int _desired_partitions_count, PartitioningAlgorithm _algorithm)
    {
        partitions_count = _desired_partitions_count;
        algorithm = _algorithm;
        current_partition = vgl_library_data.get_mpi_rank();
    }

    void run(EdgesContainer &_edges_container)
    {
        long long partition_edges_count = 0;
        int partition_vertices_count = 0;
        vector<int> partitioning_data;
        partitioning_data.resize(_edges_container.get_edges_count());

        // call main partitioning algorithm
        do_partitioning(_edges_container, partitioning_data, partition_vertices_count, partition_edges_count);

        // copy data to temporary container
        EdgesContainer local_container(partition_vertices_count, partition_edges_count);
        long long copy_pos = 0;
        for(long long i = 0; i < _edges_container.get_edges_count(); i++)
        {
            if(partitioning_data[i] == current_partition)
            {
                local_container.get_src_ids()[copy_pos] = _edges_container.get_src_ids()[i];
                local_container.get_dst_ids()[copy_pos] = _edges_container.get_dst_ids()[i];
                copy_pos++;
            }
        }

        // copy data to original container
        _edges_container.resize(local_container.get_vertices_count(), local_container.get_edges_count());
        MemoryAPI::copy(_edges_container.get_src_ids(), local_container.get_src_ids(), local_container.get_edges_count());
        MemoryAPI::copy(_edges_container.get_dst_ids(), local_container.get_dst_ids(), local_container.get_edges_count());
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
