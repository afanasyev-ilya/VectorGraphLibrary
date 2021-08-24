#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class EdgesContainer
{
private:
    int vertices_count;
    long long edges_count;
    int *src_ids;
    int *dst_ids;
public:
    int *get_src_ids() {return src_ids;};
    int *get_dst_ids() {return dst_ids;};
    int get_vertices_count() {return vertices_count;};
    long long get_edges_count() {return edges_count;};

    void transpose()
    {
        int *tmp = src_ids;
        src_ids = dst_ids;
        dst_ids = tmp;
    }

    void print()
    {
        for(long long i = 0; i < edges_count; i++)
        {
            cout << src_ids[i] << " " << dst_ids[i] << endl;
        }
    }

    void resize(int _vertices_count, long long _edges_count)
    {
        vertices_count = _vertices_count;
        edges_count = _edges_count;

        MemoryAPI::free_array(src_ids);
        MemoryAPI::free_array(dst_ids);
        MemoryAPI::allocate_array(&src_ids, edges_count);
        MemoryAPI::allocate_array(&dst_ids, edges_count);
    }

    EdgesContainer(int _vertices_count = 1, long long _edges_count = 1)
    {
        vertices_count = _vertices_count;
        edges_count = _edges_count;
        MemoryAPI::allocate_array(&src_ids, edges_count);
        MemoryAPI::allocate_array(&dst_ids, edges_count);
    }

    ~EdgesContainer()
    {
        MemoryAPI::free_array(src_ids);
        MemoryAPI::free_array(dst_ids);
    }

    bool save_to_binary_file(string _file_name)
    {
        FILE *graph_file = fopen(_file_name.c_str(), "wb");
        if(graph_file == NULL)
            return false;

        GraphFormatType graph_type = EDGES_CONTAINER;

        fwrite(reinterpret_cast<const void*>(&vertices_count), sizeof(int), 1, graph_file);
        fwrite(reinterpret_cast<const void*>(&edges_count), sizeof(long long), 1, graph_file);
        fwrite(reinterpret_cast<void*>(&graph_type), sizeof(GraphFormatType), 1, graph_file);

        fwrite(reinterpret_cast<const void*>(src_ids), sizeof(int), edges_count, graph_file);
        fwrite(reinterpret_cast<const void*>(dst_ids), sizeof(int), edges_count, graph_file);

        fclose(graph_file);
        return true;
    }

    bool load_from_binary_file(string _file_name)
    {
        FILE *graph_file = fopen(_file_name.c_str(), "rb");
        if(graph_file == NULL)
            return false;

        GraphFormatType graph_type = EDGES_CONTAINER;

        fread(reinterpret_cast<void*>(&vertices_count), sizeof(int), 1, graph_file);
        fread(reinterpret_cast<void*>(&edges_count), sizeof(long long), 1, graph_file);
        fread(reinterpret_cast<void*>(&graph_type), sizeof(GraphFormatType), 1, graph_file);

        if(graph_type != EDGES_CONTAINER)
            throw "Error in EdgesContainer::load_from_binary_file : incorrect type of graph in file";

        resize(vertices_count, edges_count);

        fread(reinterpret_cast<void*>(src_ids), sizeof(int), edges_count, graph_file);
        fread(reinterpret_cast<void*>(dst_ids), sizeof(int), edges_count, graph_file);

        fclose(graph_file);
        return true;
    }

    void preprocess_into_csr_based(int *_work_buffer, vgl_sort_indexes *_sort_buffer)
    {
        bool work_buffer_was_allocated = false;
        if(_work_buffer == NULL)
        {
            work_buffer_was_allocated = true;
            MemoryAPI::allocate_array(&_work_buffer, this->edges_count);
        }

        bool sort_buffer_was_allocated = false;
        if(_sort_buffer == NULL)
        {
            sort_buffer_was_allocated = true;
            MemoryAPI::allocate_array(&_sort_buffer, this->edges_count);
        }

        // init sort indexes
        #pragma _NEC ivdep
        #pragma omp parallel for
        for(int i = 0; i < this->edges_count; i++)
        {
            _sort_buffer[i] = i;
        }

        // sort src_ids
        Timer tm;
        tm.start();
        Sorter::sort(src_ids, _sort_buffer, this->edges_count, SORT_ASCENDING);
        tm.end();
        #ifdef __PRINT_API_PERFORMANCE_STATS__
        tm.print_time_stats("EdgesListGraph sorting (to CSR) time");
        #endif

        // reorder dst_ids
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        #pragma omp parallel for
        for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
        {
            _work_buffer[edge_pos] = dst_ids[_sort_buffer[edge_pos]];
        }

        #pragma _NEC ivdep
        #pragma omp parallel for
        for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
        {
            dst_ids[edge_pos] = _work_buffer[edge_pos];
        }

        if(work_buffer_was_allocated)
        {
            MemoryAPI::free_array(_work_buffer);
        }
        if(sort_buffer_was_allocated)
        {
            MemoryAPI::free_array(_sort_buffer);
        }
    }

    void renumber_vertices(int *_conversion_array, int *_work_buffer)
    {
        Timer tm;
        tm.start();

        bool work_buffer_was_allocated = false;
        if(_work_buffer == NULL)
        {
            work_buffer_was_allocated = true;
            MemoryAPI::allocate_array(&_work_buffer, this->edges_count);
        }

        #pragma _NEC ivdep
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        #pragma omp parallel for
        for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
        {
            _work_buffer[edge_pos] = _conversion_array[src_ids[edge_pos]];
        }

        #pragma _NEC ivdep
        #pragma omp parallel for
        for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
        {
            src_ids[edge_pos] = _work_buffer[edge_pos];
        }

        #pragma _NEC ivdep
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        #pragma omp parallel for
        for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
        {
            _work_buffer[edge_pos] = _conversion_array[dst_ids[edge_pos]];
        }

        #pragma _NEC ivdep
        #pragma omp parallel for
        for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
        {
            dst_ids[edge_pos] = _work_buffer[edge_pos];
        }

        if(work_buffer_was_allocated)
        {
            MemoryAPI::free_array(_work_buffer);
        }
    }

    void random_shuffle_edges()
    {
        srand ( unsigned ( time(0) ) );

        vector<int> reorder_ids(vertices_count);
        #pragma omp parallel for
        for (int i = 0; i < vertices_count; i++)
            reorder_ids[i] = i;
        random_shuffle(reorder_ids.begin(), reorder_ids.end() );

        #pragma _NEC ivdep
        #pragma _NEC novob
        #pragma omp parallel for
        for(long long i = 0; i < edges_count; i++)
        {
            src_ids[i] = reorder_ids[src_ids[i]];
            dst_ids[i] = reorder_ids[dst_ids[i]];
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
