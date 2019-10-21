//
//  vertex_queue.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef vertex_queue_h
#define vertex_queue_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VertexQueue
{
private:
    int *data;
    int cur_size;
    int max_size;
    int *local_offsets;
    
    void increase_size()
    {
        int *tmp_data = new int[max_size * 2];
        memcpy(tmp_data, data, cur_size * sizeof(int));
        
        delete []data;
        data = tmp_data;
        
        max_size *= 2;
    }
public:
    VertexQueue(int _max_size = 1)
    {
        max_size = _max_size;
        cur_size = 0;
        data = new int[max_size];
        local_offsets = new int[omp_get_max_threads()];
    }
    
    ~VertexQueue()
    {
        if(data != NULL)
        {
            delete []data;
            data = NULL;
        }
        if(local_offsets != NULL)
        {
            delete []local_offsets;
            local_offsets = NULL;
        }
    }
    
    inline void push_back(int _vertex)
    {
        data[cur_size] = _vertex;
        cur_size++;
    }
    
    inline void push_back(int *_vertices_reg)
    {
        int pos = 0;
        
        #ifdef __USE_NEC_SX_AURORA__
        #pragma _NEC vector
        #endif
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if(_vertices_reg[i] >= 0)
            {
                data[cur_size + pos] = _vertices_reg[i];
                pos++;
            }
        }

        cur_size += pos;
    }
    
    inline void push_back_one(int *_vertices_reg)
    {
        int pos = 0;
        
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if(_vertices_reg[i] > 0)
            {
                data[cur_size] = _vertices_reg[i];
                break;
            }
        }
        
        cur_size++;
    }
    
    inline bool empty()
    {
        if(cur_size == 0)
            return true;
        else
            return false;
    }
    
    inline void clear()
    {
        cur_size = 0;
    }
    
    void append_with_local_queues(VertexQueue &_local_queue)
    {
        #pragma omp barrier
        
        int tid = omp_get_thread_num();
        local_offsets[tid] = _local_queue.cur_size;
        vector<int> tmp_offsets(omp_get_max_threads());
        
        cur_size = 0;
        #pragma omp barrier
        
        #pragma omp atomic
        cur_size += _local_queue.cur_size;
        
        #pragma omp barrier
        
        #pragma omp master
        {
            int omp_num_threads = omp_get_max_threads();
            
            tmp_offsets[0] = 0;
            for(int i = 1; i < omp_num_threads; i++)
            {
                tmp_offsets[i] = local_offsets[i - 1] + tmp_offsets[i - 1];
            }
            
            for(int i = 0; i < omp_num_threads; i++)
            {
                local_offsets[i] = tmp_offsets[i];
            }
        }
        
        #pragma omp barrier
        
        int current_offset = local_offsets[tid];
        memcpy(&(data[current_offset]), _local_queue.data, _local_queue.cur_size * sizeof(int));
        
        #pragma omp barrier
    }
    
    void print()
    {
        for(int i = 0; i < cur_size; i++)
            cout << data[i] << " ";
        cout << endl;
    }
    
    inline int get_size() { return cur_size; };
    inline int *get_data() { return data; };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* vertex_queue_h */
