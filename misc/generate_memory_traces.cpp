//
//  generate_test_data.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 16/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include "../graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void save_to_file(vector<int> vals, string file_name)
{
    ofstream myfile;
    myfile.open(file_name.c_str());
    
    for(int i = 0; i < vals.size(); i++)
        myfile << vals[i] << "\n";
    
    myfile.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<uintptr_t> shared_trace;

void save_to_file(vector<uintptr_t> vals, string file_name)
{
    ofstream myfile;
    myfile.open(file_name.c_str());
    
    for(int i = 0; i < vals.size(); i++)
        myfile << vals[i] << "\n";
    
    myfile.close();
}

#define INIT_TRACE() \
shared_trace.clear()\

#define SAVE_ADDRESS(access) {\
uintptr_t pointer_val = reinterpret_cast<uintptr_t>(&access)/sizeof(int);\
shared_trace.push_back(pointer_val);}\

#define SAVE_TRACE(file_name)\
save_to_file(shared_trace, file_name);\

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void generate_simple_memory_profiles(EdgesListGraph<int, float> &rand_graph)
{
    int vect_size = rand_graph.get_vertices_count();
    
    int *whole_array = new int[3*rand_graph.get_vertices_count() + rand_graph.get_edges_count()];
    int *a = &whole_array[0];
    int *b = &whole_array[vect_size];
    int *c = &whole_array[2*vect_size];
    
    vector<int> saxpy_accesses;
    
    for(int i = 0; i < vect_size; i++)
    {
        a[i] = b[i] + c[i];
        saxpy_accesses.push_back(&c[i] - whole_array);
        saxpy_accesses.push_back(&b[i] - whole_array);
        saxpy_accesses.push_back(&a[i] - whole_array);
    }
    
    save_to_file(saxpy_accesses, "saxpy.txt");
    
    int edges_count = rand_graph.get_edges_count() / 2;
    int vertices_count = vect_size;
    int *distances = &whole_array[0];
    int *dst_ids = &whole_array[vertices_count];
    for(int i = 0; i < edges_count; i++)
    {
        dst_ids[i] = rand() % vertices_count;
    }
    
    vector<int> random_accesses;
    for(int i = 0; i < edges_count; i++)
    {
        int dst_id = dst_ids[i];
        random_accesses.push_back(&dst_ids[i] - whole_array);
        
        int val = distances[dst_id];
        random_accesses.push_back(&distances[dst_id] - whole_array);
    }
    
    save_to_file(random_accesses, "random.txt");
    
    for(int i = 0; i < edges_count; i++)
    {
        dst_ids[i] = rand_graph.get_dst_ids()[i];
    }
    
    vector<int> rmat_accesses;
    for(int i = 0; i < edges_count; i++)
    {
        int dst_id = dst_ids[i];
        rmat_accesses.push_back(&dst_ids[i] - whole_array);
        
        int val = distances[dst_id];
        rmat_accesses.push_back(&distances[dst_id] - whole_array);
    }
    
    save_to_file(rmat_accesses, "rmat.txt");
    
    int mat_size = 4;
    int *A = &whole_array[0];
    int *B = &whole_array[mat_size*mat_size];
    int *C = &whole_array[2*mat_size*mat_size];
    
    vector<int> first_mat_mul_accesses;
    for(int i = 0; i < mat_size; i++)
    {
        for(int j = 0; j < mat_size; j++)
        {
            for(int k = 0; k < mat_size; k++)
            {
                first_mat_mul_accesses.push_back(&A[i * mat_size + k] - whole_array);
                first_mat_mul_accesses.push_back(&B[k * mat_size + j] - whole_array);
                first_mat_mul_accesses.push_back(&C[i * mat_size + j] - whole_array);
                C[i * mat_size + j] += A[i * mat_size + k] * B[k * mat_size + j];
            }
        }
    }
    save_to_file(first_mat_mul_accesses, "first_mat_mul.txt");
    
    vector<int> second_mat_mul_accesses;
    for(int i = 0; i < mat_size; i++) // fast
    {
        for(int k = 0; k < mat_size; k++)
        {
            for(int j = 0; j < mat_size; j++)
            {
                second_mat_mul_accesses.push_back(&A[i * mat_size + k] - whole_array);
                second_mat_mul_accesses.push_back(&B[k * mat_size + j] - whole_array);
                second_mat_mul_accesses.push_back(&C[i * mat_size + j] - whole_array);
                C[i * mat_size + j] += A[i * mat_size + k] * B[k * mat_size + j];
            }
        }
    }
    save_to_file(second_mat_mul_accesses, "second_mat_mul.txt");
    
    vector<int> third_mat_mul_accesses;
    for(int j = 0; j < mat_size; j++)
    {
        for(int k = 0; k < mat_size; k++)
        {
            for(int i = 0; i < mat_size; i++)
            {
                third_mat_mul_accesses.push_back(&A[i * mat_size + k] - whole_array);
                third_mat_mul_accesses.push_back(&B[k * mat_size + j] - whole_array);
                third_mat_mul_accesses.push_back(&C[i * mat_size + j] - whole_array);
                C[i * mat_size + j] += A[i * mat_size + k] * B[k * mat_size + j];
            }
        }
    }
    save_to_file(third_mat_mul_accesses, "third_mat_mul.txt");
    
    vector<int> fourth_mat_mul_accesses;
    for(int j = 0; j < mat_size; j++)
    {
        for(int i = 0; i < mat_size; i++)
        {
            for(int k = 0; k < mat_size; k++)
            {
                fourth_mat_mul_accesses.push_back(&A[i * mat_size + k] - whole_array);
                fourth_mat_mul_accesses.push_back(&B[k * mat_size + j] - whole_array);
                fourth_mat_mul_accesses.push_back(&C[i * mat_size + j] - whole_array);
                C[i * mat_size + j] += A[i * mat_size + k] * B[k * mat_size + j];
            }
        }
    }
    save_to_file(fourth_mat_mul_accesses, "fourths_mat_mul.txt");
    
    vector<int> fith_mat_mul_accesses;
    for(int k = 0; k < mat_size; k++) // fast
    {
        for(int i = 0; i < mat_size; i++)
        {
            for(int j = 0; j < mat_size; j++)
            {
                fith_mat_mul_accesses.push_back(&A[i * mat_size + k] - whole_array);
                fith_mat_mul_accesses.push_back(&B[k * mat_size + j] - whole_array);
                fith_mat_mul_accesses.push_back(&C[i * mat_size + j] - whole_array);
                C[i * mat_size + j] += A[i * mat_size + k] * B[k * mat_size + j];
            }
        }
    }
    save_to_file(fith_mat_mul_accesses, "fith_mat_mul.txt");
    
    vector<int> sixth_mat_mul_accesses;
    for(int k = 0; k < mat_size; k++)
    {
        for(int j = 0; j < mat_size; j++)
        {
            for(int i = 0; i < mat_size; i++)
            {
                sixth_mat_mul_accesses.push_back(&A[i * mat_size + k] - whole_array);
                sixth_mat_mul_accesses.push_back(&B[k * mat_size + j] - whole_array);
                sixth_mat_mul_accesses.push_back(&C[i * mat_size + j] - whole_array);
                C[i * mat_size + j] += A[i * mat_size + k] * B[k * mat_size + j];
            }
        }
    }
    save_to_file(sixth_mat_mul_accesses, "sixth_mat_mul.txt");
    
    vector<int> first_matrix_traversal;
    for(int i = 0; i < mat_size; i++)
    {
        for(int j = 0; j < mat_size; j++)
        {
            first_matrix_traversal.push_back(&A[i * mat_size + j] - whole_array);
        }
    }
    save_to_file(first_matrix_traversal, "first_matrix_traversal.txt");
    
    vector<int> second_matrix_traversal;
    for(int j = 0; j < mat_size; j++)
    {
        for(int i = 0; i < mat_size; i++)
        {
            second_matrix_traversal.push_back(&A[i * mat_size + j] - whole_array);
        }
    }
    save_to_file(second_matrix_traversal, "second_matrix_traversal.txt");
    
    int length = 1000;
    double k = 0.4;
    int non_uniq_num = length * k;
    vector<int> k_1_test;
    int *non_repeats_array = &whole_array[0];
    int *repeats_array = &whole_array[length];
    int uniq_num = 10;
    for(int i = 0; i < length; i++)
    {
        if(i < non_uniq_num)
        {
            int val = non_repeats_array[i];
            k_1_test.push_back(&non_repeats_array[i] - whole_array);
        }
        else
        {
            int val2 = repeats_array[i % uniq_num];
            k_1_test.push_back(&repeats_array[i % uniq_num] - whole_array);
        }
    }
    save_to_file(k_1_test, "k_1_test.txt");
    std::random_shuffle(k_1_test.begin(), k_1_test.end());
    save_to_file(k_1_test, "k_1_test_shuffled.txt");
    
    length = 10000;
    non_uniq_num = length * k;
    vector<int> k_2_test;
    non_repeats_array = &whole_array[0];
    repeats_array = &whole_array[length];
    uniq_num = 10;
    for(int i = 0; i < length; i++)
    {
        if(i < non_uniq_num)
        {
            int val = non_repeats_array[i];
            k_2_test.push_back(&non_repeats_array[i] - whole_array);
        }
        else
        {
            int val2 = repeats_array[i % uniq_num];
            k_2_test.push_back(&repeats_array[i % uniq_num] - whole_array);
        }
    }
    save_to_file(k_2_test, "k_2_test.txt");
    std::random_shuffle(k_2_test.begin(), k_2_test.end());
    save_to_file(k_2_test, "k_2_test_shuffled.txt");
    
    int i, j;
    a = &whole_array[0];
    b = &whole_array[100*100];
    c = &whole_array[100*100 + 100];
    vector<int> matrix_vector_mul_simple;
    int n = 100;
    for (i = 0; i < n; i++)
    {
        c[i] = 0;
        matrix_vector_mul_simple.push_back(&c[i] - whole_array);
        for (j = 0; j < n; j++) {
            //c[i] = c[i] + a[i][j] * b[j];
            matrix_vector_mul_simple.push_back(&c[i] - whole_array);
            matrix_vector_mul_simple.push_back(&a[i * 100 + j] - whole_array);
            matrix_vector_mul_simple.push_back(&b[j] - whole_array);
            matrix_vector_mul_simple.push_back(&c[i] - whole_array);
        }
    }
    save_to_file(matrix_vector_mul_simple, "matrix_vector_mul_simple.txt");
    
    vector<int> matrix_vector_mul_blocks;
    int x, y;
    for (i = 0; i < n; i += 2)
    {
        c[i] = 0;
        matrix_vector_mul_blocks.push_back(&c[i] - whole_array);
        c[i + 1] = 0;
        matrix_vector_mul_blocks.push_back(&c[i+1] - whole_array);
        for (j = 0; j < n; j += 2)
        {
            for (x = i; x < min(i + 2, n); x++)
            {
                for (y = j; y < min(j + 2, n); y++)
                {
                    //c[x] = c[x] + a[x][y] * b[y];
                    matrix_vector_mul_blocks.push_back(&c[x] - whole_array);
                    matrix_vector_mul_blocks.push_back(&a[x * 100 + y] - whole_array);
                    matrix_vector_mul_blocks.push_back(&b[y] - whole_array);
                    matrix_vector_mul_blocks.push_back(&c[x] - whole_array);
                }
            }
        }
    }
    save_to_file(matrix_vector_mul_blocks, "matrix_vector_mul_blocks.txt");
    
    delete []whole_array;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ext_csr_trace(EdgesListGraph<int, float> &rand_graph, int VECT_LEN, float *distances)
{
    UndirectedCSRGraph<int, float> ext_graph;
    ext_graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_RANDOM_SHUFFLED, VECT_LEN, PULL_TRAVERSAL);
    cout << "ext csr graph generated" << endl;
    
    int vertices_count    = ext_graph.get_vertices_count();
    long long edges_count = ext_graph.get_edges_count   ();
    long long    *vertex_pointers    = ext_graph.get_vertex_pointers();
    int          *adjacent_ids     = ext_graph.get_adjacent_ids ();
    float        *adjacent_weights = ext_graph.get_adjacent_weights();
    
    int iters = 0;
    INIT_TRACE();
    for(int vec_start = 0; vec_start < vertices_count; vec_start += VECT_LEN)
    {
        int connections_reg[VECTOR_LENGTH];
        long long start_pos_reg[VECTOR_LENGTH];
        
        for(int i = 0; i < VECT_LEN; i++)
        {
            int src_id = vec_start + i;
            connections_reg[i] = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
            SAVE_ADDRESS(vertex_pointers[src_id]);
            SAVE_ADDRESS(vertex_pointers[src_id + 1]);
            
            start_pos_reg[i] = vertex_pointers[src_id];
        }
        
        int total_connections = 0;
        int max_connections = 0;
        for(int i = 0; i < VECT_LEN; i++)
        {
            if(max_connections < connections_reg[i])
                max_connections = connections_reg[i];
        }
        
        for(int edge_pos = 0; edge_pos < max_connections; edge_pos++)
        {
            for(int i = 0; i < VECT_LEN; i++)
            {
                int dst_id = adjacent_ids[start_pos_reg[i] + edge_pos];
                SAVE_ADDRESS(adjacent_ids[start_pos_reg[i] + edge_pos]);
                
                float weight = adjacent_weights[start_pos_reg[i] + edge_pos];
                SAVE_ADDRESS(adjacent_weights[start_pos_reg[i] + edge_pos]);
                                                          
                float dst_weight = distances[dst_id] + weight;
                SAVE_ADDRESS(distances[dst_id]);
                
                iters++;
            }
        }
    }
    
    cout << "iters: " << iters << endl;
    SAVE_TRACE("graph_ext_csr_traversal.txt");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ext_csr_trace_small_vectors(EdgesListGraph<int, float> &rand_graph, int VECT_LEN, float *distances)
{
    UndirectedCSRGraph<int, float> ext_graph;
    ext_graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_RANDOM_SHUFFLED, VECT_LEN, PULL_TRAVERSAL);
    cout << "ext csr graph small vectors generated" << endl;
    
    int vertices_count    = ext_graph.get_vertices_count();
    long long edges_count = ext_graph.get_edges_count   ();
    long long    *vertex_pointers    = ext_graph.get_vertex_pointers();
    int          *adjacent_ids     = ext_graph.get_adjacent_ids ();
    float        *adjacent_weights = ext_graph.get_adjacent_weights();
    
    INIT_TRACE();
    
    int iters = 0;
    for(int idx = 0; idx < vertices_count; idx++)
    {
        int src_id = idx;
        int connections = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
        SAVE_ADDRESS(vertex_pointers[src_id]);
        SAVE_ADDRESS(vertex_pointers[src_id + 1]);
        
        int start_pos = vertex_pointers[src_id];
        
        for(int edge_pos = 0; edge_pos < connections; edge_pos ++)
        {
            int dst_id = adjacent_ids[start_pos + edge_pos];
            SAVE_ADDRESS(adjacent_ids[start_pos + edge_pos]);
            
            float weight = adjacent_weights[start_pos + edge_pos];
            SAVE_ADDRESS(adjacent_weights[start_pos + edge_pos]);
            
            int dst_level = distances[dst_id] + weight;
            SAVE_ADDRESS(distances[dst_id]);
            
            iters++;
        }
    }
    
    cout << "iters: " << iters << endl;
    SAVE_TRACE("graph_ext_csr_traversal_small_vectors.txt");
}


void vect_csr_trace(EdgesListGraph<int, float> &rand_graph, int VECT_LEN, float *distances)
{
    VectorisedCSRGraph<int, float> vect_graph;
    vect_graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_RANDOM_SHUFFLED, VECT_LEN, PULL_TRAVERSAL, true);
    cout << "vect csr graph generated" << endl;
    
    int vertices_count                   = vect_graph.get_vertices_count                  ();
    long long edges_count            = vect_graph.get_edges_count                     ();
    int vector_segments_count            = vect_graph.get_vector_segments_count           ();
    int number_of_vertices_in_first_part = vect_graph.get_number_of_vertices_in_first_part();
    
    int           *reordered_vertex_ids = vect_graph.get_reordered_vertex_ids     ();
    long long     *first_part_ptrs      = vect_graph.get_first_part_ptrs          ();
    int           *first_part_sizes     = vect_graph.get_first_part_sizes         ();
    long long     *vector_group_ptrs    = vect_graph.get_vector_group_ptrs        ();
    int           *vector_group_sizes   = vect_graph.get_vector_group_sizes       ();
    int           *incoming_ids         = vect_graph.get_adjacent_ids             ();
    float  *incoming_weights     = vect_graph.get_adjacent_weights         ();
    
    INIT_TRACE();
    
    int iters = 0;
    cout << "number_of_vertices_in_first_part: " << number_of_vertices_in_first_part << endl;
    if(number_of_vertices_in_first_part > 0)
    {
        int local_changes = 0;
        for(int src_id = 0; src_id < number_of_vertices_in_first_part; src_id++)
        {
            long long edge_start = first_part_ptrs[src_id];
            int connections_count = first_part_sizes[src_id];
            
            SAVE_ADDRESS(first_part_ptrs[src_id]);
            SAVE_ADDRESS(first_part_sizes[src_id]);
            
            for(long long edge_pos = 0; edge_pos < connections_count; edge_pos += VECT_LEN)
            {
                for(int i = 0; i < VECT_LEN; i++)
                {
                    int dst_id = incoming_ids[edge_start + edge_pos + i];
                    SAVE_ADDRESS(incoming_ids[edge_start + edge_pos + i]);
                    
                    float weight = incoming_weights[edge_start + edge_pos + i];
                    SAVE_ADDRESS(incoming_weights[edge_start + edge_pos + i]);
                    
                    float new_weight = distances[dst_id];
                    SAVE_ADDRESS(distances[dst_id]);
                    iters++;
                }
            }
        }
    }
    
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int segment_first_vertex = cur_vector_segment * VECT_LEN + number_of_vertices_in_first_part;
        
        long long segement_edges_start = vector_group_ptrs[cur_vector_segment];
        int segment_connections_count  = vector_group_sizes[cur_vector_segment];

        SAVE_ADDRESS(vector_group_ptrs[cur_vector_segment]);
        SAVE_ADDRESS(vector_group_sizes[cur_vector_segment]);
        
        for(long long edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
        {
            for(int i = 0; i < VECT_LEN; i++)
            {
                int src_id = segment_first_vertex + i;
                int dst_id = incoming_ids[segement_edges_start + edge_pos * VECT_LEN + i];
                SAVE_ADDRESS(incoming_ids[segement_edges_start + edge_pos * VECT_LEN + i]);
                
                float weight = incoming_weights[segement_edges_start + edge_pos * VECT_LEN + i];
                SAVE_ADDRESS(incoming_weights[segement_edges_start + edge_pos * VECT_LEN + i]);
                
                float new_weight = distances[dst_id] + weight;
                SAVE_ADDRESS(distances[dst_id]);
                iters++;
            }
        }
    }
    
    cout << "iters: " << iters << endl;
    SAVE_TRACE("graph_vect_csr_traversal.txt");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void different_graph_formats_profiles(EdgesListGraph<int, float> &rand_graph)
{
    float *distances = new float[rand_graph.get_vertices_count()];
    
    int VECT_LEN = 4;
    
    ext_csr_trace(rand_graph, VECT_LEN, distances);
    ext_csr_trace_small_vectors(rand_graph, VECT_LEN, distances);
    
    vect_csr_trace(rand_graph, VECT_LEN, distances);
    
    delete []distances;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <complex>
#include <valarray>
 
const double PI = 3.141592653589793238460;
 
typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;
 
// Cooley–Tukey FFT (in-place, divide-and-conquer)
// Higher memory requirements and redundancy although more intuitive
void fft(CArray& x)
{
    const size_t N = x.size();
    if (N <= 1) return;
 
    // divide
    CArray even = x[std::slice(0, N/2, 2)];
    CArray  odd = x[std::slice(1, N/2, 2)];
 
    // conquer
    fft(even);
    fft(odd);
 
    // combine
    for (size_t k = 0; k < N/2; ++k)
    {
        Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
        SAVE_ADDRESS(odd[k]);
        x[k    ] = even[k] + t;
        SAVE_ADDRESS(even[k]);
        SAVE_ADDRESS(x[k]);
        
        x[k+N/2] = even[k] - t;
        SAVE_ADDRESS(even[k]);
        SAVE_ADDRESS(x[k+N/2]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
// Cooley-Tukey FFT (in-place, breadth-first, decimation-in-frequency)
// Better optimized but less intuitive
// !!! Warning : in some cases this code make result different from not optimased version above (need to fix bug)
// The bug is now fixed @2017/05/30
void fft_opt(CArray &x)
{
    // DFT
    unsigned int N = x.size(), k = N, n;
    double thetaT = 3.14159265358979323846264338328L / N;
    Complex phiT = Complex(cos(thetaT), -sin(thetaT)), T;
    while (k > 1)
    {
        n = k;
        k >>= 1;
        phiT = phiT * phiT;
        T = 1.0L;
        for (unsigned int l = 0; l < k; l++)
        {
            for (unsigned int a = l; a < N; a += n)
            {
                unsigned int b = a + k;
                Complex t = x[a] - x[b];
                SAVE_ADDRESS(x[b]);
                SAVE_ADDRESS(x[a]);
                x[a] += x[b];
                SAVE_ADDRESS(x[b]);
                SAVE_ADDRESS(x[a]);
                x[b] = t * T;
                SAVE_ADDRESS(x[b]);
            }
            T *= phiT;
        }
    }
    // Decimate
    unsigned int m = (unsigned int)log2(N);
    for (unsigned int a = 0; a < N; a++)
    {
        unsigned int b = a;
        // Reverse bits
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
        b = ((b >> 16) | (b << 16)) >> (32 - m);
        if (b > a)
        {
            Complex t = x[a];
            SAVE_ADDRESS(x[a]);
            x[a] = x[b];
            SAVE_ADDRESS(x[a]);
            SAVE_ADDRESS(x[b]);
            x[b] = t;
            SAVE_ADDRESS(x[b]);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
void fft_profile()
{
    const Complex test[] = { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, };
    CArray data(test, 16);
 
    // forward fft
    INIT_TRACE();
    fft(data);
    SAVE_TRACE("fft_trace.txt");
 
    INIT_TRACE();
    fft_opt(data);
    SAVE_TRACE("fft_opt_trace.txt");
}

void spmvm_csr_profile()
{
    int    *ia, *ja;
    double *a, *x, *y;
    int    row, i, j, idx, n, nnzMax, nnz, nrows;
    double ts, t, rate;

    n = 50;
    
    nrows  = n * n;
    nnzMax = nrows * 5;
    ia = (int*)malloc(nrows*sizeof(int));
    ja = (int*)malloc(nnzMax*sizeof(int));
    a  = (double*)malloc(nnzMax*sizeof(double));
    /* Allocate the source and result vectors */
    x = (double*)malloc(nrows*sizeof(double));
    y = (double*)malloc(nrows*sizeof(double));

    /* Create a pentadiagonal matrix, representing very roughly a finite
       difference approximation to the Laplacian on a square n x n mesh */
    row = 0;
    nnz = 0;
    for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
        ia[row] = nnz;
        if (i>0) { ja[nnz] = row - n; a[nnz] = -1.0; nnz++; }
        if (j>0) { ja[nnz] = row - 1; a[nnz] = -1.0; nnz++; }
        ja[nnz] = row; a[nnz] = 4.0; nnz++;
        if (j<n-1) { ja[nnz] = row + 1; a[nnz] = -1.0; nnz++; }
        if (i<n-1) { ja[nnz] = row + n; a[nnz] = -1.0; nnz++; }
        row++;
    }
    }
    ia[row] = nnz;

    /* Create the source (x) vector */
    for (i=0; i<nrows; i++)
        x[i] = 1.0;
    
    INIT_TRACE();

    /* Perform a matrix-vector multiply: y = A*x */
    /* Warning: To use this for timing, you need to (a) handle cold start
       (b) perform enough tests to make timing quantum relatively small */
    for (row=0; row<nrows; row++)
    {
        double sum = 0.0;
        SAVE_ADDRESS(ia[row]);
        SAVE_ADDRESS(ia[row+1]);
        for (idx= ia[row]; idx< ia[row+1]; idx++)
        {
            sum += a[idx] * x[ja[idx]];
            SAVE_ADDRESS(ja[idx]);
            SAVE_ADDRESS(x[ja[idx]]);
            SAVE_ADDRESS(a[idx]);
        }
        y[row] = sum;
        SAVE_ADDRESS(y[row]);
    }
    
    SAVE_TRACE("spmvm_csr_trace.txt");
    
    /* Compute with the result to keep the compiler for marking the
       code as dead */
    for (row=0; row<nrows; row++)
    {
        if (y[row] < 0)
        {
            fprintf(stderr,"y[%d]=%f, fails consistency test\n", row, y[row]);
        }
    }
    printf("Time for Sparse Ax, nrows=%d, nnz=%d, T = %f\n", nrows, nnz, t);

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void simple_linpack_trace()
{
    const int Nx = 20;
    const int Ny = 20;
    float Mat[Nx*Ny];
    float q[Nx];
    
    INIT_TRACE();

    for(int i = 0; i < Nx; i++)
        for(int j = 0; j < Ny; j++)
            Mat[Nx * i  + j] = rand()%100 - 5;

    // Triangularization
    for (int i = 0; i < Nx - 1; i++)
    {
        for (int h = i + 1; h < Nx; h++)
        {
            float t = Mat[h*Nx + i] / Mat[i*Nx + i];
            SAVE_ADDRESS(Mat[h*Nx + i]);
            SAVE_ADDRESS(Mat[i*Nx + i]);
            for (int j = 0; j <= Nx; j++)
            {
                Mat[h*Nx + j] = Mat[h*Nx + j] - t * Mat[i*Nx + j];
                SAVE_ADDRESS(Mat[h*Nx + j]);
                SAVE_ADDRESS(Mat[i*Nx + j]);
                SAVE_ADDRESS(Mat[h*Nx + j]);
            }
        }
    }

    // Resolution
    for (int i = Nx - 1; i >= 0; i--)
    {
        q[i] = Mat[i*Nx + Nx];
        SAVE_ADDRESS(Mat[i*Nx + Nx]);
        SAVE_ADDRESS(q[i]);
        for (int j = Nx - 1; j > i; j--)
        {
            q[i] = q[i] - Mat[i*Nx + j] * q[j];
            SAVE_ADDRESS(q[j]);
            SAVE_ADDRESS(Mat[i*Nx + j]);
            SAVE_ADDRESS(q[i]);
        }
        q[i] = q[i] / Mat[i*Nx + i];
        SAVE_ADDRESS(Mat[i*Nx + i]);
        SAVE_ADDRESS(q[i]);
        SAVE_ADDRESS(q[i]);
    }
    
    SAVE_TRACE("linpack_simple_trace.txt");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
int partition (int arr[], int low, int high)
{
    int pivot = arr[high]; // pivot
    SAVE_ADDRESS(arr[high])
    int i = (low - 1); // Index of smaller element
  
    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        SAVE_ADDRESS(arr[j])
        if (arr[j] < pivot)
        {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
            SAVE_ADDRESS(arr[i])
            SAVE_ADDRESS(arr[j])
        }
    }
    swap(&arr[i + 1], &arr[high]);
    SAVE_ADDRESS(arr[i + 1])
    SAVE_ADDRESS(arr[high])
    return (i + 1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
void quickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
        at right place */
        int pi = partition(arr, low, high);
  
        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void bubbleSort(int arr[], int n)
{
    int i, j;
    for (i = 0; i < n-1; i++)
    {
        // Last i elements are already in place
        for (j = 0; j < n-i-1; j++)
        {
            SAVE_ADDRESS(arr[j])
            SAVE_ADDRESS(arr[j+1])
            if (arr[j] > arr[j+1])
            {
                swap(&arr[j], &arr[j+1]);
                SAVE_ADDRESS(arr[j])
                SAVE_ADDRESS(arr[j+1])
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
// Driver Code
void sort_traces()
{
    int arr[100];
    int n = sizeof(arr) / sizeof(arr[0]);
    
    for(int i = 0; i < n; i++)
        arr[i] = rand() % 100;
    
    INIT_TRACE();
    quickSort(arr, 0, n - 1);
    SAVE_TRACE("quick_sort_trace.txt");
    
    for(int i = 0; i < n; i++)
    arr[i] = rand() % 100;
    
    INIT_TRACE();
    bubbleSort(arr, n);
    SAVE_TRACE("bubble_sort_trace.txt");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void localised_random_access(int region_size, int regions_num, int distance, int access_count)
{
    int *data = new int[(distance + region_size) * (regions_num + 1)];
    for(int idx = 0; idx < access_count; idx++)
    {
        int cur_region = rand()%regions_num;
        int pos = rand()%region_size;
        
        int global_pos = distance * cur_region + region_size * cur_region + pos;
        
        int val = data[global_pos];
        SAVE_ADDRESS(data[global_pos])
    }
    
    delete []data;
}

void random_access(int region_size, int access_count)
{
    int *data = new int[region_size];
    for(int idx = 0; idx < access_count; idx++)
    {
        int global_pos = rand() % region_size;
        
        int val = data[global_pos];
        SAVE_ADDRESS(data[global_pos])
    }
    
    delete []data;
}

void random_access_traces()
{
    INIT_TRACE();
    random_access(5000, 500);
    SAVE_TRACE("random_access.txt");
    
    INIT_TRACE();
    random_access(5000, 5000*5);
    SAVE_TRACE("random_access_with_repeats.txt");
    
    INIT_TRACE();
    localised_random_access(100, 5, 2000, 5000*5);
    SAVE_TRACE("random_access_localised.txt");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char ** argv)
{
    try
    {
        EdgesListGraph<int, float> rand_graph;
        
        GraphGenerationAPI<int, float>::R_MAT(rand_graph, pow(2.0, 12), pow(2.0, 12)*7, 57, 19, 19, 5, DIRECTED_GRAPH);
        
        generate_simple_memory_profiles(rand_graph);
        
        different_graph_formats_profiles(rand_graph);
        
        fft_profile();
        
        spmvm_csr_profile();
        
        simple_linpack_trace();
        
        sort_traces();
        
        random_access_traces();
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
