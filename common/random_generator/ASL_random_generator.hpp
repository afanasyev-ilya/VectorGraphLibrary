#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ASLRandomGenerator::ASLRandomGenerator()
{
    asl_library_initialize();
    asl_random_create(&hnd, ALGO_TYPE);
    asl_random_distribute_uniform(hnd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void ASLRandomGenerator::generate_array_of_random_uniform_values(_T *_array, asl_int_t _size, _T _max_val)
{
    /* generic implementation  */
    throw "ERROR: unsupported type";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ASLRandomGenerator::~ASLRandomGenerator()
{
    asl_random_destroy(hnd);
    asl_library_finalize();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
void ASLRandomGenerator::generate_array_of_random_uniform_values<float>(float *_array, asl_int_t _size, float _max_val)
{
    asl_random_generate_s(hnd, _size, _array);
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < _size; i++)
    {
        _array[i] = abs(_array[i]) * _max_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
void ASLRandomGenerator::generate_array_of_random_uniform_values<double>(double *_array, asl_int_t _size, double _max_val)
{
    asl_random_generate_d(hnd, _size, _array);
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < _size; i++)
    {
        _array[i] = abs(_array[i]) * _max_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
void ASLRandomGenerator::generate_array_of_random_uniform_values<int>(int *_array, asl_int_t _size, int _max_val)
{
    asl_random_generate_i(hnd, _size, (asl_int_t*)_array);
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < _size; i++)
    {
        _array[i] = abs(_array[i]) % _max_val;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
