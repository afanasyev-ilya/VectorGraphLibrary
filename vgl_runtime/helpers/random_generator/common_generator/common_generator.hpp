#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void CommonRandomGenerator::generate_array_of_random_uniform_values(_T *_array, size_t _size, _T _max_val)
{
    srand(time(NULL));
    #pragma omp parallel
    {
        unsigned int seed = int(time(NULL)) ^ rand() ^ omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < _size; ++i)
        {
            _array[i] = rand_r(&seed) % _max_val;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
void CommonRandomGenerator::generate_array_of_random_uniform_values<float>(float *_array, size_t _size, float _max_val)
{
    srand(time(NULL));
    #pragma omp parallel
    {
        unsigned int seed = int(time(NULL)) ^ rand() ^ omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < _size; ++i)
        {
            float random = ((float) rand_r(&seed)) / (float) RAND_MAX;
            _array[i] = random*_max_val;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
void CommonRandomGenerator::generate_array_of_random_uniform_values<double>(double *_array, size_t _size, double _max_val)
{
    srand(time(NULL));
    #pragma omp parallel
    {
        unsigned int seed = int(time(NULL)) ^ rand() ^ omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < _size; ++i)
        {
            float random = ((double) rand_r(&seed)) / (double) RAND_MAX;
            _array[i] = random*_max_val;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

