#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void CommonRandomGenerator::generate_array_of_random_uniform_values(_T *_array, int _size, _T _max_val)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();

        #pragma omp for
        for(int i = 0; i < _size; i++)
        {
            _array[i] = rand_r(&myseed) % ((int)_max_val);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
void CommonRandomGenerator::generate_array_of_random_uniform_values<float>(float *_array, int _size, float _max_val)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();

        #pragma omp for
        for(int i = 0; i <  _size; i++)
        {
            _array[i] = static_cast <float> (rand_r(&myseed)) / (static_cast <float> (RAND_MAX/_max_val));
        }
    }

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

