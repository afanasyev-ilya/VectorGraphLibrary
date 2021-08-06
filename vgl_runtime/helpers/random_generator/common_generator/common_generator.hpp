#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void CommonRandomGenerator::generate_array_of_random_uniform_values(_T *_array, int _size, _T _max_val)
{
    random_device r;
    std::vector<std::default_random_engine> generators;
    for (int i = 0, N = omp_get_max_threads(); i < N; ++i) {
        generators.emplace_back(default_random_engine(r()));
    }

    #pragma omp parallel for
    for (int i = 0; i < _size; ++i)
    {
        default_random_engine& engine = generators[omp_get_thread_num()];
        uniform_int_distribution<int> uniform_dist(0, _max_val);
        _array[i] = uniform_dist(engine); // I assume this is thread unsafe
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
void CommonRandomGenerator::generate_array_of_random_uniform_values<float>(float *_array, int _size, float _max_val)
{
    random_device r;
    std::vector<std::default_random_engine> generators;
    for (int i = 0, N = omp_get_max_threads(); i < N; ++i) {
        generators.emplace_back(default_random_engine(r()));
    }

    #pragma omp parallel for
    for (int i = 0; i < _size; ++i)
    {
        default_random_engine& engine = generators[omp_get_thread_num()];
        std::uniform_real_distribution<float> uniform_dist(0.0, _max_val);
        _array[i] = uniform_dist(engine); // I assume this is thread unsafe
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

