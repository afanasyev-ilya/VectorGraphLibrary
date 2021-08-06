#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void RandomGenerator::generate_array_of_random_values(_T *_array, size_t _size, _T _max_val)
{
    #ifdef __USE_ASL__
    ASL_rng.generate_array_of_random_uniform_values<_T>(_array, _size, _max_val);
    #else
    common_rng.generate_array_of_random_uniform_values<_T>(_array, _size, _max_val);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
