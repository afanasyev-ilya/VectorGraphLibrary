#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

union converter { uint32_t i; float f; };

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline uint64_t pack_values(float a, float b)
{
    union converter ca = { .f = a };
    union converter cb = { .f = b };
    return ((uint64_t)cb.i << 32) + ca.i;
}

inline uint64_t pack_values(int a, int b)
{
    return ((uint64_t)b << 32) + a;
}

inline uint64_t pack_values(float a, int b)
{
    union converter ca = { .f = a };
    return ((uint64_t)b << 32) + ca.i;
}

inline uint64_t pack_values(int a, float b)
{
    union converter cb = { .f = b };
    return ((uint64_t)cb.i << 32) + a;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void unpack_values(uint64_t packed, float *a, float *b)
{
    union converter ca = { .i = packed };
    union converter cb = { .i = packed >> 32 };
    *a = ca.f;
    *b = cb.f;
}

inline void fast_unpack_values(uint64_t packed, float *a, float *b)
{
    int ca = packed;
    int cb = packed >> 32;
    *a = *(float*)&ca;
    *b = *(float*)&cb;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T1, typename _T2>
void pack_array_data(_T1 *_first_unpacked_data, _T2 *_second_unpacked_data, uint64_t *_packed_data, int _size)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t1 = omp_get_wtime();
    #endif

    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < _size; i++)
    {
        _packed_data[i] = pack_values(_first_unpacked_data[i], _second_unpacked_data[i]);
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t2 = omp_get_wtime();
    INNER_WALL_NEC_TIME += t2 - t1;
    INNER_PACK_NEC_TIME += t2 - t1;
    cout << "pack time: " << (t2 - t1)*1000.0 << " ms" << endl;
    cout << "pack BW: " << (sizeof(_T1) + sizeof(_T2) + sizeof(uint64_t))*_size/((t2-t1)*1e9) << " GB/s" << endl << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T1, typename _T2>
void unpack_array_data(_T1 *_first_unpacked_data, _T2 *_second_unpacked_data, uint64_t *_unpacked_data, int _size)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t1 = omp_get_wtime();
    #endif

    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < _size; i++)
    {
        unpack_values(_unpacked_data[i], &_first_unpacked_data[i], &_second_unpacked_data[i]);
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t2 = omp_get_wtime();
    INNER_WALL_NEC_TIME += t2 - t1;
    INNER_PACK_NEC_TIME += t2 - t1;
    cout << "unpack time: " << (t2 - t1)*1000.0 << " ms" << endl;
    cout << "unpack BW: " << (sizeof(_T1) + sizeof(_T2) + sizeof(uint64_t))*_size/((t2-t1)*1e9) << " GB/s" << endl << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
