#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

union converter { uint32_t i; float f; };

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline uint64_t pack(float a, float b)
{
    union converter ca = { .f = a };
    union converter cb = { .f = b };
    return ((uint64_t)cb.i << 32) + ca.i;
}

inline uint64_t pack(int a, int b)
{
    return ((uint64_t)b << 32) + a;
}

inline uint64_t pack(float a, int b)
{
    union converter ca = { .f = a };
    return ((uint64_t)b << 32) + ca.i;
}

inline uint64_t pack(int a, float b)
{
    union converter cb = { .f = b };
    return ((uint64_t)cb.i << 32) + a;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void unpack(uint64_t packed, float *a, float *b) {
    union converter ca = { .i = packed };
    union converter cb = { .i = packed >> 32 };
    *a = ca.f;
    *b = cb.f;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void pack_array(float *_first_data, float *_second_data, uint64_t *_packed_data, int _size)
{
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < _size; i++)
    {
        _packed_data[i] = pack(_first_data[i], _second_data[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
