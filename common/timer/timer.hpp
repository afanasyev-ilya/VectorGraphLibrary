#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Timer::Timer()
{
    t_start = 0;
    t_end = 0;

    #ifdef __USE_GPU__
    cudaEventCreate(&cuda_event_start);
    cudaEventCreate(&cuda_event_stop);
    #endif

    start();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Timer::start()
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_INTEL__)
    #pragma omp barrier
    t_start = omp_get_wtime();
    #pragma omp barrier
    #endif

    #ifdef __USE_GPU__
    cudaEventRecord(cuda_event_start);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Timer::end()
{
    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_INTEL__)
    #pragma omp barrier
    t_end = omp_get_wtime();
    #pragma omp barrier
    #endif

    #ifdef __USE_GPU__
    cudaEventRecord(cuda_event_stop);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double Timer::get_time()
{
    #ifdef __USE_GPU__
    cudaEventSynchronize(cuda_event_stop);
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, cuda_event_start, cuda_event_stop);
    return time_ms/1000.0;
    #endif

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_INTEL__)
    return (t_end - t_start);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double Timer::get_time_in_ms()
{
    #ifdef __USE_GPU__
    cudaEventSynchronize(cuda_event_stop);
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, cuda_event_start, cuda_event_stop);
    return time_ms;
    #endif

    #if defined(__USE_NEC_SX_AURORA__) || defined(__USE_INTEL__)
    return (t_end - t_start)*1000.0;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Timer::print_bandwidth_stats(string _name, long long _elements, double _bytes_per_element)
{
    #pragma omp master
    {
        double bytes = _elements * _bytes_per_element;
        cout << _name << " BW: " << bytes/(this->get_time()*1e9) << " GB/s" << endl << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Timer::print_time_stats(string _name)
{
    #pragma omp master
    {
        cout << _name << " time: " << this->get_time() << " (s)" << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Timer::print_time_and_bandwidth_stats(string _name, long long _elements, double _bytes_per_element)
{
    #pragma omp master
    {
        double bytes = _elements * _bytes_per_element;
        cout << _name << " time: " << this->get_time() << " (s)" << endl;
        cout << _name << " BW: " << bytes / (this->get_time() * 1e9) << " (GB/s)" << endl << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
