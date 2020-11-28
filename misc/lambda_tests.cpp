//
// Created by Elijah Afanasiev on 10/02/2020.
//

#include <iostream>
#include <omp.h>
#include <stdio.h>

using namespace std;

void f1()
{
    cout << "Here I just create 8 omp threads" << endl;
    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS)
    {
        printf("Hello World... from thread = %d\n", omp_get_thread_num());
    }
    cout << endl;
}

void f2()
{
    cout << "here I don't want to create 8 threads, since FOR-loop inside lambda :(" << endl;
    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS)
    {
        auto test_lambda = [] (int _a, int _b)
        {
            int a = _a;
            int b = _b;
            int c = a + b;

            int reg_test[2];
            for(int i = 0; i < 2; i++)
                c += reg_test[i];
            cout << "C: " << c << endl;

        };
        test_lambda(10, 12);
        #pragma omp barrier
        printf("Hello World... from thread = %d\n", omp_get_thread_num());
    }
    cout << endl;
}

void f3()
{
    cout << "Here I create 8 threads again, since my lambda doesn't include any loops (loop is manually unrolled)" << endl;
    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS)
    {
        auto test_lambda = [] (int _a, int _b) {
            int a = _a;
            int b = _b;
            int c = a + b;

            int reg_test[2];
            c += reg_test[0];
            c += reg_test[1];
            cout << "C: " << c << " ";
        };
        test_lambda(10, 12);
        #pragma omp barrier
        printf("Hello World... from thread = %d\n", omp_get_thread_num());
    }
    cout << endl;
}

void f4()
{
    cout << "if/else return problem" << endl;
    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS)
    {
        auto test_lambda = [] (int _a, int _b)->int {
            if(_a > _b)
                return 1;
            else
                return 0;
        };

        cout << test_lambda(10, 12) << endl;

        #pragma omp barrier
        printf("Hello World... from thread = %d\n", omp_get_thread_num());
    }
    cout << endl;
}

void f5()
{
    cout << "if/else return NO problem" << endl;
    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS)
    {
        auto test_lambda = [] (int _a, int _b)->int {
            int res = 0;
            if(_a > _b)
                res = 1;
            else
                res = 0;
            return res;
        };

        cout << test_lambda(10, 12) << endl;

        #pragma omp barrier
        printf("Hello World... from thread = %d\n", omp_get_thread_num());
    }
    cout << endl;
}

int main(int argc, const char * argv[])
{
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(MAX_SX_AURORA_THREADS);  // Use 8 threads for all consecutive parallel regions

    f1();
    f2();
    f3();
    f4();
    f5();

    return 0;
}

