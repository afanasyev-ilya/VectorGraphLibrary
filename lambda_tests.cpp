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
    #pragma omp parallel num_threads(8)
    {
        printf("Hello World... from thread = %d\n", omp_get_thread_num());
    }
    cout << endl;
}

void f2()
{
    cout << "here I don't want to create 8 threads, since FOR-loop inside lambda :(" << endl;
    #pragma omp parallel num_threads(8)
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
    #pragma omp parallel num_threads(8)
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
    cout << "functor" << endl;
    #pragma omp parallel num_threads(8)
    {
        int test_reg[256];
        #pragma _NEC vreg(test_reg)

        struct Functor {
            int* const& rca;

            Functor(int* const& a): rca(a)
            {}

            inline bool operator()(int i1, int i2) const {
                int sum = 0;
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for(int i = 0; i < 256; i++)
                    sum += rca[i];
                cout << "lambda: " << sum + i1 + i2 << endl;
            }
        };
        Functor compare(test_reg);

        compare(10, 12);

        #pragma omp barrier
        printf("Hello World... from thread = %d\n", omp_get_thread_num());
    }
    cout << endl;
}

int main(int argc, const char * argv[])
{
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(8);  // Use 8 threads for all consecutive parallel regions

    f1();
    f2();
    f3();
    f4();

    return 0;
}

