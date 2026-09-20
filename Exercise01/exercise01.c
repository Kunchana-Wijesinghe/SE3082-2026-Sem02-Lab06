#include <omp.h>
#include <stdio.h>

int main(void) {

    #pragma omp parallel
    {
        int th = omp_get_thread_num();

        printf("Thread # %d reached the barrier\n", th);

        #pragma omp barrier

        printf("Thread # %d passed the barrier\n",
               omp_get_thread_num());

        printf("Hi again from thread # %d\n",
               omp_get_thread_num());
    }

    return 0;
}
