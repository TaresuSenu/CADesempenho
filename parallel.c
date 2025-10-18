/* parallel.c
Implementação paralela para estimar pi por Monte Carlo com openMP.
Compilar: gcc parallel.c -o parallel
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define T 8 //Define o nr de threads a usar
int main() {
    long int n;
    long int count = 0;
    double start, end, wall_clock_time;
    printf("\nn = ");
    scanf("%ld", &n);
    // Define o número de threads a serem usadas
    omp_set_num_threads(T);
    // Inicia a medição de tempo
    start = omp_get_wtime();
    // Inicia a região paralela
    #pragma omp parallel
    {
        long int local_count = 0;
        // 1. CADA thread terá seu próprio buffer e variável para o resultado.
        // Isso evita a condição de corrida.
        struct drand48_data randBuffer;
        double x, y;
        // 2. CADA thread inicializa (semeia) seu próprio buffer.
        // Usamos o tempo + ID da thread para garantir sementes únicas.
        srand48_r(time(NULL) + omp_get_thread_num(), &randBuffer);
        // O loop é dividido entre as threads
        #pragma omp for
        for(long int i = 0; i < n; ++i) {
            // A função armazena o resultado em 'x' e 'y'.
            drand48_r(&randBuffer, &x); // Gera número aleatório e armazena em x
            drand48_r(&randBuffer, &y); // Gera número aleatório e armazena em y
            // Verifica se o ponto (x,y) está dentro do círculo unitário
            if (x * x + y * y <= 1.0) {
                local_count++;
            }
        }
        // Atualiza o contador global de forma atômica para evitar conflitos
        #pragma omp atomic
        count += local_count;
    }
    // Finaliza a medição de tempo
    end = omp_get_wtime();
    // Calcula a estimativa de Pi
    long double pi = 4.0L * ((long double)count / n);
    printf("\nEstimativa de PI = %.9Lf\n", pi);
    wall_clock_time = end - start;
    printf("Tempo de execução: %f segundos\n", wall_clock_time);
    return 0;
}