/* seq.c
Implementação sequencial simples para estimar pi por Monte Carlo.
Utiliza a função drand48_r e inclui medição de tempo de execução.
Compilar: gcc seq.c -o seq -fopenmp
Observação: a flag -fopenmp será utilizada para aproveitar o recurso de
cronômetro da biblioteca.
Uso: ./seq (o número de pontos será pedido no console)
*/
#include <stdio.h>
#include <stdlib.h> // Para drand48_r, drand48_data, srand48_r, geração reentrante de números aleatórios entre 0 e 1 em ponto flutuante
#include <omp.h> // Para omp_get_wtime()
#include <time.h> // Para time()

int main() {
    int count; // Pontos dentro do círculo
    long int n; // Número total de pontos
    long int i; // Contador
    double pi; // Valor de pi
    double x, y; // Coordenadas do ponto

    // Variáveis para medir o tempo
    double start_time, end_time, wall_clock_time;

    // Estrutura para o estado do gerador reentrante
    struct drand48_data randBuffer;

    // Semente inicial para o gerador reentrante.
    printf("\nn = "); // Pergunta a quantidade de pontos
    scanf("%ld", &n); // Lê a quantidade de pontos do console
    //inicializa o contador

    count = 0;
    // Inicializa o estado do gerador reentrante com a semente.
    srand48_r(time(NULL), &randBuffer);

    // --- Início da medição de tempo ---
    start_time = omp_get_wtime();
    for (i = 0; i < n; i++) {
        // Gera números aleatórios entre 0 e 1 usando drand48_r
        drand48_r(&randBuffer, &x);
        drand48_r(&randBuffer, &y);
        if (x * x + y * y <= 1) {
            count++;
        }
    }
    // --- Fim da medição de tempo ---
    end_time = omp_get_wtime();
    wall_clock_time = end_time - start_time;
    pi = (double)count / n * 4;
    printf("\nEstimativa de PI: %.9f\n", pi);
    printf("Tempo de execução (Wall Clock Time): %f segundos\n", wall_clock_time);
    return 0;
}