#include <stdio.h>
#include <stdlib.h>     // Para rand_r, RAND_MAX, NULL
#include <omp.h>        // Para OpenMP

int main() {
    
    // Etapa 1: Leitura sequencial do número de pontos
    long long num_pontos;
    printf("Digite o numero de pontos (dardos): ");
    scanf("%lld", &num_pontos); 

    long long contagem_circulo = 0;
    
    // Variáveis para medição de tempo
    double start_time, end_time, tempo_gasto;

    // Marca o tempo de início ANTES da região paralela
    start_time = omp_get_wtime();

    // Inicia a região paralela.
    // 'reduction(+:contagem_circulo)' implementa a redução.
    #pragma omp parallel reduction(+:contagem_circulo)
    {
        // Cada thread inicializa sua própria semente (estado local)
        unsigned int seed = (unsigned int)omp_get_thread_num();

        // Mapeamento: O loop 'for' divide o total de 'num_pontos'.
        // 'schedule(guided)' aplica o mapeamento dinâmico guiado.
        #pragma omp for schedule(guided)
        for (long long i = 0; i < num_pontos; ++i) {
            
            // Tarefa Aglomerada: Fusão de "gerar" + "verificar"
            
            // 1. Gera Ponto (x, y)
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;
            
            // 2. Checa se está dentro do círculo
            if (x * x + y * y <= 1.0) {
                // Acumula no contador local (gerenciado pela cláusula 'reduction')
                contagem_circulo++; 
            }
        }
        // Uma barreira implícita existe ao final do loop 'for'.
    } 
    // Ao final da região 'parallel', os contadores locais são somados.

    // Marca o tempo de fim DEPOIS da região paralela
    end_time = omp_get_wtime();
    tempo_gasto = end_time - start_time;

    // Etapa 5: Cálculo sequencial de PI
    double pi_calculado = 4.0 * (double)contagem_circulo / num_pontos;

    // Etapa 6: Impressão sequencial
    printf("--------------------------------------\n");
    printf("Numero total de pontos:   %lld\n", num_pontos);
    printf("Pontos dentro do circulo: %lld\n", contagem_circulo);
    printf("Valor aproximado de PI:   %.10f\n", pi_calculado);
    printf("--------------------------------------\n");
    printf("Tempo de avaliacao (s):   %.6f\n", tempo_gasto); // Imprime o tempo gasto

    return 0;
}
