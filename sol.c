#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> // Para tipos de inteiros (u64, i64 etc.)
#include <time.h>

#ifndef BATCH
#define BATCH 65536 // Define o tamanho do "super-bloco" de trabalho
#endif

#ifndef MICRO
#define MICRO 1024 // Define o tamanho do "micro-bloco" (para caber no cache L1/L2)
#endif

// --- Define o número de threads a usar ---
#ifndef T
#define T 8
#endif

// --- Definições de tipos para facilitar a leitura ---
typedef uint64_t u64;
typedef int64_t  i64;
typedef uint32_t u32;
typedef int32_t  i32;

// --- Implementação do Gerador de Números Aleatórios (PRNG) xoroshiro128+ ---
// Estado interno do gerador
typedef struct { u64 s[2]; } xoroshiro128_state;

// Operação de rotação de bits (helper para o PRNG)
static inline u64 rotl(u64 x, i32 k) {
    return (x << k) | (x >> (64 - k));
}

// Usado para inicializar (semear) o estado do xoroshiro
static inline u64 splitmix64(u64* x) {
    u64 z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Gera o próximo número aleatório de 64 bits
static inline u64 xoroshiro128_next(xoroshiro128_state *s) {
    u64 s0 = s->s[0];
    u64 s1 = s->s[1];
    u64 r = s0 + s1; // O resultado é a soma dos estados
    s1 ^= s0;
    s->s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
    s->s[1] = rotl(s1, 36);
    return r;
}
// --- Fim do PRNG ---

int main(void) {
    u64 n;
    if (printf("n = ") < 0) return 1;
    if (scanf("%lu", &n) != 1) return 1; // Lê o número total de pontos

    // --- Define o número de threads explicitamente ---
    omp_set_num_threads(T);

    double t0 = omp_get_wtime(); // Inicia o timer
    
    // Constante para normalizar inteiros de 32 bits para [0, 1)
    const float inv32 = 1.0f / 4294967296.0f;
    
    // Calcula quantos "super-blocos" (BATCH) são necessários
    i64 total_blocks = (n + BATCH - 1) / BATCH;
    i64 global_count = 0; // Contador total de pontos dentro do círculo

    // Divide os "super-blocos" (total_blocks) entre as threads
    #pragma omp parallel for reduction(+:global_count) schedule(static)
    for (i64 b = 0; b < total_blocks; ++b) {
        
        // --- Inicialização do PRNG (Thread-local) ---
        xoroshiro128_state rng;
        // Garante uma semente única para cada thread/bloco
        u64 seed = (u64)time(NULL) ^ (u64)(b + omp_get_thread_num() * 1315423911ULL);
        u64 z = seed;
        // Usa splitmix64 para inicializar o estado do xoroshiro
        rng.s[0] = splitmix64(&z);
        rng.s[1] = splitmix64(&z);

        // --- Buffers locais alinhados para melhor performance SIMD ---
        u64 rnd[BATCH] __attribute__((aligned(64)));  // Armazena a geração em lote
        float xs[MICRO] __attribute__((aligned(64))); // Armazena coordenadas X do micro-bloco
        float ys[MICRO] __attribute__((aligned(64))); // Armazena coordenadas Y do micro-bloco

        // Calcula o tamanho real deste bloco (pode ser menor que BATCH no final)
        u64 start = b * BATCH;
        u64 end = start + BATCH;
        if (end > n) end = n;
        i32 block_size = (int)(end - start);

        // Fase 1: Geração de Números Aleatórios (Sequencial)
        // Gera todos os números necessários para este BATCH de uma vez.
        // Isso separa o gargalo serial (RNG) do gargalo paralelizável (cálculo).
        for (i32 j = 0; j < block_size; ++j)
            rnd[j] = xoroshiro128_next(&rng);

        // Fase 2: Processamento em Micro-Blocos (Otimizado para Cache)
        i64 local_count = 0;
        for (i32 base = 0; base < block_size; base += MICRO) {
            
            // Calcula o tamanho deste micro-bloco (pode ser < MICRO)
            i32 m = MICRO;
            if (base + m > block_size) m = block_size - base;

            // 2a: Conversão de tipos (Data Marshalling)
            // Converte os u64 do buffer 'rnd' para coordenadas 'xs' e 'ys'
            // O objetivo é encher os buffers 'xs' e 'ys' para caberem no cache L1/L2.
            for (i32 j = 0; j < m; ++j) {
                u64 u = rnd[base + j];
                u32 lo = (u32)u;       // Usa 32 bits inferiores para X
                u32 hi = (u32)(u >> 32); // Usa 32 bits superiores para Y
                // Normaliza para [-1.0, 1.0]
                xs[j] = (float)lo * inv32 * 2.0f - 1.0f;
                ys[j] = (float)hi * inv32 * 2.0f - 1.0f;
            }

            // 2b: Computação Vetorizada (SIMD)
            // Este é o loop principal. Como os dados (xs, ys) estão no cache
            // e não há mais chamadas ao RNG, o compilador pode vetorizar.
            #pragma omp simd reduction(+:local_count)
            for (int j = 0; j < m; ++j)
                // O 'local_count +=' aqui é otimizado pela redução SIMD
                local_count += (xs[j]*xs[j] + ys[j]*ys[j] <= 1.0f);
        }
        // Acumula o resultado deste BATCH no contador global (via redução OMP)
        global_count += local_count;
    }

    double t1 = omp_get_wtime(); // Para o timer
    
    // Cálculo final e impressão
    long double pi = 4.0L * ((long double)global_count / (long double)n);
    printf("\nEstimativa de PI = %.9Lf\n", pi);
    printf("Threads usadas: %d\n", T); // Imprime o número de threads
    printf("Tempo de execução: %f segundos\n", t1 - t0);
    return 0;
}