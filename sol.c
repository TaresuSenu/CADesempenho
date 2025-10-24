#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifndef BATCH
#define BATCH 65536 // tamanho do super-bloco
#endif

#ifndef MICRO
#define MICRO 1024 // tamanho do micro-bloco (para cache L1/L2)
#endif

// define num de threads
#ifndef T
#define T 8
#endif

typedef uint64_t u64;
typedef int64_t  i64;
typedef uint32_t u32;
typedef int32_t  i32;

// xoroshiro128+ 
typedef struct { u64 s[2]; } xoroshiro128_state; // estado interno do PRNG

static inline u64 rotl(u64 x, i32 k) { // helper: rotação de bits
    return (x << k) | (x >> (64 - k));
}

static inline u64 splitmix64(u64* x) { // seeder
    u64 z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline u64 xoroshiro128_next(xoroshiro128_state *s) { // gera próximo u64 aleatório
    u64 s0 = s->s[0];
    u64 s1 = s->s[1];
    u64 r = s0 + s1;
    s1 ^= s0;
    s->s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
    s->s[1] = rotl(s1, 36);
    return r;
}
// fim de xoroshiro128+ 

int main(void) {
    u64 n;
    if (printf("n = ") < 0) return 1;
    if (scanf("%lu", &n) != 1) return 1; // lê N 

    omp_set_num_threads(T); // define o número de threads

    double t0 = omp_get_wtime(); // inicia timer
    
    const float inv32 = 1.0f / 4294967296.0f; // constante para normalizar u32 -> [0, 1)
    
    i64 total_blocks = (n + BATCH - 1) / BATCH; // calcula número de super-blocos
    i64 global_count = 0; // contador total (qtd dentro do círculo)

    // divide os super-blocos (total_blocks) entre as threads
    #pragma omp parallel for reduction(+:global_count) schedule(static)
    for (i64 b = 0; b < total_blocks; ++b) {
        
        // inicializa PRNG
        xoroshiro128_state rng;
        u64 seed = (u64)time(NULL) ^ (u64)(b + omp_get_thread_num() * 1315423911ULL);
        u64 z = seed;
        rng.s[0] = splitmix64(&z);
        rng.s[1] = splitmix64(&z);

        // buffers locais alinhados para performance SIMD
        u64 rnd[BATCH] __attribute__((aligned(64)));  // buffer para u64 aleatórios
        float xs[MICRO] __attribute__((aligned(64))); // buffer para coordenadas X
        float ys[MICRO] __attribute__((aligned(64))); // buffer para coordenadas Y

        // calcula o tamanho real deste bloco
        u64 start = b * BATCH;
        u64 end = start + BATCH;
        if (end > n) end = n;
        i32 block_size = (int)(end - start);

        // gera todos os u64 aleatórios em lote (serial)
        for (i32 j = 0; j < block_size; ++j)
            rnd[j] = xoroshiro128_next(&rng);

        // processa em micro-blocos (otimizado p/ cache)
        i64 local_count = 0;
        for (i32 base = 0; base < block_size; base += MICRO) {
            
            // calcula o tamanho real do micro-bloco
            i32 m = MICRO;
            if (base + m > block_size) m = block_size - base;

            // converte u64 para coords X, Y [-1.0, 1.0]
            // enche os buffers xs/ys (que cabem no cache L1/L2)
            for (i32 j = 0; j < m; ++j) {
                u64 u = rnd[base + j];
                u32 lo = (u32)u;       // 32 bits inferiores p/ X
                u32 hi = (u32)(u >> 32); // 32 bits superiores p/ Y
                // normaliza p/ [-1, 1]
                xs[j] = (float)lo * inv32 * 2.0f - 1.0f; 
                ys[j] = (float)hi * inv32 * 2.0f - 1.0f; 
            }

            // computação vetorizada (SIMD)
            // processa os dados que estão no cache (xs, ys)
            #pragma omp simd reduction(+:local_count)
            for (int j = 0; j < m; ++j)
                // Acumula se (x,y) está no círculo
                local_count += (xs[j]*xs[j] + ys[j]*ys[j] <= 1.0f);
        }
        // acumula o resultado deste BATCH (redução)
        global_count += local_count;
    }

    double t1 = omp_get_wtime(); // para o timer
    
    // cálculo final e impressão
    long double pi = 4.0L * ((long double)global_count / (long double)n);
    printf("\nEstimativa de PI = %.9Lf\n", pi);
    printf("Threads usadas: %d\n", T);
    printf("Tempo de execução: %f segundos\n", t1 - t0);
    return 0;
}