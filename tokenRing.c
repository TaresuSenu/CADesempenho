#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

#define MAX_TOKEN 39 // colocar 1 a menos que T
#define T 40

sem_t mutex[T+1];
int token;

void *solve(void * eu){
    int Eu = *(int *)eu;
    sem_wait(&mutex[Eu]);
    int libera = Eu+1;
    token++;
    printf("%d\n", token);
    if(token==MAX_TOKEN) return;
    sem_post(&mutex[libera]);
}

int main(){
    pthread_t t[T];

    for(int i=1;i<=T;i++){
        sem_init(&mutex[i], 0, 0);
    }

    int p[T+1];
    for(int i=1;i<T;i++){
        p[i] = i;
        if (pthread_create(&t[i], 0, (void *) solve, (void *) &p[i]) != 0) {
            printf("Error creating thread! Exiting! \n");
            exit(0);
        }
    }

    // thread 0 (main) settando o valor 0 para token e liberando o 1;
    token = 0;
    printf("%d\n", token);
    sem_post(&mutex[1]);

    for(int i=1;i<T;i++){
        pthread_join(t[i], 0);
    }

    //thread 0 incrementa o token

    token++;

    printf("%d\n", token);

    return 0;
}
