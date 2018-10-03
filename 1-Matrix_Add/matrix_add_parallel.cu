#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Maximo numero de blocks sem estourar a capacidade do hardware 
// (Esse não é o limite real do hardware do parsusy, é apenas uma estimação feita impiricamente)
int MAX_N_BLOCK = 1024; 
// Maximo numero de threads sem estourar a capacidade do hardware 
// (Esse não é o limite real do hardware do parsusy, é apenas uma estimação feita impiricamente)
int MAX_N_THREADS = 1024; 

/*Função que será executada na GPU*/
__global__ 
void sum_matrix(int *A, int *B, int *C, int n_process, int n) {
	// Calcula o index baseado no index do bloco, da thread e quatidade de elementos processados
	int index_begin = threadIdx.x * n_process + blockIdx.x * blockDim.x * n_process; 
	int n_end = index_begin + n_process;
	
	if (n_end > n) n_end = n;
	for (int i = index_begin; i < n_end; ++i)
		C[i] = A[i] + B[i];
}

int main()
{
	int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    int i, j;

    //Input
    int linhas, colunas;

    scanf("%d", &linhas);
    scanf("%d", &colunas);

	//Tamanho da memoria
	int size = sizeof(int) * linhas * colunas;
	//Quantidade de elementos
	long n = linhas * colunas;

    //Alocando memória na CPU
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);
    
    //Alocando memória na GPU
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    
    //Inicializar
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            A[i*colunas+j] =  B[i*colunas+j] = i+j;
        }
    }

	// Copy inputs to device
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	/*Chama a função que será executada na GPU*/
    //Quantidade de blocos = 512 = N
    //Quantidade de threads por bloco = (linhas * colunas) / N = M 
    
    int n_threads = MAX_N_THREADS; //Quantidade de threads que será utilizadas
    int n_blocks = MAX_N_BLOCK; //Quantidade de blocos
    int n_process = 1; //Quantidade de valores que cada thread deve processar
    
   
    if (n < MAX_N_THREADS)
		n_threads = n;													//Para matriz mt pequena
	else if (n > (MAX_N_BLOCK * MAX_N_THREADS))
		n_process = ceil(double(n) / (MAX_N_BLOCK * MAX_N_THREADS));	//Para matriz muito grande
	
	//Calculando a quantidade de blocos
	n_blocks = ceil(double(n) / (n_threads * n_process));
	
    // chama a função que será executada na GPU
    sum_matrix <<< n_blocks, n_threads >>> (d_A, d_B, d_C, n_process, n); 
  	
	//Copia resultado do device para host
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    long long int somador=0;
    //Manter esta computação na CPU
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            somador+=C[i*colunas+j];   
        }
    }
    
    printf("%lli\n", somador);
    
    free(A);
    free(B);
    free(C);
    
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
    
    return 0;
}

