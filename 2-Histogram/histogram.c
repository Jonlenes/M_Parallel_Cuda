/*

Histogram
	- Um método para extrair recursos e padrões notáveis de grandes conjuntos de dados
		- Extração de recursos para reconhecimento de objetos em imagens
		- Detecção de fraudes em transações com cartão de crédito
		- Correlacionando movimentos de objetos celestes em astrofísica
	- Histogramas básicos - para cada elemento no conjunto de dados, use o valor para identificar um “contador de bin” para incrementar

A simple parallel histogram algorithm
	- Particionar a entrada em seções
	- Ter cada thread para tirar uma seção da entrada
	- Cada segmento percorre sua seção.
	- Para cada letra, incremente o contador apropriado
	
O particionamento de entrada afeta a eficiência do acesso à memória
	- Particionamento seccionado resulta em baixa eficiência de acesso à memória
		- Threads adjacentes não acessam locais de memória adjacentes
		- Acessos não são unidos
		- A largura de banda de DRAM é mal utilizada		
	- Alterar para particionamento intercalado
		- Todos os encadeamentos processam uma seção contígua de elementos
		- Todos eles se movem para a próxima seção e repetem
		- Os acessos de memória são unidos
		
Objective
	- Para entender as corridas de dados na computação paralela
		- As corridas de dados podem ocorrer durante a execução de operações de leitura-modificação-gravação
		- Corridas de dados podem causar erros difíceis de reproduzir
		- As operações atômicas são projetadas para eliminar tais raças de dados
		
Read-Modify-Write Usado em Padrões de Colaboração
	- Por exemplo, vários caixas de banco contam a quantidade total de dinheiro no cofre
	- Cada um pega uma pilha e conta
	- Ter uma exibição central do total em execução
	- Sempre que alguém terminar de contar uma pilha, leia o total corrente atual (lido) e adicione o subtotal da pilha ao total corrente (modificar-escrever)
	- Um mau resultado
		- Algumas das pilhas não foram contabilizadas no total final

A Common Parallel Service Pattern
	- Por exemplo, vários agentes de atendimento ao cliente que atendem clientes em espera
	- O sistema mantém dois números,
		- o número a ser dado ao próximo cliente entrante (I)
		- o número para o cliente ser atendido em seguida (S)
	- O sistema fornece a cada cliente entrante um número (leia I) e incrementa o número a ser dado ao próximo cliente em 1 (modificar-escrever I)
	- Um display central mostra o número do cliente a ser atendido a seguir
	- Quando um agente se torna disponível, ele chama o número (leia S) e incrementa o número de exibição em 1 (modificar-escrever S)
	- Resultados ruins
		- Vários clientes recebem o mesmo número, apenas um deles recebe serviço
		- Vários agentes atendem ao mesmo número

Objetivo das Operações Atômicas - Garantir Bons Resultados
	- Aprender a usar operações atômicas em programação paralela
		- Conceitos de operação atômica
		- Tipos de operações atômicas em CUDA
		- Funções intrínsecas
		- Um kernel básico de histograma
		
Key Concepts of Atomic Operations
	- Uma operação de leitura-modificação-gravação executada por uma única instrução de hardware em um endereço de local de memória
		- Leia o valor antigo, calcule um novo valor e escreva o novo valor no local
	- The hardware ensures that no other threads can perform another read-modify-write operation on the same location until the current atomic operation is complete	

Atomic Operations in CUDA
	- Performed by calling functions that are translated into single instructions
		- Atomic add, sub, inc, dec, min, max, exch (exchange), CAS (compare and swap)
	- Atomic Add
		- int atomicAdd(int* address, int val);
		- reads the 32-bit word old from the location pointed to by address in global or shared memory, computes (old + val), and stores the result back to memory at the same address. The function returns old.
		- unsigned int atomicAdd(unsigned int* address, unsigned int val);
		- unsigned long long int atomicAdd(unsigned long long int* address, unsigned long long int val);
		- float atomicAdd(float* address, float val);
Objective
	- Learn to write a high performance kernel by privatizing outputs
	- Privatização como uma técnica para reduzir a latência, aumentar o rendimento e reduzir a serialização
	- Um kernel de histograma privatizado de alto desempenho
	- Practical example of using shared memory and L2 cache atomic operations
Privatization
	- Cost and Benefit of Privatization
		- Cost
			- Overhead for creating and initializing private copies
			- Overhead for accumulating the contents of private copies into the final copy
		- Benefit
			- Much less contention and serialization in accessing both the private copies and the final copy
			- The overall performance can often be improved more than 10x
Shared Memory Atomics for Histogram
	- Each subset of threads are in the same block
	- Maior rendimento do que os dados atômicos DRAM (100x) ou L2 (10x)
	- Less contention – only threads in the same block can access a shared memory variable
	- This is a very important use case for shared memory!
 */


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

typedef struct {
	unsigned char red, green, blue;
} PPMPixel;

typedef struct {
	int x, y;
	PPMPixel *data;
} PPMImage;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


static PPMImage *readPPM(const char *filename) {
	char buff[16];
	PPMImage *img;
	FILE *fp;
	int c, rgb_comp_color;
	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	img = (PPMImage *) malloc(sizeof(PPMImage));
	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n')
			;
		c = getc(fp);
	}

	ungetc(c, fp);
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
				filename);
		exit(1);
	}

	if (rgb_comp_color != RGB_COMPONENT_COLOR) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n')
		;
	img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	fclose(fp);
	return img;
}


void Histogram(PPMImage *image, float *h) {

	int i, j,  k, l, x, count;
	int rows, cols;

	float n = image->y * image->x;

	cols = image->x;
	rows = image->y;

	//printf("%d, %d\n", rows, cols );

	for (i = 0; i < n; i++) {
		image->data[i].red = floor((image->data[i].red * 4) / 256);
		image->data[i].blue = floor((image->data[i].blue * 4) / 256);
		image->data[i].green = floor((image->data[i].green * 4) / 256);
	}

	count = 0;
	x = 0;
	for (j = 0; j <= 3; j++) {
		for (k = 0; k <= 3; k++) {
			for (l = 0; l <= 3; l++) {
				for (i = 0; i < n; i++) {
					if (image->data[i].red == j && image->data[i].green == k && image->data[i].blue == l) {
						count++;
					}
				}
				h[x] = count / n; //Histograma normalizado
				count = 0;
				x++;
			}				
		}
	}
}

int main(int argc, char *argv[]) {

	if( argc != 2 ) {
		printf("Too many or no one arguments supplied.\n");
	}

	double t_start, t_end;
	int i;
	char *filename = argv[1]; //Recebendo o arquivo!;
	
	//scanf("%s", filename);
	PPMImage *image = readPPM(filename);

	float *h = (float*)malloc(sizeof(float) * 64);

	//Inicializar h
	for(i=0; i < 64; i++) h[i] = 0.0;

	t_start = rtclock();
	Histogram(image, h);
	t_end = rtclock();

	for (i = 0; i < 64; i++){
		printf("%0.3f ", h[i]);
	}
	printf("\n");
	fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);  
	free(h);
}
