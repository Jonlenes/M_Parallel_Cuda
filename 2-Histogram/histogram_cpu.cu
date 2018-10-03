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

int n_block = 512;
int n_thread_block = 512;

__global__ 
void histogram_kernel(PPMPixel *data, float n, float *hist) {
	
	__shared__ float hist_private[64];
	if (threadIdx.x < 64) 
		hist_private[threadIdx.x] = 0; //Inicializa o histograma privado
	
	__syncthreads();
	
	int i, j, k, l, x, count;
	
	count = 0;
	x = 0;
	int begin = threadIdx.x + blockIdx.x * blockDim.x; //Index do inicio 
	int stride = blockDim.x * gridDim.x; // stride is total number of threads
	
	for (j = 0; j <= 3; j++) {
		for (k = 0; k <= 3; k++) {
			for (l = 0; l <= 3; l++) {
				for (i = begin; i < n; i += stride ) {
					if (data[i].red == j && data[i].green == k && data[i].blue == l)
						count++;
				}
				//printf("Bd: %d 	Bi: %03d 	Ti: %03d 	st: %d 	h: %.6f\n", blockDim.x, blockIdx.x, threadIdx.x, stride, ((float) count)/n);
				atomicAdd(hist_private + x, ((float) count)/n);
				count = 0;
				x++;
			}				
		}
	}
	
	__syncthreads();
	
	if (threadIdx.x < 64)
		atomicAdd(hist + threadIdx.x, (float) hist_private[threadIdx.x] ); //Juntando os histogramas 
}


void Histogram(PPMImage *image, float *h) {

	float n = image->y * image->x;

	for (int i = 0; i < n; i++) {
		image->data[i].red = floor((image->data[i].red * 4) / 256);
		image->data[i].blue = floor((image->data[i].blue * 4) / 256);
		image->data[i].green = floor((image->data[i].green * 4) / 256);
	}

	//Alocando memória na GPU e Coping inputs to device******************
	PPMPixel *d_data;
	float *d_h;
	int size_hist = 64 * sizeof(float);
		
	cudaMalloc((void **)&d_data, n * sizeof(PPMPixel));
	cudaMalloc((void **)&d_h, size_hist);
	
	cudaMemcpy(d_data, image->data, n * sizeof(PPMPixel), cudaMemcpyHostToDevice);
	//******************************************************************
	
	histogram_kernel <<< n_block, n_thread_block >>> (d_data, n, d_h); 
	//cudaDeviceSynchronize();
	
	//Copia resultado do device para host
	cudaMemcpy(h, d_h, size_hist, cudaMemcpyDeviceToHost);
	
	//Liberando a memória alocada
	cudaFree(d_data);
	cudaFree(d_h); 
	
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
