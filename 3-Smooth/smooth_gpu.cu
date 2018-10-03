#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

#define TILE_SIZE 32 
#define MASK_WIDTH 13

#define MASK_OFFSET ((MASK_WIDTH-1)/2)

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

void writePPM(PPMImage *img) {

    fprintf(stdout, "P6\n");
    fprintf(stdout, "# %s\n", COMMENT);
    fprintf(stdout, "%d %d\n", img->x, img->y);
    fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

    fwrite(img->data, 3 * img->x, img->y, stdout);
    fclose(stdout);
}

__global__ void Smoothing_GPU_kernel(PPMPixel *data, PPMPixel *data_copy,  int image_x, int image_y, long n) {
    
    __shared__ PPMPixel private_data[ TILE_SIZE ][ TILE_SIZE ];
    
    //Index inicial do block e index da thread
    int y_begin_block = blockDim.y * blockIdx.y;	int ty = threadIdx.y;
    int x_begin_block = blockDim.x * blockIdx.x;	int tx = threadIdx.x;

	//Elemento atual da matriz que será processado 
	int current_index = (y_begin_block + ty) * image_x + (x_begin_block + tx);
    
    //Preenchendo a memoria compartilhada
    if (current_index < n)
		private_data[threadIdx.y][threadIdx.x] = data[ current_index ];
	else {
		PPMPixel zeroPixel = {0, 0, 0};
		private_data[threadIdx.y][threadIdx.x] = zeroPixel;
	}
	
	__syncthreads();
	
	if (current_index < n) {	
		
		int total_red = 0, total_blue = 0, total_green = 0, x, y, index;
		
		for (y = ty - MASK_OFFSET; y <= (ty + MASK_OFFSET); ++y) {
			for (x = tx - MASK_OFFSET; x <= (tx + MASK_OFFSET); ++x) {
				if (x >= 0 && y >= 0 && y < TILE_SIZE && x < TILE_SIZE) {				
					total_red += private_data[y][x].red;
					total_blue += private_data[y][x].blue;
					total_green += private_data[y][x].green;
				} else {
					int x_real = x_begin_block + x;
					int y_real = y_begin_block + y;
					if (x_real >= 0 && y_real >= 0 && y_real < image_y && x_real < image_x) {
						index = y_real * image_x + x_real;
						total_red += data[ index ].red;
                        total_blue += data[ index ].blue;
                        total_green += data[ index ].green;
					}
				}
			} //for x
		} //for y
		
		data_copy[ current_index ].red = total_red / (MASK_WIDTH * MASK_WIDTH);
		data_copy[ current_index ].blue = total_blue / (MASK_WIDTH * MASK_WIDTH);
		data_copy[ current_index ].green = total_green / (MASK_WIDTH * MASK_WIDTH);
	}
}


void Smoothing_CPU_Serial(PPMImage *image) {
    
    PPMPixel *d_data, *d_data_copy;
    long n = image->y * image->x;
    
    //aloca memory
    cudaMalloc((void **) &d_data, n * sizeof(PPMPixel));
	cudaMalloc((void **) &d_data_copy, n * sizeof(PPMPixel));
	
	//copy img for device 
	cudaMemcpy(d_data, image->data, n * sizeof(PPMPixel), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data_copy, image->data, n * sizeof(PPMPixel), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(ceil(image->x / (float)TILE_SIZE), ceil(image->y / (float)TILE_SIZE));	//Number of Blocks required
    dim3 dimGrid(TILE_SIZE, TILE_SIZE);														//Number of threads in each block
    
    double t_start, t_end;
    t_start = rtclock();
    //Executando a função na GPU
	Smoothing_GPU_kernel <<< dimBlock, dimGrid >>> (d_data, d_data_copy, image->x, image->y, n);
	t_end = rtclock();
	t_end = t_end - t_start;
	
	FILE *f = fopen("out.txt", "a");
    fprintf(f, "%0.6lfs\n", t_end); 
    fclose(f);
	
	//Copiando os dados de volta
	cudaMemcpy(image->data, d_data_copy, n * sizeof(PPMPixel), cudaMemcpyDeviceToHost);
	
	//Liberando a memória alocada
	cudaFree(d_data); cudaFree(d_data_copy); 
}

int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }

    char *filename = argv[1]; //Recebendo o arquivo!;

    PPMImage *image = readPPM(filename);

    Smoothing_CPU_Serial(image);
    
    writePPM(image);
 
    free(image);
}
