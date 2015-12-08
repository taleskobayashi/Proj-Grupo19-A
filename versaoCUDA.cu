#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <time.h>

#define CHECK(r) {_check((r), __LINE__);}
using namespace cv;
using namespace std;

//Função que verifica caso houver erro na sincronização do kernel, qual é o erro
inline __host__ void PostKernelCall(void)
{
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if(errSync != cudaSuccess)
		printf("\nSync kernel error: %s.", cudaGetErrorString(errSync));
	if(errAsync != cudaSuccess)
		printf("\nAsync kernel error: %s.\n", cudaGetErrorString(errAsync));
}

int prime = 5;

Mat writeImage(Mat img, uchar *red, uchar *green, uchar *blue);

//função que checa se uma chamada de uma função de cuda teve falha e mostra qual a  falha
inline void _check(cudaError_t r, int line)
{
  if (r != cudaSuccess)
  {
    printf("CUDA error on line %d: %s\n", line, cudaGetErrorString(r));
    exit(0);
  }
}

//Função que recebe os valores RGB dos pixels da imagem
void getChannels(Mat image, uchar *red, uchar *green, uchar *blue)
{
    
    for(int i = 0; i < image.cols; i++)
    {
        for(int j = 0; j < image.rows; j++)
        {
            int index = i + j*image.cols;
            red[index] = image.at<Vec3b>(j,i)[2];
            green[index] = image.at<Vec3b>(j,i)[1];
            blue[index] = image.at<Vec3b>(j,i)[0];
        }
    }
}

//função que verifica se o computador testado possui um device compativel com CUDA
__host__ void deviceQuery()
{
	int n;
	cudaDeviceProp prop;
	cudaGetDeviceCount(&n);
	if(n < 1){ cout << "Error no cuda devices found!" << endl;  exit(-1);  }

	cudaGetDeviceProperties(&prop, 0);
	cout << prop.name << endl;
	cout << "Max threads per block allowed: " << prop.maxThreadsPerBlock << endl;
	cout << "Cuda support for: " << prop.major << "." << prop.minor << endl;
}

//função que imprime os pixels computados numa noma imagem
Mat writeImage(Mat img, uchar *red, uchar *green, uchar *blue)
{

    for(int i = 0; i < img.cols; i++)
    {
        for(int j = 0; j < img.rows; j++)
        {
            int index = i + j*img.cols;
            img.at<Vec3b>(j,i)[2] = red[index];
            img.at<Vec3b>(j,i)[1] = green[index];
            img.at<Vec3b>(j,i)[0] = blue[index];
        }
    }
    return img;
}

//função principal em CUDA
__global__ void ComputeKernel(uchar *red, uchar *green, uchar *blue,
					 uchar *out_red, uchar *out_green, uchar *out_blue, int width, int height, int pr)

{
	int size = width * height;
	int index = threadIdx.x + blockIdx.x*blockDim.x;//y +x*tamanholinha
	if(index < size) //processar apenas as threads com pixel correspondente, evitando threads fantasmas
	{
		//computa
		
		int listI[25];
		int listJ[25];
		int pixelI = index % width;
		int pixelJ = index / width;

		int amount = pr;
    	int far = (amount - 1)/2;
   		int r_i = pixelI - far;
    	int r_j, c = 0;
    	for(int i = 0; i < amount; i++)
    	{
        	r_j = pixelJ - far;
        	for(int j = 0; j < amount; j++)
        	{
            	if(r_i >= 0 && r_i < width && r_j >= 0 && r_j < height)
	            {
	                listI[c] = r_i;
	                listJ[c] = r_j;
	            }
	            else
	            {
	                listI[c] = -1;
	                listJ[c] = -1;
	            }
	            c++;
	            r_j++;
	        }
	        r_i ++;
	    }

	    int sum_r = 0, sum_g = 0, sum_b = 0;
	    int count = 0;
	    for(int i = 0; i < amount*amount; i++)
	    {
	        if(listI[i] == -1) continue;
	        int id = listI[i] + listJ[i] * width;
	        sum_r += red[id];
	        sum_g += green[id];
	        sum_b += blue[id];
	        count ++;
	    }

	    out_red[index] = sum_r/count;
	    out_green[index] = sum_g/count;
	    out_blue[index] = sum_b/count;					 
	}
}

int main(int argc, char **argv)
{
	clock_t tempoInicial, tempoFinal;
    	double tempoGasto;
    	tempoInicial = clock();
	deviceQuery();

	Mat image;
	//lê a imagem
	image = imread("imagem.jpg", 1);
	uchar *h_red, *h_green, *h_blue, *d_red, *d_green, *d_blue;
   	uchar *d_out_red, *d_out_green, *d_out_blue;



        int w = image.cols;
	int h = image.rows;
	int size = w*h;
		
	//alocando espaço para os vetores tanto para o device como para o host computar
	h_red = (uchar *)malloc(sizeof(uchar)*size);
	h_green = (uchar *)malloc(sizeof(uchar)*size);
	h_blue = (uchar *)malloc(sizeof(uchar)*size);

	CHECK(cudaMalloc((void **)&d_red, sizeof(uchar)*size));
	CHECK(cudaMalloc((void **)&d_green, sizeof(uchar)*size));
	CHECK(cudaMalloc((void **)&d_blue, sizeof(uchar)*size));
	CHECK(cudaMalloc((void **)&d_out_red, sizeof(uchar)*size));
	CHECK(cudaMalloc((void **)&d_out_green, sizeof(uchar)*size));
	CHECK(cudaMalloc((void **)&d_out_blue, sizeof(uchar)*size));

	getChannels(image, h_red, h_green, h_blue);

	CHECK(cudaMemcpy(d_red, h_red, sizeof(uchar)*size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_green, h_green, sizeof(uchar)*size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_blue, h_blue, sizeof(uchar)*size, cudaMemcpyHostToDevice));


	//           <<<numBlocks,numThreads per block>>>
	ComputeKernel<<<65000, 512>>>(d_red, d_green, d_blue, d_out_red,
										 d_out_green, d_out_blue, w, h, prime);

	PostKernelCall();
	
	CHECK(cudaMemcpy(h_red, d_out_red, sizeof(uchar)*size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(h_green, d_out_green, sizeof(uchar)*size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(h_blue, d_out_blue, sizeof(uchar)*size, cudaMemcpyDeviceToHost));

    	Mat newimg = writeImage(image.clone(), h_red, h_green, h_blue);
    	imwrite("imagemNova.jpg", newimg);
   	cudaFree(d_out_red);
    	cudaFree(d_out_green);
   	cudaFree(d_out_blue);
    	cudaFree(d_red);
    	cudaFree(d_green);
    	cudaFree(d_blue);

    	tempoFinal = clock();
    	tempoGasto = tempoFinal-tempoInicial;
    	printf("Tempo em segundos: %.2f\n", tempoGasto/CLOCKS_PER_SEC);
    	
	waitKey(0);
   	

	return 0;
}
