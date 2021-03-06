#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <omp.h>
#include <time.h>

using namespace cv;
using namespace std;

int Namount;

//estrutura usada para salvar o par de coordenadas de um pixel
typedef struct 
{
    int i, j;
}pixel;

//salva os dados computados dos pixels no vetor para criar a imagem nova
Mat writeImage(Mat img, uchar *red, uchar *green, uchar *blue)
{
    int Num_Threads,tid;
    
    for(int i = 0; i < img.cols; i++)
    {
        tid = omp_get_thread_num();
        if(tid == 0){
            Num_Threads = omp_get_num_threads();

        }
        omp_set_num_threads(4);
        #pragma omp parallel for
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

//pega as informações dos valores RGB dos pixels da imagem
void getChannels(Mat image, uchar *red, uchar *green, uchar *blue)
{
    #pragma omp parallel for
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

//calcula os pixels vizinhos do pixel central
void getNeighbors(pixel center, pixel *list, int width, int height)
{
    int amount = Namount;
    int far = (amount - 1)/2;
    int r_i = center.i - far;
    int r_j, c = 0;
    //#pragma omp parallel for private(Num_Threads)
    for(int i = 0; i < amount; i++)
    {
        r_j = center.j - far;
        /*
       */
        for(int j = 0; j < amount; j++)
        {
            if(r_i >= 0 && r_i < width && r_j >= 0 && r_j < height)
            {
                pixel current;
                current.i = r_i;
                current.j = r_j;
                list[c] = current;
                
            }
            else
            {
                list[c].i = -1;
                list[c].j = -1;
            }
            c++;
            r_j++;
        }
        r_i ++;
    }
}

//soma os valores e calcula o valor final do pixel
void avaluate(pixel *list, uchar &out_r, uchar &out_g, uchar &out_b, uchar *red, 
                                                    uchar *green, uchar *blue, int width)
{
    int amount = Namount;
    int sum_r = 0, sum_g = 0, sum_b = 0;
    int count = 0;
    for(int i = 0; i < amount*amount; i++)
    {
        if(list[i].i == -1) continue;
        int index = list[i].i + list[i].j * width;
        sum_r += red[index];
        sum_g += green[index];
        sum_b += blue[index];
        count ++;
    }

    out_r = sum_r/count;
    out_g = sum_g/count;
    out_b = sum_b/count;
}

//função que calcula o valor do pixel de acordo com seu index e chama as funções de calculo
void computeKernel(uchar *red, uchar *green, uchar *blue, int width, int height, int size, 
                                                    uchar *out_r, uchar *out_g, uchar *out_b)
{
    int amount = Namount;
    pixel *list = (pixel *)malloc(sizeof(pixel)*amount*amount);
    for(int i = 0; i < size; i++)
    {
        pixel p; 
        p.i = i % width;
        p.j = (i / width);
        getNeighbors(p, list, width, height);
        avaluate(list, out_r[i], out_g[i], out_b[i], red, green, blue, width);
    }
}

int main(int argc, char** argv)
{
    clock_t tempoInicial, tempoFinal;
    double tempoGasto;
    tempoInicial = clock();
    Mat image;
    Namount = atoi(argv[1]);
    std::string str("grey4.jpg");
    image = imread( str.c_str(), 1 );
    uchar *red, *green, *blue;
    uchar *out_red, *out_green, *out_blue;
    int WIDTH = image.cols;
    int HEIGHT = image.rows;
    int size = WIDTH * HEIGHT;

    red = (uchar *)malloc(sizeof(uchar)*size);
    green = (uchar *)malloc(sizeof(uchar)*size);
    blue = (uchar *)malloc(sizeof(uchar)*size);
    out_red = (uchar *)malloc(sizeof(uchar)*size);
    out_green = (uchar *)malloc(sizeof(uchar)*size);
    out_blue = (uchar *)malloc(sizeof(uchar)*size);
    
    getChannels(image, red, green, blue);
    computeKernel(red, green, blue, WIDTH, HEIGHT, size, out_red, out_green, out_blue);
    Mat newimg = writeImage(image.clone(), out_red, out_green, out_blue);
    
    tempoFinal = clock();
    tempoGasto = tempoFinal-tempoInicial;
    printf("Tempo em segundos: %.2f\n", tempoGasto/CLOCKS_PER_SEC);

    imwrite("imagemNova.jpg", newimg);

    waitKey(0);

    return 0;
}
