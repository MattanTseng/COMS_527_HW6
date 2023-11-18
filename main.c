#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util/img.h"
#include "neural/activations.h"
#include "neural/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"

int main() {
	srand(time(NULL));

	//TRAINING
	printf("Starting training");
	// record the start time
	time_t startTime = clock();

	int number_imgs = 10000;
	time_t startLoadTime = clock();
	Img** imgs = csv_to_imgs("./data/mnist_test.csv", number_imgs);
	time_t doneLoadTime = clock();


	NeuralNetwork* net = network_create(784, 300, 10, 0.1);
	
	time_t startTrainTime = clock();
	network_train_batch_imgs(net, imgs, number_imgs);
	time_t doneTrainTime = clock();
	
	network_save(net, "testing_net");

	time_t endTime = clock();

	double trainingTime = (double)(doneTrainTime - startTrainTime) / CLOCKS_PER_SEC;
	double imageLoadTime = (double)(doneLoadTime - startLoadTime) / CLOCKS_PER_SEC;
	double totalTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;


	printf("Image load time: %f \n", imageLoadTime);
	printf("Training time: %f \n", trainingTime);
	printf("Total time: %f \n", totalTime);


	// PREDICTING
	// int number_imgs = 3000;
	// Img** imgs = csv_to_imgs("data/mnist_test.csv", number_imgs);
	// NeuralNetwork* net = network_load("testing_net");
	// double score = network_predict_imgs(net, imgs, 1000);
	// printf("Score: %1.5f\n", score);

	// imgs_free(imgs, number_imgs);
	// network_free(net);
	return 0;
}