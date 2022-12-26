#include <iostream>
#include <limits>
#include "HigherOrderMS_1D.h"
int main(int argc, char* argv[])
{
	std::cout << "Hello World!\n";

	const int n = 15;
	const int k = 2;
	const double beta = 1;// std::numeric_limits<double>::infinity();
	const auto m = HOMS::computeSystemMatrix(n, k, beta);
	std::string sep = "\n----------------------------------------\n";
	std::cout << m << sep;
	return 0;
}