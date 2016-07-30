/*Programmer: Chris Tralie
*Purpose: To implement an implicit version of Smith-Waterman that works on
*a binary dissimilarity matrix*/
#include <stdio.h>
#include <stdlib.h>
#include "swalignimp.h"

double quadMax(double a, double b, double c, double d) {
    double max = a;
    if (b > a) max = b;
    if (c > max) max = c;
    if (d > max) max = d;
    return max;
}

/*Inputs: S (a binary N x M dissimilarity matrix)*/
/*Outputs: 1) Distance (scalar)
*2) (N+1) x (M+1) dynamic programming matrix (Optional)*/
double swalignimp(double* S, int N, int M) {
    int i, j;
    double maxD, d1, d2, d3;
    double* D;
    double matchScore = 2;
    double mismatchScore = -3;
    double gapScore = -2;
    
    N++; M++;
    D = malloc(N*M*sizeof(double));
    D[0] = 0;
    for (i = 1; i < N; i++) {
        D[i*M] = 0;
    }
    for (i = 1; i < M; i++) {
        D[i] = 0;
    }
    
    maxD = 0.0;
    for (i = 1; i < N; i++) {
        for (j = 1; j < M; j++) {
            d1 = D[(i-1)*M+j] + gapScore;
            d2 = D[i*M+(j-1)] + gapScore;
            d3 = D[(i-1)*M+(j-1)];
            d3 += (S[(i-1)*(M-1)+(j-1)] > 0)?matchScore:mismatchScore;
            D[i*M+j] = quadMax(d1, d2, d3, 0.0);
            if (D[i*M+j] > maxD) {
                maxD = D[i*M+j];
            }
        }
    }
    free(D);
    return maxD;
}

double Delta(double a, double b) {
    /*Parameters used in the paper*/
    double gapOpening = -0.5; 
    double gapExtension = -0.7;
    if (b > 0) {
        return 0;
    }
    if (b == 0 && a > 0) {
        return gapOpening;
    }
    return gapExtension;
}

double Match(double i) {
    double matchScore = 1;
    double mismatchScore = -1;
    if (i == 0) {
        return mismatchScore;
    }
    return matchScore;
}

/*Inputs: D (a binary N x M cross-similarity matrix)*/

/*Outputs: 1) Distance (scalar)
*2) (N+1) x (M+1) dynamic programming matrix (Optional)*/
double swalignimpconstrained(double* S, int N, int M) {
    double* D;
    int i, j, k;
    double maxD, d1, d2, d3, MS;
    
    N++; M++;
    D = malloc(N*M*sizeof(double));//Dynamic programming matrix
    if (N < 4 || M < 4) {
        return 0.0;
    }    
    
    /*Don't penalize as much at the beginning*/
    for (k = 0; k < 3; k++) {
        /*Initialize first 3 columns to zero*/
        for (i = 0; i < N; i++) {
            D[i*M+k] = 0;
        }
        /*Initialize first 3 rows to zero*/
        for (i = 0; i < M; i++) {
            D[k*M+i] = 0;
        }
    }

    maxD = 0.0;
    for (i = 3; i < N; i++) {
        for (j = 3; j < M; j++) {
            MS = Match(S[(i-1)*(M-1)+(j-1)]);
            /*H_(i-1, j-1) + S_(i-1, j-1) + delta(S_(i-2,j-2), S_(i-1, j-1))*/
            d1 = D[(i-1)*M+(j-1)] + MS + Delta(S[(i-2)*(M-1)+(j-2)], S[(i-1)*(M-1)+(j-1)]);
            /*H_(i-2, j-1) + S_(i-1, j-1) + delta(S_(i-3, j-2), S_(i-1, j-1))*/
            d2 = D[(i-2)*M+(j-1)] + MS + Delta(S[(i-3)*(M-1)+(j-2)], S[(i-1)*(M-1)+(j-1)]);
            /*H_(i-1, j-2) + S_(i-1, j-1) + delta(S_(i-2, j-3), S_(i-1, j-1))*/
            d3 = D[(i-1)*M+(j-2)] + MS + Delta(S[(i-2)*(M-1)+(j-3)], S[(i-1)*(M-1)+(j-1)]);
            D[i*M+j] = quadMax(d1, d2, d3, 0.0);
            if (D[i*M+j] > maxD) {
                maxD = D[i*M+j];
            }
        }
    }
    free(D);
    return maxD;
}
