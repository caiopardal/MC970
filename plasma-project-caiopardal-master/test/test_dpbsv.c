/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @generated from /home/luszczek/workspace/plasma/bitbucket/plasma/test/test_zpbsv.c, normal z -> d, Fri Sep 28 17:38:31 2018
 *
 **/
#include "test.h"
#include "flops.h"
#include "plasma.h"
#include <plasma_core_blas.h>
#include "core_lapack.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define REAL

#define A(i_, j_) A[(i_) + (size_t)lda*(j_)]

/***************************************************************************//**
 *
 * @brief Tests DPBSV.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_dpbsv(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
            //  pbtrf params
    param[PARAM_UPLO   ].used = true;
    param[PARAM_DIM    ].used = PARAM_USE_N;
    param[PARAM_KL     ].used = true;
    param[PARAM_PADA   ].used = true;
    param[PARAM_NB     ].used = true;
            //  gbtrs params for check
    param[PARAM_NRHS   ].used = true;
    param[PARAM_PADB   ].used = true;
    param[PARAM_ZEROCOL].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    char uplo_ = param[PARAM_UPLO].c;
    int pada   = param[PARAM_PADA].i;
    int n      = param[PARAM_DIM].dim.n;
    int lda    = imax(1, n + pada);

    plasma_enum_t uplo = plasma_uplo_const(uplo_);
    int kd = (uplo == PlasmaUpper ? param[PARAM_KU].i : param[PARAM_KL].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================

    // band matrix A in full storage (also used for solution check)
    double *A =
        (double*)malloc((size_t)lda*n*sizeof(double));
    assert(A != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_dlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);
    // make it SPD
    int i, j;
    for (i = 0; i < n; ++i) {
        A(i,i) = creal(A(i,i)) + n;
        for (j = 0; j < i; ++j) {
            A(j,i) = (A(i,j));
        }
    }
    // zero out elements outside the band
    for (i = 0; i < n; i++) {
        for (j = i+kd+1; j < n; j++) A(i, j) = 0.0;
    }
    for (j = 0; j < n; j++) {
        for (i = j+kd+1; i < n; i++) A(i, j) = 0.0;
    }

    int zerocol = param[PARAM_ZEROCOL].i;
    if (zerocol >= 0 && zerocol < n) {
        LAPACKE_dlaset_work(
            LAPACK_COL_MAJOR, 'F', n, 1, 0.0, 0.0, &A(0, zerocol), lda);
        LAPACKE_dlaset_work(
            LAPACK_COL_MAJOR, 'F', 1, n, 0.0, 0.0, &A(zerocol, 0), lda);
    }

    // band matrix A in LAPACK storage
    int ldab = imax(1, kd+1+pada);
    double *AB = NULL;
    AB = (double*)malloc((size_t)ldab*n*sizeof(double));
    assert(AB != NULL);

    // convert into LAPACK's symmetric band storage
    for (j = 0; j < n; j++) {
        for (i = 0; i < ldab; i++) AB[i + j*ldab] = 0.0;
        if (uplo == PlasmaUpper) {
            for (i = imax(0, j-kd); i <= j; i++) AB[i-j+kd + j*ldab] = A(i, j);
        }
        else {
            for (i = j; i <= imin(n-1, j+kd); i++) AB[i-j + j*ldab] = A(i, j);
        }
    }
    double *ABref = NULL;
    if (test) {
        ABref = (double*)malloc(
            (size_t)ldab*n*sizeof(double));
        assert(ABref != NULL);

        memcpy(ABref, AB, (size_t)ldab*n*sizeof(double));
    }

    // set up solution vector X with right-hand-side B
    int nrhs = param[PARAM_NRHS].i;
    int ldx = imax(1, n + param[PARAM_PADB].i);
    double *X = (double*)malloc(
        (size_t)ldx*nrhs*sizeof(double));
    assert(X != NULL);

    retval = LAPACKE_dlarnv(1, seed, (size_t)ldx*nrhs, X);
    assert(retval == 0);

    // copy B to X
    int ldb = ldx;
    double *B = NULL;
    if (test) {
        B = (double*)malloc(
            (size_t)ldb*nrhs*sizeof(double));
        assert(B != NULL);
        LAPACKE_dlacpy_work(LAPACK_COL_MAJOR, 'F', n, nrhs, X, ldx, B, ldb);
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    int plainfo;

    plasma_time_t start = omp_get_wtime();
    plainfo = plasma_dpbsv(uplo, n, kd, nrhs, AB, ldab, X, ldx);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = 0.0 / time / 1e9;

    //================================================================
    // Test results by computing residual norm.
    //================================================================
    if (test) {
        if (plainfo == 0) {
            double zone =   1.0;
            double zmone = -1.0;

            // compute residual vector
            cblas_dsymm(CblasColMajor, CblasLeft, (CBLAS_UPLO) uplo, n, nrhs,
                        (zmone), A, lda,
                                            X, ldx,
                        (zone),  B, ldb);

            // compute various norms
            double *work = NULL;
            work = (double*)malloc((size_t)n*sizeof(double));
            assert(work != NULL);

            double Anorm = LAPACKE_dlansy_work(
                LAPACK_COL_MAJOR, 'F', uplo_, n, A, lda, work);
            double Xnorm = LAPACKE_dlange_work(
                LAPACK_COL_MAJOR, 'I', n, nrhs, X, ldb, work);
            double Rnorm = LAPACKE_dlange_work(
                LAPACK_COL_MAJOR, 'I', n, nrhs, B, ldb, work);
            double residual = Rnorm/(n*Anorm*Xnorm);

            param[PARAM_ERROR].d = residual;
            param[PARAM_SUCCESS].i = residual < tol;

            // free arrays
            free(work);
            free(B);
        }
        else {
            int lapinfo = LAPACKE_dpbsv(
                              LAPACK_COL_MAJOR, lapack_const(uplo),
                              n, kd, nrhs, ABref, ldab, X, ldx);
            if (plainfo == lapinfo) {
                param[PARAM_ERROR].d = 0.0;
                param[PARAM_SUCCESS].i = 1;
            }
            else {
                param[PARAM_ERROR].d = INFINITY;
                param[PARAM_SUCCESS].i = 0;
            }
        }
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(X);
    free(A);
    free(AB);
    if (test)
        free(ABref);
}
