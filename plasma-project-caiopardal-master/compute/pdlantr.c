/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @generated from /home/luszczek/workspace/plasma/bitbucket/plasma/compute/pzlantr.c, normal z -> d, Fri Sep 28 17:38:13 2018
 *
 **/

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include <plasma_core_blas.h>

#define A(m, n) (double*)plasma_tile_addr(A, m, n)

/***************************************************************************//**
 *  Parallel tile calculation of max, one, infinity or Frobenius matrix norm
 *  for a triangular matrix.
 ******************************************************************************/
void plasma_pdlantr(plasma_enum_t norm, plasma_enum_t uplo, plasma_enum_t diag,
                    plasma_desc_t A, double *work, double *value,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    switch (norm) {
    double stub;
    double *workspace;
    double *scale;
    double *sumsq;
    //================
    // PlasmaMaxNorm
    //================
    case PlasmaMaxNorm:
        for (int m = 0; m < A.mt; m++) {
            int mvam = plasma_tile_mview(A, m);
            int ldam = plasma_tile_mmain(A, m);
            if (uplo == PlasmaLower) {
                for (int n = 0; n < imin(m, A.nt); n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_dlange(PlasmaMaxNorm,
                                    mvam, nvan,
                                    A(m, n), ldam,
                                    &stub, &work[A.mt*n+m],
                                    sequence, request);
                }
            }
            else { // PlasmaUpper
                for (int n = m+1; n < A.nt; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_dlange(PlasmaMaxNorm,
                                    mvam, nvan,
                                    A(m, n), ldam,
                                    &stub, &work[A.mt*n+m],
                                    sequence, request);
                }
            }
            if (m < A.nt) {
                int nvam = plasma_tile_nview(A, m);
                plasma_core_omp_dlantr(PlasmaMaxNorm, uplo, diag,
                                mvam, nvam,
                                A(m, m), ldam,
                                &stub, &work[A.mt*m+m],
                                sequence, request);
            }
        }
        #pragma omp taskwait
        plasma_core_omp_dlantr(PlasmaMaxNorm, uplo, PlasmaNonUnit,
                        A.mt, A.nt,
                        work, A.mt,
                        &stub, value,
                        sequence, request);
        break;
    //================
    // PlasmaOneNorm
    //================
    case PlasmaOneNorm:
        for (int m = 0; m < A.mt; m++) {
            int mvam = plasma_tile_mview(A, m);
            int ldam = plasma_tile_mmain(A, m);
            if (uplo == PlasmaLower) {
                for (int n = 0; n < imin(m, A.nt); n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_dlange_aux(PlasmaOneNorm,
                                        mvam, nvan,
                                        A(m, n), ldam,
                                        &work[A.n*m+n*A.nb],
                                        sequence, request);
                }
            }
            else { // PlasmaUpper
                for (int n = m+1; n < A.nt; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_dlange_aux(PlasmaOneNorm,
                                        mvam, nvan,
                                        A(m, n), ldam,
                                        &work[A.n*m+n*A.nb],
                                        sequence, request);
                }
            }
            if (m < A.nt) {
                int nvam = plasma_tile_nview(A, m);
                plasma_core_omp_dlantr_aux(PlasmaOneNorm, uplo, diag,
                                    mvam, nvam,
                                    A(m, m), ldam,
                                    &work[A.n*m+m*A.nb],
                                    sequence, request);
            }
        }
        #pragma omp taskwait
        workspace = work + A.mt*A.n;
        plasma_core_omp_dlange(PlasmaInfNorm,
                        A.n, A.mt,
                        work, A.n,
                        workspace, value,
                        sequence, request);
        break;
    //================
    // PlasmaInfNorm
    //================
    case PlasmaInfNorm:
        for (int m = 0; m < A.mt; m++) {
            int mvam = plasma_tile_mview(A, m);
            int ldam = plasma_tile_mmain(A, m);

            if (uplo == PlasmaLower) {
                for (int n = 0; n < imin(m, A.nt); n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_dlange_aux(PlasmaInfNorm,
                                        mvam, nvan,
                                        A(m, n), ldam,
                                        &work[A.m*n+m*A.mb],
                                        sequence, request);
                }
            }
            else { // PlasmaUpper
                for (int n = m+1; n < A.nt; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_dlange_aux(PlasmaInfNorm,
                                        mvam, nvan,
                                        A(m, n), ldam,
                                        &work[A.m*n+m*A.mb],
                                        sequence, request);
                }
            }
            if (m < A.nt) {
                int nvam = plasma_tile_nview(A, m);
                plasma_core_omp_dlantr_aux(PlasmaInfNorm, uplo, diag,
                                    mvam, nvam,
                                    A(m, m), ldam,
                                    &work[A.m*m+m*A.nb],
                                    sequence, request);
            }
        }
        #pragma omp taskwait
        workspace = work + A.nt*A.m;
        plasma_core_omp_dlange(PlasmaInfNorm,
                        A.m, A.nt,
                        work, A.m,
                        workspace, value,
                        sequence, request);
        break;
    //======================
    // PlasmaFrobeniusNorm
    //======================
    case PlasmaFrobeniusNorm:
        scale = work;
        sumsq = work + A.mt*A.nt;
        for (int m = 0; m < A.mt; m++) {
            int mvam = plasma_tile_mview(A, m);
            int ldam = plasma_tile_mmain(A, m);
            if (uplo == PlasmaLower) {
                for (int n = 0; n < imin(m, A.nt); n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_dgessq(mvam, nvan,
                                    A(m, n), ldam,
                                    &scale[A.mt*n+m], &sumsq[A.mt*n+m],
                                    sequence, request);
                }
            }
            else { // PlasmaUpper
                for (int n = m+1; n < A.nt; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_dgessq(mvam, nvan,
                                    A(m, n), ldam,
                                    &scale[A.mt*n+m], &sumsq[A.mt*n+m],
                                    sequence, request);
                }
            }
            if (m < A.nt) {
                int nvam = plasma_tile_nview(A, m);
                plasma_core_omp_dtrssq(uplo, diag,
                                mvam, nvam,
                                A(m, m), ldam,
                                &scale[A.mt*m+m], &sumsq[A.mt*m+m],
                                sequence, request);
            }
        }
        #pragma omp taskwait
        plasma_core_omp_dgessq_aux(A.mt*A.nt,
                            scale, sumsq,
                            value,
                            sequence, request);
        break;
    }
}
