/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @generated from /home/luszczek/workspace/plasma/bitbucket/plasma/compute/pzlascl.c, normal z -> d, Fri Sep 28 17:38:13 2018
 *
 **/

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include <plasma_core_blas.h>

#define A(m, n) (double*)plasma_tile_addr(A, m, n)

/******************************************************************************/
void plasma_pdlascl(plasma_enum_t uplo,
                    double cfrom, double cto,
                    plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    switch (uplo) {
    //==============
    // PlasmaUpper
    //==============
    case PlasmaUpper:
        for (int m = 0; m < A.mt; m++) {
            int mvam = plasma_tile_mview(A, m);
            int ldam = plasma_tile_mmain(A, m);
            if (m < A.nt) {
                int nvam = plasma_tile_nview(A, m);
                plasma_core_omp_dlascl(
                    PlasmaUpper,
                    cfrom, cto,
                    mvam, nvam,
                    A(m, m), ldam,
                    sequence, request);
            }
            for (int n = m+1; n < A.nt; n++) {
                int nvan = plasma_tile_nview(A, n);
                plasma_core_omp_dlascl(
                    PlasmaGeneral,
                    cfrom, cto,
                    mvam, nvan,
                    A(m, n), ldam,
                    sequence, request);
            }
        }
        break;
    //==============
    // PlasmaLower
    //==============
    case PlasmaLower:
        for (int m = 0; m < A.mt; m++) {
            int mvam = plasma_tile_mview(A, m);
            int ldam = plasma_tile_mmain(A, m);
            if (m < A.nt) {
                int nvam = plasma_tile_nview(A, m);
                plasma_core_omp_dlascl(
                    PlasmaLower,
                    cfrom, cto,
                    mvam, nvam,
                    A(m, m), ldam,
                    sequence, request);
            }
            for (int n = 0; n < imin(m, A.nt); n++) {
                int nvan = plasma_tile_nview(A, n);
                plasma_core_omp_dlascl(
                    PlasmaGeneral,
                    cfrom, cto,
                    mvam, nvan,
                    A(m, n), ldam,
                    sequence, request);
            }
        }
        break;
    //================
    // PlasmaGeneral
    //================
    case PlasmaGeneral:
    default:
        for (int m = 0; m < A.mt; m++) {
            int mvam  = plasma_tile_mview(A, m);
            int ldam = plasma_tile_mmain(A, m);
            for (int n = 0; n < A.nt; n++) {
                int nvan = plasma_tile_nview(A, n);
                plasma_core_omp_dlascl(
                    PlasmaGeneral,
                    cfrom, cto,
                    mvam, nvan,
                    A(m, n), ldam,
                    sequence, request);
            }
        }
    }
}
