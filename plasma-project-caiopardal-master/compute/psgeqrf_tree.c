/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @generated from /home/luszczek/workspace/plasma/bitbucket/plasma/compute/pzgeqrf_tree.c, normal z -> s, Fri Sep 28 17:38:11 2018
 *
 **/

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "plasma_tree.h"
#include <plasma_core_blas_s.h>

#define A(m, n) (float*)plasma_tile_addr(A, m, n)
#define T(m, n) (float*)plasma_tile_addr(T, m, n)
#define T2(m, n) (float*)plasma_tile_addr(T, m, n+(T.nt/2))
/***************************************************************************//**
 *  Parallel tile QR factorization based on a tree Householder reduction
 * @see plasma_omp_sgeqrf
 **/
void plasma_psgeqrf_tree(plasma_desc_t A, plasma_desc_t T,
                         plasma_workspace_t work,
                         plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    // Precompute order of QR operations.
    int *operations = NULL;
    int num_operations;
    plasma_tree_operations(A.mt, A.nt, &operations, &num_operations,
                           sequence, request);

    // Set inner blocking from the T tile row-dimension.
    int ib = T.mb;

    for (int iop = 0; iop < num_operations; iop++) {
        int j, k, kpiv;
        plasma_enum_t kernel;
        plasma_tree_get_operation(operations, iop, &kernel, &j, &k, &kpiv);

        int nvaj    = plasma_tile_nview(A, j);
        int mvak    = plasma_tile_mview(A, k);
        int ldak    = plasma_tile_mmain(A, k);

        if (kernel == PlasmaGeKernel) {
            // triangularization
            plasma_core_omp_sgeqrt(
                mvak, nvaj, ib,
                A(k, j), ldak,
                T(k, j), T.mb,
                work,
                sequence, request);

            for (int jj = j + 1; jj < A.nt; jj++) {
                int nvajj = plasma_tile_nview(A, jj);

                plasma_core_omp_sormqr(
                    PlasmaLeft, PlasmaTrans,
                    mvak, nvajj, imin(mvak, nvaj), ib,
                    A(k, j), ldak,
                    T(k, j), T.mb,
                    A(k, jj), ldak,
                    work,
                    sequence, request);
            }
        }
        else if (kernel == PlasmaTtKernel) {
            // elimination of the tile
            int mvakpiv = plasma_tile_mview(A, kpiv);
            int ldakpiv = plasma_tile_mmain(A, kpiv);

            plasma_core_omp_sttqrt(
                mvak, nvaj, ib,
                A(kpiv, j), ldakpiv,
                A(k,  j),   ldak,
                T2(k, j),   T.mb,
                work,
                sequence, request);

            for (int jj = j + 1; jj < A.nt; jj++) {
                int nvajj = plasma_tile_nview(A, jj);

                plasma_core_omp_sttmqr(
                    PlasmaLeft, PlasmaTrans,
                    mvakpiv, nvajj, mvak, nvajj, imin(mvakpiv+mvak, nvaj), ib,
                    A(kpiv, jj), ldakpiv,
                    A(k,    jj), ldak,
                    A(k,    j),  ldak,
                    T2(k,   j),  T.mb,
                    work,
                    sequence, request);
            }
        }
        else if (kernel == PlasmaTsKernel) {
            // elimination of the tile
            int mvakpiv = plasma_tile_mview(A, kpiv);
            int ldakpiv = plasma_tile_mmain(A, kpiv);

            plasma_core_omp_stsqrt(
                mvak, nvaj, ib,
                A(kpiv, j), ldakpiv,
                A(k,  j),   ldak,
                T2(k, j),   T.mb,
                work,
                sequence, request);

            for (int jj = j + 1; jj < A.nt; jj++) {
                int nvajj = plasma_tile_nview(A, jj);

                plasma_core_omp_stsmqr(
                    PlasmaLeft, PlasmaTrans,
                    mvakpiv, nvajj, mvak, nvajj, imin(mvakpiv+mvak, nvaj), ib,
                    A(kpiv, jj), ldakpiv,
                    A(k,    jj), ldak,
                    A(k,    j),  ldak,
                    T2(k,   j),  T.mb,
                    work,
                    sequence, request);
            }
        }
        else {
            plasma_error("illegal kernel");
            plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        }
    }

    free(operations);
}
