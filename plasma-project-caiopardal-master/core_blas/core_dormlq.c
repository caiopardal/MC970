/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @generated from /home/luszczek/workspace/plasma/bitbucket/plasma/core_blas/core_zunmlq.c, normal z -> d, Fri Sep 28 17:38:25 2018
 *
 **/

#include <plasma_core_blas.h>
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_lapack.h"

#include <omp.h>

/***************************************************************************//**
 *
 * @ingroup core_unmlq
 *
 *  Overwrites the general complex m-by-n tile C with
 *
 *                                    side = PlasmaLeft      side = PlasmaRight
 *    trans = PlasmaNoTrans              Q * C                  C * Q
 *    trans = PlasmaTrans         Q^T * C                  C * Q^T
 *
 *  where Q is a orthogonal matrix defined as the product of k
 *  elementary reflectors
 *    \f[
 *        Q = H(k) . . . H(2) H(1)
 *    \f]
 *  as returned by plasma_core_dgelqt. Q is of order m if side = PlasmaLeft
 *  and of order n if side = PlasmaRight.
 *
 *******************************************************************************
 *
 * @param[in] side
 *         - PlasmaLeft  : apply Q or Q^T from the Left;
 *         - PlasmaRight : apply Q or Q^T from the Right.
 *
 * @param[in] trans
 *         - PlasmaNoTrans    :  No transpose, apply Q;
 *         - PlasmaTrans :  Transpose, apply Q^T.
 *
 * @param[in] m
 *         The number of rows of the tile C.  m >= 0.
 *
 * @param[in] n
 *         The number of columns of the tile C.  n >= 0.
 *
 * @param[in] k
 *         The number of elementary reflectors whose product defines
 *         the matrix Q.
 *         If side = PlasmaLeft,  m >= k >= 0;
 *         if side = PlasmaRight, n >= k >= 0.
 *
 * @param[in] ib
 *         The inner-blocking size. ib >= 0.
 *
 * @param[in] A
 *         Dimension:  (lda,m) if SIDE = PlasmaLeft,
 *                     (lda,n) if SIDE = PlasmaRight,
 *         The i-th row must contain the vector which defines the
 *         elementary reflector H(i), for i = 1,2,...,k, as returned by
 *         plasma_core_dgelqt in the first k rows of its array argument A.
 *
 * @param[in] lda
 *         The leading dimension of the array A.  lda >= max(1,k).
 *
 * @param[in] T
 *         The ib-by-k triangular factor T of the block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] ldt
 *         The leading dimension of the array T. ldt >= ib.
 *
 * @param[in,out] C
 *         On entry, the m-by-n tile C.
 *         On exit, C is overwritten by Q*C or Q^T*C or C*Q^T or C*Q.
 *
 * @param[in] ldc
 *         The leading dimension of the array C. ldc >= max(1,m).
 *
 * @param work
 *         Auxiliary workspace array of length
 *         ldwork-by-m   if side == PlasmaLeft
 *         ldwork-by-ib  if side == PlasmaRight
 *
 * @param[in] ldwork
 *         The leading dimension of the array work.
 *             ldwork >= max(1,ib) if side == PlasmaLeft
 *             ldwork >= max(1,n)  if side == PlasmaRight
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
__attribute__((weak))
int plasma_core_dormlq(plasma_enum_t side, plasma_enum_t trans,
                int m, int n, int k, int ib,
                const double *A,    int lda,
                const double *T,    int ldt,
                      double *C,    int ldc,
                      double *work, int ldwork)
{
    // Check input arguments.
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        plasma_coreblas_error("illegal value of side");
        return -1;
    }

    int nq; // order of Q
    int nw; // dimension of work

    if (side == PlasmaLeft) {
        nq = m;
        nw = n;
    }
    else {
        nq = n;
        nw = m;
    }

    if (trans != PlasmaNoTrans && trans != PlasmaTrans) {
        plasma_coreblas_error("illegal value of trans");
        return -2;
    }
    if (m < 0) {
        plasma_coreblas_error("illegal value of m");
        return -3;
    }
    if (n < 0) {
        plasma_coreblas_error("illegal value of n");
        return -4;
    }
    if (k < 0 || k > nq) {
        plasma_coreblas_error("illegal value of k");
        return -5;
    }
    if (ib < 0) {
        plasma_coreblas_error("illegal value of ib");
        return -6;
    }
    if (A == NULL) {
        plasma_coreblas_error("NULL A");
        return -7;
    }
    if ((lda < imax(1, k)) && (k > 0)) {
        plasma_coreblas_error("illegal value of lda");
        return -8;
    }
    if (T == NULL) {
        plasma_coreblas_error("NULL T");
        return -9;
    }
    if (ldt < imax(1, ib)) {
        plasma_coreblas_error("illegal value of ldt");
        return -10;
    }
    if (C == NULL) {
        plasma_coreblas_error("NULL C");
        return -11;
    }
    if ((ldc < imax(1, m)) && (m > 0)) {
        plasma_coreblas_error("illegal value of ldc");
        return -12;
    }
    if (work == NULL) {
        plasma_coreblas_error("NULL work");
        return -13;
    }
    if ((ldwork < imax(1, nw)) && (nw > 0)) {
        plasma_coreblas_error("illegal value of ldwork");
        return -14;
    }

    // quick return
    if (m == 0 || n == 0 || k == 0)
        return PlasmaSuccess;

    int i1, i3;

    if ((side == PlasmaLeft  && trans == PlasmaNoTrans) ||
        (side == PlasmaRight && trans != PlasmaNoTrans)) {
        i1 = 0;
        i3 = ib;
    }
    else {
        i1 = ((k-1)/ib)*ib;
        i3 = -ib;
    }

    if (trans == PlasmaNoTrans)
        trans = PlasmaTrans;
    else
        trans = PlasmaNoTrans;

    for (int i = i1; i > -1 && i < k; i += i3) {
        int kb = imin(ib, k-i);
        int ic = 0;
        int jc = 0;
        int ni = n;
        int mi = m;

        if (side == PlasmaLeft) {
            // H or H^T is applied to C(i:m,1:n).
            mi = m - i;
            ic = i;
        }
        else {
            // H or H^T is applied to C(1:m,i:n).
            ni = n - i;
            jc = i;
        }

        // Apply H or H^T.
        LAPACKE_dlarfb_work(LAPACK_COL_MAJOR,
                            lapack_const(side),
                            lapack_const(trans),
                            lapack_const(PlasmaForward),
                            lapack_const(PlasmaRowwise),
                            mi, ni, kb,
                            &A[lda*i+i], lda,
                            &T[ldt*i], ldt,
                            &C[ldc*jc+ic], ldc,
                            work, ldwork);
    }

    return PlasmaSuccess;
}

/******************************************************************************/
void plasma_core_omp_dormlq(plasma_enum_t side, plasma_enum_t trans,
                     int m, int n, int k, int ib,
                     const double *A, int lda,
                     const double *T, int ldt,
                           double *C, int ldc,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    int ak;
    if (side == PlasmaLeft)
        ak = m;
    else
        ak = n;

    #pragma omp task depend(in:A[0:lda*ak]) \
                     depend(in:T[0:ib*k]) \
                     depend(inout:C[0:ldc*n])
    {
        if (sequence->status == PlasmaSuccess) {
            // Prepare workspaces.
            int tid = omp_get_thread_num();
            double *W = (double*)work.spaces[tid];
            int ldwork = side == PlasmaLeft ? n : m; // TODO: double check

            // Call the kernel.
            int info = plasma_core_dormlq(side, trans,
                                   m, n, k, ib,
                                   A, lda,
                                   T, ldt,
                                   C, ldc,
                                   W, ldwork);

            if (info != PlasmaSuccess) {
                plasma_error("core_dormlq() failed");
                plasma_request_fail(sequence, request, PlasmaErrorInternal);
            }
        }
    }
}
