// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math/cmplx"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/internal/asm/c128"
)

// Zgerc performs the rank-one operation
//  A += alpha * x * y^H
// where A is an m×n dense matrix, alpha is a scalar, x is an m element vector,
// and y is an n element vector.
func (Implementation) Zgerc(m, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int) {
	checkZMatrix('A', m, n, a, lda)
	checkZVector('x', m, x, incX)
	checkZVector('y', n, y, incY)

	if m == 0 || n == 0 || alpha == 0 {
		return
	}

	var kx, jy int
	if incX < 0 {
		kx = (1 - m) * incX
	}
	if incY < 0 {
		jy = (1 - n) * incY
	}
	for j := 0; j < n; j++ {
		if y[jy] != 0 {
			tmp := alpha * cmplx.Conj(y[jy])
			c128.AxpyInc(tmp, x, a[j:], uintptr(m), uintptr(incX), uintptr(lda), uintptr(kx), 0)
		}
		jy += incY
	}
}

// Zgeru performs the rank-one operation
//  A += alpha * x * y^T
// where A is an m×n dense matrix, alpha is a scalar, x is an m element vector,
// and y is an n element vector.
func (Implementation) Zgeru(m, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int) {
	checkZMatrix('A', m, n, a, lda)
	checkZVector('x', m, x, incX)
	checkZVector('y', n, y, incY)

	if m == 0 || n == 0 || alpha == 0 {
		return
	}

	var kx int
	if incX < 0 {
		kx = (1 - m) * incX
	}
	if incY == 1 {
		for i := 0; i < m; i++ {
			if x[kx] != 0 {
				tmp := alpha * x[kx]
				c128.AxpyUnitary(tmp, y[:n], a[i*lda:i*lda+n])
			}
			kx += incX
		}
		return
	}
	var jy int
	if incY < 0 {
		jy = (1 - n) * incY
	}
	for i := 0; i < m; i++ {
		if x[kx] != 0 {
			tmp := alpha * x[kx]
			c128.AxpyInc(tmp, y, a[i*lda:i*lda+n], uintptr(n), uintptr(incY), 1, uintptr(jy), 0)
		}
		kx += incX
	}
}

// Zher performs the Hermitian rank-one operation
//  A += alpha * x * x^H
// where A is an n×n Hermitian matrix, alpha is a real scalar, and x is an n
// element vector. On entry, the imaginary parts of the diagonal elements of A
// are ignored and assumed to be zero, on return they will be set to zero.
func (Implementation) Zher(uplo blas.Uplo, n int, alpha float64, x []complex128, incX int, a []complex128, lda int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	checkZMatrix('A', n, n, a, lda)
	checkZVector('x', n, x, incX)

	if n == 0 || alpha == 0 {
		return
	}

	var kx int
	if incX < 0 {
		kx = (1 - n) * incX
	}
	if uplo == blas.Upper {
		if incX == 1 {
			for j := 0; j < n; j++ {
				if x[j] != 0 {
					tmp := complex(alpha*real(x[j]), -alpha*imag(x[j]))
					c128.AxpyInc(tmp, x[:j], a[j:], uintptr(j), 1, uintptr(lda), 0, 0)
					ajj := real(a[j*lda+j])
					xtmp := real(x[j] * tmp)
					a[j*lda+j] = complex(ajj+xtmp, 0)
				} else {
					ajj := real(a[j*lda+j])
					a[j*lda+j] = complex(ajj, 0)
				}
			}
			return
		}

		jx := kx
		for j := 0; j < n; j++ {
			if x[jx] != 0 {
				tmp := complex(alpha*real(x[jx]), -alpha*imag(x[jx]))
				c128.AxpyInc(tmp, x, a[j:], uintptr(j), uintptr(incX), uintptr(lda), uintptr(kx), 0)
				ajj := real(a[j*lda+j])
				xtmp := real(tmp * x[jx])
				a[j*lda+j] = complex(ajj+xtmp, 0)
			} else {
				ajj := real(a[j*lda+j])
				a[j*lda+j] = complex(ajj, 0)
			}
			jx += incX
		}
		return
	}

	if incX == 1 {
		for j := 0; j < n; j++ {
			xj := x[j]
			if xj != 0 {
				tmp := complex(alpha*real(xj), -alpha*imag(xj))
				ajj := real(a[j*lda+j])
				xtmp := real(xj * tmp)
				a[j*lda+j] = complex(ajj+xtmp, 0)
				if n-j-1 > 0 {
					c128.AxpyInc(tmp, x[j+1:n], a[(j+1)*lda+j:], uintptr(n-j-1), 1, uintptr(lda), 0, 0)
				}
			} else {
				ajj := real(a[j*lda+j])
				a[j*lda+j] = complex(ajj, 0)
			}
		}
		return
	}

	jx := kx
	for j := 0; j < n; j++ {
		xj := x[jx]
		if xj != 0 {
			tmp := complex(alpha*real(xj), -alpha*imag(xj))
			ajj := real(a[j*lda+j])
			xtmp := real(xj * tmp)
			a[j*lda+j] = complex(ajj+xtmp, 0)
			if n-j-1 > 0 {
				c128.AxpyInc(tmp, x, a[(j+1)*lda+j:], uintptr(n-j-1), uintptr(incX), uintptr(lda), uintptr(jx+incX), 0)
			}
		} else {
			ajj := real(a[j*lda+j])
			a[j*lda+j] = complex(ajj, 0)
		}
		jx += incX
	}
}
