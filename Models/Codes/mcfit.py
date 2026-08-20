import numpy as np


class FFT:  # TODO 1:
    # lowring picks the low-ringing grid offset; extrap power-law-extends the input past the array
    # edges instead of zero-padding. Both are required to suppress FFTLog edge ringing ("wiggles"):
    # with them off the forward/inverse round-trip has ~50% max error at the edges, with them on it
    # drops to <1%. Building the transform objects once (rather than per call) also avoids rework.
    def __init__(self, rs=None, ks=None, lowring=True, extrap=True):
        import mcfit
        self.mcfit = mcfit

        if rs is not None:
            self._fwd = self.mcfit.xi2P(rs, lowring=lowring)          # r -> k
            self.ks = self._fwd.y                                     # k grid (independent of input)
            self._inv = self.mcfit.P2xi(self.ks, lowring=lowring)     # k -> r
            self.rs_rev = self._inv.y
            self.FFTraw = lambda funcx: self._fwd(funcx, extrap=extrap)[1]
            self.IFFTraw = lambda funck: self._inv(funck, extrap=extrap)[1]

        elif ks is not None:
            self._inv = self.mcfit.P2xi(ks, lowring=lowring)
            self.rs = self._inv.y
            self._fwd = self.mcfit.xi2P(self.rs, lowring=lowring)
            self.ks_rev = self._fwd.y
            self.FFTraw = lambda funcx: self._fwd(funcx, extrap=extrap)[1]
            self.IFFTraw = lambda funck: self._inv(funck, extrap=extrap)[1]
        
    def FFT1D(self, funcx):
        return self.FFTraw(funcx)
    
    def IFFT1D(self, funck):
        return self.IFFTraw(funck)

    def FFT3D(self, funcx):
        return np.array([[self.FFTraw(funcx[:, i, j]) for i in range(funcx.shape[1])] for j in range(funcx.shape[2])]).T
    
    def IFFT3D(self, funck):
        return np.array([[self.IFFTraw(funck[:, i, j]) for i in range(funck.shape[1])] for j in range(funck.shape[2])]).T
    
    def IFFT2D(self, funck):
        return np.array([self.IFFTraw(funck[:, i]) for i in range(funck.shape[1])]).T


def _strip(a):  # drop astropy units if present, return plain float ndarray
    return np.asarray(getattr(a, "value", a), dtype=float)


def _logtrap(x):  # trapezoid weights for integration in d(ln x), i.e. int f dx = sum f*x*w
    lnx = np.log(x)
    w = np.empty_like(lnx)
    w[1:-1] = 0.5 * (lnx[2:] - lnx[:-2])
    w[0] = 0.5 * (lnx[1] - lnx[0])
    w[-1] = 0.5 * (lnx[-1] - lnx[-2])
    return w


class DHT:
    """Direct spherical-Bessel (3D isotropic Fourier) transform by log-space quadrature.

    Drop-in replacement for FFT with the same interface (.ks, FFT1D/3D, IFFT1D/2D/3D). Unlike the
    FFTLog-based FFT, this makes no log-periodicity assumption, so it does not ring -- the result is
    smooth by construction. Cost is an O(Nk*Nr) matrix multiply (one BLAS call; fast at these sizes)
    that vectorises over the trailing (z, M) axes.

    Conventions match mcfit.xi2P / mcfit.P2xi:
        forward (r -> k):  P(k) = 4 pi        * int f(r) j0(k r) r^2 dr
        inverse (k -> r):  f(r) = 1/(2 pi^2)  * int P(k) j0(k r) k^2 dk
    with j0(x) = sin(x)/x. Transforms operate on unit-stripped arrays and return plain ndarrays,
    exactly like FFT.FFT3D (so the caller's `* prof.unit` bookkeeping is unchanged).

    The k grid defaults to log-spaced over [kpad[0]/r_max, kpad[1]/r_min] with Nk (default len(rs))
    points; widen kpad or raise Nk if a round-trip test (IFFT1D(FFT1D(f)) ~= f) isn't converged.
    """

    def __init__(self, rs=None, ks=None, Nk=None, kpad=(0.1, 10.0)):
        if rs is not None:
            self.rs = _strip(rs)
            if ks is None:
                Nk = len(self.rs) if Nk is None else Nk
                ks = np.geomspace(kpad[0] / self.rs.max(), kpad[1] / self.rs.min(), Nk)
            self.ks = _strip(ks)
        else:
            self.ks = _strip(ks)
            if rs is None:
                Nr = len(self.ks) if Nk is None else Nk
                rs = np.geomspace(kpad[0] / self.ks.max(), kpad[1] / self.ks.min(), Nr)
            self.rs = _strip(rs)

        # j0(k r) = sin(kr)/(kr) = sinc(kr/pi) with numpy's sinc(y)=sin(pi y)/(pi y)
        j0_kr = np.sinc(np.outer(self.ks, self.rs) / np.pi)  # (Nk, Nr)
        self._Wf = 4 * np.pi * j0_kr * self.rs**3 * _logtrap(self.rs)          # r -> k
        self._Wi = (1 / (2 * np.pi**2)) * j0_kr.T * self.ks**3 * _logtrap(self.ks)  # k -> r

    def FFT1D(self, funcx):   # r-axis (0) -> k-axis (0)
        return np.tensordot(self._Wf, _strip(funcx), axes=([1], [0]))

    def IFFT1D(self, funck):  # k-axis (0) -> r-axis (0)
        return np.tensordot(self._Wi, _strip(funck), axes=([1], [0]))

    # 2D/3D just contract the leading (r or k) axis; trailing (z, M) axes ride along
    FFT3D = FFT1D
    IFFT3D = IFFT1D
    IFFT2D = IFFT1D