from numba import guvectorize, float64


@guvectorize([(float64[:, :], float64[:])], '(m, n), ()', nopython=True)
def r_from_qr(arr, scalar):
    """Calculate R of a QR decomposition for matrices in an array.

    args:
        arr (np.ndarray): 3d array of [..., m, n], where m >= n.
        Arr is overwritten wtih the  R of the QR decomposition.

    The algorithm uses Givens Rotations for the triangularization and fully
    exploits the sparseness of the Rotation Matrices.

    The function is based on the following algorithm found at stackoverflow,
    but tested for correctness and optimized for speed::

        function [Q,R] = qrgivens(A)
            [m,n] = size(A);
            Q = eye(m);
            R = A;

            for j = 1:n
                for i = m:-1:(j+1)
                    G = eye(m);
                    [c,s] = givensrotation( R(i-1,j),R(i,j) );
                    G([i-1, i],[i-1, i]) = [c -s; s c];
                    R = G'*R;
                    Q = Q*G;
              end
            end
        end

        function [c,s] = givensrotation(a,b)
            if b == 0
                c = 1;
                s = 0;
            else
                if abs(b) > abs(a)
                    r = a / b;
                    s = 1 / sqrt(1 + r^2);
                    c = s*r;
            else
                r = b / a;
                c = 1 / sqrt(1 + r^2);
                s = c*r;
            end
          end

        end

    """
    m, n = arr.shape
    for j in range(n):
        for i in range(m - 1, j, -1):
            b = arr[i, j]
            if b != 0.0:
                a = arr[i - 1, j]
                if abs(b) > abs(a):
                    r = a / b
                    s = 1 / (1 + r ** 2) ** 0.5
                    c = s * r
                else:
                    r = b / a
                    c = 1 / (1 + r ** 2) ** 0.5
                    s = c * r
                for k in range(n):
                    helper1 = arr[i - 1, k]
                    helper2 = arr[i, k]
                    arr[i - 1, k] = c * helper1 + s * helper2
                    arr[i, k] = -s * helper1 + c * helper2
