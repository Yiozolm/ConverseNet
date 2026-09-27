// Per-element s1 checkpoint recomputation. Inputs are from this exact call.
// Preserve the original complex division; do not replace it with scalar divide.
template<class T>__device__ Z<T> scale1_recompute_q(Z<T> y,Z<T> p,Z<T> k,T d) {
    const Z<T> pm=product(k,p);
    return add(y,-pm)/Z<T>(d,0);
}
