template<typename T>
uint32_t branchfree_search(T *a, uint32_t n, T x) {
    using I = uint32_t;
    const T *base = a;
    while (n > 1) {
        I half = n / 2;
        __builtin_prefetch(base + half / 2, 0, 0);
        __builtin_prefetch(base + half + half / 2, 0, 0);
        base = (base[half] < x) ? base + half : base;
        n -= half;
    }
    return (*base < x) + base - a;
}
