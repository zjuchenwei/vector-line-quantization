#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH
namespace faiss { namespace gpu {

__device__ __host__ inline ushort toUShort(const float& _f) {

  float ftrans = (_f + 4.f) * (65536.f / 8.f);

  return ushort( (_f >= 4.f) ? 65535 : ((_f < -4.f) ? 0 : ftrans));

}

__device__ __host__ inline float toFloat(const ushort& _s) {

  return (float(_s)* (8.f / 65536.f) - 4.f);

}


/** given the triangle with edges a, b, c (all squared) and the ratio lambda that devides c
 * at location X, the function returns the squared distance d between C and X
 *                  C
 *          / |  \
 *        b /   /d   \ a
 *            /    |       \
 *      A -----X-------- B
 *         lc    (1-l)c
 *
 *  a2 = b2 + c2 - 2 b c cos(alpha)
 *
 *  d2 = b2 + l2 * c2 - 2 b lc  cos(alpha)
 *
 *  => d2 = b2 + l2 * c2 - 2 b lc (a2 - b2 - c2) / -2bc
 *  => d2 = b2 + l2 * c2 + l(a2 - b2 - c2)
 *
 *  if lambda < 0
 *                  C
 *          / |  \
 *        d /   /b   \ a
 *            /    |       \
 *      X -----A-------- B
 *         -lc      c
 *
 *  b2 = a2 + c2 - 2 a c cos(beta)
 *
 *  d2 = a2 + (1-l)2 * c2 - 2 a (1-l)c  cos(beta)
 *
 *  => d2 = a2 + (1-l)2 * c2 - 2 a (1-l)c (b2 - a2 - c2) / -2ac
 *  => d2 = a2 + (1-l)2 * c2 + (1-l)(b2 - a2 - c2)
 *
 *
 */

__device__ __host__ float dist2(float _a2, float _b2, float _c2, float _lambda) {

//  if (_lambda < 0.f)
//    return _a2 + (1.f - _lambda) * (1.f - _lambda) * _c2
//        + (1.f - _lambda) * (_b2 - _a2 - _c2);
//  else
  return _b2 + _lambda * _lambda * _c2 + _lambda * (_a2 - _b2 - _c2);

}

__device__ __host__ float dist3(float a, float b, float a1,const float * param) {
    // "a" = ||a||^2, "b" = ||b||^2, "c" = ||c||^2, "l" = \lambda
    // float ll = (l<0) ? 1-l : l;
    //(1-l1)*(1-l)*b + (l * l -l)*(1-l1)*c + l*(1-l1) * a+ (l1 * l1 -l1)*c1 + l1 * a1;
    return param[0] * a +param[1]*b  + param[2] * a1+param[3];
}

/** given the triangle with edges a, b, c (all squared) and it computes the ratio lambda that devides c
 * at location X which is the projection of C onto AB
 *                  C
 *          / |  \
 *        b /   |     \ a
 *            /     |       \
 *      A ------X-------- B
 *         lc    (1-l)c
 *
 *  a2 = b2 + c2 - 2 b c cos(alpha)
 *
 *  lc = b cos(alpha)
 *
 *  => lambda =  b/c * (a2-b2-c2)/(-2bc) = -0.5 * (a2-b2-c2)/c2
 */__device__ __host__ float project(float _a2, float _b2, float _c2) {
  return -0.5f * (_a2 - _b2 - _c2) / _c2;
}

/** given the triangle with edges a, b, c (all squared) and it computes the ratio lambda that devides c
 * at location X which is the projection of C onto AB
 *                  C
 *          / |  \
 *        b /   |d    \ a
 *            /     |       \
 *      A ------X-------- B
 *         lc    (1-l)c
 *
 *  a2 = b2 + c2 - 2 b c cos(alpha)
 *
 *  lc = b cos(alpha)
 *
 *  => lambda =  b/c * (a2-b2-c2)/(-2bc) = -0.5 * (a2-b2-c2)/c2
 *
 *  d2 = b2 - (b cos(alpha))2
 *  =>  d2 = b2 - 0.25 * (a2-b2-c2)2/ c2;
 */

__device__ __host__ float project(float _a2, float _b2, float _c2,
    volatile float& _d2) {

  float lambda = -0.5f * (_a2 - _b2 - _c2) / _c2;

  _d2 = _b2 - lambda * lambda * _c2;

  return lambda;
}

bool equal(float _a, float _b, float _epsilon = 0.00001) {
  return (fabs(_a - _b) < _epsilon);
}

__device__ __host__ uint8_t lambdatom(float lambda)
{
    float per=2.0/64;
    if(lambda<=-1)
		return 0;
	if(lambda>=1)
		return uint8_t(64-1);
    float cha=lambda-(-1);
    return uint8_t(floorf(cha/per));
}
__device__ __host__ float mtolambda(uint8_t m)
{
    float per=2.0/64;
    return -1+m*per+per/2.0;
}


}}

#endif
