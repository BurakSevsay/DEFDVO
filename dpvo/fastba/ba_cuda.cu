#include <torch/extension.h>
#include <vector>


#define GPU_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i<n; i += blockDim.x * gridDim.x)


#define NUM_THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + NUM_THREADS - 1) / NUM_THREADS)

const auto mdtype = torch::dtype(torch::kFloat32).device(torch::kCUDA);

typedef float mtype;

__device__ void
actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__  void
actSE3(const float *t, const float *q, const float *X, float *Y) {
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

__device__ void
adjSE3(const float *t, const float *q, const float *X, float *Y) {
  float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  actSO3(qinv, &X[0], &Y[0]);
  actSO3(qinv, &X[3], &Y[3]);

  float u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

__device__ void 
relSE3(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}

__device__ __forceinline__ void equirect_iproj(
    const float u, const float v,
    const float fx, const float fy,
    const float cx, const float cy,
    float* Xn)
{
  const float theta = (u - cx) / fx;
  const float phi = (v - cy) / fy;
  const float ctheta = cosf(theta);
  const float stheta = sinf(theta);
  const float cphi = cosf(phi);
  const float sphi = sinf(phi);

  Xn[0] = stheta * cphi;
  Xn[1] = sphi;
  Xn[2] = ctheta * cphi;
}

#define EQUIRECT_TWO_PI 6.283185307179586f

// horizontal image period in pixels (fx = W / 2pi at every scale)
__device__ __forceinline__ float equirect_x_period(const float fx) {
  return EQUIRECT_TWO_PI * fx;
}

// shortest signed horizontal distance on the azimuth circle
__device__ __forceinline__ float equirect_wrap_dx(float dx, const float Wpx) {
  dx = fmodf(dx + 0.5f * Wpx, Wpx);
  if (dx < 0.0f) dx += Wpx;
  return dx - 0.5f * Wpx;
}

__device__ __forceinline__ bool equirect_proj_with_jacobian(
    const float X, const float Y, const float Z,
    const float fx, const float fy,
    const float cx, const float cy,
    float& x, float& y,
    float Jp[2][3])
{
  const float rho2 = X * X + Z * Z;
  const float s2 = X * X + Y * Y + Z * Z;
  if (rho2 < 1e-10f || s2 < 1e-10f) {
    return false;
  }

  const float rho = sqrtf(rho2);
  const float Wpx = equirect_x_period(fx);
  // wrap azimuth into [0, W)
  x = fmodf(fx * atan2f(X, Z) + cx, Wpx);
  if (x < 0.0f) x += Wpx;
  y = fy * atan2f(Y, rho) + cy;

  Jp[0][0] = fx * Z / rho2;
  Jp[0][1] = 0.0f;
  Jp[0][2] = -fx * X / rho2;

  const float inv_rho_s2 = 1.0f / (rho * s2);
  Jp[1][0] = -fy * X * Y * inv_rho_s2;
  Jp[1][1] = fy * rho / s2;
  Jp[1][2] = -fy * Z * Y * inv_rho_s2;
  return true;
}

  
__device__ void
expSO3(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta_p4 = theta_sq * theta_sq;

  float theta = sqrtf(theta_sq);
  float imag, real;

  if (theta_sq < 1e-8) {
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

__device__ void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__device__ void
expSE3(const float *xi, float* t, float* q) {
  // SE3 exponential map

  expSO3(xi + 3, q);
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};

  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4) {
    float a = (1 - cosf(theta)) / theta_sq;
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    float b = (theta - sinf(theta)) / (theta * theta_sq);
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}


__device__ void
retrSE3(const float *xi, const float* t, const float* q, float* t1, float* q1) {
  // retraction on SE3 manifold

  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  
  expSE3(xi, dt, dq);

  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

  actSO3(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
}



__global__ void pose_retr_kernel(const int t0, const int t1,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    torch::PackedTensorAccessor32<mtype,2,torch::RestrictPtrTraits> update)
{
  GPU_1D_KERNEL_LOOP(i, t1 - t0) {
    const float t = t0 + i;
    float t1[3], t0[3] = { poses[t][0], poses[t][1], poses[t][2] };
    float q1[4], q0[4] = { poses[t][3], poses[t][4], poses[t][5], poses[t][6] };

    float xi[6] = {
      update[i][0],
      update[i][1],
      update[i][2],
      update[i][3],
      update[i][4],
      update[i][5],
    };

    retrSE3(xi, t0, q0, t1, q1);

    poses[t][0] = t1[0];
    poses[t][1] = t1[1];
    poses[t][2] = t1[2];
    poses[t][3] = q1[0];
    poses[t][4] = q1[1];
    poses[t][5] = q1[2];
    poses[t][6] = q1[3];
  }
}


__global__ void patch_retr_kernel(
    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> index,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> patches,
    torch::PackedTensorAccessor32<mtype,1,torch::RestrictPtrTraits> update)
{
  GPU_1D_KERNEL_LOOP(n, index.size(0)) {
    const int p = patches.size(2);
    const int ix = index[n];
  
    float d = patches[ix][2][0][0];
    d = d + update[n];
    d = (d > 20) ? 1.0 : d;
    d = max(d, 1e-4);

    for (int i=0; i<p; i++) {
      for (int j=0; j<p; j++) {
        patches[ix][2][i][j] = d;
      }
    }
  }
}


__global__ void reprojection_residuals_and_hessian(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> kk,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ku,
    torch::PackedTensorAccessor32<mtype,2,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<mtype,2,torch::RestrictPtrTraits> E,
    torch::PackedTensorAccessor32<mtype,1,torch::RestrictPtrTraits> C,
    torch::PackedTensorAccessor32<mtype,1,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<mtype,1,torch::RestrictPtrTraits> u, const int t0)
{

  __shared__ float fx, fy, cx, cy;
  if (threadIdx.x == 0) {
    fx = intrinsics[0][0];
    fy = intrinsics[0][1];
    cx = intrinsics[0][2];
    cy = intrinsics[0][3];
  }

  __syncthreads();

  GPU_1D_KERNEL_LOOP(n, ii.size(0)) {
    int k = ku[n]; // inverse indices
    int ix = ii[n];
    int jx = jj[n];
    int kx = kk[n]; // actual

    float ti[3] = { poses[ix][0], poses[ix][1], poses[ix][2] };
    float tj[3] = { poses[jx][0], poses[jx][1], poses[jx][2] };
    float qi[4] = { poses[ix][3], poses[ix][4], poses[ix][5], poses[ix][6] };
    float qj[4] = { poses[jx][3], poses[jx][4], poses[jx][5], poses[jx][6] };

    float Xi[4], Xj[4];
    equirect_iproj(
      patches[kx][0][1][1],
      patches[kx][1][1][1],
      fx, fy, cx, cy, Xi);
    Xi[3] = patches[kx][2][1][1];
    
    float tij[3], qij[4];
    relSE3(ti, qi, tj, qj, tij, qij);
    actSE3(tij, qij, Xi, Xj);

    const float X = Xj[0];
    const float Y = Xj[1];
    const float Z = Xj[2];
    const float W = Xj[3];

    float x1 = 0.0f, y1 = 0.0f;
    float Jp[2][3] = {};
    const bool jac_valid = equirect_proj_with_jacobian(X, Y, Z, fx, fy, cx, cy, x1, y1, Jp);

    const float Wpx = equirect_x_period(fx);
    const float rx = equirect_wrap_dx(target[n][0] - x1, Wpx);
    const float ry = target[n][1] - y1;

    // minimum range |(X,Y,Z)| / W
    const float MIN_DEPTH = 0.2f;
    const bool depth_ok = (X*X + Y*Y + Z*Z) > (MIN_DEPTH * MIN_DEPTH * W * W);

    const bool in_bounds = jac_valid && depth_ok && isfinite(x1) && isfinite(y1) &&
      (sqrt(rx*rx + ry*ry) < 128);

    const float mask = in_bounds ? 1.0 : 0.0;

    ix = ix - t0;
    jx = jx - t0;

    for (int row=0; row<2; row++) {

      float Jj[6], Ji[6], Jz, r, w;

      if (row == 0){

        r = rx;
        w = mask * weight[n][0];

        Jz = Jp[0][0] * tij[0] + Jp[0][1] * tij[1] + Jp[0][2] * tij[2];

      } else {

        r = ry;
        w = mask * weight[n][1];

        Jz = Jp[1][0] * tij[0] + Jp[1][1] * tij[1] + Jp[1][2] * tij[2];

      }

      const float* jr = Jp[row];
      Jj[0] = jr[0] * W;
      Jj[1] = jr[1] * W;
      Jj[2] = jr[2] * W;
      Jj[3] = jr[1] * (-Z) + jr[2] * Y;
      Jj[4] = jr[0] * Z + jr[2] * (-X);
      Jj[5] = jr[0] * (-Y) + jr[1] * X;

      adjSE3(tij, qij, Jj, Ji);

      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
            atomicAdd(&B[6*ix+i][6*ix+j],  w * Ji[i] * Ji[j]);
          if (jx >= 0)
            atomicAdd(&B[6*jx+i][6*jx+j],  w * Jj[i] * Jj[j]);
          if (ix >= 0 && jx >= 0) {
            atomicAdd(&B[6*ix+i][6*jx+j], -w * Ji[i] * Jj[j]);
            atomicAdd(&B[6*jx+i][6*ix+j], -w * Jj[i] * Ji[j]);
          }
        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          atomicAdd(&E[6*ix+i][k], -w * Jz * Ji[i]);
        if (jx >= 0)
          atomicAdd(&E[6*jx+i][k],  w * Jz * Jj[i]);
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          atomicAdd(&v[6*ix+i], -w * r * Ji[i]);
        if (jx >= 0)
          atomicAdd(&v[6*jx+i],  w * r * Jj[i]);
      }

      atomicAdd(&C[k], w * Jz * Jz);
      atomicAdd(&u[k], w *  r * Jz);
    }
  }
}


std::vector<torch::Tensor> cuda_ba(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor target,
    torch::Tensor weight,
    torch::Tensor lmbda,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor kk,
    const int t0, const int t1, const int iterations)
{

  auto ktuple = torch::_unique(kk, true, true);
  torch::Tensor kx = std::get<0>(ktuple);
  torch::Tensor ku = std::get<1>(ktuple);

  const int N = t1 - t0;    // number of poses
  const int M = kx.size(0); // number of patches
  const int P = patches.size(3); // patch size

  poses = poses.view({-1, 7});
  patches = patches.view({-1,3,P,P});
  intrinsics = intrinsics.view({-1, 4});

  target = target.view({-1, 2});
  weight = weight.view({-1, 2});

  torch::Tensor B = torch::empty({6*N, 6*N}, mdtype);
  torch::Tensor E = torch::empty({6*N, 1*M}, mdtype);
  torch::Tensor C = torch::empty({M}, mdtype);

  torch::Tensor v = torch::empty({6*N}, mdtype);
  torch::Tensor u = torch::empty({1*M}, mdtype);

  for (int itr=0; itr < iterations; itr++) {

    B.zero_();
    E.zero_();
    C.zero_();
    v.zero_();
    u.zero_();

    v = v.view({6*N});
    u = u.view({1*M});

    reprojection_residuals_and_hessian<<<NUM_BLOCKS(ii.size(0)), NUM_THREADS>>>(
      poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      target.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      weight.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      kk.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      ku.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      B.packed_accessor32<mtype,2,torch::RestrictPtrTraits>(),
      E.packed_accessor32<mtype,2,torch::RestrictPtrTraits>(),
      C.packed_accessor32<mtype,1,torch::RestrictPtrTraits>(),
      v.packed_accessor32<mtype,1,torch::RestrictPtrTraits>(),
      u.packed_accessor32<mtype,1,torch::RestrictPtrTraits>(), t0);

    v = v.view({6*N, 1});
    u = u.view({1*M, 1});

    torch::Tensor Q = 1.0 / (C + lmbda).view({1, M});

    if (t1 - t0 == 0) {

      torch::Tensor Qt = torch::transpose(Q, 0, 1);
      torch::Tensor dZ = Qt * u;

      dZ = dZ.view({M});

      patch_retr_kernel<<<NUM_BLOCKS(M), NUM_THREADS>>>(
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        dZ.packed_accessor32<mtype,1,torch::RestrictPtrTraits>());

    }  else {

      torch::Tensor dX, dZ, Qt = torch::transpose(Q, 0, 1);
      torch::Tensor I = torch::eye(6*N, mdtype);

      torch::Tensor EQ = E * Q;
      torch::Tensor Et = torch::transpose(E, 0, 1);

      torch::Tensor S = B - torch::matmul(EQ, Et);
      torch::Tensor y = v - torch::matmul(EQ,  u);

      S += I * (1e-4 * S + 1.0);
      torch::Tensor U = std::get<0>(at::linalg_cholesky_ex(S));
      dX = torch::cholesky_solve(y, U);
      dZ = Qt * (u - torch::matmul(Et, dX));

      dX = dX.view({N, 6});
      dZ = dZ.view({M});

      pose_retr_kernel<<<NUM_BLOCKS(N), NUM_THREADS>>>(t0, t1,
          poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
          dX.packed_accessor32<mtype,2,torch::RestrictPtrTraits>());

      patch_retr_kernel<<<NUM_BLOCKS(M), NUM_THREADS>>>(
          kx.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
          patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
          dZ.packed_accessor32<mtype,1,torch::RestrictPtrTraits>());
    }
  }

  return {};
}
