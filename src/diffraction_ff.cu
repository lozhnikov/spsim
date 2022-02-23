#include "diffraction.h"
#include "config.h"
#include "crystal.h"
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/extrema.h>
#include <thrust/partition.h>
#include <thrust/count.h>

__global__ void CUDA_scattering_at_k_ff(float* real_part,float * imag_part,
				     const float * atomic_formfactors,
             const float scattering_vector_length, const float B,
				     const float * HKL_list,const float * pos,
				     const int k,const int natoms,
				     float2 beam_center, float beam_fwhm);
__device__ float cuda_illumination_function_ff(const float * pos, float2 beam_center, float beam_fwhm);

__global__ void CUDA_scattering_from_all_atoms_ff(cufftComplex * F,float * I,const int * Z,
					       const float * pos,
					       const float * HKL_list,const int hkl_size,
					       const int start_atom, const int end_atom,
					       const int natoms,const float * atomic_formfactors,
					       const float B, float2 beam_center,
					       float beam_fwhm);
__global__ 
void CUDA_spectrum_scattering_from_all_atoms_ff(cufftComplex * F,float * I, const int * Z,
					     const float * pos, const float * HKL_list,
					     const int hkl_size,const int start_atom,
					     const int end_atom, const int natoms,
					     const float * atomic_formfactors,const float B,
					     const float wavelength, const float bandwidth,
					     const int wavelength_samples,float2 beam_center,
					     float beam_fwhm);

static float illumination_function_ff(Experiment * exper,float * pos){
  float dist2;
  float sigma;
  /* If no fwhm is defined just return 1 everywhere */
  if(!exper->beam_fwhm){
    return 1;
  }
  /* calculate distance from the center of the beam */
  dist2 = (pos[0]-exper->beam_center_x)*(pos[0]-exper->beam_center_x)+(pos[1]-exper->beam_center_y)*(pos[1]-exper->beam_center_y);
  sigma = exper->beam_fwhm/2.355;
  //printf("here\n");
  return exp(-dist2/(2*sigma*sigma));
}

Diffraction_Pattern * cuda_compute_pattern_on_list_ff(Molecule * mol, float * HKL_list, int HKL_list_size, float B,Experiment * exp,Options * opts){
#ifndef _USE_CUDA
  sp_error_fatal("Can't use cuda when not compiled for CUDA");
#else
  int timer = sp_timer_start();
  int i,j;
  float scattering_vector_length;
  Diffraction_Pattern * res = (Diffraction_Pattern *)malloc(sizeof(Diffraction_Pattern));
  int points_per_percent;

  int threads_per_block = 64;
  int number_of_blocks = (mol->natoms+threads_per_block-1)/threads_per_block;
 

  res->F = (Complex *)malloc(sizeof(Complex)*HKL_list_size);
  res->ints = (float *)malloc(sizeof(float)*HKL_list_size);
  res->HKL_list = (float *)malloc(sizeof(float)*3*HKL_list_size);
  memcpy(res->HKL_list,HKL_list,sizeof(float)*3*HKL_list_size);
  res->HKL_list_size = HKL_list_size;
  float * d_real_part;
  cutilSafeCall(cudaMalloc((void **)&d_real_part,sizeof(float)*mol->natoms)); 
  float * d_imag_part;
  cutilSafeCall(cudaMalloc((void **)&d_imag_part,sizeof(float)*mol->natoms));
  int * d_atomic_number;
  cutilSafeCall(cudaMalloc((void **)&d_atomic_number,sizeof(int)*mol->natoms));
  cutilSafeCall(cudaMemcpy(d_atomic_number,mol->atomic_number,sizeof(int)*mol->natoms,cudaMemcpyHostToDevice));
  float * d_atomic_pos;
  cutilSafeCall(cudaMalloc((void **)&d_atomic_pos,sizeof(float)*mol->natoms*3));
  cutilSafeCall(cudaMemcpy(d_atomic_pos,mol->pos,sizeof(float)*mol->natoms*3,cudaMemcpyHostToDevice));

  float * d_HKL_list;
  cutilSafeCall(cudaMalloc((void **)&d_HKL_list,sizeof(float)*3*HKL_list_size));
  cutilSafeCall(cudaMemcpy(d_HKL_list,HKL_list,sizeof(float)*3*HKL_list_size,cudaMemcpyHostToDevice));

  float* d_atomic_formfactors;
  cutilSafeCall(cudaMalloc((void **)&d_atomic_formfactors,sizeof(float)*mol->natoms*9));
  cutilSafeCall(cudaMemcpy(d_atomic_formfactors,mol->atomic_formfactors,sizeof(float)*mol->natoms*9,cudaMemcpyHostToDevice));



  points_per_percent = 1+(HKL_list_size)/100;
  for(i = 0;i<HKL_list_size;i++){
    if(i % points_per_percent == 0){
        if (opts->verbosity_level > 0) {
    		 fprintf(stderr,"%f percent done\n",(100.0*i)/(HKL_list_size));
	}		  
    }
    sp_real(res->F[i]) = 0;
    sp_imag(res->F[i]) = 0;
    scattering_vector_length = sqrt(HKL_list[3*i]*HKL_list[3*i]+HKL_list[3*i+1]*HKL_list[3*i+1]+HKL_list[3*i+2]*HKL_list[3*i+2]);


    float2 beam_center = {exp->beam_center_x,exp->beam_center_y};
    CUDA_scattering_at_k_ff<<<number_of_blocks, threads_per_block>>>(d_real_part,d_imag_part,d_atomic_formfactors, scattering_vector_length, B, d_HKL_list,d_atomic_pos,i,mol->natoms,beam_center,exp->beam_fwhm);
    thrust::device_ptr<float> begin =  thrust::device_pointer_cast(d_real_part);
    thrust::device_ptr<float> end =  thrust::device_pointer_cast(d_real_part+mol->natoms);
    sp_real(res->F[i]) = thrust::reduce(begin, end);
    begin =  thrust::device_pointer_cast(d_imag_part);
    end =  thrust::device_pointer_cast(d_imag_part+mol->natoms);
    sp_imag(res->F[i]) = thrust::reduce(begin, end);
    res->ints[i] = sp_cabs(res->F[i])*sp_cabs(res->F[i]);
  }
  cudaFree(d_real_part);
  cudaFree(d_imag_part);
  cudaFree(d_atomic_number);
  cudaFree(d_atomic_pos);
  cudaFree(d_HKL_list);
  cudaFree(d_atomic_formfactors);
  if (opts->verbosity_level > 0) {
     printf("%g atoms.pixel/s\n",1.0e6*HKL_list_size*mol->natoms/sp_timer_stop(timer));
  } 	     
  return res;  
#endif 
}


int sort_int_map_ff(const void * a,const void * b){
  if( *(int *)a < *(int *)b){
    return -1;
  }else if( *(int *)a == *(int *)b){
    return 0;
  }else{
    return 1;
  }  
}

Diffraction_Pattern * cuda_compute_pattern_on_list2_ff(Molecule * mol, float * HKL_list, int HKL_list_size, float B,Experiment * exp,Options * opts){
#ifndef _USE_CUDA
  sp_error_fatal("Can't use cuda when not compiled for CUDA");
#else
  int timer = sp_timer_start();
  Diffraction_Pattern * res = (Diffraction_Pattern *)malloc(sizeof(Diffraction_Pattern));
  dim3 threads_per_block(16,16);
  const int HKL_side = (sqrt(HKL_list_size)+1);
  dim3 number_of_blocks( (HKL_side+threads_per_block.x-1)/threads_per_block.x,
			 (HKL_side+threads_per_block.y-1)/threads_per_block.y );
  
  if (opts->verbosity_level > 0) {
     printf("Using %d blocks\n",number_of_blocks.x*number_of_blocks.y);
  }

  /* sort atoms by atomic number */
  float * sorted_pos = (float *) malloc(sizeof(float)*mol->natoms*3);
  float * sorted_formfactors = (float *) malloc(sizeof(float)*mol->natoms*9);
  int * sorted_atomic_number = (int *) malloc(sizeof(int)*mol->natoms);
  /* on the odd indexes we keep a key corresponding to the original index
     and on the even indexes we keep the atomic number */ 
  int * sorted_map = (int *) malloc(sizeof(int)*mol->natoms*2);
  for(int i = 0;i<mol->natoms;i++){
    sorted_map[2*i] = mol->atomic_number[i];
    sorted_map[2*i+1] = i;
  }
  qsort(sorted_map,mol->natoms,sizeof(int)*2,sort_int_map_ff);
  /* make use of the sorted keys to sort the positions also */
  for(int i = 0;i<mol->natoms;i++){
    sorted_atomic_number[i] = sorted_map[2*i];
    sorted_pos[3*i+0] = mol->pos[sorted_map[2*i+1]*3+0];
    sorted_pos[3*i+1] = mol->pos[sorted_map[2*i+1]*3+1];
    sorted_pos[3*i+2] = mol->pos[sorted_map[2*i+1]*3+2];

    for (int j = 0; j < 9; j++)
      sorted_formfactors[9 * i + j] = mol->atomic_formfactors[sorted_map[2 * i + 1] * 9 + j];
  }

  res->F = (Complex *)malloc(sizeof(Complex)*HKL_list_size);
  res->ints = (float *)malloc(sizeof(float)*HKL_list_size);
  res->HKL_list = (float *)malloc(sizeof(float)*3*HKL_list_size);
  memcpy(res->HKL_list,HKL_list,sizeof(float)*3*HKL_list_size);
  res->HKL_list_size = HKL_list_size;
  int * d_atomic_number;
  (cudaMalloc((void **)&d_atomic_number,sizeof(int)*mol->natoms));
  cutilSafeCall(cudaMemcpy(d_atomic_number,sorted_atomic_number,sizeof(int)*mol->natoms,cudaMemcpyHostToDevice));
  float * d_atomic_pos;
  cutilSafeCall(cudaMalloc((void **)&d_atomic_pos,sizeof(float)*mol->natoms*3));
  cutilSafeCall(cudaMemcpy(d_atomic_pos,sorted_pos,sizeof(float)*mol->natoms*3,cudaMemcpyHostToDevice));

  float * d_atomic_formfactors;
  cutilSafeCall(cudaMalloc((void **)&d_atomic_formfactors,sizeof(float)*mol->natoms*9));
  cutilSafeCall(cudaMemcpy(d_atomic_formfactors,sorted_formfactors,sizeof(float)*mol->natoms*9,cudaMemcpyHostToDevice));

  float * d_HKL_list;
  cutilSafeCall(cudaMalloc((void **)&d_HKL_list,sizeof(float)*3*HKL_list_size));
  cutilSafeCall(cudaMemcpy(d_HKL_list,HKL_list,sizeof(float)*3*HKL_list_size,cudaMemcpyHostToDevice));
  cufftComplex * d_F;
  cutilSafeCall(cudaMalloc((void **)&d_F,sizeof(cufftComplex)*HKL_list_size));
  cutilSafeCall(cudaMemset(d_F,0,sizeof(cufftComplex)*HKL_list_size));
  
  float * d_I;
  cutilSafeCall(cudaMalloc((void **)&d_I,sizeof(float)*HKL_list_size));
  cutilSafeCall(cudaMemset(d_I,0,sizeof(float)*HKL_list_size));

  /* we have to do this in chunks so we don't block the card forever */
  const int chunk_size = 100;

  /* extra loop around the wavelengths */
  if(opts->wavelength_samples >1 && (opts->wavelength_samples & 1) == 0){
    opts->wavelength_samples++;
  }

  for(int i = 0;i<mol->natoms;i+=chunk_size){ 
    if (opts->verbosity_level > 0) {	      
       printf("%f%% done\n",(100.0*i)/mol->natoms);
    }
    int end_atom = sp_min(i+chunk_size,mol->natoms);
    int start_atom = i;
    float2 beam_center = {exp->beam_center_x,exp->beam_center_y};
    if(exp->bandwidth == 0 || opts->wavelength_samples == 1){
      CUDA_scattering_from_all_atoms_ff<<<number_of_blocks, threads_per_block>>>(d_F,d_I,d_atomic_number,d_atomic_pos,d_HKL_list,HKL_list_size,start_atom,end_atom,mol->natoms,d_atomic_formfactors,B,beam_center,exp->beam_fwhm);
    }else{
      CUDA_spectrum_scattering_from_all_atoms_ff<<<number_of_blocks, threads_per_block>>>(d_F,d_I,d_atomic_number,d_atomic_pos,d_HKL_list,HKL_list_size,start_atom,end_atom,mol->natoms,d_atomic_formfactors,B,exp->wavelength,exp->bandwidth,opts->wavelength_samples,beam_center,exp->beam_fwhm);
    }
    cudaThreadSynchronize();
    sp_cuda_check_errors();
  }
  //  calculate_pattern_from_crystal_cuda(d_I,d_F,d_HKL_list, HKL_list_size, opts);
  cudaMemcpy(res->F,d_F,sizeof(cufftComplex)*HKL_list_size,cudaMemcpyDeviceToHost);
  cudaMemcpy(res->ints,d_I,sizeof(float)*HKL_list_size,cudaMemcpyDeviceToHost);
  sp_cuda_check_errors();
  if (opts->verbosity_level > 0) {
     printf("%g atoms.pixel/s\n",1.0e6*HKL_list_size*mol->natoms/sp_timer_stop(timer));
  }
  cudaFree(d_I);
  cudaFree(d_F);
  cudaFree(d_HKL_list);
  cudaFree(d_atomic_pos);
  cudaFree(d_atomic_number);
  cudaFree(d_atomic_formfactors);
  free(sorted_pos);
  free(sorted_atomic_number);
  free(sorted_map);
  free(sorted_formfactors);
  return res;  
#endif 
}

__global__ void CUDA_scattering_at_k_ff(float* real_part,float * imag_part, 
				     const float * atomic_formfactors,
             const float scattering_vector_length, const float B, const float * HKL_list,
				     const float * pos,const int k,const int natoms,
				     const float2 beam_center,
				     const float beam_fwhm){
  const int i = blockIdx.x*blockDim.x + threadIdx.x;


  if(i<natoms){
    float sf = 0;
    float d = 1e-10F * scattering_vector_length;
    /* the 0.25 is there because the 's' used by the aproxumation is 'd/2' */
    for(int j = 0; j < 4; j++){
      sf+= atomic_formfactors[i * 9 + j]*exp(-(atomic_formfactors[i * 9 + j+4]+B)*d*d*0.25F);
    }                
    sf += atomic_formfactors[i * 9 + 8]*exp(-B*d*d/0.25F);

    sf *= sqrt(cuda_illumination_function_ff(&pos[i*3], beam_center, beam_fwhm));
    float tmp = -2*3.14159265F*(HKL_list[3*k]*pos[i*3]+HKL_list[3*k+1]*pos[i*3+1]+HKL_list[3*k+2]*pos[i*3+2]);
    real_part[i] = sf*cos(tmp);
    imag_part[i] = sf*sin(tmp);
  }
}

__global__ void CUDA_scattering_from_all_atoms_ff(cufftComplex * F,float * I,const int * Z,
					       const float * pos, const float * HKL_list,
					       const int hkl_size,const int start_atom,
					       const int end_atom, const int natoms,
					       const float * atomic_formfactors,const float B,
					       float2 beam_center, float beam_fwhm){
  //  const int id = blockIdx.x*blockDim.x + threadIdx.x;
  const int id = ((blockIdx.y*blockDim.y + threadIdx.y)*gridDim.x + blockIdx.x )*blockDim.x + threadIdx.x;
  if(id<hkl_size){
    float sf = 0;
    float d = sqrt(HKL_list[3*id]*HKL_list[3*id]+HKL_list[3*id+1]*HKL_list[3*id+1]+HKL_list[3*id+2]*HKL_list[3*id+2]) * 1e-10F;
    const float hkl[3] = {HKL_list[3*id+0],
    	  	          HKL_list[3*id+1],
			  HKL_list[3*id+2]};
    for(int i = start_atom;i<end_atom;i++){ 
      
	sf = 0;
	/* the 0.25 is there because the 's' used by the approximation is 'd/2' */
	for(int j = 0;j<4;j++){
	  sf+= atomic_formfactors[i*9+j]*exp(-(atomic_formfactors[i*9+j+4]+B)*d*d*0.25F);
	}                
	sf += atomic_formfactors[i*9+8]*exp(-B*d*d/0.25F);


      float tmp = -2*3.14159265F*(hkl[0]*pos[i*3+0]+
                                  hkl[1]*pos[i*3+1]+
				  hkl[2]*pos[i*3+2]);      
      float illum = sqrt(cuda_illumination_function_ff(&pos[i*3], beam_center, beam_fwhm));
      F[id].x += illum*sf*cos(tmp);
      F[id].y += illum*sf*sin(tmp);
    }
    if(end_atom == natoms){
      I[id] =  F[id].x*F[id].x + F[id].y*F[id].y;
    }
  }    
}


__device__ float cuda_illumination_function_ff(const float * pos, float2 beam_center, float beam_fwhm){
  float dist2;
  float sigma;
  /* If no fwhm is defined just return 1 everywhere */
  if(!beam_fwhm){
    return 1;
  }
  /* calculate distance from the center of the beam */
  dist2 = (pos[0]-beam_center.x)*(pos[0]-beam_center.x)+
          (pos[1]-beam_center.y)*(pos[1]-beam_center.y);
  sigma = beam_fwhm/2.355;
  return exp(-dist2/(2*sigma*sigma));
}

__global__ 
void CUDA_spectrum_scattering_from_all_atoms_ff(cufftComplex * F,float * I, const int * Z,
					     const float * pos, const float * HKL_list,
					     const int hkl_size,const int start_atom,
					     const int end_atom, const int natoms,
					     const float * atomic_formfactors,const float B,
					     const float wavelength, const float bandwidth,
					     const int wavelength_samples,
					     const float2 beam_center,
					     const float beam_fwhm){
  //  const int id = blockIdx.x*blockDim.x + threadIdx.x;
  const int id = ((blockIdx.y*blockDim.y + threadIdx.y)*gridDim.x + blockIdx.x )*blockDim.x + threadIdx.x;
  if(id<hkl_size){
    const float H = HKL_list[3*id];
    const float K = HKL_list[3*id+1];
    const float L = HKL_list[3*id+2];
    /* The 2.0 is due to the fact that the bandwidth corresponds to 2 sigma*/
    const float w_stddev = (wavelength-(wavelength*(1-bandwidth/2)))/2.0;
    const float pi = 3.14159265;
    const float one_over_sqrt_two_pi = 1/(sqrt(2*pi));
    float sf = 0;
    float d = sqrt(HKL_list[3*id]*HKL_list[3*id]+HKL_list[3*id+1]*HKL_list[3*id+1]+HKL_list[3*id+2]*HKL_list[3*id+2]) * 1e-10F;
    for(int i = start_atom;i<end_atom;i++){ 
      	sf = 0;
	/* the 0.25 is there because the 's' used by the aproxumation is 'd/2' */
	for(int j = 0;j<4;j++){
	  sf+= atomic_formfactors[i*9+j]*exp(-(atomic_formfactors[i*9+j+4]+B)*d*d*0.25F);
	}                
	sf += atomic_formfactors[i*9+8]*exp(-B*d*d/0.25F);
      float illumination = sqrt(cuda_illumination_function_ff(&pos[i*3], beam_center, beam_fwhm));
      float total_weight = 0;
      cufftComplex f = {0,0};
      for(int j = 0;j < wavelength_samples;j++){
	float new_wavelength = (wavelength*(1-bandwidth/2))+
	  (((float)j/(wavelength_samples-1))*wavelength*bandwidth);
	const float h = H*wavelength/new_wavelength;
	const float k = K*wavelength/new_wavelength;
	const float l = L*wavelength/new_wavelength;
	const float std_deviations = (new_wavelength-wavelength)/w_stddev;
	const float weight = exp(-std_deviations*std_deviations/2);
	total_weight += weight;
	float tmp = -2*pi*(h*pos[i*3]+k*pos[i*3+1]+l*pos[i*3+2]);      
	f.x += sf*cos(tmp)*weight*illumination;
	f.y += sf*sin(tmp)*weight*illumination;
      }
      f.x = f.x/total_weight;
      f.y = f.y/total_weight;
      F[id].x += f.x;
      F[id].y += f.y;
    }
    if(end_atom == natoms){
      I[id] =  F[id].x*F[id].x + F[id].y*F[id].y;
    }
  }    
}
