/*
 * Copyright (c) 2006 Filipe Maia
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *\
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */


#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <spimage.h>

#ifdef NFFT_SUPPORT
#include <nfft3.h>

/* Taken from condor
 *  NFFT_DEFINE_MALLOC_API is a new macro that was introduced in nfft version 3.3
 */
#if defined(NFFT_DEFINE_MALLOC_API)
#define NFFT_VERSION_ABOVE_3_3 1
#else
#define NFFT_VERSION_ABOVE_3_3 0
#endif

#endif

#include "config.h"
#include "diffraction.h"
#include "mpi_comm.h"
#include "io.h"
#include "box.h"
#if defined __GNUC__ && !defined __clang__
#include "sse_mathfun.h"
#endif

#ifndef M_PI
#define M_PI     3.1415926535897932384626433832795029L
#endif
#ifndef PI
#define PI     3.1415926535897932384626433832795029L
#endif


#define ELEMENTS 100

static float sp_mod(float a, float b){
  while(a < 0){
    a += b;
  }
  while(a>=b){
    a -= b;
  }
  return a;
}

/*
  Use atomsf.lib to obtain details for all atoms in a problem.
*/

/*
static void write_ff_tables(){
  int i;
  FILE * fp = fopen("atomsf.data","w");
  fprintf(fp," {\n");
  i = 0;
  fprintf(fp,"{%e,%e,%e,%e,%e,%e,%e,%e,%e}\n",atomsf[i][0],
	  atomsf[i][1],atomsf[i][2],atomsf[i][3],
	  atomsf[i][4],atomsf[i][5],atomsf[i][6],
	  atomsf[i][7],atomsf[i][8]);
  for(i =1;i<ELEMENTS;i++){
    fprintf(fp,",{%e,%e,%e,%e,%e,%e,%e,%e,%e}\n",atomsf[i][0],
	    atomsf[i][1],atomsf[i][2],atomsf[i][3],
	    atomsf[i][4],atomsf[i][5],atomsf[i][6],
	    atomsf[i][7],atomsf[i][8]);
  };
  fprintf(fp,"};\n");   
  fclose(fp);
}
*/


/* d should be the size of the scattering vector |H| */
/* d is in m^-1 but we need it in A^-1 so divide  by 1^10*/
static float  scatt_factor_ff(float d, float B, float* formfactors){
  float res = 0;
  int i;  
  d *= 1e-10;
  /* the 0.25 is there because the 's' used by the aproxumation is 'd/2' */
  for(i = 0;i<4;i++){
    res+= formfactors[i]*exp(-(formfactors[i+4]+B)*d*d*0.25);
  }                
  res += formfactors[8]*exp(-B*d*d*0.25);
  return res;    
}


/* d should be the distance from the center of the atom */
/* d is in m but we need it in A so we multiply by 1^10*/
/* this formula results for the *spherical* fourier transform
   of the 4 gaussian aproximation of the scattering factors.
   Check the basic formulas paper for more details 

   
   The formula is:
   Sum i=1..4 8*Pi^(3/2)*ai/bi^(3/2)/exp(4*Pi^2*r^2/bi)

   And now the tricky part is what to do with c.
   I decided to add it to the gaussian with the smallest b (the widest gaussian)

   But this has all been factored in the fill_ed_table
*/
static float  electron_density_ff(float d, float B, float* formfactors){
  float atom_ed[9];
  float res = 0;
  int i;

  if(B <= 0){
    B = 1;
  }

  for(int i = 0;i<4;i++){
    atom_ed[i] = 8*pow(M_PI,1.5)*(formfactors[i])/pow((formfactors[i+4]+B),1.5);
  }
  atom_ed[4] = 8*pow(M_PI,1.5)*(formfactors[8])/pow(B,1.5);

  d *= 1e10;
  for(i = 0;i<4;i++){
    res+= atom_ed[i]*exp(-(atom_ed[i+4])*d*d);
  }
  res+= atom_ed[4]*exp(-1*d*d);
  return res;    
  }


static double illumination_function(Experiment * exper,float * pos){
  double dist2;
  double sigma;
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


#ifdef NFFT_SUPPORT
void multiply_pattern_with_scattering_factor_ff(fftw_complex * f,float* formfactors,int nx, int ny, int nz, double rs_pixel_x,double rs_pixel_y,double rs_pixel_z, float B){
  /* f is assumed to be C ordered*/
  int i = 0;
  for(int zi = -nz/2;zi<nz/2;zi++){
    float z = (float)zi/(nz)/rs_pixel_z;
    for(int yi = -ny/2;yi<ny/2;yi++){
      float y = (float)yi/(ny)/rs_pixel_y;
      for(int xi = -nx/2;xi<nx/2;xi++){
	float x = (float)xi/(nx)/rs_pixel_x;
	double distance = sqrt(x*x+y*y+z*z);
	float sf = scatt_factor_ff(distance, B, formfactors);
	f[i][0] *= sf;
	f[i][1] *= sf;
	i++;
      }
    }
  }
}

Diffraction_Pattern * compute_pattern_by_nfft_ff(Molecule * mol, CCD * det, Experiment * exp, float B,float * HKL_list,Options * opts){
  double alpha_x = atan(det->width/(2.0 * det->distance));
  double alpha_y = atan(det->height/(2.0 * det->distance));
  double alpha_z = atan(det->depth/(2.0 * det->distance));
  /* Assuming spherical detector (which is a bit weird for a 3D detector)*/
  double smax_x = tan(alpha_x)/exp->wavelength;
  double smax_y = tan(alpha_y)/exp->wavelength;
  double smax_z = tan(alpha_z)/exp->wavelength;
  float rs_pixel_x = 1/(smax_x*2);  
  float rs_pixel_y = 1/(smax_y*2);  
  float rs_pixel_z = 1/(smax_z*2);  
  int is_element_in_molecule[ELEMENTS];
  fprintf(stderr,"Pixel size x-%e y-%e z-%e\n",rs_pixel_x,rs_pixel_y,rs_pixel_z);
  int nx = det->nx;
  int ny = det->ny;
  int nz = det->nz;
  int n_el_in_mol = 0;

  int mpi_skip = 1;
  int mpi_skip_flag = 0;

/* in meters defines the limit up to which we compute the electron density */
  Diffraction_Pattern * res = malloc(sizeof(Diffraction_Pattern));
  res->HKL_list_size = nx*ny*nz;
  res->F = malloc(sizeof(Complex)*res->HKL_list_size);
  res->ints = malloc(sizeof(float)*res->HKL_list_size);
  res->HKL_list = malloc(sizeof(float)*3*res->HKL_list_size);

  nfft_plan p;
  
#ifdef MPI  
  /* Skip a number of elements equivalent to the number of computers used */
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_skip);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_skip_flag);
#endif
  
  /*  if(!atomed_initialized){
    fill_ed_tables(B);
    atomed_initialized = 1;
    }*/
  for(int j = 0 ;j< ELEMENTS;j++){
    is_element_in_molecule[j] = 0;
  }
  for(int j = 0 ;j< mol->natoms;j++){
    if(is_element_in_molecule[mol->atomic_number[j]] == 0){
      n_el_in_mol++;
    }
    is_element_in_molecule[mol->atomic_number[j]]++;
  }

  for(int k = 0;k<nx*ny*nz;k++){
    sp_real(res->F[k]) = 0;
    sp_imag(res->F[k]) = 0;
  }
  for(int Z = 0 ;Z< ELEMENTS;Z++){
    if(!is_element_in_molecule[Z]){
      continue;
    }else if(is_element_in_molecule[Z]){
      mpi_skip_flag++;
      if(mpi_skip_flag < mpi_skip){
	continue;
      }
      mpi_skip_flag = 0;
      fprintf(stderr,"Calculating Z = %d\n",Z);
      /* more than 100 atoms of that kind in the molecule use nfft*/
      nfft_init_3d(&p,nz,ny,nx,is_element_in_molecule[Z]);

      int k = 0;
      float* formfactors = NULL;
      
      for(int j = 0 ;j< mol->natoms;j++){
	if(mol->atomic_number[j] == Z){

	  p.f[k][0] = 1;
	  p.f[k][1] = 0;
	  /* We have to multiply the position with the dimension of the box because the
	     fourier sample are taken between 0..1 (equivalent to 0..0.5,-0.5..0) */

	  /* For some unknown reason I have to take the negative of the position otherwise the patterns came out inverted.
	     I have absolutely no idea why! It can even be a bug in the NFFT library. I should check this out better
	  */

	  p.x[k*3] = -mol->pos[j*3]/(rs_pixel_x)/nx; 
	  p.x[k*3+1] = -mol->pos[j*3+1]/(rs_pixel_y)/ny; 
	  p.x[k*3+2] = -mol->pos[j*3+2]/(rs_pixel_z)/nz;

    formfactors = mol->atomic_formfactors + 9 * j;

	  k++;
	}
      }
      

#if NFFT_VERSION_ABOVE_3_3
      if(p.flags & PRE_ONE_PSI){
	nfft_precompute_one_psi(&p);
      }
#else
      if(p.nfft_flags & PRE_ONE_PSI){
	nfft_precompute_one_psi(&p);
      }
#endif  
      if(is_element_in_molecule[Z] < 100){
	nfft_adjoint(&p);  
      }else{
	nfft_adjoint(&p);  
      }
      
      if(!opts->delta_atoms){
	multiply_pattern_with_scattering_factor_ff(p.f_hat, formfactors ,nx,ny,nz,
						rs_pixel_x,rs_pixel_y,rs_pixel_z,B);
      }
      for(int k = 0;k<nx*ny*nz;k++){
	sp_real(res->F[k]) += p.f_hat[k][0];
	sp_imag(res->F[k]) += p.f_hat[k][1];
      }   
      nfft_finalize(&p);
    }
  }
  for(int i = 0;i<nx*ny*nz;i++){
    res->ints[i] = sp_cabs(res->F[i])*sp_cabs(res->F[i]);
  }
  sum_patterns(res);
  return res;
}
#endif

Diffraction_Pattern * compute_pattern_by_fft_ff(Molecule * mol, CCD * det, Experiment * exp, float B){
  double alpha_x = atan(det->width/(2.0 * det->distance));
  double alpha_y = atan(det->height/(2.0 * det->distance));
  double alpha_z = atan(det->depth/(2.0 * det->distance));
  double smax_x = sin(alpha_x)/exp->wavelength;
  double smax_y = sin(alpha_y)/exp->wavelength;
  double smax_z = sin(alpha_z)/exp->wavelength;
  double rs_pixel_x = 1/(smax_x*2);  
  double rs_pixel_y = 1/(smax_y*2);  
  double rs_pixel_z = 1/(smax_z*2);  
  fprintf(stderr,"Pixel size x-%e y-%e z-%e\n",rs_pixel_x,rs_pixel_y,rs_pixel_z);
  int nx = det->nx;
  int ny = det->ny;
  int nz = det->nz;
  double max_atoms_radius = 5e-10; /* in meters defines the limit up to which we compute the electron density */
  int x_grid_radius = max_atoms_radius/rs_pixel_x;
  int y_grid_radius = max_atoms_radius/rs_pixel_y;
  int z_grid_radius = max_atoms_radius/rs_pixel_z;
  int total_el = 0;

  Image * rs = sp_image_alloc(nx,ny,nz);
  for(int j = 0 ;j< mol->natoms;j++){
    if(!mol->atomic_number[j]){
      /* undetermined atomic number */
      continue;
    }
    total_el += mol->atomic_number[j];
    float home_x = sp_mod(mol->pos[j*3]/rs_pixel_x,nx);
    float home_y = sp_mod(mol->pos[j*3+1]/rs_pixel_y,ny);
    float home_z = sp_mod(mol->pos[j*3+2]/rs_pixel_z,nz);
    for(int z = home_z-z_grid_radius;z<home_z+z_grid_radius;z++){
      for(int y = home_y-y_grid_radius;y<home_y+y_grid_radius;y++){
	for(int x = home_x-x_grid_radius;x<home_x+x_grid_radius;x++){
	  float ed = 0;
/*	  int z2 = 0;
	  int y2 = 0;
	  int x2 = 0;*/
	  for(int z2 = -1;z2<2;z2+=2){
	    for(int y2 = -1;y2<2;y2+=2){
	      for(int x2 = -1;x2<2;x2+=2){
		float dz = (home_z-(z+z2/2.0))*rs_pixel_z;
		float dy = (home_y-(y+y2/2.0))*rs_pixel_y;
		float dx = (home_x-(x+x2/2.0))*rs_pixel_x;
		float distance = sqrt(dz*dz+dy*dy+dx*dx);
		ed += electron_density_ff(distance, B, mol->atomic_formfactors + 9 * j)*rs_pixel_x*1e10*rs_pixel_y*1e10*rs_pixel_z*1e10;   
	      }
	    }
	  }
	  /* ed is the average of 27 points around the grid point */
	  ed *= rs_pixel_x*1e10*rs_pixel_y*1e10*rs_pixel_z*1e10/8;
	  /* Multiply by the voxel volume so that we have the number in electrons instead of electrons/a^3*/
/*	  if(!isnormal(ed)){
	    abort();
	  }*/
	  ed += sp_cabs(sp_image_get(rs,sp_mod(x,nx),sp_mod(y,ny),sp_mod(z,nz)));
	  sp_image_set(rs,sp_mod(x,nx),sp_mod(y,ny),sp_mod(z,nz),sp_cinit(ed,0));
	}
      }
    }
    //fprintf(stderr,"%d done\n",j);
  }
  fprintf(stderr,"Total electrons - %d\n",total_el);
  sp_image_write(rs,"ed.vtk",0);  
  sp_image_write(rs,"ed.h5",sizeof(real));  
  Image * sf = sp_image_fft(rs);
  sp_image_free(rs);
  rs = sp_image_shift(sf);
  sp_image_free(sf);
  sf = rs;
  Diffraction_Pattern * res = malloc(sizeof(Diffraction_Pattern));
  res->HKL_list_size = nx*ny*nz;
  res->F = malloc(sizeof(Complex)*res->HKL_list_size);
  res->ints = malloc(sizeof(float)*res->HKL_list_size);
  res->HKL_list = malloc(sizeof(float)*3*res->HKL_list_size);
  int i = 0;
  float norm = 1.0;
  for(int z = 0;z<sp_image_z(sf);z++){
    for(int y = 0;y<sp_image_y(sf);y++){
      for(int x = 0;x<sp_image_x(sf);x++){
	res->F[i] = sp_cscale(sp_image_get(sf,x,y,z),norm);
	res->ints[i] = sp_cabs(res->F[i])*sp_cabs(res->F[i]);
	i++;
      }
    }
  }
  sp_image_free(sf);
  return res;
}

Diffraction_Pattern * compute_pattern_on_list_ff(Molecule * mol, float * HKL_list, int HKL_list_size,float B,Experiment * exp,Options * opts){
  int timer = sp_timer_start();
  int i,j;
  float scattering_factor;
  float scattering_vector_length;
  int is_element_in_molecule[ELEMENTS];
  Diffraction_Pattern * res = malloc(sizeof(Diffraction_Pattern));
  int HKL_list_start = 0;
  int HKL_list_end = 0;
  int points_per_percent;
  float * atom_illumination = malloc(sizeof(float)*mol->natoms);
  get_my_loop_start_and_end(HKL_list_size,&HKL_list_start,&HKL_list_end);

  res->F = malloc(sizeof(Complex)*HKL_list_size);
  res->ints = malloc(sizeof(float)*HKL_list_size);
  res->HKL_list = malloc(sizeof(float)*3*HKL_list_size);
  memcpy(res->HKL_list,HKL_list,sizeof(float)*3*HKL_list_size);
  res->HKL_list_size = HKL_list_size;
  for(j = 0 ;j< ELEMENTS;j++){
    is_element_in_molecule[j] = 0;
  }
  for(j = 0 ;j< mol->natoms;j++){
    is_element_in_molecule[mol->atomic_number[j]] = 1;
    atom_illumination[j] = illumination_function(exp,&(mol->pos[j*3]));
  }

  points_per_percent = 1+(HKL_list_end-HKL_list_start)/100;
  for(i = HKL_list_start;i<HKL_list_end;i++){
#ifdef MPI    
    if(is_mpi_master()){
      if(i % points_per_percent == 0){
	if (opts->verbosity_level > 0) {
	  fprintf(stderr,"%f percent done\n",(100.0*(i-HKL_list_start))/(HKL_list_end-HKL_list_start));
	}
      }
    }
#else
      if(i % points_per_percent == 0){
	if (opts->verbosity_level > 0) {
	  fprintf(stderr,"%f percent done\n",(100.0*(i-HKL_list_start))/(HKL_list_end-HKL_list_start));
	}
      }

#endif

    sp_real(res->F[i]) = 0;
    sp_imag(res->F[i]) = 0;
    scattering_vector_length = sqrt(HKL_list[3*i]*HKL_list[3*i]+HKL_list[3*i+1]*HKL_list[3*i+1]+HKL_list[3*i+2]*HKL_list[3*i+2]);

    for(j = 0 ;j< mol->natoms;j++){
      /* Multiply the scattering factor with the illumination function (should it be the square root of it?)*/
      scattering_factor = scatt_factor_ff(scattering_vector_length,j,mol->atomic_formfactors + 9 * j) * sqrt(atom_illumination[j]);
/*      scattering_factor = 1;*/
      float tmp = -2*M_PI*(HKL_list[3*i]*mol->pos[j*3]+HKL_list[3*i+1]*mol->pos[j*3+1]+HKL_list[3*i+2]*mol->pos[j*3+2]);
      if(!opts->delta_atoms){
	sp_real(res->F[i]) += scattering_factor*cos(tmp);
	sp_imag(res->F[i]) += scattering_factor*sin(tmp);
      }else{
	sp_real(res->F[i]) += cos(tmp);
	sp_imag(res->F[i]) += sin(tmp);
      }
    }
    res->ints[i] = sp_cabs(res->F[i])*sp_cabs(res->F[i]);
  }
  syncronize_patterns(res);
  
  if (opts->verbosity_level > 0) {
    printf("%g atoms.pixel/s\n",1.0e6*HKL_list_size*mol->natoms/sp_timer_stop(timer));
  }
  return res;
}


Diffraction_Pattern * vector_compute_pattern_on_list_ff(Molecule * mol, float * HKL_list, int HKL_list_size,float B,Experiment * exp,Options * opts){
#if defined __GNUC__ && !defined __clang__
  int timer = sp_timer_start();
  int i,j;
  float scattering_factor;
  float scattering_vector_length;
  int is_element_in_molecule[ELEMENTS];
  Diffraction_Pattern * res = malloc(sizeof(Diffraction_Pattern));
  int HKL_list_start = 0;
  int HKL_list_end = 0;
  int points_per_percent;
  float * atom_illumination = malloc(sizeof(float)*mol->natoms);
  get_my_loop_start_and_end(HKL_list_size,&HKL_list_start,&HKL_list_end);

  res->F = malloc(sizeof(Complex)*HKL_list_size);
  res->ints = malloc(sizeof(float)*HKL_list_size);
  res->HKL_list = malloc(sizeof(float)*3*HKL_list_size);
  memcpy(res->HKL_list,HKL_list,sizeof(float)*3*HKL_list_size);
  res->HKL_list_size = HKL_list_size;
  for(j = 0 ;j< ELEMENTS;j++){
    is_element_in_molecule[j] = 0;
  }
  for(j = 0 ;j< mol->natoms;j++){
    is_element_in_molecule[mol->atomic_number[j]] = 1;
    atom_illumination[j] = sqrt(illumination_function(exp,&(mol->pos[j*3])));
  }

  points_per_percent = 1+(HKL_list_end-HKL_list_start)/100;
  for(i = HKL_list_start;i<HKL_list_end;i++){
#ifdef MPI    
    if(is_mpi_master()){
      if(i % points_per_percent == 0){
	if (opts->verbosity_level > 0) {
	  fprintf(stderr,"%f percent done\n",(100.0*(i-HKL_list_start))/(HKL_list_end-HKL_list_start));
	}
      }
    }
#else
      if(i % points_per_percent == 0){
	if (opts->verbosity_level > 0) {
	  fprintf(stderr,"%f percent done\n",(100.0*(i-HKL_list_start))/(HKL_list_end-HKL_list_start));
	}
      }

#endif

    sp_real(res->F[i]) = 0;
    sp_imag(res->F[i]) = 0;
    scattering_vector_length = sqrt(HKL_list[3*i]*HKL_list[3*i]+HKL_list[3*i+1]*HKL_list[3*i+1]+HKL_list[3*i+2]*HKL_list[3*i+2]);

    for(j = 0 ;j< 4*(mol->natoms/4);j+=4){
      
      /* Multiply the scattering factor with the illumination function (should it be the square root of it?)*/
      v4sf sf = {scatt_factor_ff(scattering_vector_length, B, mol->atomic_formfactors + j * 9) * sqrt(atom_illumination[j]),
                 scatt_factor_ff(scattering_vector_length, B, mol->atomic_formfactors + (j + 1) * 9) * sqrt(atom_illumination[j+1]),
                 scatt_factor_ff(scattering_vector_length, B, mol->atomic_formfactors + (j + 2) * 9) * sqrt(atom_illumination[j+2]),
                 scatt_factor_ff(scattering_vector_length, B, mol->atomic_formfactors + (j + 3) * 9) * sqrt(atom_illumination[j+3])};

      float tmp[4] = {2*M_PI*(HKL_list[3*i]*-mol->pos[j*3]+HKL_list[3*i+1]*-mol->pos[j*3+1]+HKL_list[3*i+2]*-mol->pos[j*3+2]),
		      2*M_PI*(HKL_list[3*i]*-mol->pos[(j+1)*3]+HKL_list[3*i+1]*-mol->pos[(j+1)*3+1]+HKL_list[3*i+2]*-mol->pos[(j+1)*3+2]),
		      2*M_PI*(HKL_list[3*i]*-mol->pos[(j+2)*3]+HKL_list[3*i+1]*-mol->pos[(j+2)*3+1]+HKL_list[3*i+2]*-mol->pos[(j+2)*3+2]),
		      2*M_PI*(HKL_list[3*i]*-mol->pos[(j+3)*3]+HKL_list[3*i+1]*-mol->pos[(j+3)*3+1]+HKL_list[3*i+2]*-mol->pos[(j+3)*3+2])};
      v4sf phase = __builtin_ia32_loadups(tmp);
      v4sf sin_phase;
      v4sf cos_phase;
      sincos_ps(phase,&sin_phase,&cos_phase);
      if(!opts->delta_atoms){
	sin_phase = __builtin_ia32_mulps(sin_phase,sf);
	cos_phase = __builtin_ia32_mulps(cos_phase,sf);
      }
      __builtin_ia32_storeups(tmp,cos_phase);
      float sum = 0;
      for(int ii = 0;ii<4;ii++){
	sum += tmp[ii];
      }
      sp_real(res->F[i]) += sum;
      __builtin_ia32_storeups(tmp,sin_phase);
      sum = 0;
      for(int ii = 0;ii<4;ii++){
	sum += tmp[ii];
      }
      sp_imag(res->F[i]) += sum;
    }
    for(;j< mol->natoms;j++){
      scattering_factor = scatt_factor_ff(scattering_vector_length, B, mol->atomic_formfactors + j * 9) * sqrt(atom_illumination[j]);
/*      scattering_factor = 1;*/
      float tmp = 2*M_PI*(HKL_list[3*i]*-mol->pos[j*3]+HKL_list[3*i+1]*-mol->pos[j*3+1]+HKL_list[3*i+2]*-mol->pos[j*3+2]);

      sp_real(res->F[i]) += scattering_factor*cos(tmp);
      sp_imag(res->F[i]) += scattering_factor*sin(tmp);
    }
    res->ints[i] = sp_cabs(res->F[i])*sp_cabs(res->F[i]);
  }
  syncronize_patterns(res);
  if (opts->verbosity_level > 0) {
    printf("%g atoms.pixel/s\n",1.0e6*HKL_list_size*mol->natoms/sp_timer_stop(timer));
  }
  return res;
#else
  fprintf(stderr,"Vector computation only supported with when compiling with gcc.\n Using serial version.\n");
  return compute_pattern_on_list( mol, HKL_list, HKL_list_size, B, exp, opts);
#endif
}


/*
Diffraction_Pattern_Py * get_diffraction_pattern_py(Diffraction_Pattern * pat) {
  return pat;
}

Complex_Array_Py * get_diffraction_amplitudes_py(Diffraction_Pattern * pat, Options * opts) {

}
*/

