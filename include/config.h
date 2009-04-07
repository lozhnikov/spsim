/*
 * Copyright (c) 2006 Filipe Maia
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
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


#ifndef _CONFIG_H_
#define _CONFIG_H_ 1

/* All units are SI:
 
time: seconds
length: meter
energy: joules
*/

#define CHEM_FORMULA 1
#define PDB 2

#define BOX_SPHERICAL 1
#define BOX_PARALLEL 2

#define VERSION "2.0"

typedef struct{
  int n_elements;
  int * atomic_number;
  int * quantity;
}Chem_Formula;

typedef struct{
  int natoms;
  int * atomic_number;
  /* real physical position */
  float * pos;
  /* may optionally contain the boundaries of pos */
}Molecule;

typedef struct{
  float wavelength; /* in meters */
  float exposure_time; /* in seconds */
  float beam_intensity; /* In photons/(time.area) */
  float beam_center_x; /* in meters */
  float beam_center_y; /* in meters */
  float beam_fwhm; /* in meters. Equivalent aproximately
			     to 2.355*sigma or 2*sqrt(2*ln(2))*sigma 
			     Implicitly assumes gaussian pulse profile
			  */
}Experiment;

typedef struct{
  /* number of photons which fall on each pixel */
  float * photons_per_pixel;
  /* You should multiply the diffraction pattern by this to
     get the real life values in intensity/area
     Multiply with pixel area to get the total intensity 
  */
  float * thomson_correction;
  /* pixel solid angle */
  float * solid_angle;
  /* effective photon count quantum efficiency including poisson noise */
  /* the poisson noise is calculated after multiplying the photons per
   pixel by the quantum efficiency */
  float * photon_count;

  /* electrons generated by the conversion of the photon count*/
  float * electrons_per_pixel;

  /* real life output and ideal noiseless output */
  float * real_output;
  float * noiseless_output;

  float distance; /* in meters */
  float width; /* in meters */
  float height; /* in meters */
  float depth; /* in meters */
  float pixel_width; /* in meters */
  float pixel_height; /* in meters */
  float pixel_depth; /* in meters */
  float quantum_efficiency; /* wavelength dependent */
  float electron_hole_production_energy; /* in Joules */
  float readout_noise; /* in electrons/pixel */
  float dark_current; /*in electrons/(pixel.second) */
  float linear_full_well; /* in electrons */  
  int binning_x;
  int binning_y;
  int binning_z;
  int nx;
  int ny;
  int nz;
  float maximum_value;
  int spherical;
  float gaussian_blurring;
  float real_space_blurring;
  /* Center of the detector in relation to the beam (by definition the beam follows the z axis) */
  float center_x;
  float center_y;
}CCD;


typedef struct {
  /* number of dimensions (only 2 supported at the moment) */
  int n_dims;
  /* Can only be CHEM_FORMULA for the moment, meaning chemical formula. */
  int input_type;
  /*
    All atoms will be placed on top of eachother so a spherically
    symmetrical diffracton pattern is generated. 
  */
  Chem_Formula * chem_formula;
  Experiment * experiment;
  CCD * detector;  
  char pdb_filename[1024];
  char sf_filename[1024];
  char hkl_grid_filename[1024];
  int box_type; /* spherical or parallelepipedic */
  float box_dimension; /* diameter in case of spherical or side in case of parallelepipedic */
  int use_fft_for_sf;
  int use_nfft_for_sf;
  float b_factor;
  float euler_orientation[3];
  int random_orientation;
  int n_patterns;
  int vectorize;
}Options;


void read_options_file(char * filename, Options * opt);
void parse_options(int argc, char ** argv, Options * opt);
Options * set_defaults(void);
void write_options_file(char * filename, Options * res);

#endif 
