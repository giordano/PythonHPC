#include <stdlib.h>
#include <stdio.h>

#include <mpi.h>

#include "uni.h"
#include "traffic.h"

int main(int argc, char **argv)
{
  // Set the size of the road

  long int ncell = 5120000;

  int *oldroad, *newroad;

  int i, iter, nmove, nmovelocal, ncars;
  int maxiter, printfreq;

  float density;

  double tstart, tstop;

  MPI_Status status;
  int rank, size, nlocal, rankup, rankdown;

  maxiter = 1000 /* 1.024e10/((double) ncell) */;
  printfreq = maxiter/10;

  // Set target density of cars

  density = 0.52;

  // Start MPI

  MPI_Init(NULL, NULL);

  // Find size and rank

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ncell *= size;
  nlocal = ncell/size;

  if (rank == 0)
    {
      printf("Running message-passing traffic model\n");
      printf("\nLength of road is %ld\n", ncell);
      printf("Number of iterations is %d \n", maxiter);
      printf("Target density of cars is %f \n", density);
      printf("Running on %d processes\n", size);
    }

  oldroad = (int *) malloc((nlocal+2)*sizeof(int));
  newroad = (int *) malloc((nlocal+2)*sizeof(int));

  for (i=1; i <= nlocal; i++)
    {
      oldroad[i] = 0;
      newroad[i] = 0;
    }

  if (rank == 0) {

      // Initialise road accordingly using random number generator

      printf("Initialising road ...\n");
  }

  // seed random number generator
  rinit(SEED);
  int ncars_local = 0;
  for (int i=0; i<=rank; i++) {
      ncars_local = initroad(&oldroad[1], nlocal, density);
  }
  MPI_Reduce(&ncars_local, &ncars, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
      printf("...done\n");
      printf("Actual density is %f\n", (float) ncars / (float) ncell);
      printf("Scattering data ...\n");
    }


  if (rank == 0)
      {
          printf("... done\n\n");
      }

  // Compute neighbours

  rankup   = (rank + 1) % size;
  rankdown = (rank + size - 1) % size;

  MPI_Barrier(MPI_COMM_WORLD);

  tstart = MPI_Wtime();

  for (iter=1; iter<=maxiter; iter++)
    {

      // Implement halo swaps which now includes boundary conditions

      MPI_Sendrecv(&oldroad[nlocal], 1, MPI_INT, rankup, 1,
		   &oldroad[0],      1, MPI_INT, rankdown, 1,
		   MPI_COMM_WORLD, &status);

      MPI_Sendrecv(&oldroad[1],        1, MPI_INT, rankdown, 1,
		   &oldroad[nlocal+1], 1, MPI_INT, rankup, 1,
		   MPI_COMM_WORLD, &status);

      // Apply CA rules to all cells

      nmovelocal = updateroad(newroad, oldroad, nlocal);

      // Globally sum the value

      MPI_Allreduce(&nmovelocal, &nmove, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      // Copy new to old array

      for (i=1; i<=nlocal; i++)
	{
	  oldroad[i] = newroad[i];
	}

      if (iter%printfreq == 0)
	{
	  if (rank == 0)
	    {
	      printf("At iteration %d average velocity is %f \n",
		     iter, (float) nmove / (float) ncars);
	    }
	}
    }

  MPI_Barrier(MPI_COMM_WORLD);

  tstop = MPI_Wtime();

  free(oldroad);
  free(newroad);

  if (rank == 0)
    {

      printf("\nFinished\n");
      printf("\nTime taken was  %f seconds\n", tstop-tstart);
      printf("Update rate was %f MCOPs\n\n", \
      1.e-6*((double) ncell)*((double) maxiter)/(tstop-tstart));
    }

  // Finish

  MPI_Finalize();
}
