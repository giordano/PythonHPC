program traffic

  use mpi

  use trafficlib

  implicit none

! Set the size of the road

  integer :: ncell = 5120000
  integer :: maxiter, printfreq

  integer rank, size, ierr, nlocal, rankup, rankdown
  integer, dimension(MPI_STATUS_SIZE) :: status

  integer :: i, iter, nmove, nmovelocal, ncars, ncars_local
  real    :: density

  integer, allocatable, dimension(:) :: newroad, oldroad

  double precision :: tstart, tstop

  maxiter = 200
  printfreq = maxiter/10

! Set target density of cars

  density = 0.52

! Start message passing system

  call MPI_Init(ierr)

! Compute size and rank

  call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

  ncell = ncell * size
  nlocal = ncell/size

  if (rank == 0) then

     write(*,*) 'Running message-passing traffic model'
     write(*,*)
     write(*,*) 'Length of road is ', ncell
     write(*,*) 'Number of iterations is ', maxiter
     write(*,*) 'Target density of cars is ', density
     write(*,*) 'Running on ', size, ' processes'

  end if

! Allocate arrays

  allocate(newroad(0:nlocal+1))
  allocate(oldroad(0:nlocal+1))

  do i = 1, nlocal
     oldroad(i) = 0
     newroad(i) = 0
  end do

  if (rank == 0) then
    ! Initialise road accordingly using random number generator
    write(*,*) 'Initialising ...'
  end if

  call rinit(5743)
  ncars_local = 0
  do i = 0, rank
    ncars_local = initroad(oldroad(1), nlocal, density)
  end do
  call MPI_Reduce(ncars_local, ncars, 1, MPI_INTEGER, MPI_SUM, &
      0, MPI_COMM_WORLD, ierr)

  if (rank == 0) then
     write(*,*) '... done'

     write(*,*) 'Actual density of cars is ', float(ncars)/float(ncell)
     write(*,*) 'Scattering data ...'

  end if

  if (rank == 0) then

     write(*,*) '... done'
     write(*,*)

  end if

! Compute neighbours

  rankup   = mod(rank + 1, size)
  rankdown = mod(rank + size - 1, size)

  tstart = MPI_Wtime()

  do iter = 1, maxiter

! Implement halo swaps which now includes boundary conditions

     call MPI_Sendrecv(oldroad(nlocal), 1, MPI_INTEGER, rankup, 1,   &
                       oldroad(0),      1, MPI_INTEGER, rankdown, 1, &
                       MPI_COMM_WORLD, status, ierr);

     call MPI_Sendrecv(oldroad(1),        1, MPI_INTEGER, rankdown, 1,  &
                       oldroad(nlocal+1), 1, MPI_INTEGER, rankup, 1,    &
                       MPI_COMM_WORLD, status, ierr);

     nmovelocal = updateroad(newroad, oldroad, nlocal)

! Globally sum the value

     call MPI_Allreduce(nmovelocal, nmove, 1, MPI_INTEGER, MPI_SUM, &
                        MPI_COMM_WORLD, ierr)

! Copy new to old array

     do i = 1, nlocal

        oldroad(i) = newroad(i)

     end do

     if (mod(iter, printfreq) == 0) then

        if (rank == 0) write(*,*) 'At iteration ', iter, &
             ' average velocity is ', float(nmove)/float(ncars)
     end if

  end do

  tstop = MPI_Wtime()

  deallocate(oldroad)
  deallocate(newroad)

  if (rank == 0) then

     write(*,*)
     write(*,*) 'Finished'
     write(*,*)
     write(*,*) 'Time taken was  ', tstop - tstart, ' seconds'
     write(*,*) 'Update rate was ', &
                 1.d-6*float(ncell)*float(maxiter)/(tstop-tstart), ' MCOPs'
     write(*,*)

  end if

  call MPI_Finalize(ierr)

end program traffic
