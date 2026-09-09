! Test driver for the unmodified GRASP rcsfgenerate90 GEN routine.
! Compile alongside the upstream sources; no copy of GEN is vendored here.
program gen_reference
   implicit none
   integer :: ansats(15,0:10,0:1), posn(110), posl(110)
   integer :: count, minj, maxj, cf, parity, i, n, l, branch, occupation, slot
   character(len=4096) :: output

   call get_command_argument(1, output)
   read (*, *) count, minj, maxj
   ansats = 0
   parity = 0
   do i = 1, count
      read (*, *) n, l, branch, occupation
      ansats(n,l,branch) = occupation
      parity = mod(parity + l*occupation, 2)
   end do
   slot = 0
   do n = 1, 15
      do l = 0, min(n-1,10)
         slot = slot + 1
         posn(slot) = n
         posl(slot) = l
      end do
   end do
   cf = 0
   open (7, file=trim(output), status='new', form='formatted')
   call gen(ansats, posn, posl, 20, cf, .true., minj, maxj, parity)
   close (7)
   write (*, *) cf
end program gen_reference
