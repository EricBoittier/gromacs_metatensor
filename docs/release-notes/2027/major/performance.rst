Performance improvements
^^^^^^^^^^^^^^^^^^^^^^^^

.. Note to developers!
   Please use """"""" to underline the individual entries for fixed issues in the subfolders,
   otherwise the formatting on the webpage is messed up.
   Also, please use the syntax :issue:`number` to reference issues on GitLab, without
   a space between the colon and number!


Improved performance of systems that are inhomogeneous along z
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The non-bonded kernels got very inefficient with systems that are have
a very inhomogeneous atom distribution along the z-dimension. Now
all three dimensions are considered when estimating atom density.

:issue:`5622`


The plain pairlist of an ML region only holds pairs of its atoms
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

When every module requesting a plain pairlist (the Metatomic interface does)
gives the atoms it needs, the list is built only for pairs of those atoms,
instead of for every pair of the system within range. For a pair model on an
87-atom peptide in 21,084 atoms of water this cut the plain pairlist from
4.5 to 0.03 ms per step.
