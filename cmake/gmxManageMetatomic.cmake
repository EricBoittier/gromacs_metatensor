#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2024- The GROMACS Authors
# and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
# Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# https://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at https://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.


gmx_option_multichoice(GMX_METATOMIC
  "Enable interface to metatomic atomistic models"
    AUTO
    AUTO TORCH OFF
)

# The C API does not use libtorch. Leave GMX_TORCH to NNPot.
set(GMX_METATOMIC_ACTIVE OFF)

if(NOT GMX_METATOMIC STREQUAL "OFF")
    # metatomic-config pulls metatensor. The two packages live in separate
    # prefixes when installed from wheels, so both cmake dirs must be on
    # CMAKE_PREFIX_PATH.
    # metatomic-config calls find_package(metatensor REQUIRED). Resolve
    # metatensor first so a missing dependency stays a soft failure in AUTO.
    # metatensor-core 0.3.0 has never been released; metatomic-core itself
    # requires 0.2.4 (REQUIRED_METATENSOR_VERSION), so match that floor.
    find_package(metatensor 0.2.4 CONFIG QUIET)
    if(metatensor_FOUND)
        # Deliberately no version floor. metatomic-core is unreleased
        # (metatensor/metatomic#232), and its config file pairs
        # SameMinorVersion compatibility with a version that create_development_
        # version() derives by bumping the minor: a source build of the 0.1.0
        # tree reports 0.2.0-dev, so any floor we could write here rejects
        # either every dev build or every release. The STATUS line below
        # records what was actually found instead.
        # See src/gromacs/applied_forces/metatomic/docs/dependencies.md.
        find_package(metatomic CONFIG QUIET)
    endif()
    if(metatomic_FOUND)
        set(GMX_METATOMIC_ACTIVE ON)
        # Record the versions: while metatomic-core is unreleased, the build log
        # is the only place the revision in use is written down.
        message(STATUS "Found metatomic ${metatomic_VERSION} (metatensor ${metatensor_VERSION}): "
                       "Metatomic potential support enabled.")

        list(APPEND GMX_COMMON_LIBRARIES metatomic::shared)

        # Pre-installed libraries outside the GROMACS prefix need their own
        # RPATH entry. Pip wheels put metatomic and metatensor in different
        # site-packages directories.
        foreach(_mta_target IN ITEMS metatomic::shared metatensor::shared)
            if(NOT TARGET ${_mta_target})
                continue()
            endif()
            unset(_mta_location)
            string(TOUPPER "${CMAKE_BUILD_TYPE}" _mta_config)
            foreach(_mta_property IN ITEMS IMPORTED_LOCATION_${_mta_config} IMPORTED_LOCATION IMPORTED_LOCATION_RELEASE
                                           IMPORTED_LOCATION_RELWITHDEBINFO IMPORTED_LOCATION_DEBUG)
                get_target_property(_mta_location ${_mta_target} ${_mta_property})
                if(_mta_location)
                    break()
                endif()
            endforeach()
            if(_mta_location)
                get_filename_component(_mta_libdir "${_mta_location}" DIRECTORY)
                list(APPEND CMAKE_INSTALL_RPATH "${_mta_libdir}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES CMAKE_INSTALL_RPATH)
    elseif(GMX_METATOMIC STREQUAL "TORCH")
        message(FATAL_ERROR
            "metatomic with metatensor >= 0.2.4 was not found. Install "
            "libmetatomic and add the directories containing "
            "metatomic-config.cmake and metatensor-config.cmake to "
            "CMAKE_PREFIX_PATH. Building metatomic-core is described in "
            "src/gromacs/applied_forces/metatomic/docs/dependencies.md.")
    else() # AUTO
        message(STATUS "metatomic not found. Metatomic potential support will be disabled.")
    endif()
endif()
