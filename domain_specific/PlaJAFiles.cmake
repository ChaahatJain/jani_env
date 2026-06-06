set(DOMAIN_SPECIFIC_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/blocksworld.h
        ${CMAKE_CURRENT_LIST_DIR}/domain_specific.cpp ${CMAKE_CURRENT_LIST_DIR}/domain_specific.h 
)

include(${CMAKE_CURRENT_LIST_DIR}/manual_policies/PlaJAFiles.cmake)
list(APPEND DOMAIN_SPECIFIC_SOURCES ${MANUAL_POLICY_SOURCES})