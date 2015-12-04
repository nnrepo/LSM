

SET(CMAKE_INCLUDE_PATH "${BOOST_DIR}/build/" ${CMAKE_INCLUDE_PATH})
SET(CMAKE_LIBRARY_PATH "${BOOST_DIR}/build/lib/" ${CMAKE_LIBRARY_PATH})

SET(BOOST_LIBRARYDIR "${BOOST_DIR}/build/lib/")
SET(BOOST_INCLUDEDIR "${BOOST_DIR}/build/include/")
SET(BOOST_ROOT "${BOOST_DIR}/")


SET(Boost_NO_SYSTEM_PATHS ON)
set(Boost_USE_MULTITHREADED OFF)
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MU/STATIC_RUNTIME OFF) 
SET(BOOST_MIN_VERSION "1.55.0")
add_definitions( -DBOOST_SIGNALS_NO_DEPRECATION_WARNING )
find_package (Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS system filesystem date_time thread serialization timer chrono)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})    
    link_directories(${Boost_LIBRARY_DIRS})
    set (EXTRA_LIBS ${EXTRA_LIBS} ${Boost_LIBRARIES})
    
    ADD_DEFINITIONS( "-DHAS_BOOST" )
    MESSAGE("Boost found")
    MESSAGE( "${Boost_INCLUDE_DIRS}" )
    MESSAGE( "${Boost_LIBRARY_DIRS}" )
endif (Boost_FOUND)
