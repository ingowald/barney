
#ifndef BARNEY_LIBRARY_INTERFACE_H
#define BARNEY_LIBRARY_INTERFACE_H

#ifdef ANARI_LIBRARY_BARNEY_STATIC_DEFINE
#  define BARNEY_LIBRARY_INTERFACE
#  define ANARI_LIBRARY_BARNEY_NO_EXPORT
#else
#  ifndef BARNEY_LIBRARY_INTERFACE
#    ifdef anari_library_barney_EXPORTS
        /* We are building this library */
#      define BARNEY_LIBRARY_INTERFACE __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define BARNEY_LIBRARY_INTERFACE __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef ANARI_LIBRARY_BARNEY_NO_EXPORT
#    define ANARI_LIBRARY_BARNEY_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef ANARI_LIBRARY_BARNEY_DEPRECATED
#  define ANARI_LIBRARY_BARNEY_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef ANARI_LIBRARY_BARNEY_DEPRECATED_EXPORT
#  define ANARI_LIBRARY_BARNEY_DEPRECATED_EXPORT BARNEY_LIBRARY_INTERFACE ANARI_LIBRARY_BARNEY_DEPRECATED
#endif

#ifndef ANARI_LIBRARY_BARNEY_DEPRECATED_NO_EXPORT
#  define ANARI_LIBRARY_BARNEY_DEPRECATED_NO_EXPORT ANARI_LIBRARY_BARNEY_NO_EXPORT ANARI_LIBRARY_BARNEY_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef ANARI_LIBRARY_BARNEY_NO_DEPRECATED
#    define ANARI_LIBRARY_BARNEY_NO_DEPRECATED
#  endif
#endif

#endif /* BARNEY_LIBRARY_INTERFACE_H */
