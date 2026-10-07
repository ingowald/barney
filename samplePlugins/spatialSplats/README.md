# Barney Sample "Splats" Plugin

This subdirectory serves as a sample of how to write external-source
plugins for barney. External-source plugins are a mechanism in which
barney can be told to include some additional - "external" -
out-of-source subdirectories such that this out-of-source code can
define additional geometry and spatial field types that mainline
barney does not know about.

In particular, this
current sample defines a new geometry type 'sphere' (with the same
paramters that typcal anari::sphere geometry would have), but instead
of actual 3D spheres it creates, for each logical spheres, a
volumetric spherical primitives with same origin and radius, but
linear density falloff.

Note this is obviously not very useful as an actual geometry per se;
it only shows - with a very simple, artificial example - how to write
geometries and spatial fields in a way that barney can build and
recognize them. It is expected that whoever writes their own plugins
will fill these out with more useful behavior. The reason we export
this as a 'sphere' geometry is to allow simple testing by existing
anari viewers that can already render spheres - if this plugin is
built into barney it will actually override the built-in geometry of
the same type, so the viewer will create and parameterize this
plugin's spheres instead, without even knowing about it.



