// Includes
#include "light.h"

// ---- Light Implementation ----

/**
 * default constructor
 */
__host__ __device__ Light::Light() : position(Point3D()), intensity(Vecteur3D()) {}

/**
 * @param _position position of the light
 * @param _intensity intensity of the light
 */
__host__ __device__ Light::Light(Point3D _position, Vecteur3D _intensity) : position(_position), intensity(_intensity) {}