// light.h
#ifndef LIGHT_H
#define LIGHT_H

// Includes
#include "../Geometry/geometry.h"

// Struct struct
struct Light
{
public:
    // Attributes
    Point3D position;
    Vecteur3D intensity;

    // Constructors
    __host__ __device__ Light();
    __host__ __device__ Light(Point3D _position, Vecteur3D _intensity);
};

#endif // LIGHT_H