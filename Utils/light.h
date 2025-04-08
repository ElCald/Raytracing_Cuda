// light.h
#ifndef LIGHT_H
#define LIGHT_H

// Includes
#include "../Geometry/geometry.h"

// Light class
class Light
{
public:
    // Attributes
    Point3D position;
    Vecteur3D intensity;

    // Constructor
    Light(Point3D _position, Vecteur3D _intensity);
};

#endif // LIGHT_H