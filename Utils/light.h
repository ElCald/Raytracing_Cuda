#ifndef LIGHT_H
#define LIGHT_H

#include "../Geometry/geometry.h"

class Light
{
public:
    Light(Point3D _position, Vecteur3D _intensity);
    Point3D position;
    Vecteur3D intensity;
};

#endif