#ifndef CAMERA_H
#define CAMERA_H

#include "../Geometry/geometry.h"
#include <vector>



class Camera
{
public:
    Camera(Point3D _position, Vecteur3D _direction, double _fov, int _width, int _height);

    Ray generateRay(int x, int y) const;

    int width, height;
    double fov;
    Point3D position;
    Vecteur3D direction;
    Vecteur3D right, up;
};


class Light
{
public :
    Light(Point3D _position, Vecteur3D _intensity);
    Point3D position;
    Vecteur3D intensity;

};

#endif
