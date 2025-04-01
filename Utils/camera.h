#ifndef CAMERA_H
#define CAMERA_H

#include "../Geometry/geometry.h"
#include "../GeometricObjects/formes.h"
#include <vector>

class Camera
{
public:
    Camera(Point3D _position, Vecteur3D _direction, double _fov, int _width, int _height);

    Ray generateRay(int x, int y) const;

    void rotateX(double angle);
    void rotateY(double angle);


    void rotatePosX(double angle, Point3D centre);
    void rotatePosY(double angle, Point3D centre);
    void rotatePosZ(double angle, Point3D centre);


    void translateX(double direc);
    void translateY(double direc);
    void translateZ(double direc);

    int width, height;
    double fov;
    Point3D position;
    Vecteur3D direction;
    Vecteur3D right, up;
};





#endif
