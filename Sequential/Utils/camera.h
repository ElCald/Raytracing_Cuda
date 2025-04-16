// camera.h
#ifndef CAMERA_H
#define CAMERA_H

// Includes
#include "../Geometry/geometry.h"
#include "../GeometricsObjects/forms.h"
#include <vector>

// Camera class
class Camera
{
public:
    // Attributes
    Point3D position;
    Vecteur3D direction;
    double fov;
    int width, height;
    Vecteur3D right, up;

    // Constructor
    Camera(Point3D _position, Vecteur3D _direction, double _fov, int _width, int _height);

    // Methods
    Ray generateRay(int x, int y) const;
    void rotateX(double angle);
    void rotateY(double angle);
    void rotatePosX(double angle, Point3D centre);
    void rotatePosY(double angle, Point3D centre);
    void rotatePosZ(double angle, Point3D centre);
    void translateX(double direc);
    void translateY(double direc);
    void translateZ(double direc);
};

#endif // CAMERA_H