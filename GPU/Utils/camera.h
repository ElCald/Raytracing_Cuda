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
    __host__ __device__ Camera(Point3D _position, Vecteur3D _direction, double _fov, int _width, int _height);

    // Methods
    __host__ __device__ Ray generateRay(int x, int y) const;
    __host__ __device__ void rotateX(double angle);
    __host__ __device__ void rotateY(double angle);
    __host__ __device__ void rotatePosX(double angle, Point3D centre);
    __host__ __device__ void rotatePosY(double angle, Point3D centre);
    __host__ __device__ void rotatePosZ(double angle, Point3D centre);
    __host__ __device__ void translateX(double direc);
    __host__ __device__ void translateY(double direc);
    __host__ __device__ void translateZ(double direc);
};

#endif // CAMERA_H